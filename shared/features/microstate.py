"""EEG microstate extraction and analysis.

Microstates are brief (~60-120ms) periods of stable scalp topography.
Typically 4-7 canonical microstates (A, B, C, D, ...) are identified
via modified k-means clustering of GFP peaks.
"""

from dataclasses import dataclass
from typing import Optional

import mne
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MicrostateConfig:
    """Configuration for microstate extraction."""

    n_states: int = 4
    max_iter: int = 1000
    random_state: int = 42
    min_peak_distance: int = 2
    gfp_percentile: float = 95.0


def compute_gfp(data: np.ndarray) -> np.ndarray:
    """Compute Global Field Power.

    Args:
        data: EEG data of shape (n_channels, n_times).

    Returns:
        GFP time series of shape (n_times,).
    """
    return data.std(axis=0)


def find_gfp_peaks(
    gfp: np.ndarray,
    min_distance: int = 2,
    percentile: float = 95.0,
) -> np.ndarray:
    """Find peaks in GFP above a percentile threshold."""
    from scipy.signal import find_peaks

    threshold = np.percentile(gfp, percentile)
    peaks, _ = find_peaks(gfp, height=threshold, distance=min_distance)

    logger.info(
        "found_gfp_peaks",
        n_peaks=len(peaks),
        threshold=threshold,
    )
    return peaks


def modified_kmeans(
    data: np.ndarray,
    n_states: int = 4,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Modified k-means clustering for microstate extraction.

    Polarity-invariant clustering: maps with opposite polarity
    are assigned to the same cluster.

    Args:
        data: EEG data at GFP peaks, shape (n_channels, n_peaks).
        n_states: Number of microstates to extract.
        max_iter: Maximum iterations.
        random_state: Random seed.

    Returns:
        maps: Microstate topographies, shape (n_states, n_channels).
        labels: Cluster assignment for each peak, shape (n_peaks,).
    """
    rng = np.random.RandomState(random_state)
    n_channels, n_peaks = data.shape

    # Normalize each map to unit length
    norms = np.linalg.norm(data, axis=0, keepdims=True)
    norms[norms == 0] = 1
    data_norm = data / norms

    # Initialize with random peaks
    init_idx = rng.choice(n_peaks, n_states, replace=False)
    maps = data_norm[:, init_idx].T  # (n_states, n_channels)

    for iteration in range(max_iter):
        # Assign each peak to closest map (polarity-invariant)
        correlations = np.abs(maps @ data_norm)  # (n_states, n_peaks)
        labels = correlations.argmax(axis=0)

        # Update maps
        new_maps = np.zeros_like(maps)
        for state in range(n_states):
            members = data_norm[:, labels == state]
            if members.shape[1] == 0:
                continue

            # SVD to find dominant direction (polarity-invariant)
            u, s, _ = np.linalg.svd(members, full_matrices=False)
            new_maps[state] = u[:, 0]

        # Check convergence
        similarity = np.abs(np.sum(maps * new_maps, axis=1))
        if np.all(similarity > 0.999):
            logger.info("kmeans_converged", iteration=iteration)
            break

        maps = new_maps

    logger.info(
        "extracted_microstates",
        n_states=n_states,
        n_peaks=n_peaks,
        iterations=iteration + 1,
    )
    return maps, labels


def backfit_microstates(
    data: np.ndarray,
    maps: np.ndarray,
) -> np.ndarray:
    """Assign microstate labels to all time points.

    Args:
        data: Continuous EEG, shape (n_channels, n_times).
        maps: Microstate maps, shape (n_states, n_channels).

    Returns:
        Labels for each time point, shape (n_times,).
    """
    norms = np.linalg.norm(data, axis=0, keepdims=True)
    norms[norms == 0] = 1
    data_norm = data / norms

    correlations = np.abs(maps @ data_norm)
    return correlations.argmax(axis=0)


def compute_microstate_stats(
    labels: np.ndarray,
    sfreq: float,
    n_states: int = 4,
) -> dict[str, np.ndarray]:
    """Compute microstate statistics.

    Returns:
        Dictionary with:
        - coverage: fraction of time in each state
        - duration: mean duration (ms) per state
        - occurrence: occurrences per second per state
        - transition_matrix: state transition probabilities
    """
    n_times = len(labels)

    # Coverage
    coverage = np.array([(labels == s).sum() / n_times for s in range(n_states)])

    # Duration: mean length of contiguous segments
    durations = []
    for state in range(n_states):
        runs = (
            np.diff(
                np.where(
                    np.concatenate(
                        ([labels[0] != state], labels[:-1] != labels[1:], [True])
                    )
                )[0]
            )[::2]
            if state in labels
            else []
        )
        if len(runs) > 0:
            durations.append(np.mean(runs) / sfreq * 1000)  # ms
        else:
            durations.append(0.0)
    duration = np.array(durations)

    # Occurrence rate
    segment_changes = np.sum(np.diff(labels) != 0)
    total_segments = segment_changes + 1
    total_duration_s = n_times / sfreq
    occurrence = np.array(
        [coverage[s] * total_segments / total_duration_s for s in range(n_states)]
    )

    # Transition matrix
    transitions = np.zeros((n_states, n_states))
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            transitions[labels[i], labels[i + 1]] += 1

    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transitions / row_sums

    return {
        "coverage": coverage,
        "duration": duration,
        "occurrence": occurrence,
        "transition_matrix": transition_matrix,
    }


def extract_microstate_features(
    epochs: mne.Epochs,
    config: Optional[MicrostateConfig] = None,
) -> np.ndarray:
    """Extract microstate features per epoch.

    For each epoch: extract microstates, compute coverage, duration,
    and occurrence as features.

    Returns:
        Array of shape (n_epochs, n_states * 3) — coverage + duration + occurrence.
    """
    if config is None:
        config = MicrostateConfig()

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]

    # Fit microstates on concatenated GFP peaks across all epochs
    all_data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    gfp = compute_gfp(all_data)
    peaks = find_gfp_peaks(
        gfp,
        min_distance=config.min_peak_distance,
        percentile=config.gfp_percentile,
    )

    if len(peaks) < config.n_states:
        logger.warning("too_few_gfp_peaks", n_peaks=len(peaks))
        return np.zeros((data.shape[0], config.n_states * 3))

    peak_data = all_data[:, peaks]
    maps, _ = modified_kmeans(
        peak_data,
        n_states=config.n_states,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )

    # Extract features per epoch
    features_list: list[np.ndarray] = []
    for epoch_data in data:
        labels = backfit_microstates(epoch_data, maps)
        stats = compute_microstate_stats(labels, sfreq, config.n_states)
        epoch_features = np.concatenate(
            [
                stats["coverage"],
                stats["duration"],
                stats["occurrence"],
            ]
        )
        features_list.append(epoch_features)

    features = np.array(features_list)

    logger.info(
        "extracted_microstate_features",
        shape=features.shape,
        n_states=config.n_states,
    )
    return features
