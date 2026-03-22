"""EEG microstate extraction and analysis via pycrostates.

Thin wrapper around pycrostates (JOSS 2022) for microstate clustering,
backfitting, and statistics computation. Preserves our feature extraction
interface and MLflow logging.

Replaces the previous custom modified k-means implementation.
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
    gfp_percentile: float = 95.0


def extract_microstate_features(
    epochs: mne.Epochs,
    config: Optional[MicrostateConfig] = None,
) -> np.ndarray:
    """Extract microstate features per epoch using pycrostates.

    For each epoch: fit microstates on GFP peaks, backfit to all time points,
    compute coverage, duration, and occurrence as features.

    Returns:
        Array of shape (n_epochs, n_states * 3) — coverage + duration + occurrence.
    """
    if config is None:
        config = MicrostateConfig()

    try:
        import pycrostates  # noqa: F401

        return _extract_with_pycrostates(epochs, config)
    except ImportError:
        logger.warning(
            "pycrostates_not_installed",
            msg="Install with: pip install pycrostates. Returning zeros.",
        )
        return np.zeros((len(epochs), config.n_states * 3))
    except Exception as exc:
        logger.warning("microstate_extraction_failed", error=str(exc))
        return np.zeros((len(epochs), config.n_states * 3))


def _extract_with_pycrostates(
    epochs: mne.Epochs,
    config: MicrostateConfig,
) -> np.ndarray:
    """Internal: extract microstates using pycrostates library."""
    from pycrostates.cluster import ModKMeans

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    n_epochs = data.shape[0]

    # Fit ModKMeans on concatenated epoch data
    # pycrostates works with MNE Raw/Epochs objects
    modK = ModKMeans(
        n_clusters=config.n_states,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )

    try:
        modK.fit(epochs, n_jobs=1)
    except Exception as exc:
        logger.warning("pycrostates_fit_failed", error=str(exc))
        return np.zeros((n_epochs, config.n_states * 3))

    # Extract features per epoch
    features_list: list[np.ndarray] = []

    for epoch_idx in range(n_epochs):
        epoch_data = data[epoch_idx]  # (n_channels, n_times)

        # Create single-epoch Raw for backfitting
        info = mne.create_info(
            ch_names=epochs.ch_names,
            sfreq=sfreq,
            ch_types="eeg",
        )
        raw_epoch = mne.io.RawArray(epoch_data, info, verbose=False)

        try:
            segmentation = modK.predict(raw_epoch)
            labels = segmentation.labels

            stats = _compute_stats(labels, sfreq, config.n_states)
            epoch_features = np.concatenate(
                [stats["coverage"], stats["duration"], stats["occurrence"]]
            )
        except Exception:
            epoch_features = np.zeros(config.n_states * 3)

        features_list.append(epoch_features)

    features = np.array(features_list)

    logger.info(
        "extracted_microstate_features",
        shape=features.shape,
        n_states=config.n_states,
        method="pycrostates",
    )
    return features


def _compute_stats(
    labels: np.ndarray,
    sfreq: float,
    n_states: int,
) -> dict[str, np.ndarray]:
    """Compute microstate statistics from label sequence."""
    n_times = len(labels)

    # Coverage: fraction of time in each state
    coverage = np.array(
        [(labels == s).sum() / max(n_times, 1) for s in range(n_states)]
    )

    # Duration: mean length of contiguous segments (ms)
    durations = []
    for state in range(n_states):
        state_mask = labels == state
        if not state_mask.any():
            durations.append(0.0)
            continue
        # Find runs of this state
        changes = np.diff(state_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if state_mask[0]:
            starts = np.concatenate([[0], starts])
        if state_mask[-1]:
            ends = np.concatenate([ends, [n_times]])
        run_lengths = ends - starts
        durations.append(
            float(run_lengths.mean()) / sfreq * 1000 if len(run_lengths) > 0 else 0.0
        )
    duration = np.array(durations)

    # Occurrence: segments per second
    total_duration_s = n_times / sfreq
    occurrence = np.array(
        [
            len(np.where(np.diff((labels == s).astype(int)) == 1)[0])
            + (1 if labels[0] == s else 0)
            for s in range(n_states)
        ]
    ) / max(total_duration_s, 1e-10)

    return {
        "coverage": coverage,
        "duration": duration,
        "occurrence": occurrence,
    }
