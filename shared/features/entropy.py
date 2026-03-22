"""Entropy and complexity feature extraction via antropy.

Thin wrapper around antropy (numba-accelerated) for EEG complexity features.
Useful for addiction (sample entropy), sleep (permutation entropy),
and encoding detection (spectral entropy).

antropy provides: permutation entropy, spectral entropy, SVD entropy,
sample entropy, approximate entropy, Hjorth parameters, zero-crossings,
Lempel-Ziv complexity, Higuchi fractal dimension, DFA.
"""

from dataclasses import dataclass
from typing import Optional

import mne
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EntropyConfig:
    """Configuration for entropy feature extraction."""

    # Which features to compute
    permutation_entropy: bool = True
    spectral_entropy: bool = True
    sample_entropy: bool = True
    hjorth_params: bool = True
    lziv_complexity: bool = True

    # Parameters
    perm_order: int = 3
    perm_delay: int = 1
    sample_m: int = 2
    sample_tolerance: Optional[float] = None  # None = auto (0.2 * std)


def compute_entropy_features(
    epochs: mne.Epochs,
    config: Optional[EntropyConfig] = None,
) -> np.ndarray:
    """Extract entropy/complexity features per epoch per channel.

    Returns:
        Array of shape (n_epochs, n_features).
    """
    if config is None:
        config = EntropyConfig()

    try:
        import antropy as ant
    except ImportError:
        logger.warning("antropy_not_installed", msg="Returning empty features")
        return np.zeros((len(epochs), 0))

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = data.shape
    sf = epochs.info["sfreq"]

    all_features: list[np.ndarray] = []

    for epoch_idx in range(n_epochs):
        epoch_features: list[float] = []

        for ch_idx in range(n_channels):
            signal = data[epoch_idx, ch_idx]

            if config.permutation_entropy:
                pe = ant.perm_entropy(
                    signal,
                    order=config.perm_order,
                    delay=config.perm_delay,
                    normalize=True,
                )
                epoch_features.append(pe)

            if config.spectral_entropy:
                se = ant.spectral_entropy(
                    signal,
                    sf=sf,
                    method="welch",
                    normalize=True,
                )
                epoch_features.append(se)

            if config.sample_entropy:
                try:
                    samp_e = ant.sample_entropy(signal)
                    epoch_features.append(samp_e)
                except Exception:
                    epoch_features.append(0.0)

            if config.hjorth_params:
                mobility, complexity = ant.hjorth_params(signal)
                epoch_features.extend([mobility, complexity])

            if config.lziv_complexity:
                # Binarize signal around median for LZC
                binary = (signal > np.median(signal)).astype(int)
                lzc = ant.lziv_complexity(binary, normalize=True)
                epoch_features.append(lzc)

        all_features.append(epoch_features)

    features = np.array(all_features)

    n_features_per_ch = sum(
        [
            config.permutation_entropy,
            config.spectral_entropy,
            config.sample_entropy,
            config.hjorth_params * 2,  # mobility + complexity
            config.lziv_complexity,
        ]
    )

    logger.info(
        "computed_entropy_features",
        shape=features.shape,
        n_features_per_channel=n_features_per_ch,
        method="antropy",
    )
    return features


def get_entropy_feature_names(
    ch_names: list[str],
    config: Optional[EntropyConfig] = None,
) -> list[str]:
    """Get human-readable names for entropy features."""
    if config is None:
        config = EntropyConfig()

    names: list[str] = []
    for ch in ch_names:
        if config.permutation_entropy:
            names.append(f"perm_entropy_{ch}")
        if config.spectral_entropy:
            names.append(f"spectral_entropy_{ch}")
        if config.sample_entropy:
            names.append(f"sample_entropy_{ch}")
        if config.hjorth_params:
            names.extend([f"hjorth_mobility_{ch}", f"hjorth_complexity_{ch}"])
        if config.lziv_complexity:
            names.append(f"lziv_complexity_{ch}")

    return names
