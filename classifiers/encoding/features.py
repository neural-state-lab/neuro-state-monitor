"""Encoding-specific feature extraction.

Extracts features that predict encoding success based on neuroscience literature:
- Alpha/beta desynchronization (10-30 Hz power decrease = encoding success)
- Theta power increase (frontal/hippocampal memory engagement)
- Inter-electrode correlation patterns
- Time window: 0-600ms post-stimulus onset (Duan et al. 2025)
"""

from dataclasses import dataclass, field
from typing import Optional

import mne
import numpy as np
import structlog

from shared.features.erp import ERPComponent, extract_component_amplitude, extract_mean_amplitude
from shared.features.spectral import SpectralConfig, compute_band_power, compute_band_power_ratio

logger = structlog.get_logger(__name__)

# Electrode groups by brain region
FRONTAL_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"]
CENTRAL_CHANNELS = ["C3", "C4", "Cz"]
PARIETAL_CHANNELS = ["P3", "P4", "P7", "P8", "Pz"]
TEMPORAL_CHANNELS = ["T7", "T8"]
OCCIPITAL_CHANNELS = ["O1", "O2", "Oz"]


@dataclass
class EncodingFeatureConfig:
    """Configuration for encoding-specific features."""

    # Time window for feature extraction (seconds post-stimulus)
    tmin: float = 0.0
    tmax: float = 0.6

    # Spectral bands of interest for encoding
    alpha_range: tuple[float, float] = (8.0, 13.0)
    beta_range: tuple[float, float] = (13.0, 30.0)
    theta_range: tuple[float, float] = (4.0, 8.0)

    # Desynchronization baseline (pre-stimulus)
    baseline_tmin: float = -0.2
    baseline_tmax: float = 0.0

    # Whether to compute inter-electrode correlations
    compute_correlations: bool = True

    # ERP components relevant to encoding
    erp_components: list[ERPComponent] = field(
        default_factory=lambda: [
            ERPComponent(name="P300", tmin=0.25, tmax=0.5, polarity="positive"),
            ERPComponent(name="late_positive", tmin=0.4, tmax=0.7, polarity="positive"),
        ]
    )


def compute_desynchronization(
    epochs: mne.Epochs,
    band: tuple[float, float],
    config: Optional[EncodingFeatureConfig] = None,
) -> np.ndarray:
    """Compute event-related desynchronization (ERD) for a frequency band.

    ERD = (baseline_power - active_power) / baseline_power × 100

    Positive ERD = power DECREASE during encoding (desynchronization).
    This is the key marker: alpha/beta desynchronization predicts encoding success.

    Returns:
        Array of shape (n_epochs, n_channels) with ERD values.
    """
    if config is None:
        config = EncodingFeatureConfig()

    spectral_config = SpectralConfig(
        bands={"target": band},
        normalize=False,
    )

    # Baseline period power
    baseline_epochs = epochs.copy().crop(
        tmin=config.baseline_tmin,
        tmax=config.baseline_tmax,
    )
    baseline_power = compute_band_power(baseline_epochs, spectral_config)["target"]

    # Active period power
    active_epochs = epochs.copy().crop(
        tmin=config.tmin,
        tmax=config.tmax,
    )
    active_power = compute_band_power(active_epochs, spectral_config)["target"]

    # ERD: positive = desynchronization (power decrease)
    erd = (baseline_power - active_power) / (baseline_power + 1e-10) * 100

    logger.info(
        "computed_desynchronization",
        band=band,
        mean_erd=float(erd.mean()),
    )
    return erd


def compute_alpha_beta_desync(
    epochs: mne.Epochs,
    config: Optional[EncodingFeatureConfig] = None,
) -> np.ndarray:
    """Compute alpha/beta desynchronization (10-30 Hz).

    This is the primary encoding success predictor.
    Higher desynchronization = better encoding.

    Returns:
        Array of shape (n_epochs, n_channels).
    """
    if config is None:
        config = EncodingFeatureConfig()

    alpha_erd = compute_desynchronization(epochs, config.alpha_range, config)
    beta_erd = compute_desynchronization(epochs, config.beta_range, config)

    # Combined alpha-beta desynchronization
    combined = (alpha_erd + beta_erd) / 2

    logger.info(
        "computed_alpha_beta_desync",
        mean_alpha_erd=float(alpha_erd.mean()),
        mean_beta_erd=float(beta_erd.mean()),
    )
    return combined


def compute_theta_power(
    epochs: mne.Epochs,
    config: Optional[EncodingFeatureConfig] = None,
) -> np.ndarray:
    """Compute theta power increase during encoding.

    Theta power increases in frontal and hippocampal regions during
    successful encoding. This reflects memory system engagement.

    Returns:
        Array of shape (n_epochs, n_channels).
    """
    if config is None:
        config = EncodingFeatureConfig()

    active_epochs = epochs.copy().crop(tmin=config.tmin, tmax=config.tmax)

    spectral_config = SpectralConfig(
        bands={"theta": config.theta_range},
        normalize=False,
    )
    theta_power = compute_band_power(active_epochs, spectral_config)["theta"]

    logger.info("computed_theta_power", mean=float(theta_power.mean()))
    return theta_power


def compute_theta_alpha_ratio(
    epochs: mne.Epochs,
    config: Optional[EncodingFeatureConfig] = None,
) -> np.ndarray:
    """Compute theta/alpha ratio during encoding window.

    High theta/alpha ratio = strong memory encoding.

    Returns:
        Array of shape (n_epochs, n_channels).
    """
    if config is None:
        config = EncodingFeatureConfig()

    active_epochs = epochs.copy().crop(tmin=config.tmin, tmax=config.tmax)

    spectral_config = SpectralConfig(
        bands={
            "theta": config.theta_range,
            "alpha": config.alpha_range,
        },
        normalize=False,
    )
    powers = compute_band_power(active_epochs, spectral_config)
    ratio = compute_band_power_ratio(powers, "theta", "alpha")

    logger.info("computed_theta_alpha_ratio", mean=float(ratio.mean()))
    return ratio


def compute_inter_electrode_correlation(
    epochs: mne.Epochs,
    config: Optional[EncodingFeatureConfig] = None,
) -> np.ndarray:
    """Compute inter-electrode correlation patterns during encoding.

    Measures how correlated electrode pairs are during the encoding window.
    Specific patterns of correlation predict encoding success.

    Returns:
        Array of shape (n_epochs, n_pairs) where n_pairs = n_ch*(n_ch-1)/2.
    """
    if config is None:
        config = EncodingFeatureConfig()

    active_epochs = epochs.copy().crop(tmin=config.tmin, tmax=config.tmax)
    data = active_epochs.get_data()  # (n_epochs, n_channels, n_times)

    n_epochs, n_channels, _ = data.shape
    upper_idx = np.triu_indices(n_channels, k=1)
    n_pairs = len(upper_idx[0])

    correlations = np.zeros((n_epochs, n_pairs))

    for epoch_idx in range(n_epochs):
        epoch_data = data[epoch_idx]  # (n_channels, n_times)
        corr_matrix = np.corrcoef(epoch_data)
        correlations[epoch_idx] = corr_matrix[upper_idx]

    logger.info(
        "computed_inter_electrode_correlation",
        n_pairs=n_pairs,
        mean_corr=float(correlations.mean()),
    )
    return correlations


def compute_regional_power(
    epochs: mne.Epochs,
    channels: list[str],
    band: tuple[float, float],
    config: Optional[EncodingFeatureConfig] = None,
) -> np.ndarray:
    """Compute average band power for a specific brain region.

    Averages power across channels in the region.

    Returns:
        Array of shape (n_epochs,).
    """
    if config is None:
        config = EncodingFeatureConfig()

    active_epochs = epochs.copy().crop(tmin=config.tmin, tmax=config.tmax)

    # Filter to available channels
    available = [ch for ch in channels if ch in epochs.ch_names]
    if not available:
        logger.warning("no_channels_available", requested=channels)
        return np.zeros(len(epochs))

    ch_indices = [epochs.ch_names.index(ch) for ch in available]

    spectral_config = SpectralConfig(
        bands={"target": band},
        normalize=False,
    )
    power = compute_band_power(active_epochs, spectral_config)["target"]
    regional_power = power[:, ch_indices].mean(axis=1)

    return regional_power


def extract_encoding_features(
    epochs: mne.Epochs,
    config: Optional[EncodingFeatureConfig] = None,
) -> np.ndarray:
    """Extract complete encoding feature vector per epoch.

    Combines all encoding-relevant features into a single array.

    Features:
    1. Alpha/beta desynchronization (all channels)
    2. Theta power (all channels)
    3. Theta/alpha ratio (all channels)
    4. Frontal theta power (averaged)
    5. Parietal alpha desync (averaged)
    6. ERP amplitudes (P300, late positive)
    7. Inter-electrode correlations (optional)

    Returns:
        Array of shape (n_epochs, n_features).
    """
    if config is None:
        config = EncodingFeatureConfig()

    feature_arrays: list[np.ndarray] = []
    feature_names: list[str] = []

    # 1. Alpha/beta desynchronization
    ab_desync = compute_alpha_beta_desync(epochs, config)
    feature_arrays.append(ab_desync)
    feature_names.extend([f"ab_desync_{ch}" for ch in epochs.ch_names])

    # 2. Theta power
    theta = compute_theta_power(epochs, config)
    feature_arrays.append(theta)
    feature_names.extend([f"theta_{ch}" for ch in epochs.ch_names])

    # 3. Theta/alpha ratio
    ta_ratio = compute_theta_alpha_ratio(epochs, config)
    feature_arrays.append(ta_ratio)
    feature_names.extend([f"ta_ratio_{ch}" for ch in epochs.ch_names])

    # 4. Regional summaries
    frontal_theta = compute_regional_power(
        epochs, FRONTAL_CHANNELS, config.theta_range, config,
    )
    feature_arrays.append(frontal_theta[:, np.newaxis])
    feature_names.append("frontal_theta_mean")

    parietal_alpha_desync = compute_desynchronization(
        epochs, config.alpha_range, config,
    )
    parietal_indices = [
        epochs.ch_names.index(ch)
        for ch in PARIETAL_CHANNELS
        if ch in epochs.ch_names
    ]
    if parietal_indices:
        parietal_mean = parietal_alpha_desync[:, parietal_indices].mean(axis=1)
    else:
        parietal_mean = np.zeros(len(epochs))
    feature_arrays.append(parietal_mean[:, np.newaxis])
    feature_names.append("parietal_alpha_desync_mean")

    # 5. ERP amplitudes
    for component in config.erp_components:
        amp = extract_mean_amplitude(epochs, component)
        feature_arrays.append(amp)
        feature_names.extend(
            [f"{component.name}_amp_{ch}" for ch in epochs.ch_names]
        )

    # 6. Inter-electrode correlations (optional, high-dimensional)
    if config.compute_correlations:
        corr = compute_inter_electrode_correlation(epochs, config)
        feature_arrays.append(corr)
        n_ch = len(epochs.ch_names)
        feature_names.extend(
            [f"corr_{i}_{j}" for i, j in zip(*np.triu_indices(n_ch, k=1))]
        )

    features = np.concatenate(feature_arrays, axis=1)

    logger.info(
        "extracted_encoding_features",
        shape=features.shape,
        n_feature_groups=len(feature_arrays),
    )
    return features


def get_feature_names(
    n_channels: int,
    ch_names: list[str],
    config: Optional[EncodingFeatureConfig] = None,
) -> list[str]:
    """Get human-readable feature names for the encoding feature vector."""
    if config is None:
        config = EncodingFeatureConfig()

    names: list[str] = []

    # Desync, theta, ratio — per channel
    for prefix in ["ab_desync", "theta", "ta_ratio"]:
        names.extend([f"{prefix}_{ch}" for ch in ch_names])

    # Regional summaries
    names.append("frontal_theta_mean")
    names.append("parietal_alpha_desync_mean")

    # ERP per channel
    for comp in config.erp_components:
        names.extend([f"{comp.name}_amp_{ch}" for ch in ch_names])

    # Correlations
    if config.compute_correlations:
        names.extend(
            [f"corr_{i}_{j}" for i, j in zip(*np.triu_indices(n_channels, k=1))]
        )

    return names
