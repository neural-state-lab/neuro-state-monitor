"""Addiction-specific feature extraction.

Extracts features that distinguish alcoholic from control subjects:
- Resting beta power (13-30 Hz, elevated in addiction/hyperarousal)
- Alpha band connectivity (disrupted in alcoholism)
- P300 amplitude (reduced in alcoholism — key ERP biomarker)
- Beta inter-electrode connectivity (Tian et al. 2024 approach)

References:
- Tian et al. 2024: Beta connectivity patterns in addiction
- UCI EEG dataset: Begleiter, Porjesz et al.
"""

from dataclasses import dataclass, field
from typing import Optional

import mne
import numpy as np
import structlog

from shared.features.connectivity import (
    ConnectivityConfig,
    extract_connectivity_features,
)
from shared.features.erp import (
    ERPComponent,
    extract_component_amplitude,
    extract_mean_amplitude,
)
from shared.features.spectral import (
    SpectralConfig,
    compute_band_power,
    extract_spectral_features,
)

logger = structlog.get_logger(__name__)

# Electrode groups relevant to addiction
FRONTAL_CHANNELS = ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"]
CENTRAL_CHANNELS = ["C3", "C4", "CZ"]
PARIETAL_CHANNELS = ["P3", "P4", "PZ"]


@dataclass
class AddictionFeatureConfig:
    """Configuration for addiction-specific features."""

    # Frequency bands of interest
    beta_range: tuple[float, float] = (13.0, 30.0)
    alpha_range: tuple[float, float] = (8.0, 13.0)
    theta_range: tuple[float, float] = (4.0, 8.0)

    # P300 component window
    p300: ERPComponent = field(
        default_factory=lambda: ERPComponent(
            name="P300",
            tmin=0.25,
            tmax=0.5,
            polarity="positive",
        )
    )

    # Connectivity
    connectivity_method: str = "plv"

    # Whether to include all spectral bands
    include_full_spectral: bool = True

    # Whether to include beta connectivity (Tian 2024)
    include_beta_connectivity: bool = True


def compute_resting_beta(
    epochs: mne.Epochs,
    config: Optional[AddictionFeatureConfig] = None,
) -> np.ndarray:
    """Compute resting beta power (13-30 Hz).

    Elevated resting beta is a hallmark of addiction-related
    hyperarousal. Alcoholic subjects show higher beta power at rest.

    Returns:
        Array of shape (n_epochs, n_channels).
    """
    if config is None:
        config = AddictionFeatureConfig()

    spectral_config = SpectralConfig(
        bands={"beta": config.beta_range},
        normalize=False,
    )
    powers = compute_band_power(epochs, spectral_config)

    logger.info("computed_resting_beta", mean=float(powers["beta"].mean()))
    return powers["beta"]


def compute_alpha_connectivity(
    epochs: mne.Epochs,
    config: Optional[AddictionFeatureConfig] = None,
) -> np.ndarray:
    """Compute alpha band connectivity patterns.

    Alpha connectivity is disrupted in alcoholism, reflecting
    altered functional brain networks.

    Returns:
        Flattened connectivity features, shape (n_epochs, n_features).
    """
    if config is None:
        config = AddictionFeatureConfig()

    con_config = ConnectivityConfig(
        method=config.connectivity_method,
        fmin=config.alpha_range[0],
        fmax=config.alpha_range[1],
        faverage=True,
    )

    features = extract_connectivity_features(epochs, con_config)
    logger.info("computed_alpha_connectivity", shape=features.shape)
    return features


def compute_beta_connectivity(
    epochs: mne.Epochs,
    config: Optional[AddictionFeatureConfig] = None,
) -> np.ndarray:
    """Compute beta band connectivity (Tian et al. 2024 approach).

    Beta connectivity between frontal electrode pairs is altered
    in addiction and correlates with craving intensity.

    Returns:
        Flattened connectivity features, shape (n_epochs, n_features).
    """
    if config is None:
        config = AddictionFeatureConfig()

    con_config = ConnectivityConfig(
        method=config.connectivity_method,
        fmin=config.beta_range[0],
        fmax=config.beta_range[1],
        faverage=True,
    )

    features = extract_connectivity_features(epochs, con_config)
    logger.info("computed_beta_connectivity", shape=features.shape)
    return features


def compute_p300_features(
    epochs: mne.Epochs,
    config: Optional[AddictionFeatureConfig] = None,
) -> np.ndarray:
    """Extract P300 amplitude features.

    Reduced P300 amplitude is one of the most replicated
    findings in alcoholism research. It reflects impaired
    cognitive evaluation and attention allocation.

    Returns:
        Array of shape (n_epochs, n_channels * 2) — peak + mean amplitude.
    """
    if config is None:
        config = AddictionFeatureConfig()

    peak_amp = extract_component_amplitude(epochs, config.p300)
    mean_amp = extract_mean_amplitude(epochs, config.p300)

    features = np.concatenate([peak_amp, mean_amp], axis=1)
    logger.info("computed_p300_features", shape=features.shape)
    return features


def compute_frontal_beta_ratio(
    epochs: mne.Epochs,
    config: Optional[AddictionFeatureConfig] = None,
) -> np.ndarray:
    """Compute frontal beta/alpha ratio.

    High beta/alpha ratio in frontal regions indicates hyperarousal
    state associated with addiction.

    Returns:
        Array of shape (n_epochs,).
    """
    if config is None:
        config = AddictionFeatureConfig()

    spectral_config = SpectralConfig(
        bands={
            "alpha": config.alpha_range,
            "beta": config.beta_range,
        },
        normalize=False,
    )
    powers = compute_band_power(epochs, spectral_config)

    # Find frontal channels
    frontal_idx = [
        epochs.ch_names.index(ch) for ch in FRONTAL_CHANNELS if ch in epochs.ch_names
    ]

    if not frontal_idx:
        logger.warning("no_frontal_channels_available")
        return np.zeros(len(epochs))

    frontal_beta = powers["beta"][:, frontal_idx].mean(axis=1)
    frontal_alpha = powers["alpha"][:, frontal_idx].mean(axis=1)

    ratio = frontal_beta / (frontal_alpha + 1e-10)
    logger.info("computed_frontal_beta_ratio", mean=float(ratio.mean()))
    return ratio


def extract_addiction_features(
    epochs: mne.Epochs,
    config: Optional[AddictionFeatureConfig] = None,
) -> np.ndarray:
    """Extract complete addiction feature vector per epoch.

    Combines all addiction-relevant features:
    1. Full spectral band power (all bands, all channels)
    2. Resting beta power (all channels)
    3. P300 amplitude (peak + mean, all channels)
    4. Frontal beta/alpha ratio
    5. Alpha connectivity (optional)
    6. Beta connectivity (optional, Tian 2024)

    Returns:
        Array of shape (n_epochs, n_features).
    """
    if config is None:
        config = AddictionFeatureConfig()

    feature_arrays: list[np.ndarray] = []

    # 1. Full spectral features
    if config.include_full_spectral:
        spectral = extract_spectral_features(epochs)
        feature_arrays.append(spectral)

    # 2. Resting beta (redundant with spectral but useful standalone)
    beta = compute_resting_beta(epochs, config)
    feature_arrays.append(beta)

    # 3. P300 features
    p300 = compute_p300_features(epochs, config)
    feature_arrays.append(p300)

    # 4. Frontal beta/alpha ratio
    ba_ratio = compute_frontal_beta_ratio(epochs, config)
    feature_arrays.append(ba_ratio[:, np.newaxis])

    # 5. Alpha connectivity
    try:
        alpha_conn = compute_alpha_connectivity(epochs, config)
        feature_arrays.append(alpha_conn)
    except Exception as exc:
        logger.warning("alpha_connectivity_failed", error=str(exc))

    # 6. Beta connectivity (Tian 2024)
    if config.include_beta_connectivity:
        try:
            beta_conn = compute_beta_connectivity(epochs, config)
            feature_arrays.append(beta_conn)
        except Exception as exc:
            logger.warning("beta_connectivity_failed", error=str(exc))

    features = np.concatenate(feature_arrays, axis=1)

    logger.info(
        "extracted_addiction_features",
        shape=features.shape,
        n_feature_groups=len(feature_arrays),
    )
    return features
