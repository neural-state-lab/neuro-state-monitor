"""Connectivity feature extraction for EEG data.

Computes functional connectivity measures: phase locking value (PLV),
coherence, and band-specific connectivity matrices.
"""

from dataclasses import dataclass
from typing import Optional

import mne
import numpy as np
import structlog
from mne_connectivity import spectral_connectivity_epochs

logger = structlog.get_logger(__name__)


@dataclass
class ConnectivityConfig:
    """Configuration for connectivity feature extraction."""

    method: str = "plv"
    fmin: float = 4.0
    fmax: float = 30.0
    faverage: bool = True
    n_jobs: int = 1

    # Band-specific connectivity
    band_ranges: Optional[dict[str, tuple[float, float]]] = None


def compute_connectivity(
    epochs: mne.Epochs,
    config: Optional[ConnectivityConfig] = None,
) -> np.ndarray:
    """Compute pairwise connectivity between all channels.

    Returns:
        Connectivity matrix of shape (n_epochs, n_channels, n_channels)
        if faverage=True, else (n_epochs, n_channels, n_channels, n_freqs).
    """
    if config is None:
        config = ConnectivityConfig()

    indices = None  # all-to-all

    con = spectral_connectivity_epochs(
        epochs,
        method=config.method,
        fmin=config.fmin,
        fmax=config.fmax,
        faverage=config.faverage,
        n_jobs=config.n_jobs,
    )

    con_data = con.get_data(output="dense")

    logger.info(
        "computed_connectivity",
        method=config.method,
        shape=con_data.shape,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    return con_data


def compute_band_connectivity(
    epochs: mne.Epochs,
    config: Optional[ConnectivityConfig] = None,
) -> dict[str, np.ndarray]:
    """Compute connectivity separately for each frequency band.

    Returns:
        Dictionary mapping band name to connectivity matrix
        of shape (n_channels, n_channels).
    """
    if config is None:
        config = ConnectivityConfig()

    bands = config.band_ranges or {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }

    band_con: dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in bands.items():
        band_config = ConnectivityConfig(
            method=config.method,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            n_jobs=config.n_jobs,
        )
        con_data = compute_connectivity(epochs, band_config)
        band_con[band_name] = con_data

    logger.info(
        "computed_band_connectivity",
        bands=list(band_con.keys()),
        method=config.method,
    )
    return band_con


def extract_connectivity_features(
    epochs: mne.Epochs,
    config: Optional[ConnectivityConfig] = None,
) -> np.ndarray:
    """Extract flattened upper-triangle connectivity features.

    Takes the upper triangle of the connectivity matrix
    (excluding diagonal) as features.

    Returns:
        Array of shape (n_epochs, n_features) where
        n_features = n_channels * (n_channels - 1) / 2.
    """
    if config is None:
        config = ConnectivityConfig()

    con_data = compute_connectivity(epochs, config)

    # Handle different output shapes
    if con_data.ndim == 4:
        # (n_epochs, n_channels, n_channels, n_freqs) -> average across freqs
        con_data = con_data.mean(axis=-1)

    if con_data.ndim == 3:
        # (n_epochs, n_channels, n_channels) or (n_channels, n_channels, 1)
        if con_data.shape[-1] == 1:
            con_data = con_data.squeeze(-1)

    n_channels = con_data.shape[-1]
    upper_idx = np.triu_indices(n_channels, k=1)

    if con_data.ndim == 2:
        # Single connectivity matrix — expand to (1, n_features)
        features = con_data[upper_idx][np.newaxis, :]
    else:
        features = np.array([m[upper_idx] for m in con_data])

    logger.info(
        "extracted_connectivity_features",
        shape=features.shape,
        method=config.method,
    )
    return features
