"""Spectral feature extraction for EEG data.

Computes power spectral density across standard frequency bands
(delta, theta, alpha, beta, gamma) using Welch's method.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import mne
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

STANDARD_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class SpectralConfig:
    """Configuration for spectral feature extraction."""

    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(STANDARD_BANDS)
    )
    method: str = "welch"
    n_fft: int = 256
    n_overlap: Optional[int] = None
    fmin: float = 0.5
    fmax: float = 45.0
    normalize: bool = True


def compute_psd(
    epochs: mne.Epochs,
    config: Optional[SpectralConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density for epochs.

    Returns:
        psd: Array of shape (n_epochs, n_channels, n_freqs)
        freqs: Array of frequency values
    """
    if config is None:
        config = SpectralConfig()

    psd_kwargs: dict[str, Any] = {
        "method": config.method,
        "fmin": config.fmin,
        "fmax": config.fmax,
        "n_fft": config.n_fft,
    }
    if config.n_overlap is not None:
        psd_kwargs["n_overlap"] = config.n_overlap

    spectrum = epochs.compute_psd(**psd_kwargs)

    psd = spectrum.get_data()
    freqs = spectrum.freqs

    logger.info(
        "computed_psd",
        shape=psd.shape,
        freq_range=(freqs[0], freqs[-1]),
        method=config.method,
    )
    return psd, freqs


def compute_band_power(
    epochs: mne.Epochs,
    config: Optional[SpectralConfig] = None,
) -> dict[str, np.ndarray]:
    """Compute average power in each frequency band.

    Returns:
        Dictionary mapping band name to array of shape (n_epochs, n_channels).
    """
    if config is None:
        config = SpectralConfig()

    psd, freqs = compute_psd(epochs, config)
    band_powers: dict[str, np.ndarray] = {}

    for band_name, (fmin, fmax) in config.bands.items():
        freq_mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(freq_mask):
            logger.warning("empty_band", band=band_name, fmin=fmin, fmax=fmax)
            continue

        power = psd[:, :, freq_mask].mean(axis=2)
        band_powers[band_name] = power

    if config.normalize:
        total_power = sum(band_powers.values())
        band_powers = {name: power / total_power for name, power in band_powers.items()}

    logger.info(
        "computed_band_power",
        bands=list(band_powers.keys()),
        n_epochs=psd.shape[0],
    )
    return band_powers


def compute_band_power_ratio(
    band_powers: dict[str, np.ndarray],
    numerator: str,
    denominator: str,
) -> np.ndarray:
    """Compute power ratio between two bands (e.g., theta/alpha).

    Returns:
        Array of shape (n_epochs, n_channels).
    """
    if numerator not in band_powers or denominator not in band_powers:
        raise ValueError(f"Band not found. Available: {list(band_powers.keys())}")

    ratio = band_powers[numerator] / (band_powers[denominator] + 1e-10)
    logger.info("computed_band_ratio", ratio=f"{numerator}/{denominator}")
    return ratio


def extract_spectral_features(
    epochs: mne.Epochs,
    config: Optional[SpectralConfig] = None,
) -> np.ndarray:
    """Extract a flat spectral feature vector per epoch.

    Concatenates mean band power across all channels into a single
    feature vector per epoch.

    Returns:
        Array of shape (n_epochs, n_bands * n_channels).
    """
    if config is None:
        config = SpectralConfig()

    band_powers = compute_band_power(epochs, config)

    features = np.concatenate(
        [band_powers[band] for band in sorted(band_powers.keys())],
        axis=1,
    )

    logger.info(
        "extracted_spectral_features",
        shape=features.shape,
        bands=sorted(band_powers.keys()),
    )
    return features


def compute_specparam_features(
    epochs: mne.Epochs,
    config: Optional[SpectralConfig] = None,
) -> np.ndarray:
    """Extract parameterized spectral features using specparam (FOOOF).

    Separates neural power spectra into:
    - Aperiodic (1/f): offset and exponent per channel
    - Periodic: peak power and center frequency per band per channel

    This is more scientifically accurate than raw Welch PSD because
    1/f slope changes between groups (common in addiction/PTSD) confound
    raw band power measurements.

    Returns:
        Array of shape (n_epochs, n_features).
    """
    if config is None:
        config = SpectralConfig()

    try:
        from specparam import SpectralGroupModel
    except ImportError:
        logger.warning("specparam_not_installed", msg="Returning empty features")
        return np.zeros((len(epochs), 0))

    psd, freqs = compute_psd(epochs, config)
    n_epochs, n_channels, _ = psd.shape

    all_features: list[np.ndarray] = []

    for epoch_idx in range(n_epochs):
        epoch_psd = psd[epoch_idx]  # (n_channels, n_freqs)

        fg = SpectralGroupModel(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=5,
            min_peak_height=0.1,
            aperiodic_mode="fixed",
            verbose=False,
        )

        try:
            fg.fit(freqs, epoch_psd, [config.fmin, config.fmax])

            # Aperiodic features: offset + exponent per channel
            aperiodic = fg.get_params("aperiodic_params")  # (n_channels, 2)
            offsets = aperiodic[:, 0]
            exponents = aperiodic[:, 1]

            # Periodic: extract peak power for each standard band
            band_peaks = []
            for band_name, (fmin, fmax) in config.bands.items():
                peak_powers = np.zeros(n_channels)
                for ch_idx in range(n_channels):
                    result = fg.get_results()[ch_idx]
                    if result.peak_params is not None and len(result.peak_params) > 0:
                        peaks_in_band = [
                            p for p in result.peak_params if fmin <= p[0] <= fmax
                        ]
                        if peaks_in_band:
                            peak_powers[ch_idx] = max(p[1] for p in peaks_in_band)
                band_peaks.append(peak_powers)

            epoch_features = np.concatenate([offsets, exponents] + band_peaks)
        except Exception:
            # Fallback: zeros if specparam fails for this epoch
            n_features = n_channels * (2 + len(config.bands))
            epoch_features = np.zeros(n_features)

        all_features.append(epoch_features)

    features = np.array(all_features)

    logger.info(
        "computed_specparam_features",
        shape=features.shape,
        method="specparam",
    )
    return features
