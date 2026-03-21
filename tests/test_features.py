"""Tests for shared feature extraction modules."""

import mne
import numpy as np
import pytest

from shared.features.erp import (
    ERPComponent,
    ERPConfig,
    extract_component_amplitude,
    extract_component_latency,
    extract_erp_features,
    extract_mean_amplitude,
)
from shared.features.spectral import (
    SpectralConfig,
    compute_band_power,
    compute_psd,
    extract_spectral_features,
)


class TestSpectralFeatures:
    def test_compute_psd(self, synthetic_epochs: mne.Epochs) -> None:
        psd, freqs = compute_psd(synthetic_epochs)
        assert psd.ndim == 3  # (n_epochs, n_channels, n_freqs)
        assert len(freqs) > 0
        assert psd.shape[0] == len(synthetic_epochs)

    def test_compute_band_power(self, synthetic_epochs: mne.Epochs) -> None:
        powers = compute_band_power(synthetic_epochs)
        assert "alpha" in powers
        assert "beta" in powers
        assert "theta" in powers
        assert powers["alpha"].shape[0] == len(synthetic_epochs)

    def test_band_power_normalized(self, synthetic_epochs: mne.Epochs) -> None:
        config = SpectralConfig(normalize=True)
        powers = compute_band_power(synthetic_epochs, config)
        total = sum(powers.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_extract_spectral_features(self, synthetic_epochs: mne.Epochs) -> None:
        features = extract_spectral_features(synthetic_epochs)
        assert features.ndim == 2
        assert features.shape[0] == len(synthetic_epochs)
        assert features.shape[1] > 0

    def test_custom_bands(self, synthetic_epochs: mne.Epochs) -> None:
        config = SpectralConfig(
            bands={"low": (0.5, 10.0), "high": (10.0, 45.0)},
            normalize=False,
        )
        powers = compute_band_power(synthetic_epochs, config)
        assert "low" in powers
        assert "high" in powers
        assert len(powers) == 2


class TestERPFeatures:
    def test_component_amplitude(self, synthetic_epochs: mne.Epochs) -> None:
        p300 = ERPComponent(name="P300", tmin=0.25, tmax=0.5, polarity="positive")
        amp = extract_component_amplitude(synthetic_epochs, p300)
        assert amp.shape[0] == len(synthetic_epochs)

    def test_component_latency(self, synthetic_epochs: mne.Epochs) -> None:
        p300 = ERPComponent(name="P300", tmin=0.25, tmax=0.5, polarity="positive")
        lat = extract_component_latency(synthetic_epochs, p300)
        assert lat.shape[0] == len(synthetic_epochs)
        assert np.all(lat >= 0.25)
        assert np.all(lat <= 0.5)

    def test_mean_amplitude(self, synthetic_epochs: mne.Epochs) -> None:
        p300 = ERPComponent(name="P300", tmin=0.25, tmax=0.5)
        mean_amp = extract_mean_amplitude(synthetic_epochs, p300)
        assert mean_amp.shape[0] == len(synthetic_epochs)

    def test_extract_erp_features(self, synthetic_epochs: mne.Epochs) -> None:
        features = extract_erp_features(synthetic_epochs)
        assert features.ndim == 2
        assert features.shape[0] == len(synthetic_epochs)
        # 2 components * 3 features (amp, lat, mean) * n_channels
        assert features.shape[1] > 0

    def test_negative_component(self, synthetic_epochs: mne.Epochs) -> None:
        n400 = ERPComponent(name="N400", tmin=0.3, tmax=0.6, polarity="negative")
        amp = extract_component_amplitude(synthetic_epochs, n400)
        # Negative polarity should return min values
        assert amp.shape[0] == len(synthetic_epochs)
