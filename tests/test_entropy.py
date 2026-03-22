"""Tests for entropy feature extraction via antropy."""

import mne
import numpy as np

from shared.features.entropy import (
    EntropyConfig,
    compute_entropy_features,
    get_entropy_feature_names,
)


class TestEntropyFeatures:
    def test_default_features(self, synthetic_epochs: mne.Epochs) -> None:
        features = compute_entropy_features(synthetic_epochs)
        assert features.ndim == 2
        assert features.shape[0] == len(synthetic_epochs)
        assert features.shape[1] > 0

    def test_feature_count_matches_config(self, synthetic_epochs: mne.Epochs) -> None:
        config = EntropyConfig(
            permutation_entropy=True,
            spectral_entropy=True,
            sample_entropy=False,
            hjorth_params=True,
            lziv_complexity=False,
        )
        features = compute_entropy_features(synthetic_epochs, config)
        n_ch = len(synthetic_epochs.ch_names)
        # 1 (perm) + 1 (spectral) + 2 (hjorth) = 4 per channel
        expected_features = n_ch * 4
        assert features.shape[1] == expected_features

    def test_all_features_enabled(self, synthetic_epochs: mne.Epochs) -> None:
        config = EntropyConfig()
        features = compute_entropy_features(synthetic_epochs, config)
        n_ch = len(synthetic_epochs.ch_names)
        # 1 + 1 + 1 + 2 + 1 = 6 per channel
        expected_features = n_ch * 6
        assert features.shape[1] == expected_features

    def test_feature_names(self) -> None:
        ch_names = ["Fp1", "Fp2", "F3"]
        names = get_entropy_feature_names(ch_names)
        assert "perm_entropy_Fp1" in names
        assert "spectral_entropy_Fp2" in names
        assert "hjorth_mobility_F3" in names
        assert "lziv_complexity_F3" in names

    def test_values_are_finite(self, synthetic_epochs: mne.Epochs) -> None:
        features = compute_entropy_features(synthetic_epochs)
        assert np.all(np.isfinite(features))
