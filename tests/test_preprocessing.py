"""Tests for shared preprocessing pipeline."""

import mne
import numpy as np

from shared.preprocessing.pipeline import (
    PreprocessingConfig,
    apply_filters,
    create_epochs,
    interpolate_bad_channels,
    preprocess_raw,
    set_reference,
)


class TestPreprocessingConfig:
    def test_default_config(self) -> None:
        config = PreprocessingConfig()
        assert config.l_freq == 0.5
        assert config.h_freq == 45.0
        assert config.notch_freq == 60.0
        assert config.ref_channels == "average"

    def test_to_dict(self) -> None:
        config = PreprocessingConfig()
        d = config.to_dict()
        assert "preproc_l_freq" in d
        assert "preproc_h_freq" in d
        assert d["preproc_l_freq"] == 0.5


class TestApplyFilters:
    def test_bandpass_filter(self, synthetic_raw: mne.io.RawArray) -> None:
        config = PreprocessingConfig(l_freq=1.0, h_freq=40.0, notch_freq=None)
        filtered = apply_filters(synthetic_raw, config)
        assert filtered is not synthetic_raw  # returns copy
        assert len(filtered.ch_names) == len(synthetic_raw.ch_names)

    def test_notch_filter(self, synthetic_raw: mne.io.RawArray) -> None:
        config = PreprocessingConfig(notch_freq=60.0)
        filtered = apply_filters(synthetic_raw, config)
        assert filtered is not synthetic_raw

    def test_no_notch(self, synthetic_raw: mne.io.RawArray) -> None:
        config = PreprocessingConfig(notch_freq=None)
        filtered = apply_filters(synthetic_raw, config)
        assert len(filtered.ch_names) == len(synthetic_raw.ch_names)


class TestInterpolateBadChannels:
    def test_no_bads(self, synthetic_raw: mne.io.RawArray) -> None:
        result = interpolate_bad_channels(synthetic_raw)
        assert len(result.info["bads"]) == 0

    def test_with_bads(self, synthetic_raw: mne.io.RawArray) -> None:
        raw = synthetic_raw.copy()
        raw.info["bads"] = [raw.ch_names[0]]
        result = interpolate_bad_channels(raw)
        assert len(result.info["bads"]) == 0


class TestSetReference:
    def test_average_reference(self, synthetic_raw: mne.io.RawArray) -> None:
        config = PreprocessingConfig(ref_channels="average")
        rerefed = set_reference(synthetic_raw, config)
        assert rerefed is not synthetic_raw


class TestCreateEpochs:
    def test_creates_epochs(
        self,
        synthetic_raw: mne.io.RawArray,
        synthetic_events: np.ndarray,
    ) -> None:
        config = PreprocessingConfig(
            event_id={"target": 1, "nontarget": 2},
            tmin=-0.2,
            tmax=0.8,
        )
        epochs = create_epochs(synthetic_raw, synthetic_events, config)
        assert isinstance(epochs, mne.Epochs)
        assert len(epochs) > 0


class TestPreprocessRaw:
    def test_full_pipeline(self, synthetic_raw: mne.io.RawArray) -> None:
        config = PreprocessingConfig(
            notch_freq=None,  # skip notch for speed
            ica_n_components=3,
        )
        result = preprocess_raw(synthetic_raw, config)
        assert isinstance(result, mne.io.BaseRaw)
        assert len(result.ch_names) == len(synthetic_raw.ch_names)

    def test_with_resample(self, synthetic_raw: mne.io.RawArray) -> None:
        config = PreprocessingConfig(
            notch_freq=None,
            ica_n_components=3,
            resample_sfreq=128.0,
        )
        result = preprocess_raw(synthetic_raw, config)
        assert result.info["sfreq"] == 128.0
