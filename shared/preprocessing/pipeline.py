"""Shared EEG preprocessing pipeline using MNE-Python.

All classifiers share this pipeline. Each step is configurable
and logged to MLflow for reproducibility.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for the EEG preprocessing pipeline."""

    # Filtering
    l_freq: float = 0.5
    h_freq: float = 45.0
    notch_freq: Optional[float] = 60.0
    notch_freqs: Optional[list[float]] = None

    # ICA
    ica_n_components: Optional[int] = None
    ica_method: str = "fastica"
    ica_max_iter: int = 500
    ica_random_state: int = 42

    # Re-referencing
    ref_channels: str = "average"

    # Epoching
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: Optional[tuple[Optional[float], float]] = (None, 0)
    event_id: Optional[dict[str, int]] = None

    # Bad channel handling
    interpolate_bads: bool = True

    # Resampling
    resample_sfreq: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert config to dict for MLflow logging."""
        return {
            "preproc_l_freq": self.l_freq,
            "preproc_h_freq": self.h_freq,
            "preproc_notch_freq": self.notch_freq,
            "preproc_ica_method": self.ica_method,
            "preproc_ica_n_components": self.ica_n_components,
            "preproc_ref_channels": self.ref_channels,
            "preproc_tmin": self.tmin,
            "preproc_tmax": self.tmax,
            "preproc_resample_sfreq": self.resample_sfreq,
        }


def load_raw(filepath: Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load raw EEG data from various formats.

    Supports: .set (EEGLAB), .edf (European Data Format),
    .fif (MNE), .csv (generic).
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    loaders = {
        ".set": mne.io.read_raw_eeglab,
        ".edf": mne.io.read_raw_edf,
        ".fif": mne.io.read_raw_fif,
        ".bdf": mne.io.read_raw_bdf,
        ".vhdr": mne.io.read_raw_brainvision,
    }

    if suffix in loaders:
        raw = loaders[suffix](str(filepath), preload=preload)
    elif suffix == ".csv":
        raw = _load_csv_as_raw(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info(
        "loaded_raw_eeg",
        filepath=str(filepath),
        n_channels=len(raw.ch_names),
        sfreq=raw.info["sfreq"],
        duration_s=raw.times[-1],
    )
    return raw


def _load_csv_as_raw(filepath: Path) -> mne.io.RawArray:
    """Load CSV file as MNE RawArray.

    Expects columns as channels, rows as time samples.
    First row = channel names, optional 'time' column excluded.
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    if "time" in df.columns:
        df = df.drop(columns=["time"])

    ch_names = list(df.columns)
    data = df.values.T  # MNE expects (n_channels, n_samples)

    # Default to 256 Hz if not specified
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    return mne.io.RawArray(data, info)


def apply_filters(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> mne.io.BaseRaw:
    """Apply band-pass and notch filters."""
    raw = raw.copy()

    # Band-pass filter
    raw.filter(
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        fir_design="firwin",
    )
    logger.info(
        "applied_bandpass",
        l_freq=config.l_freq,
        h_freq=config.h_freq,
    )

    # Notch filter
    notch_freqs = config.notch_freqs or (
        [config.notch_freq] if config.notch_freq else []
    )
    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs)
        logger.info("applied_notch", freqs=notch_freqs)

    return raw


def interpolate_bad_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Interpolate channels marked as bad."""
    if raw.info["bads"]:
        logger.info("interpolating_bads", bads=raw.info["bads"])
        raw = raw.copy()
        raw.interpolate_bads(reset_bads=True)
    return raw


def apply_ica(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> mne.io.BaseRaw:
    """Run ICA for artifact rejection (eye blinks, muscle)."""
    ica = mne.preprocessing.ICA(
        n_components=config.ica_n_components,
        method=config.ica_method,
        max_iter=config.ica_max_iter,
        random_state=config.ica_random_state,
    )
    ica.fit(raw)

    # Auto-detect EOG artifacts
    eog_indices: list[int] = []
    eog_channels = [ch for ch in raw.ch_names if "eog" in ch.lower()]
    if eog_channels:
        eog_indices, _ = ica.find_bads_eog(raw)

    # Auto-detect muscle artifacts via high-frequency power
    muscle_indices: list[int] = []
    try:
        muscle_indices, _ = ica.find_bads_muscle(raw)
    except Exception:
        logger.warning("muscle_artifact_detection_failed")

    exclude = list(set(eog_indices + muscle_indices))
    ica.exclude = exclude

    logger.info(
        "applied_ica",
        n_components=ica.n_components_,
        excluded=exclude,
        n_eog=len(eog_indices),
        n_muscle=len(muscle_indices),
    )

    raw = raw.copy()
    ica.apply(raw)
    return raw


def set_reference(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> mne.io.BaseRaw:
    """Apply re-referencing."""
    raw = raw.copy()
    if config.ref_channels == "average":
        raw.set_eeg_reference("average", projection=True)
        raw.apply_proj()
    else:
        raw.set_eeg_reference(config.ref_channels)

    logger.info("applied_reference", ref=config.ref_channels)
    return raw


def create_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    config: PreprocessingConfig,
) -> mne.Epochs:
    """Create epochs from continuous data."""
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=config.event_id,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=config.baseline,
        preload=True,
    )

    logger.info(
        "created_epochs",
        n_epochs=len(epochs),
        tmin=config.tmin,
        tmax=config.tmax,
    )
    return epochs


def preprocess_raw(
    raw: mne.io.BaseRaw,
    config: Optional[PreprocessingConfig] = None,
) -> mne.io.BaseRaw:
    """Run the full preprocessing pipeline on continuous data.

    Steps: bad channel interpolation → filter → ICA → re-reference → resample.
    """
    if config is None:
        config = PreprocessingConfig()

    if config.interpolate_bads:
        raw = interpolate_bad_channels(raw)

    raw = apply_filters(raw, config)
    raw = apply_ica(raw, config)
    raw = set_reference(raw, config)

    if config.resample_sfreq:
        raw.resample(config.resample_sfreq)
        logger.info("resampled", sfreq=config.resample_sfreq)

    return raw


def preprocess_and_epoch(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    config: Optional[PreprocessingConfig] = None,
) -> mne.Epochs:
    """Full pipeline: preprocess raw data then create epochs."""
    if config is None:
        config = PreprocessingConfig()

    raw = preprocess_raw(raw, config)
    return create_epochs(raw, events, config)


def log_config_to_mlflow(config: PreprocessingConfig) -> None:
    """Log preprocessing configuration to MLflow."""
    try:
        import mlflow

        mlflow.log_params(config.to_dict())
        logger.info("logged_config_to_mlflow")
    except Exception as exc:
        logger.warning("mlflow_logging_failed", error=str(exc))
