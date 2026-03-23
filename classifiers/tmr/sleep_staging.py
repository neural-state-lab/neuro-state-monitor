"""Automated sleep staging via YASA.

YASA (eLife 2021, 381 citations) provides pre-trained LightGBM classifiers
for automated sleep staging achieving ~85% accuracy on healthy adults.

Stages: Wake (W), N1, N2, N3 (deep/slow-wave), REM.

DO NOT build custom sleep staging — YASA already matches human scorer accuracy.
"""

from dataclasses import dataclass
from typing import Optional

import mne
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SleepStagingConfig:
    """Configuration for YASA sleep staging."""

    eeg_name: Optional[str] = None  # auto-detect if None
    eog_name: Optional[str] = None  # optional, improves accuracy
    emg_name: Optional[str] = None  # optional, improves accuracy
    metadata: Optional[dict] = None  # age, male (for model calibration)


@dataclass
class SleepStagingResult:
    """Result of automated sleep staging."""

    hypnogram: np.ndarray  # per-epoch stage labels (W, N1, N2, N3, R)
    probabilities: np.ndarray  # (n_epochs, 5) probabilities per stage
    confidence: np.ndarray  # max probability per epoch
    stage_durations_min: dict[str, float]  # minutes in each stage
    sleep_efficiency: float  # % of recording spent asleep
    n2_n3_epochs: np.ndarray  # indices of N2/N3 epochs (TMR targets)


def run_sleep_staging(
    raw: mne.io.BaseRaw,
    config: Optional[SleepStagingConfig] = None,
) -> SleepStagingResult:
    """Run automated sleep staging using YASA.

    Args:
        raw: MNE Raw object (continuous sleep EEG recording).
        config: Optional configuration.

    Returns:
        SleepStagingResult with hypnogram and statistics.
    """
    if config is None:
        config = SleepStagingConfig()

    try:
        import yasa
    except ImportError:
        raise RuntimeError("YASA not installed. Install with: pip install yasa")

    # Auto-detect EEG channel if not specified
    eeg_name = config.eeg_name
    if eeg_name is None:
        eeg_candidates = ["C4", "C3", "Cz", "F4", "F3"]
        for candidate in eeg_candidates:
            if candidate in raw.ch_names:
                eeg_name = candidate
                break
        if eeg_name is None:
            eeg_name = raw.ch_names[0]

    logger.info(
        "starting_sleep_staging",
        eeg=eeg_name,
        eog=config.eog_name,
        emg=config.emg_name,
        duration_min=raw.times[-1] / 60,
    )

    # Run YASA sleep staging
    sls = yasa.SleepStaging(
        raw,
        eeg_name=eeg_name,
        eog_name=config.eog_name,
        emg_name=config.emg_name,
        metadata=config.metadata,
    )

    hypnogram = sls.predict()  # string labels: W, N1, N2, N3, R
    probabilities = sls.predict_proba()  # (n_epochs, 5) DataFrame

    prob_array = probabilities.values
    confidence = prob_array.max(axis=1)

    # Compute stage durations (YASA uses 30-second epochs)
    epoch_duration = 30.0  # seconds (YASA standard)
    stage_counts = {}
    for stage in ["W", "N1", "N2", "N3", "R"]:
        count = np.sum(hypnogram == stage)
        stage_counts[stage] = count
    stage_durations = {
        stage: count * epoch_duration / 60 for stage, count in stage_counts.items()
    }

    # Sleep efficiency
    total_epochs = len(hypnogram)
    sleep_epochs = np.sum(hypnogram != "W")
    sleep_efficiency = sleep_epochs / total_epochs * 100 if total_epochs > 0 else 0

    # Find N2/N3 epochs (TMR target stages)
    n2_n3_mask = (hypnogram == "N2") | (hypnogram == "N3")
    n2_n3_epochs = np.where(n2_n3_mask)[0]

    result = SleepStagingResult(
        hypnogram=hypnogram,
        probabilities=prob_array,
        confidence=confidence,
        stage_durations_min=stage_durations,
        sleep_efficiency=sleep_efficiency,
        n2_n3_epochs=n2_n3_epochs,
    )

    logger.info(
        "sleep_staging_complete",
        stages=stage_durations,
        efficiency=f"{sleep_efficiency:.1f}%",
        n_tmr_epochs=len(n2_n3_epochs),
    )
    return result
