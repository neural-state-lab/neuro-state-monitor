"""Sleep event detection via YASA.

Detects sleep spindles and slow oscillations — the two key events
for Targeted Memory Reactivation (TMR).

Sleep spindles (12-15 Hz, ~0.5-2s bursts in N2/N3):
- Thalamocortical loops that facilitate memory consolidation
- TMR cues are most effective during/just after spindles

Slow oscillations (SO, <1 Hz, in N3):
- Large amplitude cortical up/down states
- SO up-state coupled with spindles = optimal consolidation

DO NOT build custom detectors — YASA's are validated against expert scoring.
"""

from dataclasses import dataclass
from typing import Optional

import mne
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EventDetectionConfig:
    """Configuration for sleep event detection."""

    # Spindle detection
    spindle_freq: tuple[float, float] = (12.0, 15.0)
    spindle_duration: tuple[float, float] = (0.5, 2.0)
    spindle_min_distance: float = 0.5  # seconds between spindles

    # Slow oscillation detection
    so_freq: tuple[float, float] = (0.3, 1.5)
    so_duration: tuple[float, float] = (0.8, 2.0)
    so_amp_percentile: float = 75.0


@dataclass
class SpindleResult:
    """Detected sleep spindles."""

    events: pd.DataFrame  # YASA spindle dataframe
    n_spindles: int
    density: float  # spindles per minute
    mean_frequency: float  # Hz
    mean_duration: float  # seconds
    mean_amplitude: float  # μV


@dataclass
class SlowOscillationResult:
    """Detected slow oscillations."""

    events: pd.DataFrame  # YASA SO dataframe
    n_events: int
    density: float  # SOs per minute
    mean_amplitude: float  # μV
    mean_duration: float  # seconds


def detect_spindles(
    raw: mne.io.BaseRaw,
    hypnogram: Optional[np.ndarray] = None,
    config: Optional[EventDetectionConfig] = None,
) -> SpindleResult:
    """Detect sleep spindles using YASA.

    Args:
        raw: Continuous sleep EEG.
        hypnogram: If provided, only detects in N2/N3 stages (recommended).
        config: Detection parameters.
    """
    if config is None:
        config = EventDetectionConfig()

    try:
        import yasa
    except ImportError:
        raise RuntimeError("YASA not installed. Install with: pip install yasa")

    # Convert string hypnogram to YASA int format if needed
    hyp_int = None
    if hypnogram is not None:
        stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
        hyp_int = yasa.Hypnogram(
            np.array([stage_map.get(s, 0) for s in hypnogram]),
            freq="30s",
        )

    sp = yasa.spindles_detect(
        raw,
        hypno=hyp_int,
        freq_sp=config.spindle_freq,
        duration=config.spindle_duration,
        min_distance=config.spindle_min_distance,
    )

    if sp is None:
        logger.warning("no_spindles_detected")
        return SpindleResult(
            events=pd.DataFrame(),
            n_spindles=0,
            density=0.0,
            mean_frequency=0.0,
            mean_duration=0.0,
            mean_amplitude=0.0,
        )

    summary = sp.summary()
    duration_min = raw.times[-1] / 60

    result = SpindleResult(
        events=summary,
        n_spindles=len(summary),
        density=len(summary) / duration_min if duration_min > 0 else 0,
        mean_frequency=float(summary["Frequency"].mean()) if len(summary) > 0 else 0,
        mean_duration=float(summary["Duration"].mean()) if len(summary) > 0 else 0,
        mean_amplitude=float(summary["Amplitude"].mean()) if len(summary) > 0 else 0,
    )

    logger.info(
        "spindles_detected",
        n=result.n_spindles,
        density=f"{result.density:.1f}/min",
        mean_freq=f"{result.mean_frequency:.1f} Hz",
    )
    return result


def detect_slow_oscillations(
    raw: mne.io.BaseRaw,
    hypnogram: Optional[np.ndarray] = None,
    config: Optional[EventDetectionConfig] = None,
) -> SlowOscillationResult:
    """Detect slow oscillations using YASA.

    Args:
        raw: Continuous sleep EEG.
        hypnogram: If provided, only detects in N2/N3 stages (recommended).
        config: Detection parameters.
    """
    if config is None:
        config = EventDetectionConfig()

    try:
        import yasa
    except ImportError:
        raise RuntimeError("YASA not installed. Install with: pip install yasa")

    hyp_int = None
    if hypnogram is not None:
        stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
        hyp_int = yasa.Hypnogram(
            np.array([stage_map.get(s, 0) for s in hypnogram]),
            freq="30s",
        )

    sw = yasa.sw_detect(
        raw,
        hypno=hyp_int,
        freq_sw=config.so_freq,
        dur_neg=config.so_duration,
        amp_neg=(None, None),
        amp_pos=(None, None),
    )

    if sw is None:
        logger.warning("no_slow_oscillations_detected")
        return SlowOscillationResult(
            events=pd.DataFrame(),
            n_events=0,
            density=0.0,
            mean_amplitude=0.0,
            mean_duration=0.0,
        )

    summary = sw.summary()
    duration_min = raw.times[-1] / 60

    result = SlowOscillationResult(
        events=summary,
        n_events=len(summary),
        density=len(summary) / duration_min if duration_min > 0 else 0,
        mean_amplitude=float(summary["PTP"].mean()) if len(summary) > 0 else 0,
        mean_duration=float(summary["Duration"].mean()) if len(summary) > 0 else 0,
    )

    logger.info(
        "slow_oscillations_detected",
        n=result.n_events,
        density=f"{result.density:.1f}/min",
    )
    return result
