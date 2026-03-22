"""EEG signal quality monitoring.

Computes per-channel signal quality index (SQI) based on:
- Spectral entropy (via antropy) — high = broadband noise
- Kurtosis — high = muscle artifacts or electrode pops
- Amplitude range — too flat or too large = bad contact

No mature EEG-specific SQI library exists, so this is a thin
custom wrapper — a publishable contribution.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
from prometheus_client import Gauge

logger = structlog.get_logger(__name__)

SIGNAL_QUALITY = Gauge(
    "neuro_signal_quality",
    "Per-channel signal quality index (0=bad, 1=good)",
    ["channel"],
)
OVERALL_QUALITY = Gauge(
    "neuro_overall_signal_quality",
    "Mean signal quality across all channels",
)


@dataclass
class QualityConfig:
    """Configuration for signal quality monitoring."""

    # Spectral entropy thresholds
    entropy_min: float = 0.3  # Below = too regular (likely artifact)
    entropy_max: float = 0.95  # Above = pure noise

    # Kurtosis thresholds
    kurtosis_max: float = 10.0  # Above = spiky (muscle/pops)

    # Amplitude thresholds (microvolts)
    amplitude_min_uv: float = 1.0  # Below = flat line
    amplitude_max_uv: float = 200.0  # Above = saturation

    # Sampling frequency (for entropy computation)
    sfreq: float = 256.0


@dataclass
class ChannelQuality:
    """Quality metrics for a single channel."""

    channel: str
    sqi: float  # 0.0 (bad) to 1.0 (good)
    spectral_entropy: float
    kurtosis: float
    amplitude_range_uv: float
    is_good: bool


def compute_channel_quality(
    data: np.ndarray,
    channel_name: str,
    config: Optional[QualityConfig] = None,
) -> ChannelQuality:
    """Compute signal quality index for a single channel.

    Args:
        data: 1D signal array (n_samples,).
        channel_name: Channel name for logging.
        config: Quality thresholds.

    Returns:
        ChannelQuality with SQI and component metrics.
    """
    if config is None:
        config = QualityConfig()

    scores: list[float] = []

    # 1. Spectral entropy
    try:
        import antropy as ant

        se = ant.spectral_entropy(data, sf=config.sfreq, method="welch", normalize=True)
        se_score = 1.0 if config.entropy_min <= se <= config.entropy_max else 0.0
        scores.append(se_score)
    except Exception:
        se = 0.0
        scores.append(0.5)

    # 2. Kurtosis
    from scipy.stats import kurtosis

    kurt = float(kurtosis(data))
    kurt_score = 1.0 if abs(kurt) < config.kurtosis_max else 0.0
    scores.append(kurt_score)

    # 3. Amplitude range
    amp_range = (np.max(data) - np.min(data)) * 1e6  # convert to μV
    amp_score = (
        1.0 if config.amplitude_min_uv < amp_range < config.amplitude_max_uv else 0.0
    )
    scores.append(amp_score)

    sqi = float(np.mean(scores))
    is_good = sqi > 0.5

    return ChannelQuality(
        channel=channel_name,
        sqi=sqi,
        spectral_entropy=float(se),
        kurtosis=kurt,
        amplitude_range_uv=float(amp_range),
        is_good=is_good,
    )


def compute_signal_quality(
    data: np.ndarray,
    ch_names: list[str],
    config: Optional[QualityConfig] = None,
) -> dict:
    """Compute signal quality for all channels.

    Args:
        data: EEG data of shape (n_channels, n_samples).
        ch_names: Channel names.
        config: Quality thresholds.

    Returns:
        Dict with per-channel quality, overall SQI, and bad channels.
    """
    if config is None:
        config = QualityConfig()

    channel_results: list[ChannelQuality] = []

    for i, ch_name in enumerate(ch_names):
        if i < data.shape[0]:
            cq = compute_channel_quality(data[i], ch_name, config)
            channel_results.append(cq)

            # Update Prometheus
            SIGNAL_QUALITY.labels(channel=ch_name).set(cq.sqi)

    sqis = [cq.sqi for cq in channel_results]
    overall_sqi = float(np.mean(sqis)) if sqis else 0.0
    OVERALL_QUALITY.set(overall_sqi)

    bad_channels = [cq.channel for cq in channel_results if not cq.is_good]
    good_channels = [cq.channel for cq in channel_results if cq.is_good]

    result = {
        "overall_sqi": overall_sqi,
        "n_good_channels": len(good_channels),
        "n_bad_channels": len(bad_channels),
        "bad_channels": bad_channels,
        "channel_details": {
            cq.channel: {
                "sqi": cq.sqi,
                "spectral_entropy": cq.spectral_entropy,
                "kurtosis": cq.kurtosis,
                "amplitude_range_uv": cq.amplitude_range_uv,
                "is_good": cq.is_good,
            }
            for cq in channel_results
        },
    }

    logger.info(
        "signal_quality_computed",
        overall_sqi=overall_sqi,
        n_good=len(good_channels),
        n_bad=len(bad_channels),
    )

    return result
