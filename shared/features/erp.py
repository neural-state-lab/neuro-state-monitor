"""Event-related potential (ERP) feature extraction.

Extracts ERP components (P300, N400, etc.) by measuring amplitude
and latency within defined time windows.
"""

from dataclasses import dataclass, field
from typing import Optional

import mne
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ERPComponent:
    """Definition of an ERP component."""

    name: str
    tmin: float
    tmax: float
    polarity: str = "positive"  # "positive" or "negative"
    channels: Optional[list[str]] = None


@dataclass
class ERPConfig:
    """Configuration for ERP feature extraction."""

    components: list[ERPComponent] = field(
        default_factory=lambda: [
            ERPComponent(name="P300", tmin=0.25, tmax=0.5, polarity="positive"),
            ERPComponent(name="N400", tmin=0.3, tmax=0.6, polarity="negative"),
        ]
    )


def compute_erp(epochs: mne.Epochs) -> mne.Evoked:
    """Compute the grand average ERP from epochs."""
    evoked = epochs.average()
    logger.info(
        "computed_erp",
        n_epochs=epochs.events.shape[0],
        n_channels=len(evoked.ch_names),
    )
    return evoked


def extract_component_amplitude(
    epochs: mne.Epochs,
    component: ERPComponent,
) -> np.ndarray:
    """Extract peak amplitude of an ERP component per epoch.

    Returns:
        Array of shape (n_epochs, n_channels) with peak amplitudes.
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    times = epochs.times

    time_mask = (times >= component.tmin) & (times <= component.tmax)
    window_data = data[:, :, time_mask]

    if component.polarity == "positive":
        amplitudes = window_data.max(axis=2)
    else:
        amplitudes = window_data.min(axis=2)

    # If specific channels requested, filter
    if component.channels:
        ch_indices = [
            epochs.ch_names.index(ch)
            for ch in component.channels
            if ch in epochs.ch_names
        ]
        if ch_indices:
            amplitudes = amplitudes[:, ch_indices]

    logger.info(
        "extracted_component_amplitude",
        component=component.name,
        shape=amplitudes.shape,
        polarity=component.polarity,
    )
    return amplitudes


def extract_component_latency(
    epochs: mne.Epochs,
    component: ERPComponent,
) -> np.ndarray:
    """Extract peak latency of an ERP component per epoch.

    Returns:
        Array of shape (n_epochs, n_channels) with peak latencies in seconds.
    """
    data = epochs.get_data()
    times = epochs.times

    time_mask = (times >= component.tmin) & (times <= component.tmax)
    window_data = data[:, :, time_mask]
    window_times = times[time_mask]

    if component.polarity == "positive":
        peak_indices = window_data.argmax(axis=2)
    else:
        peak_indices = window_data.argmin(axis=2)

    latencies = window_times[peak_indices]

    logger.info(
        "extracted_component_latency",
        component=component.name,
        shape=latencies.shape,
    )
    return latencies


def extract_mean_amplitude(
    epochs: mne.Epochs,
    component: ERPComponent,
) -> np.ndarray:
    """Extract mean amplitude within component time window.

    More robust than peak amplitude for noisy single-trial data.

    Returns:
        Array of shape (n_epochs, n_channels).
    """
    data = epochs.get_data()
    times = epochs.times

    time_mask = (times >= component.tmin) & (times <= component.tmax)
    window_data = data[:, :, time_mask]
    mean_amp = window_data.mean(axis=2)

    if component.channels:
        ch_indices = [
            epochs.ch_names.index(ch)
            for ch in component.channels
            if ch in epochs.ch_names
        ]
        if ch_indices:
            mean_amp = mean_amp[:, ch_indices]

    logger.info(
        "extracted_mean_amplitude",
        component=component.name,
        shape=mean_amp.shape,
    )
    return mean_amp


def extract_erp_features(
    epochs: mne.Epochs,
    config: Optional[ERPConfig] = None,
) -> np.ndarray:
    """Extract all ERP features as a flat feature vector per epoch.

    For each component, extracts peak amplitude, peak latency,
    and mean amplitude, then concatenates.

    Returns:
        Array of shape (n_epochs, n_features).
    """
    if config is None:
        config = ERPConfig()

    feature_arrays: list[np.ndarray] = []

    for component in config.components:
        amp = extract_component_amplitude(epochs, component)
        lat = extract_component_latency(epochs, component)
        mean_amp = extract_mean_amplitude(epochs, component)
        feature_arrays.extend([amp, lat, mean_amp])

    features = np.concatenate(feature_arrays, axis=1)

    logger.info(
        "extracted_erp_features",
        shape=features.shape,
        components=[c.name for c in config.components],
    )
    return features
