"""Shared test fixtures and synthetic EEG data generators."""

import mne
import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.RandomState:
    """Seeded random number generator for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def n_channels() -> int:
    return 8


@pytest.fixture
def sfreq() -> float:
    return 256.0


@pytest.fixture
def duration() -> float:
    """Duration in seconds."""
    return 10.0


@pytest.fixture
def synthetic_raw(
    rng: np.random.RandomState,
    n_channels: int,
    sfreq: float,
    duration: float,
) -> mne.io.RawArray:
    """Create synthetic raw EEG data with realistic-ish properties.

    Generates a mix of sine waves at standard EEG frequencies
    plus white noise.
    """
    n_samples = int(sfreq * duration)
    times = np.arange(n_samples) / sfreq

    # Use real 10-20 names so montage/digitization works
    standard_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
                      "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8"]
    ch_names = standard_names[:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Build signal: sum of sinusoids at typical EEG frequencies
    data = np.zeros((n_channels, n_samples))
    freqs_and_amps = [
        (2.0, 5e-6),    # delta
        (6.0, 8e-6),    # theta
        (10.0, 10e-6),  # alpha
        (20.0, 4e-6),   # beta
        (35.0, 2e-6),   # gamma
    ]

    for ch in range(n_channels):
        for freq, amp in freqs_and_amps:
            phase = rng.uniform(0, 2 * np.pi)
            # Vary amplitude slightly per channel
            ch_amp = amp * (0.8 + 0.4 * rng.random())
            data[ch] += ch_amp * np.sin(2 * np.pi * freq * times + phase)

        # Add white noise
        data[ch] += rng.randn(n_samples) * 1e-6

    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw


@pytest.fixture
def synthetic_events(
    sfreq: float,
    duration: float,
) -> np.ndarray:
    """Create synthetic event array with two event types."""
    n_events = 20
    # Space events evenly, starting at 0.5s
    event_samples = np.linspace(
        int(0.5 * sfreq),
        int((duration - 1) * sfreq),
        n_events,
        dtype=int,
    )
    events = np.column_stack([
        event_samples,
        np.zeros(n_events, dtype=int),
        np.tile([1, 2], n_events // 2),  # alternating event IDs
    ])
    return events


@pytest.fixture
def synthetic_epochs(
    synthetic_raw: mne.io.RawArray,
    synthetic_events: np.ndarray,
) -> mne.Epochs:
    """Create synthetic epochs from raw data and events."""
    epochs = mne.Epochs(
        synthetic_raw,
        events=synthetic_events,
        event_id={"target": 1, "nontarget": 2},
        tmin=-0.2,
        tmax=0.8,
        baseline=(None, 0),
        preload=True,
        verbose=False,
    )
    return epochs


@pytest.fixture
def synthetic_labels(synthetic_epochs: mne.Epochs) -> np.ndarray:
    """Binary labels aligned with synthetic epochs."""
    return (synthetic_epochs.events[:, 2] == 1).astype(int)


@pytest.fixture
def synthetic_subjects(synthetic_epochs: mne.Epochs) -> np.ndarray:
    """Subject IDs for synthetic epochs (3 pseudo-subjects)."""
    n = len(synthetic_epochs)
    return np.array([f"sub-{i % 3:02d}" for i in range(n)])
