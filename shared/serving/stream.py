"""Real-time EEG streaming via MNE-LSL.

Bridges Lab Streaming Layer (LSL) hardware streams with
the MNE-Python preprocessing pipeline for live inference.

MNE-LSL (JOSS 2025) replaces the older pylsl bindings with
native MNE integration.

Requires: pip install mne-lsl (optional dependency)

Usage:
    stream = EEGStreamProcessor(
        stream_name="MyEEG",
        window_seconds=1.0,
        classifier="encoding",
    )
    stream.start(callback=my_prediction_handler)
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StreamConfig:
    """Configuration for real-time EEG streaming."""

    # LSL stream settings
    stream_name: Optional[str] = None
    stream_type: str = "EEG"

    # Windowing
    window_seconds: float = 1.0
    overlap_seconds: float = 0.5
    buffer_seconds: float = 5.0

    # Preprocessing
    l_freq: float = 0.5
    h_freq: float = 45.0
    notch_freq: float = 60.0

    # Prediction
    classifier: str = "encoding"
    prediction_interval: float = 0.5  # seconds between predictions


class EEGStreamProcessor:
    """Real-time EEG stream processor using MNE-LSL.

    Reads from an LSL stream, applies sliding window epoching,
    runs preprocessing and feature extraction, and calls a
    prediction callback with results.
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
    ) -> None:
        self.config = config or StreamConfig()
        self._running = False
        self._stream = None

    def start(
        self,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """Start processing the EEG stream.

        Args:
            callback: Function called with prediction results dict.
                      Keys: predictions, probabilities, latency_ms, window_idx
        """
        if callback is None:
            callback = _default_callback

        try:
            from mne_lsl.stream import StreamLSL
        except ImportError:
            raise RuntimeError(
                "mne-lsl not installed. Install with: pip install mne-lsl"
            )

        # Connect to LSL stream
        self._stream = StreamLSL(
            bufsize=self.config.buffer_seconds,
            name=self.config.stream_name,
            stype=self.config.stream_type,
        )
        self._stream.connect(processing_flags="all")

        sfreq = self._stream.info["sfreq"]
        n_window = int(self.config.window_seconds * sfreq)
        n_overlap = int(self.config.overlap_seconds * sfreq)
        n_step = n_window - n_overlap

        logger.info(
            "stream_started",
            sfreq=sfreq,
            n_channels=self._stream.info["nchan"],
            window_samples=n_window,
            step_samples=n_step,
        )

        self._running = True
        window_idx = 0

        import time

        while self._running:
            # Get latest data from buffer
            data, timestamps = self._stream.get_data(
                winsize=self.config.window_seconds,
            )

            if data is None or data.shape[1] < n_window:
                time.sleep(0.01)
                continue

            # Process window
            result = self._process_window(data, sfreq, window_idx)

            if result is not None:
                callback(result)

            window_idx += 1
            time.sleep(self.config.prediction_interval)

    def stop(self) -> None:
        """Stop the stream processor."""
        self._running = False
        if self._stream is not None:
            self._stream.disconnect()
            logger.info("stream_stopped")

    def _process_window(
        self,
        data: np.ndarray,
        sfreq: float,
        window_idx: int,
    ) -> Optional[dict]:
        """Process a single window of EEG data.

        Args:
            data: EEG data of shape (n_channels, n_samples).
            sfreq: Sampling frequency.
            window_idx: Window counter.

        Returns:
            Dict with signal quality metrics and raw data for prediction.
        """
        import time

        import mne

        start = time.monotonic()

        # Basic signal quality check
        signal_power = np.mean(data**2, axis=1)
        quality_ok = np.all(signal_power > 1e-12) and np.all(signal_power < 1e-2)

        if not quality_ok:
            logger.warning("poor_signal_quality", window=window_idx)
            return None

        # Create MNE Raw from window
        n_channels = data.shape[0]
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw_window = mne.io.RawArray(data, info, verbose=False)

        # Apply basic filtering
        raw_window.filter(
            l_freq=self.config.l_freq,
            h_freq=self.config.h_freq,
            verbose=False,
        )

        latency = time.monotonic() - start

        return {
            "window_idx": window_idx,
            "data": raw_window.get_data(),
            "sfreq": sfreq,
            "n_channels": n_channels,
            "n_samples": data.shape[1],
            "signal_quality": float(np.mean(signal_power)),
            "preprocessing_latency_ms": round(latency * 1000, 2),
        }


def _default_callback(result: dict) -> None:
    """Default callback — just logs the result."""
    logger.info(
        "stream_window_processed",
        window=result["window_idx"],
        quality=result["signal_quality"],
        latency_ms=result["preprocessing_latency_ms"],
    )
