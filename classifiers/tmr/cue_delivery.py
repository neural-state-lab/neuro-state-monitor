"""TMR cue delivery logic.

Determines WHEN to play audio cues during sleep for
Targeted Memory Reactivation. Cues should be delivered:
1. During N2 or N3 sleep (not wake, not REM)
2. Ideally during or just after a sleep spindle
3. During the UP state of a slow oscillation
4. Not too frequently (minimum 5s between cues)

This module makes the DECISION of when to cue.
Actual audio playback requires hardware integration (out of scope).
"""

from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CueConfig:
    """Configuration for TMR cue delivery."""

    # Sleep stage requirements
    allowed_stages: tuple[str, ...] = ("N2", "N3")

    # Timing constraints
    min_interval_sec: float = 5.0  # minimum seconds between cues
    max_cues_per_minute: int = 6

    # Spindle coupling
    prefer_spindle_coupling: bool = True
    spindle_window_sec: float = 1.0  # deliver within 1s after spindle onset

    # Confidence thresholds
    min_stage_confidence: float = 0.7  # minimum staging confidence to deliver


@dataclass
class CueDecision:
    """Decision about whether to deliver a TMR cue."""

    should_cue: bool
    reason: str
    current_stage: str
    stage_confidence: float
    seconds_since_last_cue: float
    spindle_nearby: bool


class TMRCueController:
    """Controls TMR cue delivery timing.

    Call check_cue_opportunity() at regular intervals (e.g., every 100ms)
    during real-time sleep monitoring. It returns a CueDecision indicating
    whether now is a good time to play a memory cue.
    """

    def __init__(self, config: Optional[CueConfig] = None) -> None:
        self.config = config or CueConfig()
        self._last_cue_time: Optional[float] = None
        self._cues_this_minute: int = 0
        self._minute_start: float = 0.0
        self._total_cues: int = 0

    def check_cue_opportunity(
        self,
        current_time: float,
        current_stage: str,
        stage_confidence: float,
        spindle_detected: bool = False,
    ) -> CueDecision:
        """Check if now is a good time to deliver a TMR cue.

        Args:
            current_time: Current time in seconds from recording start.
            current_stage: Current sleep stage ("W", "N1", "N2", "N3", "R").
            stage_confidence: Confidence of stage classification (0-1).
            spindle_detected: Whether a spindle was just detected.

        Returns:
            CueDecision with recommendation and reasoning.
        """
        # Reset per-minute counter
        if current_time - self._minute_start > 60:
            self._cues_this_minute = 0
            self._minute_start = current_time

        seconds_since_last = (
            current_time - self._last_cue_time
            if self._last_cue_time is not None
            else float("inf")
        )

        # Check: correct sleep stage?
        if current_stage not in self.config.allowed_stages:
            return CueDecision(
                should_cue=False,
                reason=f"wrong_stage ({current_stage})",
                current_stage=current_stage,
                stage_confidence=stage_confidence,
                seconds_since_last_cue=seconds_since_last,
                spindle_nearby=spindle_detected,
            )

        # Check: staging confident enough?
        if stage_confidence < self.config.min_stage_confidence:
            return CueDecision(
                should_cue=False,
                reason=f"low_confidence ({stage_confidence:.2f})",
                current_stage=current_stage,
                stage_confidence=stage_confidence,
                seconds_since_last_cue=seconds_since_last,
                spindle_nearby=spindle_detected,
            )

        # Check: enough time since last cue?
        if seconds_since_last < self.config.min_interval_sec:
            return CueDecision(
                should_cue=False,
                reason=f"too_soon ({seconds_since_last:.1f}s)",
                current_stage=current_stage,
                stage_confidence=stage_confidence,
                seconds_since_last_cue=seconds_since_last,
                spindle_nearby=spindle_detected,
            )

        # Check: rate limit
        if self._cues_this_minute >= self.config.max_cues_per_minute:
            return CueDecision(
                should_cue=False,
                reason="rate_limited",
                current_stage=current_stage,
                stage_confidence=stage_confidence,
                seconds_since_last_cue=seconds_since_last,
                spindle_nearby=spindle_detected,
            )

        # Prefer spindle coupling if configured
        if self.config.prefer_spindle_coupling and not spindle_detected:
            return CueDecision(
                should_cue=False,
                reason="waiting_for_spindle",
                current_stage=current_stage,
                stage_confidence=stage_confidence,
                seconds_since_last_cue=seconds_since_last,
                spindle_nearby=False,
            )

        # All checks passed — deliver cue!
        self._last_cue_time = current_time
        self._cues_this_minute += 1
        self._total_cues += 1

        logger.info(
            "cue_delivered",
            time=current_time,
            stage=current_stage,
            confidence=stage_confidence,
            spindle=spindle_detected,
            total_cues=self._total_cues,
        )

        return CueDecision(
            should_cue=True,
            reason="all_checks_passed",
            current_stage=current_stage,
            stage_confidence=stage_confidence,
            seconds_since_last_cue=seconds_since_last,
            spindle_nearby=spindle_detected,
        )

    def get_stats(self) -> dict:
        """Get TMR session statistics."""
        return {
            "total_cues_delivered": self._total_cues,
            "cues_this_minute": self._cues_this_minute,
        }
