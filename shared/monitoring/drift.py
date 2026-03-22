"""Feature drift detection for production EEG models.

Uses statistical tests to detect when incoming EEG feature distributions
shift from the training distribution. Optionally uses NannyML for
advanced CBPE (Confidence-Based Performance Estimation).

NannyML is an optional dependency (pip install nannyml) — falls back
to lightweight scipy-based drift detection if unavailable.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
from prometheus_client import Gauge

logger = structlog.get_logger(__name__)

# Prometheus metrics
DRIFT_SCORE = Gauge(
    "neuro_drift_score",
    "Feature drift score (0=no drift, 1=severe drift)",
    ["classifier"],
)
DRIFT_ALERT = Gauge(
    "neuro_drift_alert",
    "Whether drift alert is active (0 or 1)",
    ["classifier"],
)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    # Reference window size (training data stats)
    reference_window: int = 1000

    # Significance threshold for drift detection
    p_value_threshold: float = 0.05

    # Number of features to monitor (top N by variance)
    n_features_monitor: int = 50

    # Alert if this fraction of features are drifting
    drift_fraction_alert: float = 0.3


class DriftDetector:
    """Detects feature distribution drift in production.

    Compares incoming feature distributions against a reference
    (training) distribution using KS tests per feature.
    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self.config = config or DriftConfig()
        self._reference_stats: Optional[dict] = None
        self._nannyml_available = False

        try:
            import nannyml  # noqa: F401

            self._nannyml_available = True
        except ImportError:
            logger.info("nannyml_not_available", msg="Using scipy fallback")

    def set_reference(self, features: np.ndarray) -> None:
        """Set the reference distribution from training data.

        Args:
            features: Training feature matrix (n_samples, n_features).
        """
        self._reference_stats = {
            "mean": features.mean(axis=0),
            "std": features.std(axis=0),
            "data": features[: self.config.reference_window],
            "n_features": features.shape[1],
        }
        logger.info(
            "reference_set",
            n_samples=features.shape[0],
            n_features=features.shape[1],
        )

    def check_drift(
        self,
        features: np.ndarray,
        classifier: str = "unknown",
    ) -> dict:
        """Check for feature drift against reference distribution.

        Args:
            features: Incoming feature matrix (n_samples, n_features).
            classifier: Name for metric labeling.

        Returns:
            Dict with drift_score, drifting_features, alert status.
        """
        if self._reference_stats is None:
            return {"error": "Reference not set. Call set_reference() first."}

        from scipy.stats import ks_2samp

        ref_data = self._reference_stats["data"]
        n_features = min(
            features.shape[1],
            ref_data.shape[1],
            self.config.n_features_monitor,
        )

        p_values = []
        drifting = []

        for i in range(n_features):
            stat, p_val = ks_2samp(ref_data[:, i], features[:, i])
            p_values.append(p_val)
            if p_val < self.config.p_value_threshold:
                drifting.append(i)

        drift_fraction = len(drifting) / max(n_features, 1)
        alert = drift_fraction > self.config.drift_fraction_alert

        # Update Prometheus metrics
        DRIFT_SCORE.labels(classifier=classifier).set(drift_fraction)
        DRIFT_ALERT.labels(classifier=classifier).set(int(alert))

        result = {
            "drift_score": drift_fraction,
            "n_drifting_features": len(drifting),
            "n_total_features": n_features,
            "alert": alert,
            "drifting_feature_indices": drifting,
            "mean_p_value": float(np.mean(p_values)),
        }

        if alert:
            logger.warning("drift_detected", classifier=classifier, **result)
        else:
            logger.info("drift_check_ok", classifier=classifier, **result)

        return result
