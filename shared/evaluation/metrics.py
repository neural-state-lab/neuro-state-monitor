"""Evaluation metrics for brain state classifiers.

Standard metrics (accuracy, AUC, F1) plus EEG-specific metrics
like cross-subject generalization score.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = structlog.get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation metrics."""

    accuracy: float
    auc: float
    f1: float
    precision: float
    recall: float
    cross_subject_generalization: Optional[float] = None

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for MLflow logging."""
        metrics = {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }
        if self.cross_subject_generalization is not None:
            metrics["cross_subject_generalization"] = self.cross_subject_generalization
        return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
) -> ClassificationMetrics:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for AUC).
        average: Averaging strategy for multiclass ('binary', 'macro', 'weighted').
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)

    # AUC requires probability scores
    if y_prob is not None:
        try:
            if average == "binary":
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
        except ValueError:
            logger.warning("auc_computation_failed")
            auc = 0.0
    else:
        auc = 0.0

    metrics = ClassificationMetrics(
        accuracy=acc,
        auc=auc,
        f1=f1,
        precision=prec,
        recall=rec,
    )

    logger.info(
        "computed_metrics",
        accuracy=acc,
        auc=auc,
        f1=f1,
    )
    return metrics


def compute_cross_subject_generalization(
    within_subject_scores: list[float],
    cross_subject_scores: list[float],
) -> float:
    """Compute cross-subject generalization score.

    Ratio of cross-subject to within-subject performance.
    Score of 1.0 = perfect generalization. Lower = subject-specific.

    Args:
        within_subject_scores: Per-subject within-subject accuracies.
        cross_subject_scores: Per-subject leave-one-out accuracies.
    """
    mean_within = np.mean(within_subject_scores)
    mean_cross = np.mean(cross_subject_scores)

    if mean_within == 0:
        return 0.0

    gen_score = mean_cross / mean_within

    logger.info(
        "cross_subject_generalization",
        within=mean_within,
        cross=mean_cross,
        ratio=gen_score,
    )
    return gen_score


def compare_with_baseline(
    metrics: ClassificationMetrics,
    baseline: dict[str, float],
) -> dict[str, float]:
    """Compare metrics against a published baseline.

    Returns dict of differences (positive = better than baseline).
    """
    current = metrics.to_dict()
    diffs = {}

    for key, baseline_val in baseline.items():
        if key in current:
            diffs[f"{key}_vs_baseline"] = current[key] - baseline_val

    logger.info("baseline_comparison", diffs=diffs)
    return diffs
