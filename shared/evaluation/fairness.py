"""Fairness analysis for brain state classifiers.

Evaluates model performance across demographic subgroups
(age, gender, etc.) to detect bias.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

from shared.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)

logger = structlog.get_logger(__name__)


@dataclass
class SubgroupResult:
    """Metrics for a single demographic subgroup."""

    group_name: str
    group_value: str
    n_samples: int
    metrics: ClassificationMetrics


@dataclass
class FairnessReport:
    """Complete fairness analysis report."""

    subgroup_results: list[SubgroupResult]
    max_disparity: dict[str, float]
    flagged_metrics: list[str]

    def to_dict(self) -> dict[str, float]:
        """Flatten for MLflow logging."""
        result: dict[str, float] = {}
        for sr in self.subgroup_results:
            prefix = f"fairness_{sr.group_name}_{sr.group_value}"
            for key, val in sr.metrics.to_dict().items():
                result[f"{prefix}_{key}"] = val
            result[f"{prefix}_n"] = sr.n_samples

        for key, val in self.max_disparity.items():
            result[f"max_disparity_{key}"] = val

        return result


def analyze_subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    group_name: str,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
    disparity_threshold: float = 0.1,
) -> FairnessReport:
    """Analyze classifier performance across demographic subgroups.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        group_labels: Demographic group for each sample.
        group_name: Name of the demographic dimension (e.g., "gender", "age_group").
        y_prob: Predicted probabilities.
        average: Metric averaging strategy.
        disparity_threshold: Flag metrics with disparity above this.
    """
    unique_groups = np.unique(group_labels)
    subgroup_results: list[SubgroupResult] = []

    for group_val in unique_groups:
        mask = group_labels == group_val
        n_samples = mask.sum()

        if n_samples < 5:
            logger.warning(
                "small_subgroup",
                group=group_name,
                value=str(group_val),
                n=n_samples,
            )
            continue

        group_prob = y_prob[mask] if y_prob is not None else None
        metrics = compute_classification_metrics(
            y_true[mask], y_pred[mask], group_prob, average=average
        )

        subgroup_results.append(
            SubgroupResult(
                group_name=group_name,
                group_value=str(group_val),
                n_samples=int(n_samples),
                metrics=metrics,
            )
        )

    # Compute max disparity per metric
    max_disparity: dict[str, float] = {}
    flagged: list[str] = []

    if len(subgroup_results) >= 2:
        metric_names = ["accuracy", "auc", "f1", "precision", "recall"]
        for metric_name in metric_names:
            values = [
                getattr(sr.metrics, metric_name) for sr in subgroup_results
            ]
            disparity = max(values) - min(values)
            max_disparity[metric_name] = disparity

            if disparity > disparity_threshold:
                flagged.append(metric_name)
                logger.warning(
                    "fairness_disparity",
                    group=group_name,
                    metric=metric_name,
                    disparity=disparity,
                )

    report = FairnessReport(
        subgroup_results=subgroup_results,
        max_disparity=max_disparity,
        flagged_metrics=flagged,
    )

    logger.info(
        "fairness_analysis_complete",
        group=group_name,
        n_subgroups=len(subgroup_results),
        n_flagged=len(flagged),
    )
    return report


def analyze_multiple_groups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    demographics: dict[str, np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
) -> dict[str, FairnessReport]:
    """Run fairness analysis across multiple demographic dimensions.

    Args:
        demographics: Dict mapping group name to group labels array.
                      e.g., {"gender": gender_array, "age_group": age_array}
    """
    reports: dict[str, FairnessReport] = {}

    for group_name, group_labels in demographics.items():
        reports[group_name] = analyze_subgroup_performance(
            y_true, y_pred, group_labels, group_name, y_prob, average
        )

    return reports
