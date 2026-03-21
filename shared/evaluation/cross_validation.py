"""Cross-validation strategies for EEG classifiers.

Provides leave-one-subject-out and grouped k-fold CV
to prevent data leakage across subjects.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import structlog
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)

from shared.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)

logger = structlog.get_logger(__name__)


@dataclass
class CVResult:
    """Result of cross-validation."""

    fold_metrics: list[ClassificationMetrics]
    mean_metrics: ClassificationMetrics
    std_metrics: dict[str, float]
    fold_subjects: list[list[str]]

    def to_dict(self) -> dict[str, float]:
        """Flatten to dict for MLflow logging."""
        result = {}
        for key, val in self.mean_metrics.to_dict().items():
            result[f"cv_mean_{key}"] = val
        for key, val in self.std_metrics.items():
            result[f"cv_std_{key}"] = val
        result["cv_n_folds"] = len(self.fold_metrics)
        return result


def leave_one_subject_out(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    fit_predict_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, Optional[np.ndarray]],
    ],
    average: str = "binary",
) -> CVResult:
    """Leave-one-subject-out cross-validation.

    Args:
        X: Features, shape (n_samples, n_features).
        y: Labels, shape (n_samples,).
        subjects: Subject IDs, shape (n_samples,).
        fit_predict_fn: Callable(X_train, y_train, X_test) -> (y_pred, y_prob).
        average: Metric averaging strategy.

    Returns:
        CVResult with per-fold and aggregated metrics.
    """
    logo = LeaveOneGroupOut()
    fold_metrics: list[ClassificationMetrics] = []
    fold_subjects: list[list[str]] = []

    unique_subjects = np.unique(subjects)
    logger.info("starting_loso_cv", n_subjects=len(unique_subjects))

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=subjects)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subj = list(np.unique(subjects[test_idx]))

        y_pred, y_prob = fit_predict_fn(X_train, y_train, X_test)
        metrics = compute_classification_metrics(
            y_test, y_pred, y_prob, average=average
        )
        fold_metrics.append(metrics)
        fold_subjects.append(test_subj)

        logger.info(
            "loso_fold_complete",
            fold=fold_idx,
            test_subject=test_subj,
            accuracy=metrics.accuracy,
        )

    return _aggregate_cv_results(fold_metrics, fold_subjects)


def grouped_kfold(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    fit_predict_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, Optional[np.ndarray]],
    ],
    n_folds: int = 5,
    stratified: bool = True,
    average: str = "binary",
) -> CVResult:
    """K-fold cross-validation with subject grouping.

    Ensures no subject appears in both train and test within a fold.

    Args:
        X: Features, shape (n_samples, n_features).
        y: Labels, shape (n_samples,).
        subjects: Subject IDs, shape (n_samples,).
        fit_predict_fn: Callable(X_train, y_train, X_test) -> (y_pred, y_prob).
        n_folds: Number of folds.
        stratified: Whether to stratify by label distribution.
        average: Metric averaging strategy.
    """
    if stratified:
        cv = StratifiedGroupKFold(n_splits=n_folds)
    else:
        cv = GroupKFold(n_splits=n_folds)

    fold_metrics: list[ClassificationMetrics] = []
    fold_subjects: list[list[str]] = []

    logger.info(
        "starting_grouped_kfold",
        n_folds=n_folds,
        stratified=stratified,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=subjects)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subj = list(np.unique(subjects[test_idx].astype(str)))

        y_pred, y_prob = fit_predict_fn(X_train, y_train, X_test)
        metrics = compute_classification_metrics(
            y_test, y_pred, y_prob, average=average
        )
        fold_metrics.append(metrics)
        fold_subjects.append(test_subj)

        logger.info(
            "kfold_complete",
            fold=fold_idx,
            n_test_subjects=len(test_subj),
            accuracy=metrics.accuracy,
        )

    return _aggregate_cv_results(fold_metrics, fold_subjects)


def _aggregate_cv_results(
    fold_metrics: list[ClassificationMetrics],
    fold_subjects: list[list[str]],
) -> CVResult:
    """Aggregate per-fold metrics into mean/std."""
    metric_dicts = [m.to_dict() for m in fold_metrics]
    keys = metric_dicts[0].keys()

    mean_values = {key: np.mean([d[key] for d in metric_dicts]) for key in keys}
    std_values = {key: np.std([d[key] for d in metric_dicts]) for key in keys}

    mean_metrics = ClassificationMetrics(
        accuracy=mean_values["accuracy"],
        auc=mean_values["auc"],
        f1=mean_values["f1"],
        precision=mean_values["precision"],
        recall=mean_values["recall"],
    )

    logger.info(
        "cv_complete",
        n_folds=len(fold_metrics),
        mean_accuracy=mean_metrics.accuracy,
        mean_auc=mean_metrics.auc,
    )

    return CVResult(
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        std_metrics=std_values,
        fold_subjects=fold_subjects,
    )
