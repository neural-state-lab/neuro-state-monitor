"""End-to-end training script for the encoding classifier.

Pipeline: load data → preprocess → extract features → train → evaluate → log to MLflow.
Configurable via YAML. Supports cross-subject validation.

Usage:
    python -m classifiers.encoding.train --config experiments/encoding_config.yaml
    python -m classifiers.encoding.train --model svm --features spectral
"""

import argparse
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import structlog
import yaml

from classifiers.encoding.data import (
    EncodingDataset,
    PEERSConfig,
    load_peers_dataset,
    merge_datasets,
    split_by_subject,
)
from classifiers.encoding.features import (
    EncodingFeatureConfig,
    extract_encoding_features,
)
from classifiers.encoding.models import BaseEncodingModel, ModelResult, create_model
from shared.evaluation.cross_validation import CVResult, grouped_kfold, leave_one_subject_out
from shared.evaluation.metrics import (
    ClassificationMetrics,
    compare_with_baseline,
    compute_classification_metrics,
)
from shared.experiment import log_metrics, log_model_metadata, setup_experiment

logger = structlog.get_logger(__name__)

# Published baseline to beat
EEGITNET_BASELINE = {
    "auc": 0.68,
    "accuracy": 0.65,
    "f1": 0.63,
}


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load training configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "experiments" / "base_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("loaded_training_config", path=str(config_path))
    return config


def prepare_features(
    dataset: EncodingDataset,
    model_type: str,
    feature_config: Optional[EncodingFeatureConfig] = None,
) -> np.ndarray:
    """Prepare features based on model type.

    SVM models need extracted feature vectors.
    CNN models work on raw epoch data directly.
    """
    if model_type == "svm":
        return extract_encoding_features(dataset.epochs, feature_config)
    else:
        # CNN models use raw epoch data: (n_epochs, n_channels, n_times)
        return dataset.epochs.get_data().astype(np.float32)


def train_and_evaluate(
    train_data: EncodingDataset,
    test_data: EncodingDataset,
    model_type: str = "svm",
    model_kwargs: Optional[dict] = None,
    feature_config: Optional[EncodingFeatureConfig] = None,
) -> tuple[BaseEncodingModel, ClassificationMetrics, ModelResult]:
    """Train a model and evaluate on test data.

    Returns the trained model, metrics, and predictions.
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Prepare features
    X_train = prepare_features(train_data, model_type, feature_config)
    X_test = prepare_features(test_data, model_type, feature_config)
    y_train = train_data.labels
    y_test = test_data.labels

    # Create and train model
    if model_type != "svm":
        model_kwargs["n_channels"] = X_train.shape[1]
        model_kwargs["n_times"] = X_train.shape[2]

    model = create_model(model_type, **model_kwargs)
    model.fit(X_train, y_train)

    # Predict and evaluate
    result = model.predict(X_test)
    metrics = compute_classification_metrics(
        y_test, result.y_pred, result.y_prob
    )

    logger.info(
        "train_and_evaluate_complete",
        model=model_type,
        accuracy=metrics.accuracy,
        auc=metrics.auc,
        f1=metrics.f1,
    )

    return model, metrics, result


def run_cross_validation(
    dataset: EncodingDataset,
    model_type: str = "svm",
    model_kwargs: Optional[dict] = None,
    feature_config: Optional[EncodingFeatureConfig] = None,
    cv_method: str = "loso",
    n_folds: int = 5,
) -> CVResult:
    """Run cross-subject validation on the dataset.

    Args:
        cv_method: "loso" (leave-one-subject-out) or "kfold" (grouped k-fold).
    """
    if model_kwargs is None:
        model_kwargs = {}

    X = prepare_features(dataset, model_type, feature_config)
    y = dataset.labels
    subjects = dataset.subjects

    def fit_predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        kwargs = dict(model_kwargs)
        if model_type != "svm":
            kwargs["n_channels"] = X_train.shape[1]
            kwargs["n_times"] = X_train.shape[2]

        model = create_model(model_type, **kwargs)
        model.fit(X_train, y_train)
        result = model.predict(X_test)
        return result.y_pred, result.y_prob

    if cv_method == "loso":
        cv_result = leave_one_subject_out(X, y, subjects, fit_predict_fn)
    else:
        cv_result = grouped_kfold(
            X, y, subjects, fit_predict_fn, n_folds=n_folds
        )

    logger.info(
        "cross_validation_complete",
        method=cv_method,
        mean_auc=cv_result.mean_metrics.auc,
        mean_accuracy=cv_result.mean_metrics.accuracy,
    )

    return cv_result


def run_experiment(
    config_path: Optional[Path] = None,
    model_type: str = "svm",
    cv_method: str = "kfold",
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "encoding-classifier",
) -> dict[str, Any]:
    """Run a complete encoding classifier experiment.

    Full pipeline: load data → preprocess → features → train → evaluate → log.

    Returns dict with metrics, cv_results, and model.
    """
    # Load config
    config = load_config(config_path)

    # Setup MLflow
    try:
        setup_experiment(experiment_name, tracking_uri)
        use_mlflow = True
    except Exception:
        logger.warning("mlflow_unavailable", msg="Running without MLflow tracking")
        use_mlflow = False

    # Load data
    peers_config = PEERSConfig()
    datasets = load_peers_dataset(peers_config)
    dataset = merge_datasets(datasets)

    # Split by subject
    split = split_by_subject(dataset, peers_config)

    # Feature config
    feature_config = EncodingFeatureConfig()

    # Start MLflow run
    run_context = mlflow.start_run(run_name=f"encoding-{model_type}") if use_mlflow else _noop_context()

    with run_context:
        if use_mlflow:
            # Log parameters
            mlflow.log_params({
                "model_type": model_type,
                "cv_method": cv_method,
                "n_train_subjects": len(split.train_subjects),
                "n_val_subjects": len(split.val_subjects),
                "n_test_subjects": len(split.test_subjects),
                "n_train_epochs": len(split.train.epochs),
                "n_test_epochs": len(split.test.epochs),
                "feature_tmin": feature_config.tmin,
                "feature_tmax": feature_config.tmax,
            })

            log_model_metadata(
                dataset_accession="ds004106",
                preprocessing_version="v1.0",
                participant_count=len(np.unique(dataset.subjects)),
                classifier_type=model_type,
                target_state="encoding_success",
            )

        # Train and evaluate on held-out test set
        model, test_metrics, predictions = train_and_evaluate(
            train_data=split.train,
            test_data=split.test,
            model_type=model_type,
            feature_config=feature_config,
        )

        # Cross-validation on training data
        train_val = merge_datasets([split.train, split.val])
        cv_result = run_cross_validation(
            dataset=train_val,
            model_type=model_type,
            feature_config=feature_config,
            cv_method=cv_method,
        )

        # Compare with baseline
        baseline_diff = compare_with_baseline(test_metrics, EEGITNET_BASELINE)

        # Log everything to MLflow
        if use_mlflow:
            log_metrics(test_metrics.to_dict())
            log_metrics(cv_result.to_dict())
            for key, val in baseline_diff.items():
                mlflow.log_metric(key, val)

            # Log model
            if model_type == "svm":
                mlflow.sklearn.log_model(model.pipeline, "model")

        # Print results
        logger.info(
            "experiment_complete",
            model=model_type,
            test_accuracy=test_metrics.accuracy,
            test_auc=test_metrics.auc,
            test_f1=test_metrics.f1,
            cv_mean_auc=cv_result.mean_metrics.auc,
            baseline_diff=baseline_diff,
        )

    return {
        "model": model,
        "test_metrics": test_metrics,
        "cv_result": cv_result,
        "baseline_diff": baseline_diff,
        "predictions": predictions,
    }


class _noop_context:
    """No-op context manager when MLflow is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train encoding classifier"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=["svm", "eegnet", "shallownet"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="kfold",
        choices=["loso", "kfold"],
        help="Cross-validation method",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="encoding-classifier",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        model_type=args.model,
        cv_method=args.cv,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()
