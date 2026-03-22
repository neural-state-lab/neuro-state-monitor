"""End-to-end training script for the addiction classifier.

Pipeline: load UCI data → preprocess → extract features → train → evaluate → log.
Configurable via YAML. Supports cross-subject validation.

Usage:
    python -m classifiers.addiction.train --model svm --cv kfold
    python -m classifiers.addiction.train --model eegnet --cv loso
"""

import argparse
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import structlog
import yaml

from classifiers.addiction.data import (
    AddictionConfig,
    AddictionDataset,
    dataset_to_epochs,
    load_uci_dataset,
    split_by_subject,
)
from classifiers.addiction.features import (
    AddictionFeatureConfig,
    extract_addiction_features,
)
from classifiers.addiction.models import BaseAddictionModel, create_model
from shared.evaluation.cross_validation import (
    CVResult,
    grouped_kfold,
    leave_one_subject_out,
)
from shared.evaluation.metrics import (
    compare_with_baseline,
    compute_classification_metrics,
)
from shared.experiment import log_metrics, log_model_metadata, setup_experiment

logger = structlog.get_logger(__name__)

# Published baselines
PUBLISHED_BASELINES = {
    "cnn_uci": {"accuracy": 0.95, "auc": 0.97, "f1": 0.95},
    "svm_spectral": {"accuracy": 0.85, "auc": 0.90, "f1": 0.84},
}


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load training configuration from YAML."""
    if config_path is None:
        config_path = (
            Path(__file__).parents[2] / "experiments" / "addiction_config.yaml"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("loaded_training_config", path=str(config_path))
    return config


def prepare_features(
    dataset: AddictionDataset,
    model_type: str,
    feature_config: Optional[AddictionFeatureConfig] = None,
) -> np.ndarray:
    """Prepare features based on model type."""
    if model_type in ("svm", "random_forest"):
        # Feature-based models need extracted features
        epochs = dataset_to_epochs(dataset, preprocess=True)
        return extract_addiction_features(epochs, feature_config)
    else:
        # CNN models use raw epoch data
        return dataset.data.astype(np.float32)


def train_and_evaluate(
    train_data: AddictionDataset,
    test_data: AddictionDataset,
    model_type: str = "svm",
    model_kwargs: Optional[dict] = None,
    feature_config: Optional[AddictionFeatureConfig] = None,
) -> tuple[BaseAddictionModel, Any, Any]:
    """Train a model and evaluate on test data."""
    if model_kwargs is None:
        model_kwargs = {}

    X_train = prepare_features(train_data, model_type, feature_config)
    X_test = prepare_features(test_data, model_type, feature_config)
    y_train = train_data.labels
    y_test = test_data.labels

    if model_type not in ("svm", "random_forest"):
        model_kwargs["n_channels"] = X_train.shape[1]
        model_kwargs["n_times"] = X_train.shape[2]

    model = create_model(model_type, **model_kwargs)
    model.fit(X_train, y_train)

    result = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, result.y_pred, result.y_prob)

    logger.info(
        "train_and_evaluate_complete",
        model=model_type,
        accuracy=metrics.accuracy,
        auc=metrics.auc,
        f1=metrics.f1,
    )

    return model, metrics, result


def run_cross_validation(
    dataset: AddictionDataset,
    model_type: str = "svm",
    model_kwargs: Optional[dict] = None,
    feature_config: Optional[AddictionFeatureConfig] = None,
    cv_method: str = "kfold",
    n_folds: int = 5,
) -> CVResult:
    """Run cross-subject validation."""
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
        if model_type not in ("svm", "random_forest"):
            kwargs["n_channels"] = X_train.shape[1]
            kwargs["n_times"] = X_train.shape[2]

        model = create_model(model_type, **kwargs)
        model.fit(X_train, y_train)
        result = model.predict(X_test)
        return result.y_pred, result.y_prob

    if cv_method == "loso":
        return leave_one_subject_out(X, y, subjects, fit_predict_fn)
    else:
        return grouped_kfold(X, y, subjects, fit_predict_fn, n_folds=n_folds)


def run_experiment(
    config_path: Optional[Path] = None,
    model_type: str = "svm",
    cv_method: str = "kfold",
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "addiction-classifier",
) -> dict[str, Any]:
    """Run a complete addiction classifier experiment."""
    _config = load_config(config_path)  # noqa: F841

    try:
        setup_experiment(experiment_name, tracking_uri)
        use_mlflow = True
    except Exception:
        logger.warning("mlflow_unavailable")
        use_mlflow = False

    # Load data
    data_config = AddictionConfig()
    dataset = load_uci_dataset(data_config)
    split = split_by_subject(dataset, data_config)

    feature_config = AddictionFeatureConfig()

    run_context = (
        mlflow.start_run(run_name=f"addiction-{model_type}")
        if use_mlflow
        else _noop_context()
    )

    with run_context:
        if use_mlflow:
            mlflow.log_params(
                {
                    "model_type": model_type,
                    "cv_method": cv_method,
                    "n_train_subjects": len(split.train_subjects),
                    "n_test_subjects": len(split.test_subjects),
                    "n_train_trials": len(split.train.labels),
                    "n_test_trials": len(split.test.labels),
                }
            )
            log_model_metadata(
                dataset_accession="uci-eeg-121",
                preprocessing_version="v1.0",
                participant_count=len(np.unique(dataset.subjects)),
                classifier_type=model_type,
                target_state="alcoholism",
            )

        model, test_metrics, predictions = train_and_evaluate(
            train_data=split.train,
            test_data=split.test,
            model_type=model_type,
            feature_config=feature_config,
        )

        cv_result = run_cross_validation(
            dataset=split.train,
            model_type=model_type,
            feature_config=feature_config,
            cv_method=cv_method,
        )

        baseline_diff = compare_with_baseline(
            test_metrics, PUBLISHED_BASELINES["cnn_uci"]
        )

        if use_mlflow:
            log_metrics(test_metrics.to_dict())
            log_metrics(cv_result.to_dict())
            for key, val in baseline_diff.items():
                mlflow.log_metric(key, val)

            if model_type in ("svm", "random_forest"):
                mlflow.sklearn.log_model(model.pipeline, "model")

        logger.info(
            "experiment_complete",
            model=model_type,
            test_accuracy=test_metrics.accuracy,
            test_auc=test_metrics.auc,
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
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train addiction classifier")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=["svm", "random_forest", "eegnet", "dscnet"],
    )
    parser.add_argument("--cv", type=str, default="kfold", choices=["loso", "kfold"])
    parser.add_argument("--tracking-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--experiment", type=str, default="addiction-classifier")

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
