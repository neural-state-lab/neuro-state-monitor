"""End-to-end training script for the PTSD classifier.

Usage:
    python -m classifiers.trauma.train --fmri-dir data/raw/enigma --model svm
"""

import argparse
from pathlib import Path
from typing import Any

import mlflow
import structlog

from classifiers.trauma.data import TraumaConfig, load_fmri_connectivity
from classifiers.trauma.features import (
    TraumaFeatureConfig,
    extract_connectivity_features,
)
from classifiers.trauma.models import create_model
from shared.evaluation.metrics import (
    compute_classification_metrics,
)
from shared.experiment import log_metrics, log_model_metadata, setup_experiment

logger = structlog.get_logger(__name__)

PUBLISHED_BASELINE = {"auc": 0.75, "accuracy": 0.72, "f1": 0.70}


def run_experiment(
    fmri_dir: Path,
    model_type: str = "svm",
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "ptsd-classifier",
) -> dict[str, Any]:
    """Run a complete PTSD classifier experiment."""
    config = TraumaConfig(fmri_dir=fmri_dir)
    dataset = load_fmri_connectivity(fmri_dir, config)

    feature_config = TraumaFeatureConfig()
    features = extract_connectivity_features(
        dataset.connectivity_matrices, feature_config
    )

    # Simple train/test split by subject
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        dataset.labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=dataset.labels,
    )

    try:
        setup_experiment(experiment_name, tracking_uri)
        use_mlflow = True
    except Exception:
        use_mlflow = False

    run_context = (
        mlflow.start_run(run_name=f"ptsd-{model_type}") if use_mlflow else _noop()
    )

    with run_context:
        model = create_model(model_type)
        model.fit(X_train, y_train)
        result = model.predict(X_test)
        metrics = compute_classification_metrics(y_test, result.y_pred, result.y_prob)

        if use_mlflow:
            log_metrics(metrics.to_dict())
            log_model_metadata(
                dataset_accession="enigma-ptsd",
                preprocessing_version="fmriprep-v23",
                participant_count=len(dataset.subjects),
                classifier_type=model_type,
                target_state="ptsd",
            )

        logger.info(
            "experiment_complete",
            accuracy=metrics.accuracy,
            auc=metrics.auc,
            f1=metrics.f1,
        )

    return {"model": model, "metrics": metrics}


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PTSD classifier")
    parser.add_argument("--fmri-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="svm")
    parser.add_argument("--tracking-uri", type=str, default="http://localhost:5000")
    args = parser.parse_args()
    run_experiment(args.fmri_dir, args.model, args.tracking_uri)


if __name__ == "__main__":
    main()
