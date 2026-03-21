"""MLflow experiment tracking utilities.

Provides helpers for consistent experiment logging across all classifiers.
"""

from pathlib import Path
from typing import Any, Optional

import mlflow
import structlog
import yaml

logger = structlog.get_logger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("loaded_config", path=str(config_path))
    return config


def setup_experiment(
    experiment_name: str,
    tracking_uri: str = "http://localhost:5000",
    tags: Optional[dict[str, str]] = None,
) -> str:
    """Set up MLflow experiment and return experiment ID."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id

    logger.info(
        "setup_experiment",
        name=experiment_name,
        id=experiment_id,
        tracking_uri=tracking_uri,
    )
    return experiment_id


def start_run(
    run_name: str,
    experiment_name: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
) -> mlflow.ActiveRun:
    """Start an MLflow run with optional config logging."""
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    run = mlflow.start_run(run_name=run_name, tags=tags)

    if config:
        _log_nested_params(config)

    logger.info("started_run", name=run_name, run_id=run.info.run_id)
    return run


def _log_nested_params(
    params: dict[str, Any], prefix: str = ""
) -> None:
    """Flatten and log nested config dict as MLflow params."""
    for key, value in params.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            _log_nested_params(value, prefix=f"{full_key}.")
        elif isinstance(value, list):
            mlflow.log_param(full_key, str(value))
        else:
            mlflow.log_param(full_key, value)


def log_model_metadata(
    dataset_accession: str,
    preprocessing_version: str,
    participant_count: int,
    classifier_type: str,
    target_state: str,
) -> None:
    """Log standardized model registry metadata."""
    mlflow.log_params(
        {
            "dataset_accession": dataset_accession,
            "preprocessing_version": preprocessing_version,
            "participant_count": participant_count,
            "classifier_type": classifier_type,
            "target_state": target_state,
        }
    )


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to the active MLflow run."""
    mlflow.log_metrics(metrics, step=step)
    logger.info("logged_metrics", metrics=metrics, step=step)


def log_artifact(filepath: Path) -> None:
    """Log a file as an MLflow artifact."""
    mlflow.log_artifact(str(filepath))
    logger.info("logged_artifact", path=str(filepath))
