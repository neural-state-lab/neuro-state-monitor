"""BentoML service for real-time EEG inference.

Serves encoding and addiction classifiers via REST API.
Handles mixed model types (SVM + CNN) with Python preprocessing
in the serving path.

Usage:
    bentoml serve shared.serving.service:svc
    # or via Docker after bentoml build
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger(__name__)

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "neuro_predictions_total",
    "Total predictions made",
    ["classifier", "result"],
)
PREDICTION_LATENCY = Histogram(
    "neuro_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["classifier"],
)


@bentoml.service(
    name="neuro-state-monitor",
    traffic={"timeout": 30},
    resources={"cpu": "2"},
)
class NeuroStateService:
    """Unified EEG brain state classification service.

    Loads trained models for encoding and addiction classifiers,
    applies preprocessing and feature extraction in the serving path,
    and returns predictions with confidence scores.
    """

    def __init__(self) -> None:
        """Load models from MLflow or local paths."""
        self.models: dict[str, Any] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load available classifier models."""
        import os

        # Try loading from environment-configured paths
        encoding_path = os.environ.get("ENCODING_MODEL_PATH")
        addiction_path = os.environ.get("ADDICTION_MODEL_PATH")
        encoding_uri = os.environ.get("ENCODING_MODEL_URI")
        addiction_uri = os.environ.get("ADDICTION_MODEL_URI")

        if encoding_path and Path(encoding_path).exists():
            import joblib

            self.models["encoding"] = joblib.load(encoding_path)
            logger.info("loaded_model", classifier="encoding", source=encoding_path)

        if addiction_path and Path(addiction_path).exists():
            import joblib

            self.models["addiction"] = joblib.load(addiction_path)
            logger.info("loaded_model", classifier="addiction", source=addiction_path)

        if encoding_uri and "encoding" not in self.models:
            try:
                import mlflow

                self.models["encoding"] = mlflow.sklearn.load_model(encoding_uri)
                logger.info("loaded_model", classifier="encoding", source=encoding_uri)
            except Exception as exc:
                logger.warning(
                    "model_load_failed", classifier="encoding", error=str(exc)
                )

        if addiction_uri and "addiction" not in self.models:
            try:
                import mlflow

                self.models["addiction"] = mlflow.sklearn.load_model(addiction_uri)
                logger.info(
                    "loaded_model", classifier="addiction", source=addiction_uri
                )
            except Exception as exc:
                logger.warning(
                    "model_load_failed", classifier="addiction", error=str(exc)
                )

        logger.info("models_loaded", available=list(self.models.keys()))

    @bentoml.api
    def predict_encoding(
        self,
        features: np.ndarray,
    ) -> dict[str, Any]:
        """Predict encoding success from extracted feature vector.

        Args:
            features: Feature array of shape (n_epochs, n_features).

        Returns:
            Dict with predictions, probabilities, and metadata.
        """
        return self._predict("encoding", features)

    @bentoml.api
    def predict_addiction(
        self,
        features: np.ndarray,
    ) -> dict[str, Any]:
        """Predict alcoholic vs control from extracted feature vector.

        Args:
            features: Feature array of shape (n_epochs, n_features).

        Returns:
            Dict with predictions, probabilities, and metadata.
        """
        return self._predict("addiction", features)

    @bentoml.api
    def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "models_loaded": list(self.models.keys()),
            "n_models": len(self.models),
        }

    @bentoml.api
    def list_models(self) -> dict[str, Any]:
        """List available models and their metadata."""
        model_info = {}
        for name, model in self.models.items():
            model_info[name] = {
                "type": type(model).__name__,
                "loaded": True,
            }
        return {"models": model_info}

    def _predict(
        self,
        classifier: str,
        features: np.ndarray,
    ) -> dict[str, Any]:
        """Internal prediction logic shared by all classifiers."""
        if classifier not in self.models:
            return {
                "error": f"Model '{classifier}' not loaded",
                "available_models": list(self.models.keys()),
            }

        start = time.monotonic()

        model = self.models[classifier]
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]

        latency = time.monotonic() - start

        # Track metrics
        PREDICTION_LATENCY.labels(classifier=classifier).observe(latency)
        for pred in predictions:
            PREDICTION_COUNT.labels(
                classifier=classifier,
                result=str(int(pred)),
            ).inc()

        result = {
            "classifier": classifier,
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "n_epochs": len(predictions),
            "latency_ms": round(latency * 1000, 2),
            "mean_confidence": float(
                np.where(probabilities > 0.5, probabilities, 1 - probabilities).mean()
            ),
        }

        logger.info("prediction_served", **result)
        return result
