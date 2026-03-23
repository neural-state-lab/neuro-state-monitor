"""PTSD classifier models.

SVM and deep learning classifiers for PTSD vs control
using fMRI connectivity features.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = structlog.get_logger(__name__)


@dataclass
class ModelResult:
    """Container for model predictions."""

    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    model_name: str = ""


class SVMTraumaClassifier:
    """SVM classifier on fMRI connectivity features."""

    name = "svm"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ) -> None:
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        class_weight=class_weight,
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.pipeline.fit(X, y)
        logger.info("svm_trauma_trained", n_samples=len(y), n_features=X.shape[1])

    def predict(self, X: np.ndarray) -> ModelResult:
        y_pred = self.pipeline.predict(X)
        y_prob = self.pipeline.predict_proba(X)[:, 1]
        return ModelResult(y_pred=y_pred, y_prob=y_prob, model_name=self.name)

    def get_params(self) -> dict:
        svm = self.pipeline.named_steps["svm"]
        return {
            "type": "svm",
            "kernel": svm.kernel,
            "C": svm.C,
            "gamma": svm.gamma,
        }


def create_model(model_type: str = "svm", **kwargs) -> SVMTraumaClassifier:
    """Factory function for trauma models."""
    if model_type == "svm":
        return SVMTraumaClassifier(**kwargs)
    raise ValueError(f"Unknown model type: {model_type}")
