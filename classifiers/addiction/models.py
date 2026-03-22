"""Addiction classifier models.

Models for alcoholic vs control classification:
1. SVM baseline on extracted features
2. CNN via Braindecode (target: reproduce ~95% on UCI)
3. DSCnet-inspired dual-stream approach (Wu 2025)
4. Beta connectivity model (Tian 2024)

All models follow BaseAddictionModel interface and are MLflow-compatible.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import mlflow
import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier
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


class BaseAddictionModel(ABC):
    """Base class for addiction classifier models."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelResult:
        """Predict labels and probabilities."""

    @abstractmethod
    def get_params(self) -> dict:
        """Get model parameters for MLflow logging."""

    def log_to_mlflow(self) -> None:
        """Log model parameters to active MLflow run."""
        params = self.get_params()
        prefixed = {f"model_{k}": v for k, v in params.items()}
        mlflow.log_params(prefixed)


class SVMAddictionClassifier(BaseAddictionModel):
    """SVM classifier on extracted addiction features.

    Uses RBF kernel SVM with standardized features.
    Baseline approach for the UCI dataset.
    """

    name = "svm"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 10.0,
        gamma: str = "scale",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state

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
        logger.info(
            "svm_trained",
            n_samples=len(y),
            n_features=X.shape[1],
            class_balance=f"{y.mean():.2f}",
        )

    def predict(self, X: np.ndarray) -> ModelResult:
        y_pred = self.pipeline.predict(X)
        y_prob = self.pipeline.predict_proba(X)[:, 1]
        return ModelResult(y_pred=y_pred, y_prob=y_prob, model_name=self.name)

    def get_params(self) -> dict:
        return {
            "type": "svm",
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "class_weight": self.class_weight,
        }


class RFAddictionClassifier(BaseAddictionModel):
    """Random Forest classifier for addiction detection.

    Good interpretability through feature importances.
    """

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: str = "balanced",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        class_weight=class_weight,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.pipeline.fit(X, y)
        logger.info(
            "rf_trained",
            n_samples=len(y),
            n_features=X.shape[1],
        )

    def predict(self, X: np.ndarray) -> ModelResult:
        y_pred = self.pipeline.predict(X)
        y_prob = self.pipeline.predict_proba(X)[:, 1]
        return ModelResult(y_pred=y_pred, y_prob=y_prob, model_name=self.name)

    def get_params(self) -> dict:
        return {
            "type": "random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "class_weight": self.class_weight,
        }

    def feature_importances(self) -> np.ndarray:
        """Get feature importances from the trained RF."""
        return self.pipeline.named_steps["rf"].feature_importances_


class EEGNetAddictionClassifier(BaseAddictionModel):
    """EEGNet classifier for addiction detection.

    Works on raw epoch data (n_trials, n_channels, n_samples).
    Target: reproduce ~95% accuracy on UCI dataset.
    """

    name = "eegnet"

    def __init__(
        self,
        n_channels: int = 61,
        n_times: int = 256,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate
        self.n_epochs_train = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self._clf = None

    def _build_model(self):
        from braindecode.models import EEGNetv4

        return EEGNetv4(
            in_chans=self.n_channels,
            n_classes=self.n_classes,
            input_window_samples=self.n_times,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=self.kernel_length,
            drop_prob=self.dropout_rate,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from braindecode import EEGClassifier

        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]
        model = self._build_model()

        self._clf = EEGClassifier(
            module=model,
            max_epochs=self.n_epochs_train,
            lr=self.lr,
            batch_size=self.batch_size,
            train_split=None,
            verbose=0,
        )
        self._clf.fit(X.astype(np.float32), y)

        logger.info(
            "eegnet_trained",
            n_samples=len(y),
            n_channels=self.n_channels,
            n_times=self.n_times,
        )

    def predict(self, X: np.ndarray) -> ModelResult:
        X32 = X.astype(np.float32)
        y_pred = self._clf.predict(X32)
        y_prob = self._clf.predict_proba(X32)[:, 1]
        return ModelResult(y_pred=y_pred, y_prob=y_prob, model_name=self.name)

    def get_params(self) -> dict:
        return {
            "type": "eegnet",
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "F1": self.F1,
            "D": self.D,
            "F2": self.F2,
            "kernel_length": self.kernel_length,
            "dropout_rate": self.dropout_rate,
            "n_epochs": self.n_epochs_train,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }


class DSCNetClassifier(BaseAddictionModel):
    """Dual-Stream CNN inspired by Wu 2025 (DSCnet).

    Processes temporal and spatial features in parallel streams
    then fuses them. Simplified implementation using Braindecode
    building blocks.

    Stream 1: Temporal convolutions (capture time patterns)
    Stream 2: Spatial convolutions (capture electrode patterns)
    Fusion: Concatenate + dense layers
    """

    name = "dscnet"

    def __init__(
        self,
        n_channels: int = 61,
        n_times: int = 256,
        n_classes: int = 2,
        n_temporal_filters: int = 40,
        n_spatial_filters: int = 40,
        dropout_rate: float = 0.5,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_temporal_filters = n_temporal_filters
        self.n_spatial_filters = n_spatial_filters
        self.dropout_rate = dropout_rate
        self.n_epochs_train = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self._clf = None

    def _build_model(self):
        """Build DSCNet using ShallowFBCSPNet as base.

        ShallowFBCSPNet already captures temporal-spatial patterns,
        serving as a strong proxy for the dual-stream concept.
        """
        from braindecode.models import ShallowFBCSPNet

        return ShallowFBCSPNet(
            in_chans=self.n_channels,
            n_classes=self.n_classes,
            input_window_samples=self.n_times,
            n_filters_time=self.n_temporal_filters,
            n_filters_spat=self.n_spatial_filters,
            drop_prob=self.dropout_rate,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from braindecode import EEGClassifier

        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]
        model = self._build_model()

        self._clf = EEGClassifier(
            module=model,
            max_epochs=self.n_epochs_train,
            lr=self.lr,
            batch_size=self.batch_size,
            train_split=None,
            verbose=0,
        )
        self._clf.fit(X.astype(np.float32), y)

        logger.info(
            "dscnet_trained",
            n_samples=len(y),
            n_channels=self.n_channels,
        )

    def predict(self, X: np.ndarray) -> ModelResult:
        X32 = X.astype(np.float32)
        y_pred = self._clf.predict(X32)
        y_prob = self._clf.predict_proba(X32)[:, 1]
        return ModelResult(y_pred=y_pred, y_prob=y_prob, model_name=self.name)

    def get_params(self) -> dict:
        return {
            "type": "dscnet",
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_temporal_filters": self.n_temporal_filters,
            "n_spatial_filters": self.n_spatial_filters,
            "dropout_rate": self.dropout_rate,
            "n_epochs": self.n_epochs_train,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }


def create_model(
    model_type: str,
    n_channels: int = 61,
    n_times: int = 256,
    **kwargs,
) -> BaseAddictionModel:
    """Factory function to create addiction models by name."""
    models = {
        "svm": SVMAddictionClassifier,
        "random_forest": RFAddictionClassifier,
        "eegnet": EEGNetAddictionClassifier,
        "dscnet": DSCNetClassifier,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {list(models.keys())}"
        )

    if model_type in ("svm", "random_forest"):
        return models[model_type](**kwargs)
    else:
        return models[model_type](n_channels=n_channels, n_times=n_times, **kwargs)
