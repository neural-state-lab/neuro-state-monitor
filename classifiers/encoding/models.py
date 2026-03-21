"""Encoding classifier models.

Multiple architectures for predicting encoding success:
1. Baseline: SVM on spectral features (reproduce published results)
2. CNN: Braindecode ShallowFBCSPNet or EEGNet
3. Target: beat AUC 0.68 (EEG-ITNet baseline from literature)

All models follow a common interface and are registered in MLflow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import mlflow
import mne
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


class BaseEncodingModel(ABC):
    """Base class for encoding classifier models."""

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


class SVMClassifier(BaseEncodingModel):
    """SVM baseline classifier on extracted features.

    Uses RBF kernel SVM with standardized features.
    This is the baseline approach — should reproduce published results.
    """

    name = "svm"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
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
        """Train SVM on feature vectors."""
        self.pipeline.fit(X, y)
        logger.info(
            "svm_trained",
            n_samples=len(y),
            n_features=X.shape[1],
            class_balance=f"{y.mean():.2f}",
        )

    def predict(self, X: np.ndarray) -> ModelResult:
        """Predict with SVM, returning labels and probabilities."""
        y_pred = self.pipeline.predict(X)
        y_prob = self.pipeline.predict_proba(X)[:, 1]

        return ModelResult(
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=self.name,
        )

    def get_params(self) -> dict:
        return {
            "type": "svm",
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "class_weight": self.class_weight,
        }


class EEGNetClassifier(BaseEncodingModel):
    """EEGNet classifier using Braindecode.

    EEGNet is a compact CNN designed for EEG classification.
    Works directly on raw epoch data (no manual feature extraction needed).

    Reference: Lawhern et al. 2018, "EEGNet: a compact convolutional
    neural network for EEG-based brain-computer interfaces"
    """

    name = "eegnet"

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 256,
        n_classes: int = 2,
        # EEGNet hyperparameters
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
        # Training
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
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
        self.model = None

    def _build_model(self) -> None:
        """Build EEGNet using Braindecode."""
        from braindecode.models import EEGNetv4

        self.model = EEGNetv4(
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
        """Train EEGNet on raw epoch data.

        Args:
            X: Raw epoch data of shape (n_epochs, n_channels, n_times).
            y: Labels of shape (n_epochs,).
        """
        import torch
        from braindecode import EEGClassifier

        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]
        self._build_model()

        clf = EEGClassifier(
            module=self.model,
            max_epochs=self.n_epochs_train,
            lr=self.lr,
            batch_size=self.batch_size,
            train_split=None,
            verbose=0,
        )

        X_tensor = X.astype(np.float32)
        clf.fit(X_tensor, y)

        self._clf = clf
        logger.info(
            "eegnet_trained",
            n_samples=len(y),
            n_channels=self.n_channels,
            n_times=self.n_times,
            epochs=self.n_epochs_train,
        )

    def predict(self, X: np.ndarray) -> ModelResult:
        """Predict with EEGNet."""
        X_tensor = X.astype(np.float32)
        y_pred = self._clf.predict(X_tensor)
        y_prob = self._clf.predict_proba(X_tensor)[:, 1]

        return ModelResult(
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=self.name,
        )

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


class ShallowNetClassifier(BaseEncodingModel):
    """ShallowFBCSPNet classifier using Braindecode.

    A shallow CNN that learns temporal and spatial filters,
    inspired by Filter Bank Common Spatial Patterns (FBCSP).
    Often outperforms EEGNet on motor imagery and encoding tasks.

    Reference: Schirrmeister et al. 2017
    """

    name = "shallownet"

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 256,
        n_classes: int = 2,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout_rate: float = 0.5,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.dropout_rate = dropout_rate
        self.n_epochs_train = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def _build_model(self) -> None:
        """Build ShallowFBCSPNet using Braindecode."""
        from braindecode.models import ShallowFBCSPNet

        self.model = ShallowFBCSPNet(
            in_chans=self.n_channels,
            n_classes=self.n_classes,
            input_window_samples=self.n_times,
            n_filters_time=self.n_filters_time,
            filter_time_length=self.filter_time_length,
            n_filters_spat=self.n_filters_spat,
            pool_time_length=self.pool_time_length,
            pool_time_stride=self.pool_time_stride,
            drop_prob=self.dropout_rate,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train ShallowNet on raw epoch data."""
        from braindecode import EEGClassifier

        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]
        self._build_model()

        clf = EEGClassifier(
            module=self.model,
            max_epochs=self.n_epochs_train,
            lr=self.lr,
            batch_size=self.batch_size,
            train_split=None,
            verbose=0,
        )

        X_tensor = X.astype(np.float32)
        clf.fit(X_tensor, y)

        self._clf = clf
        logger.info(
            "shallownet_trained",
            n_samples=len(y),
            n_channels=self.n_channels,
            n_times=self.n_times,
        )

    def predict(self, X: np.ndarray) -> ModelResult:
        """Predict with ShallowNet."""
        X_tensor = X.astype(np.float32)
        y_pred = self._clf.predict(X_tensor)
        y_prob = self._clf.predict_proba(X_tensor)[:, 1]

        return ModelResult(
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=self.name,
        )

    def get_params(self) -> dict:
        return {
            "type": "shallownet",
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_filters_time": self.n_filters_time,
            "filter_time_length": self.filter_time_length,
            "n_filters_spat": self.n_filters_spat,
            "pool_time_length": self.pool_time_length,
            "pool_time_stride": self.pool_time_stride,
            "dropout_rate": self.dropout_rate,
            "n_epochs": self.n_epochs_train,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }


def create_model(
    model_type: str,
    n_channels: int = 64,
    n_times: int = 256,
    **kwargs,
) -> BaseEncodingModel:
    """Factory function to create encoding models by name.

    Args:
        model_type: One of "svm", "eegnet", "shallownet".
        n_channels: Number of EEG channels (for CNN models).
        n_times: Number of time samples per epoch (for CNN models).
        **kwargs: Additional model-specific parameters.
    """
    models = {
        "svm": SVMClassifier,
        "eegnet": EEGNetClassifier,
        "shallownet": ShallowNetClassifier,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {list(models.keys())}"
        )

    if model_type == "svm":
        return SVMClassifier(**kwargs)
    else:
        return models[model_type](
            n_channels=n_channels,
            n_times=n_times,
            **kwargs,
        )
