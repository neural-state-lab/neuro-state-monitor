"""Inference script for the encoding classifier.

Loads a trained model from MLflow and runs prediction on new EEG data.

Usage:
    python -m classifiers.encoding.predict --model-uri runs:/<run_id>/model --input data.edf
    python -m classifiers.encoding.predict --model-path models/encoding_svm.pkl --input data.edf
"""

import argparse
from pathlib import Path
from typing import Optional

import mlflow
import mne
import numpy as np
import structlog

from classifiers.encoding.features import (
    EncodingFeatureConfig,
    extract_encoding_features,
)
from shared.preprocessing.pipeline import (
    PreprocessingConfig,
    load_raw,
    preprocess_and_epoch,
)

logger = structlog.get_logger(__name__)


def predict_from_raw(
    raw_path: Path,
    model_uri: Optional[str] = None,
    model_path: Optional[Path] = None,
    events: Optional[np.ndarray] = None,
    preproc_config: Optional[PreprocessingConfig] = None,
    feature_config: Optional[EncodingFeatureConfig] = None,
) -> dict:
    """Run encoding prediction on a raw EEG file.

    Args:
        raw_path: Path to raw EEG file.
        model_uri: MLflow model URI (e.g., "runs:/<run_id>/model").
        model_path: Local path to saved model (alternative to model_uri).
        events: Event array. If None, extracted from raw data.
        preproc_config: Preprocessing configuration.
        feature_config: Feature extraction configuration.

    Returns:
        Dict with predictions, probabilities, and metadata.
    """
    if model_uri is None and model_path is None:
        raise ValueError("Provide either model_uri or model_path")

    # Load model
    if model_uri:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("loaded_model_from_mlflow", uri=model_uri)
    else:
        import joblib

        model = joblib.load(model_path)
        logger.info("loaded_model_from_disk", path=str(model_path))

    # Load and preprocess EEG
    raw = load_raw(raw_path)

    if events is None:
        events = mne.find_events(raw, shortest_event=1)
        if len(events) == 0:
            events, _ = mne.events_from_annotations(raw)

    if preproc_config is None:
        preproc_config = PreprocessingConfig()

    epochs = preprocess_and_epoch(raw, events, preproc_config)

    if len(epochs) == 0:
        logger.warning("no_epochs_extracted")
        return {"predictions": [], "probabilities": [], "n_epochs": 0}

    # Extract features
    if feature_config is None:
        feature_config = EncodingFeatureConfig()
    features = extract_encoding_features(epochs, feature_config)

    # Predict
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    results = {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "n_epochs": len(epochs),
        "n_predicted_recalled": int(predictions.sum()),
        "n_predicted_forgotten": int((1 - predictions).sum()),
        "mean_confidence": float(probabilities.mean()),
    }

    logger.info(
        "prediction_complete",
        n_epochs=results["n_epochs"],
        n_recalled=results["n_predicted_recalled"],
        n_forgotten=results["n_predicted_forgotten"],
        mean_confidence=results["mean_confidence"],
    )

    return results


def predict_from_epochs(
    epochs: mne.Epochs,
    model_uri: Optional[str] = None,
    model_path: Optional[Path] = None,
    feature_config: Optional[EncodingFeatureConfig] = None,
) -> dict:
    """Run encoding prediction on preprocessed epochs.

    For when data is already preprocessed.
    """
    if model_uri is None and model_path is None:
        raise ValueError("Provide either model_uri or model_path")

    # Load model
    if model_uri:
        model = mlflow.sklearn.load_model(model_uri)
    else:
        import joblib

        model = joblib.load(model_path)

    # Extract features
    if feature_config is None:
        feature_config = EncodingFeatureConfig()
    features = extract_encoding_features(epochs, feature_config)

    # Predict
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "n_epochs": len(epochs),
        "n_predicted_recalled": int(predictions.sum()),
        "n_predicted_forgotten": int((1 - predictions).sum()),
        "mean_confidence": float(probabilities.mean()),
    }


def main() -> None:
    """CLI entry point for prediction."""
    parser = argparse.ArgumentParser(description="Run encoding prediction on EEG data")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw EEG file",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="MLflow model URI",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local path to saved model",
    )

    args = parser.parse_args()

    results = predict_from_raw(
        raw_path=args.input,
        model_uri=args.model_uri,
        model_path=args.model_path,
    )

    print(f"\nResults:")
    print(f"  Epochs analyzed: {results['n_epochs']}")
    print(f"  Predicted recalled: {results['n_predicted_recalled']}")
    print(f"  Predicted forgotten: {results['n_predicted_forgotten']}")
    print(f"  Mean confidence: {results['mean_confidence']:.3f}")


if __name__ == "__main__":
    main()
