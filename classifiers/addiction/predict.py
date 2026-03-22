"""Inference script for the addiction classifier.

Loads a trained model and runs prediction on new EEG data.

Usage:
    python -m classifiers.addiction.predict --model-path models/addiction_svm.pkl --input data.edf
"""

import argparse
from pathlib import Path
from typing import Optional

import mne
import structlog

from classifiers.addiction.features import (
    AddictionFeatureConfig,
    extract_addiction_features,
)
from shared.preprocessing.pipeline import PreprocessingConfig, load_raw, preprocess_raw

logger = structlog.get_logger(__name__)


def predict_from_raw(
    raw_path: Path,
    model_uri: Optional[str] = None,
    model_path: Optional[Path] = None,
    preproc_config: Optional[PreprocessingConfig] = None,
    feature_config: Optional[AddictionFeatureConfig] = None,
) -> dict:
    """Run addiction prediction on a raw EEG file."""
    import mlflow

    if model_uri is None and model_path is None:
        raise ValueError("Provide either model_uri or model_path")

    if model_uri:
        model = mlflow.sklearn.load_model(model_uri)
    else:
        import joblib

        model = joblib.load(model_path)

    raw = load_raw(raw_path)

    if preproc_config is None:
        preproc_config = PreprocessingConfig()

    raw = preprocess_raw(raw, preproc_config)

    # Create epochs from continuous data (fixed-length)
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=1.0 - 1 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )

    if len(epochs) == 0:
        return {"predictions": [], "n_epochs": 0}

    if feature_config is None:
        feature_config = AddictionFeatureConfig()
    features = extract_addiction_features(epochs, feature_config)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    mean_prob = float(probabilities.mean())
    classification = "alcoholic" if mean_prob > 0.5 else "control"

    results = {
        "classification": classification,
        "confidence": mean_prob if mean_prob > 0.5 else 1 - mean_prob,
        "epoch_predictions": predictions.tolist(),
        "epoch_probabilities": probabilities.tolist(),
        "n_epochs": len(epochs),
        "n_alcoholic_epochs": int(predictions.sum()),
        "n_control_epochs": int((1 - predictions).sum()),
    }

    logger.info("prediction_complete", **results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run addiction prediction")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-uri", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=None)

    args = parser.parse_args()
    results = predict_from_raw(
        raw_path=args.input,
        model_uri=args.model_uri,
        model_path=args.model_path,
    )

    print(f"\nClassification: {results['classification']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Epochs: {results['n_epochs']}")


if __name__ == "__main__":
    main()
