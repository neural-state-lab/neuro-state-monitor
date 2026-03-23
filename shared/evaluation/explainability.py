"""Model explainability for EEG deep learning models.

Uses captum for Integrated Gradients and DeepLIFT — the only two
methods that pass systematic interpretability benchmarks for EEG DL.

DO NOT use: saliency maps, guided backprop — they fail validity tests
(per ecosystem audit, March 2026).

Produces electrode-level contribution maps showing which channels
drive predictions — directly maps to neuroscience interpretations.

Requires: pip install captum (optional dependency)
"""

from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def compute_integrated_gradients(
    model,
    X: np.ndarray,
    target_class: int = 1,
    n_steps: int = 50,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Integrated Gradients attribution for EEG epochs.

    Integrated Gradients measures the contribution of each input
    feature (each channel × time point) to the model's prediction
    by integrating gradients along a path from a baseline to the input.

    Args:
        model: PyTorch model (EEGNet, ShallowNet, etc.)
        X: Input epochs, shape (n_epochs, n_channels, n_times).
        target_class: Which class to explain (1 = positive).
        n_steps: Number of integration steps (more = more accurate).
        baseline: Reference input (default: zeros).

    Returns:
        Attribution map, shape (n_epochs, n_channels, n_times).
        Positive = pushes toward target class.
        Negative = pushes away from target class.
    """
    try:
        import torch
        from captum.attr import IntegratedGradients
    except ImportError:
        raise RuntimeError("captum not installed. Install with: pip install captum")

    device = next(model.parameters()).device
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    if baseline is None:
        baseline_tensor = torch.zeros_like(X_tensor)
    else:
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=device)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        X_tensor,
        baselines=baseline_tensor,
        target=target_class,
        n_steps=n_steps,
    )

    attr_np = attributions.detach().cpu().numpy()

    logger.info(
        "computed_integrated_gradients",
        shape=attr_np.shape,
        target_class=target_class,
        n_steps=n_steps,
    )
    return attr_np


def compute_deeplift(
    model,
    X: np.ndarray,
    target_class: int = 1,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute DeepLIFT attribution for EEG epochs.

    DeepLIFT computes importance scores by comparing neuron activations
    to a reference (baseline) activation, propagating differences
    back through the network.

    Args:
        model: PyTorch model.
        X: Input epochs, shape (n_epochs, n_channels, n_times).
        target_class: Which class to explain.
        baseline: Reference input (default: zeros).

    Returns:
        Attribution map, shape (n_epochs, n_channels, n_times).
    """
    try:
        import torch
        from captum.attr import DeepLift
    except ImportError:
        raise RuntimeError("captum not installed. Install with: pip install captum")

    device = next(model.parameters()).device
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    if baseline is None:
        baseline_tensor = torch.zeros_like(X_tensor)
    else:
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=device)

    dl = DeepLift(model)
    attributions = dl.attribute(
        X_tensor,
        baselines=baseline_tensor,
        target=target_class,
    )

    attr_np = attributions.detach().cpu().numpy()

    logger.info(
        "computed_deeplift",
        shape=attr_np.shape,
        target_class=target_class,
    )
    return attr_np


def channel_importance(
    attributions: np.ndarray,
    ch_names: list[str],
) -> dict[str, float]:
    """Compute per-channel importance from attribution maps.

    Averages absolute attributions across time and epochs
    to get a single importance score per channel.

    Args:
        attributions: Shape (n_epochs, n_channels, n_times).
        ch_names: Channel names.

    Returns:
        Dict mapping channel name to importance score.
    """
    # Mean absolute attribution per channel
    importance = np.abs(attributions).mean(axis=(0, 2))  # (n_channels,)

    # Normalize to sum to 1
    total = importance.sum()
    if total > 0:
        importance = importance / total

    result = {ch: float(imp) for ch, imp in zip(ch_names, importance)}

    # Log top 5 channels
    top_5 = sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(
        "channel_importance",
        top_channels={ch: f"{imp:.4f}" for ch, imp in top_5},
    )
    return result


def temporal_importance(
    attributions: np.ndarray,
    sfreq: float,
    tmin: float = -0.2,
) -> dict[str, np.ndarray]:
    """Compute time-resolved importance from attribution maps.

    Averages absolute attributions across channels and epochs
    to get importance at each time point.

    Args:
        attributions: Shape (n_epochs, n_channels, n_times).
        sfreq: Sampling frequency.
        tmin: Epoch start time (seconds).

    Returns:
        Dict with 'times' and 'importance' arrays.
    """
    importance = np.abs(attributions).mean(axis=(0, 1))  # (n_times,)
    times = tmin + np.arange(len(importance)) / sfreq

    # Find peak importance time
    peak_idx = np.argmax(importance)
    peak_time = times[peak_idx]

    logger.info(
        "temporal_importance",
        peak_time_ms=f"{peak_time * 1000:.0f}ms",
        peak_importance=f"{importance[peak_idx]:.6f}",
    )

    return {
        "times": times,
        "importance": importance,
        "peak_time": peak_time,
    }
