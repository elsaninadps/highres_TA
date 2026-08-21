from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn import metrics


def validate_quantiles(quantiles: Sequence[float]) -> np.ndarray:
    """Return a validated, strictly increasing one-dimensional quantile grid."""
    values = np.asarray(quantiles, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("quantiles must be a non-empty one-dimensional sequence.")
    if np.any(~np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
        raise ValueError("quantiles must contain finite values strictly between 0 and 1.")
    if np.any(np.diff(values) <= 0):
        raise ValueError("quantiles must be unique and strictly increasing.")
    return values


def quantile_column(quantiles: Sequence[float], alpha: float) -> int:
    """Return the unique prediction-column index for ``alpha``."""
    values = validate_quantiles(quantiles)
    matches = np.flatnonzero(np.isclose(values, alpha))
    if len(matches) != 1:
        raise KeyError(f"Quantile {alpha:g} is not uniquely present in the grid.")
    return int(matches[0])


def validate_quantile_prediction(
    y_true, prediction, quantiles: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize arrays used by the quantile metrics."""
    alpha = validate_quantiles(quantiles)
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(prediction, dtype=float)
    expected_shape = (len(observed), len(alpha))
    if predicted.shape != expected_shape:
        raise ValueError(f"prediction must have shape {expected_shape}, got {predicted.shape}.")
    return observed, predicted, alpha


def pinball_loss_by_quantile(y_true, prediction, quantiles: Sequence[float]) -> np.ndarray:
    """Return mean pinball loss at each requested quantile."""
    observed, predicted, alpha = validate_quantile_prediction(y_true, prediction, quantiles)
    error = observed[:, None] - predicted
    return np.mean(np.maximum(alpha * error, (alpha - 1.0) * error), axis=0)


def quantile_crps(y_true, prediction, quantiles: Sequence[float]) -> float:
    """Approximate mean CRPS by integrating pinball loss over the grid."""
    alpha = validate_quantiles(quantiles)
    losses = pinball_loss_by_quantile(y_true, prediction, alpha)
    return float(2.0 * np.trapezoid(losses, alpha))


def quantile_calibration(y_true, prediction, quantiles: Sequence[float]) -> np.ndarray:
    """Return the empirical fraction below each predicted quantile."""
    observed, predicted, _ = validate_quantile_prediction(y_true, prediction, quantiles)
    return np.mean(observed[:, None] <= predicted, axis=0)


def quantile_crossing_rate(prediction) -> float:
    """Return the fraction of rows containing at least one quantile crossing."""
    predicted = np.asarray(prediction, dtype=float)
    if predicted.ndim != 2:
        raise ValueError("prediction must be two-dimensional.")
    return float(np.mean(np.any(np.diff(predicted, axis=1) < 0, axis=1)))


def score_quantile_predictions(
    y_true,
    prediction,
    quantiles: Sequence[float],
    intervals: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Compute point, distribution, calibration, and interval diagnostics."""
    observed, predicted, alpha = validate_quantile_prediction(y_true, prediction, quantiles)
    median = predicted[:, quantile_column(alpha, 0.5)]
    residual = median - observed
    calibration = quantile_calibration(observed, predicted, alpha)
    pinball = pinball_loss_by_quantile(observed, predicted, alpha)

    scores = {
        "count": float(len(observed)),
        "median_rmse": float(metrics.root_mean_squared_error(observed, median)),
        "quantile_crps": quantile_crps(observed, predicted, alpha),
        "mean_pinball_loss": float(np.mean(pinball)),
        "median_mae": float(metrics.mean_absolute_error(observed, median)),
        "median_bias_mean": float(np.mean(residual)),
        "median_bias_median": float(np.median(residual)),
        "median_r2": float(metrics.r2_score(observed, median)),
        "calibration_mae": float(np.mean(np.abs(calibration - alpha))),
        "quantile_crossing_rate": quantile_crossing_rate(predicted),
    }

    for label, (lower_alpha, upper_alpha) in (intervals or {}).items():
        lower = predicted[:, quantile_column(alpha, lower_alpha)]
        upper = predicted[:, quantile_column(alpha, upper_alpha)]
        prefix = f"interval_{label}"
        scores[f"{prefix}_coverage"] = float(np.mean((observed >= lower) & (observed <= upper)))
        scores[f"{prefix}_mean_width"] = float(np.mean(upper - lower))
        scores[f"{prefix}_ordered_fraction"] = float(np.mean(lower <= upper))

    return scores


__all__ = [
    "pinball_loss_by_quantile",
    "quantile_calibration",
    "quantile_column",
    "quantile_crossing_rate",
    "quantile_crps",
    "score_quantile_predictions",
    "validate_quantile_prediction",
    "validate_quantiles",
]
