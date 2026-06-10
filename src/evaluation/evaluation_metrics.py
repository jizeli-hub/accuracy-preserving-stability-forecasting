"""Forecast accuracy metrics for M5-style demand forecasting."""

import numpy as np


def mean_absolute_error(y_true, y_pred) -> float:
    """Compute mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred) -> float:
    """Compute root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Compute MAPE with a small denominator guard."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)))


def weighted_root_mean_squared_scaled_error(
    y_true,
    y_pred,
    scale,
    weight=None,
    epsilon: float = 1e-8,
) -> float:
    """Compute a compact WRMSSE-style metric component.

    This helper expects precomputed series-level scale and optional weights.
    Full M5 hierarchy aggregation can be layered on top later.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scale = np.asarray(scale, dtype=float)

    rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=-1) / np.maximum(scale, epsilon))
    if weight is None:
        return float(np.mean(rmsse))

    weight = np.asarray(weight, dtype=float)
    return float(np.sum(weight * rmsse) / np.sum(weight))

