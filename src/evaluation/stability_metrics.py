"""Stability metrics for comparing forecast robustness across settings."""

import numpy as np
import pandas as pd


def prediction_volatility(predictions, axis: int = -1) -> float:
    """Measure average forecast variation across a horizon or repeated runs."""
    predictions = np.asarray(predictions, dtype=float)
    return float(np.mean(np.std(predictions, axis=axis)))


def mean_absolute_forecast_change(predictions, axis: int = -1) -> float:
    """Measure average absolute step-to-step forecast changes."""
    predictions = np.asarray(predictions, dtype=float)
    return float(np.mean(np.abs(np.diff(predictions, axis=axis))))


def forecast_stability_score(predictions) -> float:
    """Compute FSS as mean(abs(pred_t - pred_t_minus_1))."""
    return mean_absolute_forecast_change(predictions, axis=-1)


def grouped_forecast_stability_score(
    predictions,
    groups,
    order_values,
) -> float:
    """Compute FSS within each forecast series, then average changes."""
    frame = pd.DataFrame(
        {
            "prediction": np.asarray(predictions, dtype=float),
            "group": groups,
            "order": order_values,
        }
    ).sort_values(["group", "order"])
    changes = frame.groupby("group", sort=False)["prediction"].diff().abs()
    return float(changes.dropna().mean())


def grouped_forecast_volatility(
    predictions,
    groups,
    order_values,
) -> float:
    """Compute mean within-series forecast standard deviation."""
    frame = pd.DataFrame(
        {
            "prediction": np.asarray(predictions, dtype=float),
            "group": groups,
            "order": order_values,
        }
    ).sort_values(["group", "order"])
    volatility = frame.groupby("group", sort=False)["prediction"].std(ddof=0)
    return float(volatility.fillna(0).mean())


def rank_stability(score_matrix) -> float:
    """Estimate stability of model rankings across folds or seeds.

    Lower values indicate more stable rankings.
    """
    scores = np.asarray(score_matrix, dtype=float)
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
    return float(np.mean(np.std(ranks, axis=0)))
