"""Stability-aware post-processing for forecast sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd


def exponential_smooth_predictions(
    validation: pd.DataFrame,
    raw_predictions,
    alpha: float,
    group_col: str = "id",
    order_col: str = "d_int",
) -> np.ndarray:
    """Apply recursive exponential smoothing within each forecast series.

    final_pred_t = alpha * raw_pred_t + (1 - alpha) * final_pred_t_minus_1
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1].")

    frame = validation[[group_col, order_col]].copy()
    frame["_raw_prediction"] = np.asarray(raw_predictions, dtype=float)
    frame["_original_order"] = np.arange(len(frame))
    frame = frame.sort_values([group_col, order_col])

    smoothed_parts = []
    for _, group in frame.groupby(group_col, sort=False):
        raw_values = group["_raw_prediction"].to_numpy(dtype=float)
        final_values = np.empty_like(raw_values)
        final_values[0] = raw_values[0]

        for index in range(1, len(raw_values)):
            final_values[index] = alpha * raw_values[index] + (1 - alpha) * final_values[index - 1]

        group = group.copy()
        group["_smoothed_prediction"] = final_values
        smoothed_parts.append(group)

    smoothed = pd.concat(smoothed_parts, axis=0).sort_values("_original_order")
    return smoothed["_smoothed_prediction"].to_numpy(dtype=float)
