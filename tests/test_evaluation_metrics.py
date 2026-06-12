"""Tests for the accuracy, stability, and smoothing components used in the paper.

These cover the pure-Python metric and post-processing functions that produce the
reported MAE/RMSE, Forecast Stability Score (FSS), within-series volatility, and
the exponential-smoothing baseline. They depend only on numpy/pandas (no xgboost),
so they run in any environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluation_metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    weighted_root_mean_squared_scaled_error,
)
from src.evaluation.stability_metrics import (
    forecast_stability_score,
    grouped_forecast_stability_score,
    grouped_forecast_volatility,
    mean_absolute_forecast_change,
    prediction_volatility,
)
from src.models.stability_aware import exponential_smooth_predictions


# --------------------------------------------------------------------------- #
# Accuracy metrics
# --------------------------------------------------------------------------- #
def test_mae_and_rmse_basic():
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [2.0, 2.0, 2.0, 2.0]
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2.0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(2.0)


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = [0.0, 0.0]
    y_pred = [0.0, 4.0]
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2.0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(8.0))


def test_mape_uses_denominator_guard_for_zeros():
    # Zero actuals must not raise; they are guarded by epsilon (motivates the
    # paper's switch from MAPE to WAPE for intermittent demand).
    value = mean_absolute_percentage_error([0.0, 4.0], [1.0, 2.0])
    assert np.isfinite(value)


def test_wrmsse_component_weighted_and_unweighted():
    y_true = np.array([[0.0, 0.0], [0.0, 0.0]])
    y_pred = np.array([[1.0, 1.0], [3.0, 3.0]])
    scale = np.array([1.0, 9.0])
    # Per-series RMSSE: sqrt(1/1)=1.0 and sqrt(9/9)=1.0 -> unweighted mean 1.0
    assert weighted_root_mean_squared_scaled_error(y_true, y_pred, scale) == pytest.approx(1.0)
    # Weighting by [3, 1] keeps both components at 1.0 -> still 1.0
    assert weighted_root_mean_squared_scaled_error(
        y_true, y_pred, scale, weight=[3.0, 1.0]
    ) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Stability metrics
# --------------------------------------------------------------------------- #
def test_mean_absolute_forecast_change_and_fss():
    preds = [1.0, 3.0, 2.0]  # steps: |2|, |-1| -> mean 1.5
    assert mean_absolute_forecast_change(preds) == pytest.approx(1.5)
    assert forecast_stability_score(preds) == pytest.approx(1.5)


def test_constant_forecast_has_zero_movement_and_volatility():
    preds = [[5.0, 5.0, 5.0]]
    assert forecast_stability_score(preds) == pytest.approx(0.0)
    assert prediction_volatility(preds) == pytest.approx(0.0)


def test_grouped_fss_only_diffs_within_series():
    preds = [1.0, 3.0, 10.0, 7.0]
    groups = ["A", "A", "B", "B"]
    order = [1, 2, 1, 2]
    # Within A: |3-1|=2 ; within B: |7-10|=3 ; mean of [2, 3] = 2.5
    assert grouped_forecast_stability_score(preds, groups, order) == pytest.approx(2.5)


def test_grouped_volatility_averages_within_series_std():
    preds = [1.0, 3.0, 10.0, 7.0]
    groups = ["A", "A", "B", "B"]
    order = [1, 2, 1, 2]
    # std(ddof=0): A -> 1.0, B -> 1.5 ; mean -> 1.25
    assert grouped_forecast_volatility(preds, groups, order) == pytest.approx(1.25)


def test_grouped_fss_is_order_invariant_in_input():
    # Same series/values supplied in a shuffled row order must give the same FSS,
    # because the function sorts by (group, order) internally.
    preds = [3.0, 10.0, 1.0, 7.0]
    groups = ["A", "B", "A", "B"]
    order = [2, 1, 1, 2]
    assert grouped_forecast_stability_score(preds, groups, order) == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# Exponential smoothing baseline
# --------------------------------------------------------------------------- #
def _validation_frame(ids, days):
    return pd.DataFrame({"id": ids, "d_int": days})


def test_smoothing_alpha_one_is_identity():
    frame = _validation_frame(["A", "A", "A"], [1, 2, 3])
    raw = np.array([2.0, 9.0, 4.0])
    out = exponential_smooth_predictions(frame, raw, alpha=1.0)
    np.testing.assert_allclose(out, raw)


def test_smoothing_recursion_value():
    frame = _validation_frame(["A", "A"], [1, 2])
    raw = np.array([0.0, 10.0])
    # final[0]=0 ; final[1]=0.5*10 + 0.5*0 = 5
    out = exponential_smooth_predictions(frame, raw, alpha=0.5)
    np.testing.assert_allclose(out, [0.0, 5.0])


def test_smoothing_is_per_series_and_preserves_input_order():
    # Two interleaved series in the input rows.
    frame = _validation_frame(["A", "B", "A", "B"], [1, 1, 2, 2])
    raw = np.array([0.0, 100.0, 10.0, 80.0])
    out = exponential_smooth_predictions(frame, raw, alpha=0.5)
    # A: [0, 5] ; B: [100, 90] ; returned in the original interleaved order.
    np.testing.assert_allclose(out, [0.0, 100.0, 5.0, 90.0])


def test_smoothing_reduces_within_series_movement():
    # Mirrors the paper's claim that smoothing lowers raw FSS for alpha < 1.
    frame = _validation_frame(["A"] * 6, list(range(1, 7)))
    raw = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    smoothed = exponential_smooth_predictions(frame, raw, alpha=0.5)
    groups = frame["id"].to_numpy()
    order = frame["d_int"].to_numpy()
    assert grouped_forecast_stability_score(smoothed, groups, order) < grouped_forecast_stability_score(
        raw, groups, order
    )


@pytest.mark.parametrize("bad_alpha", [0.0, -0.1, 1.5])
def test_smoothing_rejects_alpha_outside_unit_interval(bad_alpha):
    frame = _validation_frame(["A", "A"], [1, 2])
    with pytest.raises(ValueError):
        exponential_smooth_predictions(frame, np.array([1.0, 2.0]), alpha=bad_alpha)
