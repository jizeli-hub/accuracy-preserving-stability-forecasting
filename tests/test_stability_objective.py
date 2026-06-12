"""Correctness tests for the stability-aware XGBoost objective.

These tests validate the central algorithmic contribution of the paper: that the
custom objective returns the correct analytic gradients and Hessians for the
within-series total-variation (consecutive-change) stability penalty. The key
check compares the analytic gradient against numerical finite differences of the
exact total loss, for both the L1 and the optional Huber-smoothed variant.

Run from the repository root:

    python -m pytest tests/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# The module under test imports xgboost at load time. Skip cleanly if xgboost (or
# its native library) cannot be imported in the current environment; a broken
# native build raises XGBoostError rather than ImportError, so catch broadly.
try:
    import xgboost  # noqa: F401
except Exception:  # pragma: no cover - environment without a working xgboost build
    pytest.skip("xgboost is not importable in this environment", allow_module_level=True)

from src.models.stability_objective import (  # noqa: E402
    build_stability_context,
    make_stability_objective,
    stability_loss_mean_abs_change,
)


def _toy_frame() -> pd.DataFrame:
    """Two series; series B has a calendar gap (7 -> 9) that breaks adjacency."""
    return pd.DataFrame(
        {
            "id": ["A", "A", "A", "B", "B", "B", "B"],
            "d_int": [1, 2, 3, 5, 6, 7, 9],
        }
    )


def _reference_total_loss(
    predictions: np.ndarray,
    labels: np.ndarray,
    context,
    stability_lambda: float,
    huber_delta: float | None,
) -> float:
    """Scalar loss whose gradient ``make_stability_objective`` is expected to return."""
    sample_count = len(labels)
    pair_count = max(context.pair_count, 1)
    valid_positions = np.flatnonzero(context.valid_pairs)
    previous_positions = context.previous_index[valid_positions]

    mse = float(np.sum((predictions - labels) ** 2))
    diffs = predictions[valid_positions] - predictions[previous_positions]
    if huber_delta is None:
        penalty = np.abs(diffs)
    else:
        absd = np.abs(diffs)
        penalty = np.where(
            absd <= huber_delta,
            diffs ** 2 / (2.0 * huber_delta),
            absd - huber_delta / 2.0,
        )
    coefficient = stability_lambda * sample_count / pair_count
    return mse + coefficient * float(np.sum(penalty))


def test_build_stability_context_identifies_consecutive_same_series_pairs():
    context = build_stability_context(_toy_frame())
    # Valid adjacent pairs: (A1,A2), (A2,A3), (B5,B6), (B6,B7); the 7->9 gap and the
    # A->B series boundary are excluded.
    assert context.valid_pairs.tolist() == [False, True, True, False, True, True, False]
    assert context.pair_count == 4


def test_stability_loss_respects_series_boundaries_and_gaps():
    preds = np.array([0.0, 1.0, 1.5, 4.0, 4.2, 4.5, 9.0])
    groups = _toy_frame()["id"].to_numpy()
    order = _toy_frame()["d_int"].to_numpy()
    # Mean abs consecutive change over the 4 valid pairs only:
    # |1-0|, |1.5-1|, |4.2-4.0|, |4.5-4.2| = 1.0, 0.5, 0.2, 0.3 -> mean 0.5
    assert stability_loss_mean_abs_change(preds, groups, order) == pytest.approx(0.5)


@pytest.mark.parametrize("huber_delta", [None, 0.5])
def test_objective_gradient_matches_finite_difference(huber_delta):
    context = build_stability_context(_toy_frame())
    labels = np.array([0.3, 0.9, 1.1, 4.8, 5.2, 6.9, 7.1])
    # Predictions chosen so that, for the Huber case, some |delta| < delta (quadratic
    # branch) and some > delta (linear branch); for L1 all |delta| >> epsilon.
    preds = np.array([0.0, 1.0, 1.2, 5.0, 5.1, 7.0, 7.05])
    stability_lambda = 0.05

    objective = make_stability_objective(
        labels, context, stability_lambda, huber_delta=huber_delta
    )
    gradient, hessian = objective(preds, None)

    step = 1e-6
    numerical = np.zeros_like(preds)
    for i in range(len(preds)):
        up = preds.copy()
        up[i] += step
        down = preds.copy()
        down[i] -= step
        numerical[i] = (
            _reference_total_loss(up, labels, context, stability_lambda, huber_delta)
            - _reference_total_loss(down, labels, context, stability_lambda, huber_delta)
        ) / (2.0 * step)

    np.testing.assert_allclose(gradient, numerical, atol=1e-5, rtol=1e-5)
    assert np.all(hessian > 0.0)


def test_zero_lambda_recovers_plain_mse_gradient():
    context = build_stability_context(_toy_frame())
    labels = np.array([0.3, 0.9, 1.1, 4.8, 5.2, 6.9, 7.1])
    preds = np.array([0.0, 1.0, 1.2, 5.0, 5.1, 7.0, 7.05])
    objective = make_stability_objective(labels, context, stability_lambda=0.0)
    gradient, hessian = objective(preds, None)
    np.testing.assert_allclose(gradient, 2.0 * (preds - labels))
    np.testing.assert_allclose(hessian, np.full_like(preds, 2.0))
