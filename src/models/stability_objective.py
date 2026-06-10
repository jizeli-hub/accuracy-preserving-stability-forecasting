"""Stability-aware XGBoost training objective."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.config import RANDOM_SEED


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StabilityObjectiveContext:
    """Index metadata needed to compute adjacent forecast penalties."""

    previous_index: np.ndarray
    valid_pairs: np.ndarray
    pair_count: int


def forecast_loss_mse(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute the MSE forecasting loss component."""
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    return float(np.mean((predictions - labels) ** 2))


def stability_loss_mean_abs_change(
    predictions: np.ndarray,
    groups,
    order_values,
) -> float:
    """Compute mean(abs(pred_t - pred_t_minus_1)) within each series."""
    frame = pd.DataFrame(
        {
            "prediction": np.asarray(predictions, dtype=float),
            "group": groups,
            "order": order_values,
        }
    ).sort_values(["group", "order"])
    return float(frame.groupby("group", sort=False)["prediction"].diff().abs().dropna().mean())


def prediction_scale_summary(raw_predictions: np.ndarray, clipped_predictions: np.ndarray) -> dict[str, float]:
    """Summarize raw and non-negative clipped prediction scales."""
    raw_predictions = np.asarray(raw_predictions, dtype=float)
    clipped_predictions = np.asarray(clipped_predictions, dtype=float)
    return {
        "raw_pred_min": float(np.min(raw_predictions)),
        "raw_pred_mean": float(np.mean(raw_predictions)),
        "raw_pred_max": float(np.max(raw_predictions)),
        "scaled_pred_min": float(np.min(clipped_predictions)),
        "scaled_pred_mean": float(np.mean(clipped_predictions)),
        "scaled_pred_max": float(np.max(clipped_predictions)),
    }


def build_stability_context(
    frame: pd.DataFrame,
    group_col: str = "id",
    order_col: str = "d_int",
) -> StabilityObjectiveContext:
    """Build adjacency metadata for a table sorted by forecast series and time."""
    sorted_frame = frame[[group_col, order_col]].reset_index(drop=True)
    groups = sorted_frame[group_col].to_numpy()
    order_values = sorted_frame[order_col].to_numpy()

    previous_index = np.arange(len(sorted_frame)) - 1
    valid_pairs = np.r_[False, (groups[1:] == groups[:-1]) & (order_values[1:] - order_values[:-1] == 1)]
    pair_count = int(valid_pairs.sum())

    return StabilityObjectiveContext(
        previous_index=previous_index,
        valid_pairs=valid_pairs,
        pair_count=pair_count,
    )


def make_stability_objective(
    labels: np.ndarray,
    context: StabilityObjectiveContext,
    stability_lambda: float,
    epsilon: float = 1e-6,
):
    """Create a custom objective for MSE plus adjacent forecast-change penalty."""
    labels = np.asarray(labels, dtype=float)
    sample_count = len(labels)
    valid_positions = np.flatnonzero(context.valid_pairs)
    previous_positions = context.previous_index[valid_positions]
    pair_count = max(context.pair_count, 1)

    def objective(predictions: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        del dmatrix
        predictions = np.asarray(predictions, dtype=float)

        gradient = 2.0 * (predictions - labels)
        hessian = np.full(sample_count, 2.0, dtype=float)

        if stability_lambda > 0 and len(valid_positions) > 0:
            diffs = predictions[valid_positions] - predictions[previous_positions]
            signs = diffs / np.maximum(np.abs(diffs), epsilon)
            stability_grad = stability_lambda * sample_count * signs / pair_count
            np.add.at(gradient, valid_positions, stability_grad)
            np.add.at(gradient, previous_positions, -stability_grad)
            hessian += stability_lambda * epsilon

        return gradient, hessian

    return objective


def train_stability_aware_xgboost(
    train: pd.DataFrame,
    feature_columns: list[str],
    stability_lambda: float,
    num_boost_round: int = 300,
    random_seed: int = RANDOM_SEED,
) -> xgb.Booster:
    """Train XGBoost with a stability-aware custom objective."""
    ordered_train = train.sort_values(["id", "d_int"]).reset_index(drop=True)
    labels = ordered_train["demand"].to_numpy(dtype=float)
    context = build_stability_context(ordered_train)

    dtrain = xgb.DMatrix(
        ordered_train[feature_columns],
        label=labels,
        enable_categorical=True,
    )
    params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": random_seed,
        "verbosity": 0,
    }

    LOGGER.info("Training stability-aware XGBoost with lambda=%s", stability_lambda)
    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        obj=make_stability_objective(labels, context, stability_lambda),
    )


def predict_stability_aware_xgboost(
    model: xgb.Booster,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Predict with a stability-aware XGBoost booster."""
    dmatrix = xgb.DMatrix(frame[feature_columns], enable_categorical=True)
    return model.predict(dmatrix).clip(min=0)


def predict_stability_aware_xgboost_with_scale(
    model: xgb.Booster,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw booster predictions and non-negative demand-scale predictions."""
    dmatrix = xgb.DMatrix(frame[feature_columns], enable_categorical=True)
    raw_predictions = model.predict(dmatrix)
    clipped_predictions = raw_predictions.clip(min=0)
    return raw_predictions, clipped_predictions
