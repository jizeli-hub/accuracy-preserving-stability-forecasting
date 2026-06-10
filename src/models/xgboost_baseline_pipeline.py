"""Baseline XGBoost pipeline for M5 demand forecasting."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.data_processing.data_loader import load_m5_forecasting_inputs
from src.data_processing.feature_engineering import build_baseline_features
from src.data_processing.preprocessing import preprocess_m5
from src.evaluation.evaluation_metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from src.evaluation.stability_metrics import grouped_forecast_stability_score, grouped_forecast_volatility
from src.models.stability_aware import exponential_smooth_predictions
from src.models.stability_objective import (
    forecast_loss_mse,
    predict_stability_aware_xgboost,
    predict_stability_aware_xgboost_with_scale,
    prediction_scale_summary,
    stability_loss_mean_abs_change,
    train_stability_aware_xgboost,
)
from src.models.timemixer_encoder import (
    TimeMixerConfig,
    TimeMixerTemporalEncoder,
    add_demand_history_sequences,
    history_feature_columns,
)
from src.utils.config import DATA_DIR, FIGURES_DIR, FORECAST_HORIZON, RESULTS_DIR, RANDOM_SEED
from src.utils.io import ensure_dir, save_table


LOGGER = logging.getLogger(__name__)


CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

NUMERIC_FEATURES = [
    "d_int",
    "sell_price",
    "sell_price_lag_1",
    "sell_price_change_1",
    "sell_price_pct_change_1",
    "sell_price_rolling_mean_28",
    "sell_price_relative_to_28d_mean",
    "dayofweek",
    "month",
    "year",
    "day",
    "weekofyear",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "snap_any",
    "has_event_name_1",
    "has_event_type_1",
    "has_event_name_2",
    "has_event_type_2",
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
    "demand_rolling_mean_7",
    "demand_rolling_mean_14",
    "demand_rolling_mean_28",
]

DEMAND_FEATURES = [
    "d_int",
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
    "demand_rolling_mean_7",
    "demand_rolling_mean_14",
    "demand_rolling_mean_28",
]

STABILITY_ALPHAS = (0.5, 0.7, 0.9)
STABILITY_LAMBDAS = (0.0, 0.01, 0.05, 0.1, 0.5)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the baseline experiment."""
    parser = argparse.ArgumentParser(description="Run an M5 XGBoost baseline forecasting pipeline.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing M5 CSV files.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Directory for metric outputs.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Directory for plots.")
    parser.add_argument("--validation-days", type=int, default=FORECAST_HORIZON, help="Validation horizon.")
    parser.add_argument("--max-series", type=int, default=None, help="Optional number of item-store series to use.")
    parser.add_argument("--use-evaluation", action="store_true", help="Use sales_train_evaluation.csv if present.")
    parser.add_argument(
        "--stability-lambdas",
        type=float,
        nargs="+",
        default=list(STABILITY_LAMBDAS),
        help="Lambda values for stability-aware objective training.",
    )
    parser.add_argument("--timemixer-history-length", type=int, default=28, help="Demand history window length.")
    parser.add_argument("--timemixer-embedding-dim", type=int, default=16, help="Temporal embedding dimension.")
    parser.add_argument("--timemixer-max-iter", type=int, default=80, help="Maximum TimeMixer training iterations.")
    parser.add_argument(
        "--timemixer-max-train-rows",
        type=int,
        default=50_000,
        help="Maximum sampled rows for fitting the TimeMixer encoder.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise runtime logging for experiment runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_and_merge(
    data_dir: Path,
    use_evaluation: bool = False,
    max_series: int | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Load sales, calendar, and price files, then merge them into one panel."""
    datasets = load_m5_forecasting_inputs(data_dir, use_evaluation=use_evaluation)
    sales_key = "sales_train_evaluation" if use_evaluation else "sales_train_validation"
    sales = datasets[sales_key]

    if max_series is not None:
        sample_size = min(max_series, len(sales))
        if random_seed is None:
            sales = sales.head(sample_size).copy()
        else:
            sales = sales.sample(n=sample_size, random_state=random_seed).sort_index().copy()

    return preprocess_m5(
        sales=sales,
        calendar=datasets["calendar"],
        sell_prices=datasets["sell_prices"],
    )


def prepare_modeling_table(
    df: pd.DataFrame,
    max_series: int | None = None,
    history_length: int = 28,
) -> tuple[pd.DataFrame, list[str]]:
    """Create features and return a clean model-ready table."""
    if max_series is not None:
        selected_ids = df["id"].drop_duplicates().head(max_series)
        df = df[df["id"].isin(selected_ids)].copy()

    featured = build_baseline_features(df)
    featured = add_demand_history_sequences(featured, history_length=history_length)
    feature_columns = [column for column in CATEGORICAL_FEATURES + NUMERIC_FEATURES if column in featured.columns]
    modeling = featured.dropna(
        subset=["demand", "demand_lag_28", "demand_rolling_mean_28", f"demand_history_lag_{history_length}"]
    ).copy()

    for column in feature_columns:
        if column not in CATEGORICAL_FEATURES:
            modeling[column] = modeling[column].fillna(0)

    return modeling, feature_columns


def apply_train_fitted_categorical_encoding(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit categorical levels on train only and apply them to validation."""
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    train = train.copy()
    validation = validation.copy()

    for column in categorical_features:
        if column not in train.columns or column not in validation.columns:
            continue
        categories = pd.Index(train[column].dropna().astype("object").unique())
        train[column] = pd.Categorical(train[column], categories=categories)
        validation[column] = pd.Categorical(validation[column], categories=categories)

    return train, validation


def split_train_validation(
    df: pd.DataFrame,
    validation_days: int = FORECAST_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by the final validation window to preserve time ordering."""
    validation_start = df["d_int"].max() - validation_days + 1
    train = df[df["d_int"] < validation_start].copy()
    validation = df[df["d_int"] >= validation_start].copy()
    return apply_train_fitted_categorical_encoding(train, validation)


def train_xgboost(
    train: pd.DataFrame,
    feature_columns: list[str],
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 300,
) -> XGBRegressor:
    """Train a compact XGBoost regression baseline."""
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
    )
    model.fit(train[feature_columns], train["demand"])
    return model


def evaluate_predictions(validation: pd.DataFrame, predictions) -> dict[str, float]:
    """Compute accuracy and stability metrics."""
    return {
        "mae": mean_absolute_error(validation["demand"], predictions),
        "rmse": root_mean_squared_error(validation["demand"], predictions),
        "mape": mean_absolute_percentage_error(validation["demand"], predictions),
        "fss": grouped_forecast_stability_score(predictions, validation["id"], validation["d_int"]),
        "forecast_volatility": grouped_forecast_volatility(predictions, validation["id"], validation["d_int"]),
    }


def predict_with_feature_set(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 300,
) -> np.ndarray:
    """Train XGBoost and return non-negative validation predictions."""
    model = train_xgboost(train, feature_columns, random_seed=random_seed, n_estimators=n_estimators)
    return model.predict(validation[feature_columns]).clip(min=0)


def build_method_comparison(
    validation: pd.DataFrame,
    raw_predictions: np.ndarray,
    alphas: tuple[float, ...] = STABILITY_ALPHAS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate raw XGBoost and stability-aware smoothed forecasts."""
    predictions_by_method = {"xgboost_baseline": raw_predictions}
    rows = [
        {
            "method": "xgboost_baseline",
            "alpha": np.nan,
            **evaluate_predictions(validation, raw_predictions),
        }
    ]

    for alpha in alphas:
        smoothed = exponential_smooth_predictions(validation, raw_predictions, alpha=alpha)
        method_name = f"xgboost_smoothing_alpha_{alpha}"
        predictions_by_method[method_name] = smoothed
        rows.append(
            {
                "method": method_name,
                "alpha": alpha,
                **evaluate_predictions(validation, smoothed),
            }
        )

    return pd.DataFrame(rows), predictions_by_method


def build_stability_objective_comparison(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    lambdas: list[float],
    random_seed: int = RANDOM_SEED,
    num_boost_round: int = 300,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Train and evaluate stability-aware objective models across lambda values."""
    rows = []
    predictions_by_lambda = {}

    for stability_lambda in lambdas:
        model = train_stability_aware_xgboost(
            train,
            feature_columns,
            stability_lambda=stability_lambda,
            random_seed=random_seed,
            num_boost_round=num_boost_round,
        )
        raw_predictions, predictions = predict_stability_aware_xgboost_with_scale(model, validation, feature_columns)
        method = f"stability_objective_lambda_{stability_lambda:g}"
        predictions_by_lambda[method] = predictions
        metrics = evaluate_predictions(validation, predictions)
        forecast_loss = forecast_loss_mse(validation["demand"], predictions)
        stability_loss = stability_loss_mean_abs_change(predictions, validation["id"], validation["d_int"])
        total_loss = forecast_loss + stability_lambda * stability_loss
        scale_summary = prediction_scale_summary(raw_predictions, predictions)
        rows.append(
            {
                "method": method,
                "lambda": stability_lambda,
                "forecast_loss_mse": forecast_loss,
                "stability_loss": stability_loss,
                "total_loss": total_loss,
                **scale_summary,
                **metrics,
            }
        )
        LOGGER.info(
            (
                "lambda=%s | forecast_loss=%.6f | stability_loss=%.6f | "
                "total_loss=%.6f | raw_range=[%.4f, %.4f] | scaled_range=[%.4f, %.4f]"
            ),
            stability_lambda,
            forecast_loss,
            stability_loss,
            total_loss,
            scale_summary["raw_pred_min"],
            scale_summary["raw_pred_max"],
            scale_summary["scaled_pred_min"],
            scale_summary["scaled_pred_max"],
        )

    return pd.DataFrame(rows), predictions_by_lambda


def build_three_method_comparison(
    validation: pd.DataFrame,
    baseline_predictions: np.ndarray,
    smoothing_comparison: pd.DataFrame,
    smoothing_predictions: dict[str, np.ndarray],
    objective_comparison: pd.DataFrame,
    objective_predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compare baseline, best smoothing-only, and best stability-objective methods."""
    smoothing_method, smoothing_pred = select_best_smoothed_predictions(
        smoothing_comparison,
        smoothing_predictions,
    )
    objective_best_row = objective_comparison.sort_values(["rmse", "fss"]).iloc[0]
    objective_method = str(objective_best_row["method"])
    objective_pred = objective_predictions[objective_method]

    methods = {
        "baseline XGBoost": baseline_predictions,
        f"smoothing-only ({smoothing_method})": smoothing_pred,
        f"stability-aware objective ({objective_method})": objective_pred,
    }

    rows = []
    for method, predictions in methods.items():
        rows.append({"method": method, **evaluate_predictions(validation, predictions)})

    return pd.DataFrame(rows), methods


def fit_timemixer_encoder(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: TimeMixerConfig,
) -> tuple[TimeMixerTemporalEncoder, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Train TimeMixer and return train/validation embeddings plus direct forecasts."""
    columns = history_feature_columns(config.history_length)
    encoder = TimeMixerTemporalEncoder(config)
    encoder.fit(train[columns].to_numpy(dtype=float), train["demand"].to_numpy(dtype=float))

    train_embeddings = encoder.transform(train[columns].to_numpy(dtype=float))
    validation_sequences = validation[columns].to_numpy(dtype=float)
    validation_embeddings = encoder.transform(validation_sequences)
    timemixer_predictions = encoder.predict(validation_sequences)
    return encoder, train_embeddings, validation_embeddings, timemixer_predictions


def append_temporal_embeddings(
    frame: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Append TimeMixer embeddings to a modeling frame."""
    embedding_columns = list(embeddings.columns)
    output = frame.reset_index(drop=True).copy()
    output = pd.concat([output, embeddings.reset_index(drop=True)], axis=1)
    return output, embedding_columns


def build_temporal_embedding_analysis(
    validation: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize learned temporal embedding dimensions for inspection."""
    rows = []
    for column in embeddings.columns:
        values = embeddings[column].to_numpy(dtype=float)
        demand_values = validation["demand"].to_numpy(dtype=float)
        corr_with_demand = (
            float(np.corrcoef(values, demand_values)[0, 1])
            if np.std(values) > 0 and np.std(demand_values) > 0
            else 0.0
        )
        rows.append(
            {
                "embedding": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "corr_with_demand": corr_with_demand,
            }
        )
    return pd.DataFrame(rows)


def build_hybrid_model_comparison(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    baseline_predictions: np.ndarray,
    lambdas: list[float],
    timemixer_config: TimeMixerConfig,
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Compare XGBoost, TimeMixer-only, hybrid, and hybrid stability-objective models."""
    _, train_embeddings, validation_embeddings, timemixer_predictions = fit_timemixer_encoder(
        train,
        validation,
        timemixer_config,
    )
    hybrid_train, embedding_columns = append_temporal_embeddings(train, train_embeddings)
    hybrid_validation, _ = append_temporal_embeddings(validation, validation_embeddings)
    hybrid_feature_columns = feature_columns + embedding_columns

    LOGGER.info("Training Hybrid TimeMixer + XGBoost")
    hybrid_predictions = predict_with_feature_set(
        hybrid_train,
        hybrid_validation,
        hybrid_feature_columns,
        random_seed=random_seed,
        n_estimators=n_estimators,
    )

    LOGGER.info("Training Hybrid + Stability Objective models")
    hybrid_objective_comparison, hybrid_objective_predictions = build_stability_objective_comparison(
        hybrid_train,
        hybrid_validation,
        hybrid_feature_columns,
        lambdas=lambdas,
        random_seed=random_seed,
        num_boost_round=n_estimators,
    )
    best_hybrid_objective_row = hybrid_objective_comparison.sort_values(["rmse", "fss"]).iloc[0]
    best_hybrid_objective_method = str(best_hybrid_objective_row["method"])
    best_hybrid_objective_predictions = hybrid_objective_predictions[best_hybrid_objective_method]

    predictions_by_method = {
        "XGBoost baseline": baseline_predictions,
        "TimeMixer only": timemixer_predictions,
        "Hybrid TimeMixer + XGBoost": hybrid_predictions,
        f"Hybrid + Stability Objective ({best_hybrid_objective_method})": best_hybrid_objective_predictions,
    }

    rows = []
    for method, predictions in predictions_by_method.items():
        rows.append({"method": method, **evaluate_predictions(validation, predictions)})

    comparison = pd.DataFrame(rows)
    embedding_analysis = build_temporal_embedding_analysis(validation, validation_embeddings)
    embedding_analysis["best_hybrid_objective_lambda"] = float(best_hybrid_objective_row["lambda"])
    return comparison, embedding_analysis, predictions_by_method


def select_best_smoothed_predictions(
    comparison: pd.DataFrame,
    predictions_by_method: dict[str, np.ndarray],
) -> tuple[str, np.ndarray]:
    """Select the smoothed setting with the lowest RMSE."""
    smoothed_rows = comparison[comparison["method"].str.contains("smoothing_alpha")]
    best_method = smoothed_rows.sort_values(["rmse", "fss"]).iloc[0]["method"]
    return str(best_method), predictions_by_method[str(best_method)]


def build_ablation_table(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    full_feature_columns: list[str],
    full_raw_predictions: np.ndarray | None = None,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run a compact ablation study over features and smoothing."""
    baseline_features = [
        column
        for column in CATEGORICAL_FEATURES + DEMAND_FEATURES
        if column in full_feature_columns
    ]

    baseline_raw = predict_with_feature_set(train, validation, baseline_features, random_seed=random_seed)
    full_raw = (
        full_raw_predictions
        if full_raw_predictions is not None
        else predict_with_feature_set(train, validation, full_feature_columns, random_seed=random_seed)
    )

    baseline_smoothed_comparison, baseline_smoothed_predictions = build_method_comparison(
        validation,
        baseline_raw,
        alphas=STABILITY_ALPHAS,
    )
    full_smoothed_comparison, full_smoothed_predictions = build_method_comparison(
        validation,
        full_raw,
        alphas=STABILITY_ALPHAS,
    )

    baseline_smoothing_method, baseline_smoothing_pred = select_best_smoothed_predictions(
        baseline_smoothed_comparison,
        baseline_smoothed_predictions,
    )
    full_method_name, full_method_pred = select_best_smoothed_predictions(
        full_smoothed_comparison,
        full_smoothed_predictions,
    )

    methods = {
        "baseline": baseline_raw,
        "baseline + calendar/price features": full_raw,
        "baseline + smoothing": baseline_smoothing_pred,
        "full method": full_method_pred,
    }

    rows = []
    for method, predictions in methods.items():
        smoothing_alpha = np.nan
        if method == "baseline + smoothing":
            smoothing_alpha = float(baseline_smoothing_method.rsplit("_", 1)[-1])
        if method == "full method":
            smoothing_alpha = float(full_method_name.rsplit("_", 1)[-1])

        rows.append(
            {
                "ablation": method,
                "smoothing_alpha": smoothing_alpha,
                **evaluate_predictions(validation, predictions),
            }
        )

    return pd.DataFrame(rows), methods


def save_outputs(
    validation: pd.DataFrame,
    predictions,
    metrics: dict[str, float],
    feature_columns: list[str],
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Save metrics, validation predictions, and baseline diagnostic plots."""
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    predictions_df = validation[["id", "item_id", "store_id", "date", "d_int", "demand"]].copy()
    predictions_df["prediction"] = predictions
    predictions_df["absolute_error"] = (predictions_df["demand"] - predictions_df["prediction"]).abs()

    save_table(pd.DataFrame([metrics]), results_dir / "xgboost_baseline_metrics.csv")
    save_table(predictions_df, results_dir / "xgboost_baseline_predictions.csv")
    save_table(pd.DataFrame({"feature": feature_columns}), results_dir / "xgboost_baseline_features.csv")

    daily = predictions_df.groupby("date", as_index=False)[["demand", "prediction"]].sum()
    plt.figure(figsize=(10, 5))
    plt.plot(daily["date"], daily["demand"], label="Actual demand")
    plt.plot(daily["date"], daily["prediction"], label="Predicted demand")
    plt.title("M5 XGBoost Baseline: Validation Demand")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "xgboost_baseline_validation_demand.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(predictions_df["absolute_error"], bins=50)
    plt.title("M5 XGBoost Baseline: Absolute Error Distribution")
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "xgboost_baseline_absolute_error.png", dpi=150)
    plt.close()


def save_stability_outputs(
    validation: pd.DataFrame,
    method_comparison: pd.DataFrame,
    predictions_by_method: dict[str, np.ndarray],
    ablation_table: pd.DataFrame,
    ablation_predictions: dict[str, np.ndarray],
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Save stability-aware tables and comparison plots."""
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    save_table(method_comparison, results_dir / "xgboost_stability_alpha_comparison.csv")
    save_table(ablation_table, results_dir / "xgboost_ablation_study.csv")

    predictions_df = validation[["id", "item_id", "store_id", "date", "d_int", "demand"]].copy()
    for method, predictions in predictions_by_method.items():
        predictions_df[method] = predictions
    save_table(predictions_df, results_dir / "xgboost_stability_predictions.csv")

    daily = predictions_df.groupby("date", as_index=False).sum(numeric_only=True)
    plt.figure(figsize=(11, 5))
    plt.plot(daily["date"], daily["demand"], label="Actual demand", linewidth=2)
    for method in predictions_by_method:
        plt.plot(daily["date"], daily[method], label=method)
    plt.title("M5 Forecast Comparison: Raw vs Stability-Aware Predictions")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "xgboost_stability_forecast_comparison.png", dpi=150)
    plt.close()

    metric_columns = ["mae", "rmse", "mape", "fss", "forecast_volatility"]
    plot_data = method_comparison.set_index("method")[metric_columns]
    axes = plot_data.plot(kind="bar", subplots=True, layout=(3, 2), figsize=(12, 9), legend=False)
    for axis in axes.ravel():
        axis.tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / "xgboost_stability_metric_comparison.png", dpi=150)
    plt.close()

    ablation_df = validation[["id", "item_id", "store_id", "date", "d_int", "demand"]].copy()
    for method, predictions in ablation_predictions.items():
        ablation_df[method] = predictions
    daily_ablation = ablation_df.groupby("date", as_index=False).sum(numeric_only=True)

    plt.figure(figsize=(11, 5))
    plt.plot(daily_ablation["date"], daily_ablation["demand"], label="Actual demand", linewidth=2)
    for method in ablation_predictions:
        plt.plot(daily_ablation["date"], daily_ablation[method], label=method)
    plt.title("M5 Ablation Study: Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "xgboost_ablation_forecast_comparison.png", dpi=150)
    plt.close()


def save_stability_objective_outputs(
    validation: pd.DataFrame,
    objective_comparison: pd.DataFrame,
    objective_predictions: dict[str, np.ndarray],
    three_method_comparison: pd.DataFrame,
    three_method_predictions: dict[str, np.ndarray],
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Save stability-objective tables, summaries, and tradeoff plots."""
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    save_table(three_method_comparison, results_dir / "stability_objective_comparison.csv")
    save_table(objective_comparison, results_dir / "stability_lambda_ablation.csv")

    prediction_frame = validation[["id", "item_id", "store_id", "date", "d_int", "demand"]].copy()
    for method, predictions in objective_predictions.items():
        prediction_frame[method] = predictions
    save_table(prediction_frame, results_dir / "stability_objective_predictions.csv")

    summary_lines = [
        "Stability-aware objective experiment summary",
        "",
        "Three-method comparison:",
        three_method_comparison.to_string(index=False),
        "",
        "Lambda ablation:",
        objective_comparison.to_string(index=False),
    ]
    (results_dir / "stability_objective_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.plot(objective_comparison["fss"], objective_comparison["rmse"], marker="o")
    for _, row in objective_comparison.iterrows():
        plt.annotate(f"lambda={row['lambda']:g}", (row["fss"], row["rmse"]), fontsize=8)
    plt.title("Stability-Accuracy Tradeoff")
    plt.xlabel("Forecast Stability Score")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(figures_dir / "stability_tradeoff_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(objective_comparison["forecast_volatility"], objective_comparison["mae"])
    for _, row in objective_comparison.iterrows():
        plt.annotate(f"lambda={row['lambda']:g}", (row["forecast_volatility"], row["mae"]), fontsize=8)
    plt.title("Forecast Volatility vs Accuracy")
    plt.xlabel("Forecast Volatility")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(figures_dir / "stability_vs_accuracy.png", dpi=150)
    plt.close()

    daily = validation[["date", "demand"]].copy()
    for method, predictions in three_method_predictions.items():
        daily[method] = predictions
    daily = daily.groupby("date", as_index=False).sum(numeric_only=True)

    plt.figure(figsize=(11, 5))
    plt.plot(daily["date"], daily["demand"], label="Actual demand", linewidth=2)
    for method in three_method_predictions:
        plt.plot(daily["date"], daily[method], label=method)
    plt.title("Baseline vs Smoothing vs Stability-Aware Objective")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "stability_objective_method_comparison.png", dpi=150)
    plt.close()


def save_hybrid_outputs(
    validation: pd.DataFrame,
    hybrid_comparison: pd.DataFrame,
    embedding_analysis: pd.DataFrame,
    hybrid_predictions: dict[str, np.ndarray],
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Save hybrid architecture tables and figures."""
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    save_table(hybrid_comparison, results_dir / "hybrid_model_comparison.csv")
    save_table(embedding_analysis, results_dir / "temporal_embedding_analysis.csv")

    prediction_frame = validation[["id", "item_id", "store_id", "date", "d_int", "demand"]].copy()
    for method, predictions in hybrid_predictions.items():
        prediction_frame[method] = predictions
    save_table(prediction_frame, results_dir / "hybrid_model_predictions.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(hybrid_comparison["fss"], hybrid_comparison["rmse"], marker="o")
    for _, row in hybrid_comparison.iterrows():
        plt.annotate(row["method"], (row["fss"], row["rmse"]), fontsize=8)
    plt.title("Hybrid Stability-Accuracy Tradeoff")
    plt.xlabel("Forecast Stability Score")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(figures_dir / "hybrid_tradeoff_curve.png", dpi=150)
    plt.close()

    daily = validation[["date", "demand"]].copy()
    for method, predictions in hybrid_predictions.items():
        daily[method] = predictions
    daily = daily.groupby("date", as_index=False).sum(numeric_only=True)

    plt.figure(figsize=(11, 5))
    plt.plot(daily["date"], daily["demand"], label="Actual demand", linewidth=2)
    for method in hybrid_predictions:
        plt.plot(daily["date"], daily[method], label=method)
    plt.title("Hybrid Forecast Visualization")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hybrid_forecast_visualization.png", dpi=150)
    plt.close()


def main() -> None:
    """Run the end-to-end baseline pipeline."""
    configure_logging()
    args = parse_args()
    LOGGER.info("Loading and merging M5 data from %s", args.data_dir)
    merged = load_and_merge(
        args.data_dir,
        use_evaluation=args.use_evaluation,
        max_series=args.max_series,
    )
    LOGGER.info("Preparing features")
    modeling_table, feature_columns = prepare_modeling_table(
        merged,
        history_length=args.timemixer_history_length,
    )
    train, validation = split_train_validation(modeling_table, validation_days=args.validation_days)

    if train.empty or validation.empty:
        raise ValueError("Train or validation split is empty. Check data files and validation window.")

    LOGGER.info("Training baseline XGBoost")
    predictions = predict_with_feature_set(train, validation, feature_columns)
    metrics = evaluate_predictions(validation, predictions)
    save_outputs(validation, predictions, metrics, feature_columns, args.results_dir, args.figures_dir)

    LOGGER.info("Evaluating smoothing-only methods")
    method_comparison, predictions_by_method = build_method_comparison(validation, predictions)
    ablation_table, ablation_predictions = build_ablation_table(
        train,
        validation,
        feature_columns,
        full_raw_predictions=predictions,
    )
    save_stability_outputs(
        validation,
        method_comparison,
        predictions_by_method,
        ablation_table,
        ablation_predictions,
        args.results_dir,
        args.figures_dir,
    )

    LOGGER.info("Training stability-aware objective models")
    objective_comparison, objective_predictions = build_stability_objective_comparison(
        train,
        validation,
        feature_columns,
        lambdas=args.stability_lambdas,
    )
    three_method_comparison, three_method_predictions = build_three_method_comparison(
        validation,
        predictions,
        method_comparison,
        predictions_by_method,
        objective_comparison,
        objective_predictions,
    )
    save_stability_objective_outputs(
        validation,
        objective_comparison,
        objective_predictions,
        three_method_comparison,
        three_method_predictions,
        args.results_dir,
        args.figures_dir,
    )

    LOGGER.info("Running hybrid TimeMixer experiments")
    timemixer_config = TimeMixerConfig(
        history_length=args.timemixer_history_length,
        embedding_dim=args.timemixer_embedding_dim,
        max_iter=args.timemixer_max_iter,
        max_train_rows=args.timemixer_max_train_rows,
    )
    hybrid_comparison, embedding_analysis, hybrid_predictions = build_hybrid_model_comparison(
        train,
        validation,
        feature_columns,
        predictions,
        lambdas=args.stability_lambdas,
        timemixer_config=timemixer_config,
    )
    save_hybrid_outputs(
        validation,
        hybrid_comparison,
        embedding_analysis,
        hybrid_predictions,
        args.results_dir,
        args.figures_dir,
    )

    print("Baseline XGBoost metrics")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.6f}")
    print("\nStability-aware comparison")
    print(method_comparison.to_string(index=False))
    print("\nAblation study")
    print(ablation_table.to_string(index=False))
    print("\nStability-aware objective comparison")
    print(three_method_comparison.to_string(index=False))
    print("\nLambda ablation")
    print(objective_comparison.to_string(index=False))
    print("\nHybrid model comparison")
    print(hybrid_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
