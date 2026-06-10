"""Publication-oriented experiment runner for the M5 hybrid framework."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.timemixer_encoder import TimeMixerConfig
from src.models.xgboost_baseline_pipeline import (
    build_ablation_table,
    build_hybrid_model_comparison,
    build_method_comparison,
    build_stability_objective_comparison,
    evaluate_predictions,
    load_and_merge,
    prepare_modeling_table,
    predict_with_feature_set,
    split_train_validation,
)
from src.utils.config import DATA_DIR, FORECAST_HORIZON, RANDOM_SEED
from src.utils.io import ensure_dir, save_table


LOGGER = logging.getLogger(__name__)
PAPER_RESULTS_DIR = Path("paper_results")
PAPER_FIGURES_DIR = Path("paper_figures")
LOG_DIR = Path("logs")


def parse_args() -> argparse.Namespace:
    """Parse publication experiment options."""
    parser = argparse.ArgumentParser(description="Run publication-ready M5 hybrid experiments.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing M5 CSV files.")
    parser.add_argument("--series-counts", type=int, nargs="+", default=[500, 1000], help="Series counts to run.")
    parser.add_argument("--results-dir", type=Path, default=PAPER_RESULTS_DIR, help="Final table output directory.")
    parser.add_argument("--figures-dir", type=Path, default=PAPER_FIGURES_DIR, help="Final figure output directory.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="Reproducible run log directory.")
    parser.add_argument("--validation-days", type=int, default=FORECAST_HORIZON, help="Validation horizon.")
    parser.add_argument("--stability-lambdas", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.5])
    parser.add_argument("--timemixer-history-length", type=int, default=28)
    parser.add_argument("--timemixer-embedding-dim", type=int, default=16)
    parser.add_argument("--timemixer-max-iter", type=int, default=80)
    parser.add_argument("--timemixer-max-train-rows", type=int, default=50_000)
    return parser.parse_args()


def configure_logging(log_dir: Path) -> Path:
    """Configure console and file logging for reproducibility."""
    ensure_dir(log_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"paper_experiments_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def set_publication_style() -> None:
    """Apply restrained, publication-friendly plotting defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def timed_step(name: str, func, runtime_rows: list[dict], series_count: int):
    """Run a step and record wall-clock runtime."""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    runtime_rows.append({"series_count": series_count, "step": name, "runtime_seconds": elapsed})
    LOGGER.info("%s | series=%s | runtime=%.2fs", name, series_count, elapsed)
    return result


def run_single_series_count(args: argparse.Namespace, series_count: int) -> dict[str, pd.DataFrame]:
    """Run all methods for one series-count setting."""
    runtime_rows: list[dict] = []
    LOGGER.info("Starting publication experiment for %s series", series_count)

    merged = timed_step(
        "load_and_merge",
        lambda: load_and_merge(args.data_dir, max_series=series_count),
        runtime_rows,
        series_count,
    )
    modeling_table, feature_columns = timed_step(
        "prepare_features",
        lambda: prepare_modeling_table(merged, history_length=args.timemixer_history_length),
        runtime_rows,
        series_count,
    )
    train, validation = split_train_validation(modeling_table, validation_days=args.validation_days)

    baseline_predictions = timed_step(
        "xgboost_baseline",
        lambda: predict_with_feature_set(train, validation, feature_columns),
        runtime_rows,
        series_count,
    )
    baseline_metrics = evaluate_predictions(validation, baseline_predictions)

    smoothing_comparison, smoothing_predictions = timed_step(
        "smoothing_only",
        lambda: build_method_comparison(validation, baseline_predictions),
        runtime_rows,
        series_count,
    )
    ablation_table, _ = timed_step(
        "ablation_study",
        lambda: build_ablation_table(train, validation, feature_columns, full_raw_predictions=baseline_predictions),
        runtime_rows,
        series_count,
    )
    objective_comparison, _ = timed_step(
        "stability_objective",
        lambda: build_stability_objective_comparison(
            train,
            validation,
            feature_columns,
            lambdas=args.stability_lambdas,
        ),
        runtime_rows,
        series_count,
    )

    timemixer_config = TimeMixerConfig(
        history_length=args.timemixer_history_length,
        embedding_dim=args.timemixer_embedding_dim,
        max_iter=args.timemixer_max_iter,
        max_train_rows=args.timemixer_max_train_rows,
    )
    hybrid_comparison, embedding_analysis, hybrid_predictions = timed_step(
        "hybrid_framework",
        lambda: build_hybrid_model_comparison(
            train,
            validation,
            feature_columns,
            baseline_predictions,
            lambdas=args.stability_lambdas,
            timemixer_config=timemixer_config,
        ),
        runtime_rows,
        series_count,
    )

    main_table = hybrid_comparison.copy()
    main_table.insert(0, "series_count", series_count)
    main_table["baseline_rmse"] = baseline_metrics["rmse"]
    main_table["rmse_improvement_pct"] = 100 * (baseline_metrics["rmse"] - main_table["rmse"]) / baseline_metrics["rmse"]
    main_table["fss_gain_pct"] = 100 * (baseline_metrics["fss"] - main_table["fss"]) / max(baseline_metrics["fss"], 1e-8)

    stability_tradeoff = objective_comparison.copy()
    stability_tradeoff.insert(0, "series_count", series_count)

    ablation_table = ablation_table.copy()
    ablation_table.insert(0, "series_count", series_count)

    embedding_analysis = embedding_analysis.copy()
    embedding_analysis.insert(0, "series_count", series_count)

    prediction_frame = validation[["id", "item_id", "store_id", "date", "d_int", "demand"]].copy()
    for method, predictions in hybrid_predictions.items():
        prediction_frame[method] = predictions
    objective_method = next(method for method in hybrid_predictions if method.startswith("Hybrid + Stability Objective"))
    prediction_frame["Hybrid + Stability Objective"] = hybrid_predictions[objective_method]
    prediction_frame["hybrid_stability_objective_label"] = objective_method
    prediction_frame.insert(0, "series_count", series_count)

    smoothing_best = smoothing_comparison.sort_values(["rmse", "fss"]).iloc[0].to_dict()
    summary = pd.DataFrame(
        [
            {
                "series_count": series_count,
                "best_method": main_table.sort_values(["rmse", "fss"]).iloc[0]["method"],
                "best_rmse": main_table["rmse"].min(),
                "baseline_rmse": baseline_metrics["rmse"],
                "average_rmse_improvement_pct": main_table.loc[
                    main_table["method"] != "XGBoost baseline", "rmse_improvement_pct"
                ].mean(),
                "best_relative_stability_gain_pct": main_table["fss_gain_pct"].max(),
                "best_smoothing_method": smoothing_best["method"],
            }
        ]
    )

    return {
        "main": main_table,
        "ablation": ablation_table,
        "stability": stability_tradeoff,
        "embedding": embedding_analysis,
        "predictions": prediction_frame,
        "runtime": pd.DataFrame(runtime_rows),
        "summary": summary,
    }


def save_combined_tables(outputs: list[dict[str, pd.DataFrame]], results_dir: Path) -> dict[str, pd.DataFrame]:
    """Save publication-ready combined tables."""
    ensure_dir(results_dir)
    combined = {
        "main_result_table": pd.concat([output["main"] for output in outputs], ignore_index=True),
        "ablation_study_table": pd.concat([output["ablation"] for output in outputs], ignore_index=True),
        "stability_tradeoff_table": pd.concat([output["stability"] for output in outputs], ignore_index=True),
        "temporal_embedding_analysis": pd.concat([output["embedding"] for output in outputs], ignore_index=True),
        "runtime_comparison": pd.concat([output["runtime"] for output in outputs], ignore_index=True),
        "experiment_summary_statistics": pd.concat([output["summary"] for output in outputs], ignore_index=True),
        "hybrid_predictions": pd.concat([output["predictions"] for output in outputs], ignore_index=True),
    }
    combined["prediction_distribution_diagnostics"] = build_prediction_distribution_diagnostics(
        combined["hybrid_predictions"]
    )

    for name, table in combined.items():
        save_table(table, results_dir / f"{name}.csv")

    return combined


def build_prediction_distribution_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare prediction scales across baseline, hybrid, and hybrid stability methods."""
    method_columns = ["XGBoost baseline", "Hybrid TimeMixer + XGBoost", "Hybrid + Stability Objective"]
    rows = []
    for series_count, group in predictions.groupby("series_count"):
        actual = group["demand"].to_numpy(dtype=float)
        rows.append(_distribution_row(series_count, "Actual demand", actual))
        for column in method_columns:
            rows.append(_distribution_row(series_count, column, group[column].to_numpy(dtype=float)))
    return pd.DataFrame(rows)


def _distribution_row(series_count: int, method: str, values: np.ndarray) -> dict[str, float | int | str]:
    values = np.asarray(values, dtype=float)
    return {
        "series_count": series_count,
        "method": method,
        "min": float(np.nanmin(values)),
        "p05": float(np.nanquantile(values, 0.05)),
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "p95": float(np.nanquantile(values, 0.95)),
        "max": float(np.nanmax(values)),
        "zero_share": float(np.mean(np.nan_to_num(values, nan=0.0) == 0)),
    }


def save_experiment_metadata(args: argparse.Namespace, log_path: Path, results_dir: Path) -> None:
    """Persist configuration metadata for reproducibility."""
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "log_path": str(log_path),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (results_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_publication_figures(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    """Save high-resolution PNG and PDF figures for paper drafts."""
    ensure_dir(figures_dir)
    main = tables["main_result_table"]
    stability = tables["stability_tradeoff_table"]
    runtime = tables["runtime_comparison"]
    predictions = tables["hybrid_predictions"]
    diagnostics = tables["prediction_distribution_diagnostics"]

    _save_metric_barplot(main, figures_dir / "main_result_table")
    _save_tradeoff_plot(stability, figures_dir / "stability_tradeoff")
    _save_runtime_plot(runtime, figures_dir / "runtime_comparison")
    _save_forecast_plot(predictions, figures_dir / "hybrid_forecast_visualization")
    _save_prediction_distribution_plot(predictions, figures_dir / "prediction_distribution_histograms")
    _save_loss_component_plot(stability, figures_dir / "loss_component_curves")
    _save_prediction_scale_plot(diagnostics, figures_dir / "prediction_scale_comparison")


def _save_all_formats(base_path: Path) -> None:
    """Save the current Matplotlib figure as PNG and PDF."""
    plt.tight_layout()
    plt.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _save_metric_barplot(main: pd.DataFrame, base_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ["rmse", "fss"], ["Forecast Accuracy", "Forecast Stability"]):
        pivot = main.pivot(index="method", columns="series_count", values=metric)
        pivot.plot(kind="bar", ax=axis, width=0.72)
        axis.set_title(title)
        axis.set_ylabel(metric.upper())
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelrotation=30)
        axis.legend(title="Series", frameon=False)
    _save_all_formats(base_path)


def _save_tradeoff_plot(stability: pd.DataFrame, base_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for series_count, group in stability.groupby("series_count"):
        ordered = group.sort_values("lambda")
        plt.plot(ordered["fss"], ordered["rmse"], marker="o", label=f"{series_count} series")
        for _, row in ordered.iterrows():
            plt.annotate(f"{row['lambda']:g}", (row["fss"], row["rmse"]), fontsize=8)
    plt.title("Stability Objective Tradeoff")
    plt.xlabel("Forecast Stability Score")
    plt.ylabel("RMSE")
    plt.legend(frameon=False)
    _save_all_formats(base_path)


def _save_loss_component_plot(stability: pd.DataFrame, base_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for series_count, group in stability.groupby("series_count"):
        ordered = group.sort_values("lambda")
        if {"forecast_loss_mse", "stability_loss"}.issubset(ordered.columns):
            plt.plot(ordered["lambda"], ordered["forecast_loss_mse"], marker="o", label=f"MSE {series_count}")
            plt.plot(ordered["lambda"], ordered["stability_loss"], marker="s", label=f"Stability {series_count}")
    plt.title("Loss Component Magnitudes")
    plt.xlabel("Stability Lambda")
    plt.ylabel("Loss Component Value")
    plt.legend(frameon=False, ncol=2)
    _save_all_formats(base_path)


def _save_runtime_plot(runtime: pd.DataFrame, base_path: Path) -> None:
    plt.figure(figsize=(9, 4.5))
    pivot = runtime.pivot_table(index="step", columns="series_count", values="runtime_seconds", aggfunc="sum")
    pivot.plot(kind="bar", ax=plt.gca(), width=0.75)
    plt.title("Runtime Comparison")
    plt.xlabel("")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Series", frameon=False)
    _save_all_formats(base_path)


def _save_forecast_plot(predictions: pd.DataFrame, base_path: Path) -> None:
    largest_series = predictions["series_count"].max()
    plot_data = predictions[predictions["series_count"] == largest_series].copy()
    method_columns = [
        "XGBoost baseline",
        "Hybrid TimeMixer + XGBoost",
        "Hybrid + Stability Objective",
    ]
    daily = plot_data[["date", "demand", *method_columns]].groupby("date", as_index=False).sum(numeric_only=True)

    plt.figure(figsize=(10, 4.8))
    plt.plot(daily["date"], daily["demand"], label="Actual", linewidth=2.2, color="black")
    for column in method_columns:
        plt.plot(daily["date"], daily[column], label=column, linewidth=1.6)
    plt.title(f"Hybrid Forecast Visualization ({largest_series} Series)")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend(frameon=False, ncol=2)
    _save_all_formats(base_path)


def _save_prediction_distribution_plot(predictions: pd.DataFrame, base_path: Path) -> None:
    method_columns = ["XGBoost baseline", "Hybrid TimeMixer + XGBoost", "Hybrid + Stability Objective"]
    largest_series = predictions["series_count"].max()
    plot_data = predictions[predictions["series_count"] == largest_series]

    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, np.nanquantile(plot_data["demand"], 0.99), 40)
    plt.hist(plot_data["demand"], bins=bins, alpha=0.35, label="Actual demand", density=True)
    for column in method_columns:
        plt.hist(plot_data[column], bins=bins, alpha=0.35, label=column, density=True)
    plt.title(f"Prediction Distribution Diagnostics ({largest_series} Series)")
    plt.xlabel("Demand / prediction")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    _save_all_formats(base_path)


def _save_prediction_scale_plot(diagnostics: pd.DataFrame, base_path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    plot_data = diagnostics[diagnostics["method"] != "Actual demand"].copy()
    for series_count, group in plot_data.groupby("series_count"):
        plt.plot(group["method"], group["mean"], marker="o", label=f"Mean {series_count}")
        plt.plot(group["method"], group["p95"], marker="s", linestyle="--", label=f"P95 {series_count}")
    plt.title("Prediction Scale Comparison")
    plt.xlabel("")
    plt.ylabel("Prediction magnitude")
    plt.xticks(rotation=25, ha="right")
    plt.legend(frameon=False, ncol=2)
    _save_all_formats(base_path)


def main() -> None:
    """Run publication-ready experiments."""
    args = parse_args()
    log_path = configure_logging(args.log_dir)
    set_publication_style()
    ensure_dir(args.results_dir)
    ensure_dir(args.figures_dir)

    LOGGER.info("Paper experiment configuration: %s", vars(args))
    outputs = [run_single_series_count(args, series_count) for series_count in args.series_counts]
    tables = save_combined_tables(outputs, args.results_dir)
    save_experiment_metadata(args, log_path, args.results_dir)
    save_publication_figures(tables, args.figures_dir)

    LOGGER.info("Saved paper tables to %s", args.results_dir)
    LOGGER.info("Saved paper figures to %s", args.figures_dir)
    print(tables["experiment_summary_statistics"].to_string(index=False))


if __name__ == "__main__":
    main()
