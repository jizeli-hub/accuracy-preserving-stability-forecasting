"""Large-scale and seed-robust experiments for the M5 hybrid framework."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.timemixer_encoder import TimeMixerConfig
from src.models.xgboost_baseline_pipeline import (
    append_temporal_embeddings,
    build_stability_objective_comparison,
    evaluate_predictions,
    fit_timemixer_encoder,
    load_and_merge,
    prepare_modeling_table,
    predict_with_feature_set,
    split_train_validation,
)
from src.utils.config import DATA_DIR, FORECAST_HORIZON
from src.utils.io import ensure_dir, save_table


LOGGER = logging.getLogger(__name__)
DEFAULT_SEEDS = [42, 123, 2024]
DEFAULT_SERIES_COUNTS = ["1000", "3000", "5000"]
METRIC_COLUMNS = [
    "mae",
    "rmse",
    "mape",
    "fss",
    "forecast_volatility",
    "runtime_seconds",
    "relative_stability_improvement_pct",
    "relative_rmse_change_pct",
]


def parse_args() -> argparse.Namespace:
    """Parse large-scale experiment options."""
    parser = argparse.ArgumentParser(description="Run large-scale M5 hybrid robustness experiments.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing M5 CSV files.")
    parser.add_argument(
        "--max-series",
        nargs="+",
        default=DEFAULT_SERIES_COUNTS,
        help="Series counts to evaluate, e.g. 1000 3000 5000, or full.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        nargs="+",
        default=None,
        help="One or more seeds. Defaults to 42 123 2024.",
    )
    parser.add_argument("--num-runs", type=int, default=3, help="Number of seed runs per series count.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Root directory for paper outputs.")
    parser.add_argument("--validation-days", type=int, default=FORECAST_HORIZON, help="Validation horizon.")
    parser.add_argument("--stability-lambda", type=float, default=0.05, help="Selected stability objective lambda.")
    parser.add_argument("--timemixer-history-length", type=int, default=28)
    parser.add_argument("--timemixer-embedding-dim", type=int, default=16)
    parser.add_argument("--timemixer-max-iter", type=int, default=80)
    parser.add_argument("--timemixer-max-train-rows", type=int, default=50_000)
    parser.add_argument("--xgboost-n-estimators", type=int, default=300, help="Number of boosting rounds/trees.")
    parser.add_argument(
        "--xgboost-max-train-rows",
        type=int,
        default=None,
        help="Optional reproducible training-row cap for memory-safe large-scale runs.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path) -> Path:
    """Configure reproducible console and file logging."""
    ensure_dir(log_dir)
    log_path = log_dir / f"large_scale_experiments_{time.strftime('%Y%m%d_%H%M%S')}.log"
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
    """Set clean plotting defaults for publication figures."""
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


def resolve_series_counts(values: list[str]) -> list[int | None]:
    """Resolve integer series counts and optional full-run sentinel."""
    resolved: list[int | None] = []
    for value in values:
        if str(value).lower() in {"full", "all", "none"}:
            resolved.append(None)
        else:
            resolved.append(int(value))
    return resolved


def resolve_seeds(seed_values: list[int] | None, num_runs: int) -> list[int]:
    """Resolve default or user-provided seeds for repeated experiments."""
    seeds = list(seed_values) if seed_values else DEFAULT_SEEDS.copy()
    if len(seeds) < num_runs:
        rng = np.random.default_rng(seeds[0])
        while len(seeds) < num_runs:
            candidate = int(rng.integers(1, 1_000_000))
            if candidate not in seeds:
                seeds.append(candidate)
    return seeds[:num_runs]


def memory_usage_mb() -> float | None:
    """Return process memory usage when the platform exposes it."""
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return None


def series_label(series_count: int | None) -> str:
    """Return a stable label for a series-count setting."""
    return "full" if series_count is None else str(series_count)


def timed_call(step: str, func, runtime_rows: list[dict], context: dict):
    """Run a function while logging runtime and memory."""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    memory_mb = memory_usage_mb()
    row = {**context, "step": step, "runtime_seconds": elapsed, "memory_mb": memory_mb}
    runtime_rows.append(row)
    LOGGER.info(
        "%s | series=%s | seed=%s | runtime=%.2fs | memory=%s MB",
        step,
        context["series_count"],
        context["seed"],
        elapsed,
        f"{memory_mb:.1f}" if memory_mb is not None else "unavailable",
    )
    return result


def run_one_setting(
    args: argparse.Namespace,
    series_count: int | None,
    seed: int,
    log_dir: Path,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one series-count/seed experiment and return method and runtime rows."""
    context = {"series_count": series_label(series_count), "max_series": series_count, "seed": seed}
    runtime_rows: list[dict] = []
    LOGGER.info("Starting run | series=%s | seed=%s", context["series_count"], seed)

    merged = timed_call(
        "load_and_merge",
        lambda: load_and_merge(args.data_dir, max_series=series_count, random_seed=seed),
        runtime_rows,
        context,
    )
    modeling_table, feature_columns = timed_call(
        "prepare_features",
        lambda: prepare_modeling_table(merged, history_length=args.timemixer_history_length),
        runtime_rows,
        context,
    )
    train, validation = split_train_validation(modeling_table, validation_days=args.validation_days)
    del merged, modeling_table
    gc.collect()

    fit_train = sample_training_rows(train, args.xgboost_max_train_rows, seed, "shared")
    if fit_train is not train:
        del train
        gc.collect()

    baseline_start = time.perf_counter()
    baseline_predictions = predict_with_feature_set(
        fit_train,
        validation,
        feature_columns,
        random_seed=seed,
        n_estimators=args.xgboost_n_estimators,
    )
    baseline_runtime = time.perf_counter() - baseline_start
    baseline_metrics = evaluate_predictions(validation, baseline_predictions)

    timemixer_config = TimeMixerConfig(
        history_length=args.timemixer_history_length,
        embedding_dim=args.timemixer_embedding_dim,
        max_iter=args.timemixer_max_iter,
        max_train_rows=args.timemixer_max_train_rows,
        random_seed=seed,
    )

    timemixer_start = time.perf_counter()
    _, train_embeddings, validation_embeddings, timemixer_predictions = fit_timemixer_encoder(
        fit_train,
        validation,
        timemixer_config,
    )
    timemixer_runtime = time.perf_counter() - timemixer_start

    hybrid_train, embedding_columns = append_temporal_embeddings(fit_train, train_embeddings)
    hybrid_validation, _ = append_temporal_embeddings(validation, validation_embeddings)
    hybrid_feature_columns = feature_columns + embedding_columns

    hybrid_start = time.perf_counter()
    hybrid_predictions = predict_with_feature_set(
        hybrid_train,
        hybrid_validation,
        hybrid_feature_columns,
        random_seed=seed,
        n_estimators=args.xgboost_n_estimators,
    )
    hybrid_runtime = time.perf_counter() - hybrid_start

    stability_start = time.perf_counter()
    stability_comparison, stability_predictions = build_stability_objective_comparison(
        hybrid_train,
        hybrid_validation,
        hybrid_feature_columns,
        lambdas=[args.stability_lambda],
        random_seed=seed,
        num_boost_round=args.xgboost_n_estimators,
    )
    stability_runtime = time.perf_counter() - stability_start
    stability_method = str(stability_comparison.iloc[0]["method"])
    hybrid_stability_predictions = stability_predictions[stability_method]

    method_rows = [
        build_method_row(context, "XGBoost baseline", baseline_metrics, baseline_metrics, baseline_runtime),
        build_method_row(
            context,
            "TimeMixer only",
            evaluate_predictions(validation, timemixer_predictions),
            baseline_metrics,
            timemixer_runtime,
        ),
        build_method_row(
            context,
            "Hybrid TimeMixer + XGBoost",
            evaluate_predictions(validation, hybrid_predictions),
            baseline_metrics,
            timemixer_runtime + hybrid_runtime,
        ),
        build_method_row(
            context,
            f"Hybrid + Stability Objective (lambda={args.stability_lambda:g})",
            evaluate_predictions(validation, hybrid_stability_predictions),
            baseline_metrics,
            timemixer_runtime + stability_runtime,
        ),
    ]
    method_results = pd.DataFrame(method_rows)

    runtime_rows.extend(
        [
            {**context, "step": "XGBoost baseline", "runtime_seconds": baseline_runtime, "memory_mb": memory_usage_mb()},
            {**context, "step": "TimeMixer only", "runtime_seconds": timemixer_runtime, "memory_mb": memory_usage_mb()},
            {
                **context,
                "step": "Hybrid TimeMixer + XGBoost",
                "runtime_seconds": timemixer_runtime + hybrid_runtime,
                "memory_mb": memory_usage_mb(),
            },
            {
                **context,
                "step": f"Hybrid + Stability Objective (lambda={args.stability_lambda:g})",
                "runtime_seconds": timemixer_runtime + stability_runtime,
                "memory_mb": memory_usage_mb(),
            },
        ]
    )
    runtime_results = pd.DataFrame(runtime_rows)

    save_run_summary(
        args=args,
        method_results=method_results,
        runtime_results=runtime_results,
        log_dir=log_dir,
        results_dir=results_dir,
        context=context,
    )

    del fit_train, validation, hybrid_train, hybrid_validation
    gc.collect()
    return method_results, runtime_results


def sample_training_rows(
    train: pd.DataFrame,
    max_rows: int | None,
    seed: int,
    label: str,
) -> pd.DataFrame:
    """Return a reproducible training subset when a row cap is requested."""
    if max_rows is None or len(train) <= max_rows:
        return train

    sampled = train.sample(n=max_rows, random_state=seed).sort_values(["id", "d_int"]).copy()
    LOGGER.info(
        "Applied %s XGBoost training row cap: %s -> %s rows",
        label,
        len(train),
        len(sampled),
    )
    return sampled


def build_method_row(
    context: dict,
    method: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    runtime_seconds: float,
) -> dict:
    """Build a single method result row with relative metrics."""
    return {
        **context,
        "method": method,
        **metrics,
        "runtime_seconds": runtime_seconds,
        "relative_stability_improvement_pct": 100
        * (baseline_metrics["fss"] - metrics["fss"])
        / max(baseline_metrics["fss"], 1e-8),
        "relative_rmse_change_pct": 100
        * (metrics["rmse"] - baseline_metrics["rmse"])
        / max(baseline_metrics["rmse"], 1e-8),
    }


def save_run_summary(
    args: argparse.Namespace,
    method_results: pd.DataFrame,
    runtime_results: pd.DataFrame,
    log_dir: Path,
    results_dir: Path,
    context: dict,
) -> None:
    """Save a text summary for a single run."""
    ensure_dir(log_dir)
    summary_path = log_dir / f"large_scale_series_{context['series_count']}_seed_{context['seed']}.txt"
    lines = [
        "Large-scale hybrid forecasting run summary",
        "",
        f"series_count: {context['series_count']}",
        f"seed: {context['seed']}",
        f"output_path: {results_dir}",
        f"configuration: {json.dumps(serializable_args(args), sort_keys=True)}",
        "",
        "Method results:",
        method_results.to_string(index=False),
        "",
        "Runtime results:",
        runtime_results.to_string(index=False),
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def serializable_args(args: argparse.Namespace) -> dict:
    """Convert argparse namespace into JSON-friendly metadata."""
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def aggregate_results(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metric means and standard deviations across seeds."""
    grouped = per_run.groupby(["series_count", "method"], sort=False)
    aggregated = grouped[METRIC_COLUMNS].agg(["mean", "std"]).reset_index()
    aggregated.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in aggregated.columns
    ]
    for column in aggregated.columns:
        if column.endswith("_std"):
            aggregated[column] = aggregated[column].fillna(0)
    aggregated["num_runs"] = grouped.size().to_numpy()
    return aggregated


def build_ablation_results(main_results: pd.DataFrame) -> pd.DataFrame:
    """Create a component ablation table from the four large-scale methods."""
    component_map = {
        "XGBoost baseline": "structured features only",
        "TimeMixer only": "temporal encoder only",
        "Hybrid TimeMixer + XGBoost": "temporal + structured fusion",
    }
    ablation = main_results.copy()
    ablation["component_setting"] = ablation["method"].map(component_map).fillna(
        "temporal + structured fusion + stability objective"
    )
    return ablation


def build_seed_robustness_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed sensitivity for each scale and method."""
    summary = aggregate_results(per_run)
    for metric in ["rmse", "fss", "forecast_volatility"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        summary[f"{metric}_cv_pct"] = 100 * summary[std_col].fillna(0) / summary[mean_col].abs().clip(lower=1e-8)
    return summary


def save_outputs(
    per_run: pd.DataFrame,
    runtime_per_run: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Save final large-scale CSV outputs and figures."""
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    main_results = aggregate_results(per_run)
    ablation_results = build_ablation_results(main_results)
    runtime_results = aggregate_runtime(runtime_per_run)
    seed_summary = build_seed_robustness_summary(per_run)

    save_table(main_results, results_dir / "large_scale_main_results.csv")
    save_table(ablation_results, results_dir / "large_scale_ablation_results.csv")
    save_table(runtime_results, results_dir / "large_scale_runtime_results.csv")
    save_table(seed_summary, results_dir / "seed_robustness_summary.csv")
    save_table(per_run, results_dir / "large_scale_per_run_results.csv")
    save_table(runtime_per_run, results_dir / "large_scale_runtime_per_run.csv")

    save_large_scale_figures(main_results, runtime_results, figures_dir)


def aggregate_runtime(runtime_per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runtime rows across seeds."""
    grouped = runtime_per_run.groupby(["series_count", "step"], sort=False)
    runtime = grouped[["runtime_seconds", "memory_mb"]].agg(["mean", "std", "max"]).reset_index()
    runtime.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in runtime.columns
    ]
    for column in runtime.columns:
        if column.endswith("_std"):
            runtime[column] = runtime[column].fillna(0)
    runtime["num_runs"] = grouped.size().to_numpy()
    return runtime


def save_large_scale_figures(
    main_results: pd.DataFrame,
    runtime_results: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Generate publication-ready large-scale figures."""
    save_metric_comparison(
        main_results,
        metric="rmse",
        ylabel="RMSE",
        title="Large-Scale RMSE Comparison",
        path=figures_dir / "large_scale_rmse_comparison.png",
    )
    save_metric_comparison(
        main_results,
        metric="fss",
        ylabel="Forecast Stability Score",
        title="Large-Scale Stability Comparison",
        path=figures_dir / "large_scale_stability_comparison.png",
    )
    save_tradeoff_figure(main_results, figures_dir / "accuracy_stability_tradeoff_large_scale.png")
    save_runtime_figure(runtime_results, figures_dir / "runtime_scalability_curve.png")


def save_metric_comparison(
    main_results: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """Save grouped bar chart with seed standard deviation error bars."""
    pivot_mean = main_results.pivot(index="method", columns="series_count", values=f"{metric}_mean")
    pivot_std = main_results.pivot(index="method", columns="series_count", values=f"{metric}_std").fillna(0)
    axis = pivot_mean.plot(kind="bar", yerr=pivot_std, figsize=(11, 4.8), capsize=3, width=0.74)
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", labelrotation=25)
    axis.legend(title="Series", frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_tradeoff_figure(main_results: pd.DataFrame, path: Path) -> None:
    """Save accuracy-stability tradeoff scatter plot."""
    plt.figure(figsize=(8, 5))
    for series_count, group in main_results.groupby("series_count", sort=False):
        plt.scatter(group["fss_mean"], group["rmse_mean"], label=f"{series_count} series", s=48)
        for _, row in group.iterrows():
            plt.annotate(short_method_label(row["method"]), (row["fss_mean"], row["rmse_mean"]), fontsize=8)
    plt.title("Accuracy-Stability Tradeoff at Scale")
    plt.xlabel("Forecast Stability Score")
    plt.ylabel("RMSE")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_runtime_figure(runtime_results: pd.DataFrame, path: Path) -> None:
    """Save runtime scalability curve by method."""
    method_steps = runtime_results[
        runtime_results["step"].isin(["XGBoost baseline", "TimeMixer only", "Hybrid TimeMixer + XGBoost"])
        | runtime_results["step"].str.startswith("Hybrid + Stability Objective")
    ].copy()
    method_steps["series_numeric"] = method_steps["series_count"].replace("full", np.nan).astype(float)

    plt.figure(figsize=(8.5, 5))
    for step, group in method_steps.groupby("step", sort=False):
        ordered = group.sort_values("series_numeric")
        plt.plot(ordered["series_numeric"], ordered["runtime_seconds_mean"], marker="o", label=step)
    plt.title("Runtime Scalability")
    plt.xlabel("Number of series")
    plt.ylabel("Runtime (seconds)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def short_method_label(method: str) -> str:
    """Shorten method labels for annotations."""
    return (
        method.replace("XGBoost baseline", "XGB")
        .replace("TimeMixer only", "TM")
        .replace("Hybrid TimeMixer + XGBoost", "Hybrid")
        .replace("Hybrid + Stability Objective", "Hybrid+Stable")
    )


def main() -> None:
    """Run large-scale robustness experiments."""
    args = parse_args()
    output_dir = args.output_dir
    results_dir = output_dir / "paper_results"
    figures_dir = output_dir / "paper_figures"
    log_dir = output_dir / "logs"

    ensure_dir(results_dir)
    ensure_dir(figures_dir)
    ensure_dir(log_dir)
    configure_logging(log_dir)
    set_publication_style()

    series_counts = resolve_series_counts(args.max_series)
    seeds = resolve_seeds(args.random_seed, args.num_runs)
    LOGGER.info("Configuration: %s", serializable_args(args))
    LOGGER.info("Resolved series counts: %s", [series_label(count) for count in series_counts])
    LOGGER.info("Resolved seeds: %s", seeds)

    all_method_results = []
    all_runtime_results = []
    for series_count in series_counts:
        for seed in seeds:
            method_results, runtime_results = run_one_setting(args, series_count, seed, log_dir, results_dir)
            all_method_results.append(method_results)
            all_runtime_results.append(runtime_results)

    per_run = pd.concat(all_method_results, ignore_index=True)
    runtime_per_run = pd.concat(all_runtime_results, ignore_index=True)
    save_outputs(per_run, runtime_per_run, results_dir, figures_dir)

    metadata = {
        "configuration": serializable_args(args),
        "resolved_series_counts": [series_label(count) for count in series_counts],
        "resolved_seeds": seeds,
        "output_dir": str(output_dir),
    }
    (results_dir / "large_scale_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Saved large-scale results to %s", results_dir)
    LOGGER.info("Saved large-scale figures to %s", figures_dir)
    print(pd.read_csv(results_dir / "large_scale_main_results.csv").to_string(index=False))


if __name__ == "__main__":
    main()
