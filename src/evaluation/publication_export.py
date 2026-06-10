"""Publication-quality figure and table export for the forecasting paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.utils.io import ensure_dir, save_table


METHOD_LABELS = {
    "XGBoost baseline": "XGBoost",
    "TimeMixer only": "TimeMixer",
    "Hybrid TimeMixer + XGBoost": "Hybrid",
    "Hybrid + Stability Objective (lambda=0.05)": "Hybrid + Stable",
}
METHOD_COLORS = {
    "XGBoost baseline": "#4C78A8",
    "TimeMixer only": "#F58518",
    "Hybrid TimeMixer + XGBoost": "#54A24B",
    "Hybrid + Stability Objective (lambda=0.05)": "#B279A2",
}
METHOD_MARKERS = {
    "XGBoost baseline": "o",
    "TimeMixer only": "s",
    "Hybrid TimeMixer + XGBoost": "^",
    "Hybrid + Stability Objective (lambda=0.05)": "D",
}
METHOD_ORDER = list(METHOD_LABELS)
SHORT_FORECAST_LABELS = {
    "Actual demand": "Actual",
    "XGBoost baseline": "XGBoost",
    "TimeMixer only": "TimeMixer",
    "Hybrid TimeMixer + XGBoost": "Hybrid",
    "Hybrid + Stability Objective": "Hybrid + Stable",
}
FORECAST_COLORS = {
    "Actual demand": "#222222",
    **METHOD_COLORS,
    "Hybrid + Stability Objective": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    """Parse export options."""
    parser = argparse.ArgumentParser(description="Export publication-ready paper figures and tables.")
    parser.add_argument("--results-dir", type=Path, default=Path("paper_results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("paper_figures_final"))
    parser.add_argument("--tables-dir", type=Path, default=Path("paper_tables_final"))
    return parser.parse_args()


def set_conference_style() -> None:
    """Apply a consistent conference-paper plotting style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_current_figure(base_path: Path) -> None:
    """Save the active figure as high-resolution PNG and PDF."""
    plt.tight_layout()
    plt.savefig(base_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#6B7280")


def method_marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


def display_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def add_panel_label(axis, label: str) -> None:
    """Add a bold subfigure label."""
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_table(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def optional_table(results_dir: Path, name: str) -> pd.DataFrame | None:
    """Load a table if the upstream experiment produced it."""
    path = results_dir / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def format_fixed(df: pd.DataFrame, columns: list[str], digits: int = 3) -> pd.DataFrame:
    """Format numeric columns without changing the underlying source files."""
    formatted = df.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.{digits}f}")
    return formatted


def export_tables(results_dir: Path, tables_dir: Path) -> None:
    """Create publication-ready formatted tables."""
    ensure_dir(tables_dir)
    statistical = load_table(results_dir, "statistical_summary_table.csv")
    stability = load_table(results_dir, "stability_improvement_summary.csv")
    main = load_table(results_dir, "large_scale_main_results.csv")
    runtime = load_table(results_dir, "large_scale_runtime_results.csv")
    leakage = load_table(results_dir, "data_split_summary.csv")

    statistical_clean = statistical.rename(
        columns={
            "series_count": "Series",
            "method": "Method",
            "num_runs": "Runs",
            "Forecast Volatility": "Volatility",
        }
    )
    statistical_clean["Method"] = statistical_clean["Method"].map(display_method)
    save_table(statistical_clean, tables_dir / "table_main_mean_std.csv")

    stability_clean = stability[
        [
            "series_count",
            "relative_stability_improvement_pct",
            "relative_rmse_change_pct",
            "relative_volatility_improvement_pct",
            "stability_improved_seed_count",
            "num_runs",
        ]
    ].rename(
        columns={
            "series_count": "Series",
            "relative_stability_improvement_pct": "FSS Improvement (%)",
            "relative_rmse_change_pct": "RMSE Change (%)",
            "relative_volatility_improvement_pct": "Volatility Improvement (%)",
            "stability_improved_seed_count": "Seeds Improved",
            "num_runs": "Runs",
        }
    )
    for column in ["FSS Improvement (%)", "RMSE Change (%)", "Volatility Improvement (%)"]:
        stability_clean[column] = stability_clean[column].map(lambda value: f"{value:.2f}")
    save_table(stability_clean, tables_dir / "table_stability_summary.csv")

    runtime_clean = runtime[runtime["step"].isin(METHOD_ORDER)].copy()
    runtime_clean["Method"] = runtime_clean["step"].map(display_method)
    runtime_clean = runtime_clean[
        ["series_count", "Method", "runtime_seconds_mean", "runtime_seconds_std", "memory_mb_max"]
    ].rename(
        columns={
            "series_count": "Series",
            "runtime_seconds_mean": "Runtime Mean (s)",
            "runtime_seconds_std": "Runtime Std (s)",
            "memory_mb_max": "Peak Memory (MB)",
        }
    )
    for column in ["Runtime Mean (s)", "Runtime Std (s)", "Peak Memory (MB)"]:
        runtime_clean[column] = runtime_clean[column].map(lambda value: f"{value:.2f}")
    save_table(runtime_clean, tables_dir / "table_runtime_scalability.csv")

    main_clean = main.copy()
    main_clean["method"] = main_clean["method"].map(display_method)
    main_clean = main_clean.rename(
        columns={
            "series_count": "Series",
            "method": "Method",
            "rmse_mean": "RMSE",
            "fss_mean": "FSS",
            "forecast_volatility_mean": "Volatility",
            "relative_stability_improvement_pct_mean": "FSS Improvement (%)",
            "relative_rmse_change_pct_mean": "RMSE Change (%)",
        }
    )
    for column in ["RMSE", "FSS", "Volatility"]:
        main_clean[column] = main_clean[column].map(lambda value: f"{value:.3f}")
    for column in ["FSS Improvement (%)", "RMSE Change (%)"]:
        main_clean[column] = main_clean[column].map(lambda value: f"{value:.2f}")
    save_table(
        main_clean[["Series", "Method", "RMSE", "FSS", "Volatility", "FSS Improvement (%)", "RMSE Change (%)"]],
        tables_dir / "table_large_scale_results.csv",
    )

    leakage_clean = leakage.rename(
        columns={
            "train_rows": "Train Rows",
            "validation_rows": "Validation Rows",
            "train_series": "Train Series",
            "validation_series": "Validation Series",
            "train_min_date": "Train Min Date",
            "train_max_date": "Train Max Date",
            "validation_min_date": "Validation Min Date",
            "validation_max_date": "Validation Max Date",
        }
    )
    save_table(leakage_clean, tables_dir / "table_data_split_audit.csv")

    ablation = optional_table(results_dir, "large_scale_ablation_results.csv")
    if ablation is not None:
        ablation_clean = ablation.copy()
        ablation_clean["Method"] = ablation_clean["method"].map(display_method)
        ablation_clean = ablation_clean.rename(
            columns={
                "series_count": "Series",
                "rmse_mean": "RMSE",
                "rmse_std": "RMSE Std",
                "fss_mean": "FSS",
                "fss_std": "FSS Std",
                "forecast_volatility_mean": "Volatility",
                "forecast_volatility_std": "Volatility Std",
                "component_setting": "Component Setting",
            }
        )
        ablation_clean = format_fixed(
            ablation_clean,
            ["RMSE", "RMSE Std", "FSS", "FSS Std", "Volatility", "Volatility Std"],
            digits=3,
        )
        save_table(
            ablation_clean[
                ["Series", "Method", "Component Setting", "RMSE", "RMSE Std", "FSS", "FSS Std", "Volatility"]
            ],
            tables_dir / "table_ablation_study.csv",
        )

    objective = optional_table(results_dir, "stability_tradeoff_table.csv")
    if objective is not None:
        objective_clean = objective.rename(
            columns={
                "series_count": "Series",
                "lambda": "Lambda",
                "forecast_loss_mse": "Forecast Loss",
                "stability_loss": "Stability Loss",
                "total_loss": "Total Loss",
                "rmse": "RMSE",
                "fss": "FSS",
                "forecast_volatility": "Volatility",
            }
        )
        objective_clean = format_fixed(
            objective_clean,
            ["Lambda", "Forecast Loss", "Stability Loss", "Total Loss", "RMSE", "FSS", "Volatility"],
            digits=3,
        )
        save_table(
            objective_clean[["Series", "Lambda", "Forecast Loss", "Stability Loss", "Total Loss", "RMSE", "FSS", "Volatility"]],
            tables_dir / "table_stability_objective_ablation.csv",
        )

    embedding = optional_table(results_dir, "temporal_embedding_analysis.csv")
    if embedding is not None:
        embedding_clean = embedding.rename(
            columns={
                "series_count": "Series",
                "embedding": "Embedding",
                "mean": "Mean",
                "std": "Std",
                "corr_with_demand": "Demand Correlation",
                "best_hybrid_objective_lambda": "Selected Lambda",
            }
        )
        embedding_clean = format_fixed(embedding_clean, ["Mean", "Std", "Demand Correlation", "Selected Lambda"], digits=3)
        save_table(
            embedding_clean[["Series", "Embedding", "Mean", "Std", "Demand Correlation", "Selected Lambda"]],
            tables_dir / "table_temporal_embedding_analysis.csv",
        )

    diagnostics = optional_table(results_dir, "prediction_distribution_diagnostics.csv")
    if diagnostics is not None:
        diagnostics_clean = diagnostics.copy()
        diagnostics_clean["Method"] = diagnostics_clean["method"].map(lambda value: SHORT_FORECAST_LABELS.get(value, display_method(value)))
        diagnostics_clean = diagnostics_clean.rename(
            columns={
                "series_count": "Series",
                "p05": "P05",
                "mean": "Mean",
                "median": "Median",
                "p95": "P95",
                "zero_share": "Zero Share",
            }
        )
        diagnostics_clean = format_fixed(diagnostics_clean, ["min", "P05", "Mean", "Median", "P95", "max", "Zero Share"], digits=3)
        save_table(
            diagnostics_clean[["Series", "Method", "min", "P05", "Mean", "Median", "P95", "max", "Zero Share"]],
            tables_dir / "table_prediction_scale_diagnostics.csv",
        )

    notes = optional_table(results_dir, "large_scale_execution_notes.csv")
    if notes is not None:
        notes_clean = notes.rename(
            columns={
                "series_count": "Series",
                "random_seed": "Seed",
                "num_runs": "Runs",
                "xgboost_n_estimators": "XGBoost Trees",
                "xgboost_max_train_rows": "XGBoost Train Row Cap",
                "note": "Execution Note",
            }
        )
        save_table(notes_clean, tables_dir / "table_execution_notes.csv")

    write_markdown_tables(tables_dir)


def write_markdown_tables(tables_dir: Path) -> None:
    """Export Markdown versions for quick paper drafting."""
    for csv_path in sorted(tables_dir.glob("table_*.csv")):
        df = pd.read_csv(csv_path)
        md_path = csv_path.with_suffix(".md")
        md_path.write_text(df.to_markdown(index=False), encoding="utf-8")


def plot_main_results(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Plot RMSE and FSS mean/std comparison."""
    ordered = summary.copy()
    ordered["method"] = pd.Categorical(ordered["method"], categories=METHOD_ORDER, ordered=True)
    ordered = ordered.sort_values(["series_count", "method"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=False)
    for axis, metric, ylabel, label in [
        (axes[0], "rmse", "RMSE", "(a)"),
        (axes[1], "fss", "Forecast Stability Score", "(b)"),
    ]:
        for method in METHOD_ORDER:
            group = ordered[ordered["method"] == method]
            axis.errorbar(
                group["series_count"],
                group[f"{metric}_mean"],
                yerr=group[f"{metric}_std"],
                color=method_color(method),
                marker=method_marker(method),
                capsize=3,
                label=display_method(method),
                linewidth=2.0,
            )
        axis.set_xlabel("Number of series")
        axis.set_ylabel(ylabel)
        add_panel_label(axis, label)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    save_current_figure(figures_dir / "main_results_rmse_fss")


def plot_tradeoff(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Plot accuracy-stability tradeoff."""
    fig, axis = plt.subplots(figsize=(4.8, 3.5))
    for method in METHOD_ORDER:
        group = summary[summary["method"] == method].sort_values("series_count")
        axis.plot(
            group["fss_mean"],
            group["rmse_mean"],
            color=method_color(method),
            marker=method_marker(method),
            label=display_method(method),
        )
        for _, row in group.iterrows():
            axis.annotate(str(int(row["series_count"])), (row["fss_mean"], row["rmse_mean"]), fontsize=7)
    axis.set_xlabel("Forecast Stability Score")
    axis.set_ylabel("RMSE")
    axis.set_title("Accuracy-Stability Trade-off")
    axis.legend(frameon=False, loc="best")
    save_current_figure(figures_dir / "accuracy_stability_tradeoff")


def plot_runtime(runtime: pd.DataFrame, figures_dir: Path) -> None:
    """Plot runtime scalability."""
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    runtime = runtime[runtime["step"].isin(METHOD_ORDER)].copy()
    for method in METHOD_ORDER:
        group = runtime[runtime["step"] == method].sort_values("series_count")
        axis.plot(
            group["series_count"],
            group["runtime_seconds_mean"],
            color=method_color(method),
            marker=method_marker(method),
            label=display_method(method),
        )
    axis.set_xlabel("Number of series")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title("Runtime Scalability")
    axis.legend(frameon=False, loc="upper left")
    save_current_figure(figures_dir / "runtime_scalability")


def plot_seed_robustness(per_run: pd.DataFrame, figures_dir: Path) -> None:
    """Plot per-seed robustness for the stability-aware model."""
    stable = per_run[per_run["method"] == "Hybrid + Stability Objective (lambda=0.05)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for axis, metric, ylabel, label in [
        (axes[0], "rmse", "RMSE", "(a)"),
        (axes[1], "fss", "Forecast Stability Score", "(b)"),
    ]:
        for seed, group in stable.groupby("seed"):
            group = group.sort_values("series_count")
            axis.plot(group["series_count"], group[metric], marker="o", linewidth=1.8, label=f"Seed {seed}")
        axis.set_xlabel("Number of series")
        axis.set_ylabel(ylabel)
        add_panel_label(axis, label)
    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")
    save_current_figure(figures_dir / "seed_robustness")


def plot_stability_gain(stability: pd.DataFrame, figures_dir: Path) -> None:
    """Plot stability and RMSE change for the final model."""
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(stability))
    width = 0.36
    axis.bar(
        x - width / 2,
        stability["relative_stability_improvement_pct"],
        width=width,
        color="#54A24B",
        label="FSS improvement",
    )
    axis.bar(
        x + width / 2,
        stability["relative_rmse_change_pct"],
        width=width,
        color="#E45756",
        label="RMSE change",
    )
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(stability["series_count"].astype(str))
    axis.set_xlabel("Number of series")
    axis.set_ylabel("Relative change (%)")
    axis.set_title("Stability Gains vs. Accuracy Change")
    axis.legend(frameon=False, loc="upper right")
    save_current_figure(figures_dir / "stability_gain_summary")


def plot_ablation(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Plot ablation metrics from the large-scale experiment summary."""
    ablation = summary.copy()
    ablation["method"] = pd.Categorical(ablation["method"], categories=METHOD_ORDER, ordered=True)
    ablation = ablation.sort_values(["series_count", "method"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for axis, metric, ylabel, label in [
        (axes[0], "rmse_mean", "RMSE", "(a)"),
        (axes[1], "fss_mean", "Forecast Stability Score", "(b)"),
    ]:
        for method in METHOD_ORDER:
            group = ablation[ablation["method"] == method]
            axis.plot(
                group["series_count"],
                group[metric],
                color=method_color(method),
                marker=method_marker(method),
                label=display_method(method),
            )
        axis.set_xlabel("Number of series")
        axis.set_ylabel(ylabel)
        add_panel_label(axis, label)
    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")
    save_current_figure(figures_dir / "ablation_rmse_fss")


def plot_stability_objective_losses(objective: pd.DataFrame, figures_dir: Path) -> None:
    """Plot stability-objective loss components across lambda values."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for series_count, group in objective.groupby("series_count"):
        group = group.sort_values("lambda")
        axes[0].plot(group["lambda"], group["forecast_loss_mse"], marker="o", linewidth=1.8, label=f"{series_count} series")
        axes[1].plot(group["lambda"], group["stability_loss"], marker="o", linewidth=1.8, label=f"{series_count} series")
    axes[0].set_xlabel("Stability weight lambda")
    axes[0].set_ylabel("Forecast loss (MSE)")
    axes[1].set_xlabel("Stability weight lambda")
    axes[1].set_ylabel("Stability loss")
    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")
    save_current_figure(figures_dir / "stability_objective_loss_components")


def plot_stability_objective_tradeoff(objective: pd.DataFrame, figures_dir: Path) -> None:
    """Plot RMSE and FSS sensitivity to the stability-objective weight."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for series_count, group in objective.groupby("series_count"):
        group = group.sort_values("lambda")
        axes[0].plot(group["lambda"], group["rmse"], marker="o", linewidth=1.8, label=f"{series_count} series")
        axes[1].plot(group["lambda"], group["fss"], marker="o", linewidth=1.8, label=f"{series_count} series")
    axes[0].set_xlabel("Stability weight lambda")
    axes[0].set_ylabel("RMSE")
    axes[1].set_xlabel("Stability weight lambda")
    axes[1].set_ylabel("Forecast Stability Score")
    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")
    save_current_figure(figures_dir / "stability_objective_lambda_tradeoff")


def plot_runtime_breakdown(runtime: pd.DataFrame, figures_dir: Path) -> None:
    """Plot detailed runtime components from the earlier experiment stage."""
    ordered_steps = [
        "load_and_merge",
        "prepare_features",
        "xgboost_baseline",
        "smoothing_only",
        "ablation_study",
        "stability_objective",
        "hybrid_framework",
    ]
    step_labels = {
        "load_and_merge": "Load/Merge",
        "prepare_features": "Features",
        "xgboost_baseline": "XGBoost",
        "smoothing_only": "Smoothing",
        "ablation_study": "Ablation",
        "stability_objective": "Stable Obj.",
        "hybrid_framework": "Hybrid",
    }
    fig, axis = plt.subplots(figsize=(6.4, 3.5))
    runtime = runtime[runtime["step"].isin(ordered_steps)].copy()
    runtime["step"] = pd.Categorical(runtime["step"], categories=ordered_steps, ordered=True)
    runtime = runtime.sort_values(["series_count", "step"])
    x = np.arange(runtime["series_count"].nunique())
    width = 0.11
    series_values = sorted(runtime["series_count"].unique())
    for index, step in enumerate(ordered_steps):
        group = runtime[runtime["step"] == step].set_index("series_count").reindex(series_values)
        axis.bar(x + (index - 3) * width, group["runtime_seconds"], width=width, label=step_labels[step])
    axis.set_xticks(x)
    axis.set_xticklabels([str(value) for value in series_values])
    axis.set_xlabel("Number of series")
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title("Experiment Runtime Breakdown")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    save_current_figure(figures_dir / "runtime_breakdown")


def _prediction_panel_columns(predictions: pd.DataFrame) -> list[str]:
    columns = ["demand", "XGBoost baseline", "TimeMixer only", "Hybrid TimeMixer + XGBoost", "Hybrid + Stability Objective"]
    return [column for column in columns if column in predictions.columns]


def plot_hybrid_forecast(predictions: pd.DataFrame, figures_dir: Path) -> None:
    """Plot aggregate validation-horizon forecasts."""
    selected = predictions[predictions["series_count"] == predictions["series_count"].max()].copy()
    columns = _prediction_panel_columns(selected)
    grouped = selected.groupby("date", as_index=False)[columns].mean()
    grouped["date"] = pd.to_datetime(grouped["date"])
    fig, axis = plt.subplots(figsize=(7.2, 3.5))
    label_map = {"demand": "Actual demand"}
    for column in columns:
        paper_name = label_map.get(column, column)
        axis.plot(
            grouped["date"],
            grouped[column],
            color=FORECAST_COLORS.get(paper_name, "#6B7280"),
            linewidth=2.1 if column == "demand" else 1.8,
            label=SHORT_FORECAST_LABELS.get(paper_name, display_method(paper_name)),
        )
    axis.set_xlabel("Validation date")
    axis.set_ylabel("Mean demand")
    axis.set_title(f"Aggregate Forecast Visualization ({int(selected['series_count'].max())} series)")
    axis.legend(frameon=False, ncol=3, loc="upper left")
    fig.autofmt_xdate(rotation=25)
    save_current_figure(figures_dir / "hybrid_forecast_visualization")


def plot_prediction_distributions(predictions: pd.DataFrame, figures_dir: Path) -> None:
    """Plot prediction distribution histograms using validation predictions."""
    selected = predictions[predictions["series_count"] == predictions["series_count"].max()].copy()
    columns = _prediction_panel_columns(selected)
    label_map = {"demand": "Actual demand"}
    values = [selected[column].dropna().to_numpy() for column in columns]
    upper = np.nanpercentile(np.concatenate(values), 99.5)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    axes = axes.ravel()
    for axis, column, array in zip(axes, columns, values):
        paper_name = label_map.get(column, column)
        clipped = np.clip(array, 0, upper)
        axis.hist(clipped, bins=36, density=True, color=FORECAST_COLORS.get(paper_name, "#6B7280"), alpha=0.82)
        axis.set_title(SHORT_FORECAST_LABELS.get(paper_name, display_method(paper_name)))
        axis.set_xlabel("Demand")
        axis.set_ylabel("Density")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for axis in axes[len(columns):]:
        axis.axis("off")
    save_current_figure(figures_dir / "prediction_distribution_histograms")


def plot_prediction_scale(diagnostics: pd.DataFrame, figures_dir: Path) -> None:
    """Plot compact prediction scale diagnostics from quantile summaries."""
    selected = diagnostics[diagnostics["series_count"] == diagnostics["series_count"].max()].copy()
    selected["display"] = selected["method"].map(lambda value: SHORT_FORECAST_LABELS.get(value, display_method(value)))
    fig, axis = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(selected))
    axis.vlines(x, selected["p05"], selected["p95"], color="#6B7280", linewidth=4, alpha=0.45, label="P05-P95")
    axis.scatter(x, selected["median"], color="#222222", s=24, zorder=3, label="Median")
    axis.scatter(x, selected["mean"], color="#E45756", s=24, zorder=3, label="Mean")
    axis.set_xticks(x)
    axis.set_xticklabels(selected["display"], rotation=20, ha="right")
    axis.set_ylabel("Demand scale")
    axis.set_title("Prediction Scale Comparison")
    axis.legend(frameon=False, loc="upper left")
    save_current_figure(figures_dir / "prediction_scale_comparison")


def plot_temporal_embeddings(embedding: pd.DataFrame, figures_dir: Path) -> None:
    """Plot strongest temporal embedding correlations with demand."""
    selected = embedding[embedding["series_count"] == embedding["series_count"].max()].copy()
    selected = selected.reindex(selected["corr_with_demand"].abs().sort_values(ascending=False).index).head(8)
    fig, axis = plt.subplots(figsize=(5.4, 3.4))
    colors = np.where(selected["corr_with_demand"] >= 0, "#4C78A8", "#E45756")
    axis.barh(selected["embedding"], selected["corr_with_demand"], color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Correlation with demand")
    axis.set_ylabel("Temporal embedding")
    axis.set_title("Temporal Embedding Signal")
    axis.invert_yaxis()
    save_current_figure(figures_dir / "temporal_embedding_analysis")


def plot_leakage_audit(checks: pd.DataFrame, figures_dir: Path) -> None:
    """Plot compact leakage audit status summary."""
    counts = checks["status"].value_counts().reindex(["PASS", "FAIL"], fill_value=0)
    fig, axis = plt.subplots(figsize=(3.6, 2.8))
    axis.bar(counts.index, counts.values, color=["#54A24B", "#E45756"], width=0.55)
    for index, value in enumerate(counts.values):
        axis.text(index, value + 0.5, str(int(value)), ha="center", va="bottom", fontsize=9)
    axis.set_ylabel("Number of checks")
    axis.set_title("Data Leakage Audit")
    save_current_figure(figures_dir / "data_leakage_audit_summary")


def write_captions(figures_dir: Path) -> None:
    """Write concise captions for final figures."""
    captions = {
        "main_results_rmse_fss": "Main large-scale results. Error bars denote standard deviation across three random seeds.",
        "accuracy_stability_tradeoff": "Accuracy-stability trade-off across model variants and evaluated scales. Point annotations indicate the number of series.",
        "runtime_scalability": "Runtime scalability by model family. Hybrid variants include temporal representation learning overhead.",
        "seed_robustness": "Seed robustness of the stability-aware hybrid model across scales.",
        "stability_gain_summary": "Relative stability improvement and RMSE change of the stability-aware hybrid model versus XGBoost.",
        "data_leakage_audit_summary": "Summary of automated data leakage audit checks.",
        "ablation_rmse_fss": "Ablation study comparing predictive accuracy and forecast stability across model components.",
        "stability_objective_loss_components": "Forecast and stability loss components across stability-objective weights.",
        "stability_objective_lambda_tradeoff": "Sensitivity of RMSE and Forecast Stability Score to the stability-objective weight.",
        "runtime_breakdown": "Detailed runtime by pipeline stage for earlier-scale experiments.",
        "hybrid_forecast_visualization": "Aggregate validation-horizon forecast trajectories for the hybrid forecasting variants.",
        "prediction_distribution_histograms": "Validation prediction distributions for actual demand and forecast variants.",
        "prediction_scale_comparison": "Prediction scale diagnostics based on median, mean, and central quantile range.",
        "temporal_embedding_analysis": "Temporal embedding dimensions with the strongest demand correlations.",
    }
    lines = []
    for name, caption in captions.items():
        lines.append(f"{name}: {caption}")
    (figures_dir / "figure_captions.txt").write_text("\n".join(lines), encoding="utf-8")


def export_figures(results_dir: Path, figures_dir: Path) -> None:
    """Create all final figures."""
    ensure_dir(figures_dir)
    summary = pd.read_csv(results_dir / "mean_std_results.csv")
    runtime = pd.read_csv(results_dir / "large_scale_runtime_results.csv")
    per_run = pd.read_csv(results_dir / "large_scale_per_run_results.csv")
    stability = pd.read_csv(results_dir / "stability_improvement_summary.csv")
    checks = pd.read_csv(results_dir / "feature_leakage_checks.csv")

    plot_main_results(summary, figures_dir)
    plot_tradeoff(summary, figures_dir)
    plot_runtime(runtime, figures_dir)
    plot_seed_robustness(per_run, figures_dir)
    plot_stability_gain(stability, figures_dir)
    plot_leakage_audit(checks, figures_dir)

    ablation = optional_table(results_dir, "large_scale_ablation_results.csv")
    if ablation is not None:
        plot_ablation(ablation, figures_dir)
    objective = optional_table(results_dir, "stability_tradeoff_table.csv")
    if objective is not None:
        plot_stability_objective_losses(objective, figures_dir)
        plot_stability_objective_tradeoff(objective, figures_dir)
    runtime_breakdown = optional_table(results_dir, "runtime_comparison.csv")
    if runtime_breakdown is not None:
        plot_runtime_breakdown(runtime_breakdown, figures_dir)
    predictions = optional_table(results_dir, "hybrid_predictions.csv")
    if predictions is not None:
        plot_hybrid_forecast(predictions, figures_dir)
        plot_prediction_distributions(predictions, figures_dir)
    diagnostics = optional_table(results_dir, "prediction_distribution_diagnostics.csv")
    if diagnostics is not None:
        plot_prediction_scale(diagnostics, figures_dir)
    embedding = optional_table(results_dir, "temporal_embedding_analysis.csv")
    if embedding is not None:
        plot_temporal_embeddings(embedding, figures_dir)
    write_captions(figures_dir)


def main() -> None:
    """Run final publication export."""
    args = parse_args()
    set_conference_style()
    ensure_dir(args.figures_dir)
    ensure_dir(args.tables_dir)
    export_tables(args.results_dir, args.tables_dir)
    export_figures(args.results_dir, args.figures_dir)
    print(f"Final figures written to {args.figures_dir}")
    print(f"Final tables written to {args.tables_dir}")


if __name__ == "__main__":
    main()
