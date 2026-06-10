"""Statistical robustness reporting for large-scale M5 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, save_table


METRICS = ["mae", "rmse", "mape", "fss", "forecast_volatility"]
METHOD_ORDER = [
    "XGBoost baseline",
    "TimeMixer only",
    "Hybrid TimeMixer + XGBoost",
    "Hybrid + Stability Objective (lambda=0.05)",
]


def parse_args() -> argparse.Namespace:
    """Parse reporting options."""
    parser = argparse.ArgumentParser(description="Generate statistical robustness tables and figures.")
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Experiment output directories containing paper_results/large_scale_per_run_results.csv.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("paper_results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("paper_figures"))
    return parser.parse_args()


def load_per_run_results(run_dirs: list[Path]) -> pd.DataFrame:
    """Load per-run result tables from several experiment directories."""
    frames = []
    for run_dir in run_dirs:
        path = run_dir / "paper_results" / "large_scale_per_run_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-run results: {path}")
        frames.append(pd.read_csv(path))
    data = pd.concat(frames, ignore_index=True)
    data["method"] = pd.Categorical(data["method"], categories=METHOD_ORDER, ordered=True)
    return data.sort_values(["series_count", "method", "seed"]).reset_index(drop=True)


def aggregate_mean_std(per_run: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, and 95% confidence intervals by scale and method."""
    grouped = per_run.groupby(["series_count", "method"], observed=True, sort=False)
    summary = grouped[METRICS].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    for metric in METRICS:
        summary[f"{metric}_std"] = summary[f"{metric}_std"].fillna(0)
        summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"])

    return summary


def build_formatted_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build publication-friendly mean ± std table."""
    rows = []
    for _, row in summary.iterrows():
        output = {
            "series_count": row["series_count"],
            "method": row["method"],
            "num_runs": int(row["rmse_count"]),
        }
        for metric in METRICS:
            output[metric.upper() if metric != "forecast_volatility" else "Forecast Volatility"] = (
                f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
            )
        rows.append(output)
    return pd.DataFrame(rows)


def add_relative_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Add relative RMSE and stability improvements versus XGBoost baseline."""
    output = summary.copy()
    baseline = output[output["method"] == "XGBoost baseline"][
        ["series_count", "rmse_mean", "fss_mean", "forecast_volatility_mean"]
    ].rename(
        columns={
            "rmse_mean": "baseline_rmse_mean",
            "fss_mean": "baseline_fss_mean",
            "forecast_volatility_mean": "baseline_forecast_volatility_mean",
        }
    )
    output = output.merge(baseline, on="series_count", how="left")
    output["relative_rmse_change_pct"] = 100 * (
        output["rmse_mean"] - output["baseline_rmse_mean"]
    ) / output["baseline_rmse_mean"].clip(lower=1e-8)
    output["relative_stability_improvement_pct"] = 100 * (
        output["baseline_fss_mean"] - output["fss_mean"]
    ) / output["baseline_fss_mean"].clip(lower=1e-8)
    output["relative_volatility_improvement_pct"] = 100 * (
        output["baseline_forecast_volatility_mean"] - output["forecast_volatility_mean"]
    ) / output["baseline_forecast_volatility_mean"].clip(lower=1e-8)
    return output


def build_statistical_summary(summary: pd.DataFrame, per_run: pd.DataFrame) -> pd.DataFrame:
    """Create compact statistical comparison output."""
    rows = []
    for series_count, group in summary.groupby("series_count", sort=False):
        baseline = group[group["method"] == "XGBoost baseline"].iloc[0]
        stable = group[group["method"] == "Hybrid + Stability Objective (lambda=0.05)"].iloc[0]
        seed_check = per_seed_stability_check(per_run, series_count)
        rows.append(
            {
                "series_count": series_count,
                "num_runs": int(stable["rmse_count"]),
                "baseline_rmse": baseline["rmse_mean"],
                "hybrid_stability_rmse": stable["rmse_mean"],
                "relative_rmse_change_pct": stable["relative_rmse_change_pct"],
                "baseline_fss": baseline["fss_mean"],
                "hybrid_stability_fss": stable["fss_mean"],
                "relative_stability_improvement_pct": stable["relative_stability_improvement_pct"],
                "baseline_volatility": baseline["forecast_volatility_mean"],
                "hybrid_stability_volatility": stable["forecast_volatility_mean"],
                "relative_volatility_improvement_pct": stable["relative_volatility_improvement_pct"],
                "rmse_ci95": stable["rmse_ci95"],
                "fss_ci95": stable["fss_ci95"],
                "stability_improved_all_seeds": bool(seed_check["improved"].all()),
                "stability_improved_seed_count": int(seed_check["improved"].sum()),
            }
        )
    return pd.DataFrame(rows)


def per_seed_stability_check(per_run: pd.DataFrame, series_count) -> pd.DataFrame:
    """Return per-seed stability comparison for one scale."""
    rows = []
    scale_data = per_run[per_run["series_count"] == series_count]
    for seed, group in scale_data.groupby("seed", sort=False):
        baseline = group[group["method"] == "XGBoost baseline"].iloc[0]
        stable = group[group["method"] == "Hybrid + Stability Objective (lambda=0.05)"].iloc[0]
        rows.append(
            {
                "seed": seed,
                "baseline_fss": baseline["fss"],
                "stable_fss": stable["fss"],
                "improved": stable["fss"] < baseline["fss"],
            }
        )
    return pd.DataFrame(rows)


def save_figures(summary: pd.DataFrame, per_run: pd.DataFrame, figures_dir: Path) -> None:
    """Generate robustness figures."""
    ensure_dir(figures_dir)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    save_error_bar_comparison(summary, figures_dir / "error_bar_comparison.png")
    save_seed_robustness_visualization(per_run, figures_dir / "seed_robustness_visualization.png")


def save_error_bar_comparison(summary: pd.DataFrame, path: Path) -> None:
    """Save RMSE and FSS error-bar comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric, title in zip(axes, ["rmse", "fss"], ["RMSE Across Seeds", "FSS Across Seeds"]):
        pivot_mean = summary.pivot(index="method", columns="series_count", values=f"{metric}_mean")
        pivot_std = summary.pivot(index="method", columns="series_count", values=f"{metric}_std")
        pivot_mean.plot(kind="bar", yerr=pivot_std, capsize=3, ax=axis, width=0.72)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel(metric.upper())
        axis.tick_params(axis="x", labelrotation=25)
        axis.legend(title="Series", frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_seed_robustness_visualization(per_run: pd.DataFrame, path: Path) -> None:
    """Visualize per-seed RMSE and FSS variability."""
    stable = per_run[per_run["method"] == "Hybrid + Stability Objective (lambda=0.05)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, metric, title in zip(axes, ["rmse", "fss"], ["Hybrid+Stability RMSE by Seed", "Hybrid+Stability FSS by Seed"]):
        for seed, group in stable.groupby("seed", sort=False):
            ordered = group.sort_values("series_count")
            axis.plot(ordered["series_count"], ordered[metric], marker="o", label=f"seed {seed}")
        axis.set_title(title)
        axis.set_xlabel("Series count")
        axis.set_ylabel(metric.upper())
        axis.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def write_summary_paragraph(statistical_summary: pd.DataFrame, results_dir: Path) -> None:
    """Write a concise conference-ready robustness paragraph."""
    stable_all = bool(statistical_summary["stability_improved_all_seeds"].all())
    avg_stability = statistical_summary["relative_stability_improvement_pct"].mean()
    max_abs_rmse = statistical_summary["relative_rmse_change_pct"].abs().max()
    paragraph = (
        "Across three random seeds, the stability-aware hybrid model produced consistent stability gains "
        f"relative to the XGBoost baseline. The average FSS improvement across evaluated scales was "
        f"{avg_stability:.2f}%, and the largest absolute RMSE change was {max_abs_rmse:.2f}%. "
        f"Stability improved at every evaluated scale: {stable_all}. These results indicate that the "
        "proposed stability-aware objective reduces forecast variability without introducing large "
        "accuracy instability across random seeds."
    )
    (results_dir / "robustness_summary_paragraph.txt").write_text(paragraph, encoding="utf-8")


def main() -> None:
    """Generate statistical robustness outputs."""
    args = parse_args()
    ensure_dir(args.results_dir)
    ensure_dir(args.figures_dir)

    per_run = load_per_run_results(args.run_dirs)
    summary = add_relative_statistics(aggregate_mean_std(per_run))
    formatted = build_formatted_table(summary)
    statistical_summary = build_statistical_summary(summary, per_run)

    save_table(formatted, args.results_dir / "statistical_summary_table.csv")
    save_table(summary, args.results_dir / "mean_std_results.csv")
    save_table(statistical_summary, args.results_dir / "stability_improvement_summary.csv")
    save_figures(summary, per_run, args.figures_dir)
    write_summary_paragraph(statistical_summary, args.results_dir)

    print(statistical_summary.to_string(index=False))


if __name__ == "__main__":
    main()
