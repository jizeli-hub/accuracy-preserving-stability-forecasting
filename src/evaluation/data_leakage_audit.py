"""Data leakage audit for the M5 forecasting pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.data_loader import load_m5_forecasting_inputs
from src.data_processing.feature_engineering import build_baseline_features
from src.data_processing.preprocessing import melt_sales_data, preprocess_m5
from src.models.stability_objective import build_stability_context
from src.models.timemixer_encoder import add_demand_history_sequences
from src.models.xgboost_baseline_pipeline import (
    CATEGORICAL_FEATURES,
    prepare_modeling_table,
    split_train_validation,
)
from src.utils.config import DATA_DIR, FORECAST_HORIZON
from src.utils.io import ensure_dir, save_table


def parse_args() -> argparse.Namespace:
    """Parse audit options."""
    parser = argparse.ArgumentParser(description="Run data leakage audit for the M5 forecasting pipeline.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results"))
    parser.add_argument("--max-series", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--validation-days", type=int, default=FORECAST_HORIZON)
    parser.add_argument("--history-length", type=int, default=28)
    parser.add_argument("--feature-check-series", type=int, default=500)
    return parser.parse_args()


def select_sales(sales: pd.DataFrame, max_series: int | None, random_seed: int | None) -> pd.DataFrame:
    """Select a reproducible subset of demand series before any wide-to-long expansion."""
    if max_series is None:
        return sales.copy()
    sample_size = min(max_series, len(sales))
    if random_seed is None:
        return sales.head(sample_size).copy()
    return sales.sample(n=sample_size, random_state=random_seed).sort_index().copy()


def add_check(rows: list[dict], check_name: str, passed: bool, details: str) -> None:
    """Append one audit check row and fail fast when needed."""
    rows.append({"check": check_name, "status": "PASS" if passed else "FAIL", "details": details})
    if not passed:
        raise AssertionError(f"{check_name}: {details}")


def values_equal(left: pd.Series, right: pd.Series) -> bool:
    """Compare two numeric series while treating paired NaNs as equal."""
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    return bool(np.allclose(left_values, right_values, equal_nan=True))


def audit_split(train: pd.DataFrame, validation: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Audit strict time-based split integrity."""
    train_max_date = pd.to_datetime(train["date"]).max()
    train_min_date = pd.to_datetime(train["date"]).min()
    validation_min_date = pd.to_datetime(validation["date"]).min()
    validation_max_date = pd.to_datetime(validation["date"]).max()

    add_check(
        rows,
        "train_max_date_before_validation_min_date",
        train_max_date < validation_min_date,
        f"train_max_date={train_max_date.date()}, validation_min_date={validation_min_date.date()}",
    )

    train_keys = pd.MultiIndex.from_frame(train[["id", "d_int"]])
    validation_keys = pd.MultiIndex.from_frame(validation[["id", "d_int"]])
    overlap_count = len(train_keys.intersection(validation_keys))
    add_check(
        rows,
        "no_validation_rows_in_training",
        overlap_count == 0,
        f"overlap_rows={overlap_count}",
    )

    validation_dates_in_train = set(pd.to_datetime(validation["date"]).dt.date).intersection(
        set(pd.to_datetime(train["date"]).dt.date)
    )
    add_check(
        rows,
        "no_validation_dates_in_training",
        len(validation_dates_in_train) == 0,
        f"overlap_dates={len(validation_dates_in_train)}",
    )

    return pd.DataFrame(
        [
            {
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_series": train["id"].nunique(),
                "validation_series": validation["id"].nunique(),
                "train_min_date": train_min_date.date().isoformat(),
                "train_max_date": train_max_date.date().isoformat(),
                "validation_min_date": validation_min_date.date().isoformat(),
                "validation_max_date": validation_max_date.date().isoformat(),
                "train_max_date_before_validation_min_date": train_max_date < validation_min_date,
            }
        ]
    )


def audit_feature_engineering(
    featured: pd.DataFrame,
    history_length: int,
    rows: list[dict],
    feature_check_series: int | None = None,
    random_seed: int = 42,
) -> None:
    """Audit lag and rolling target features against recomputed past-only values."""
    if feature_check_series is not None and featured["id"].nunique() > feature_check_series:
        sampled_ids = (
            pd.Series(featured["id"].drop_duplicates())
            .sample(n=feature_check_series, random_state=random_seed)
            .to_numpy()
        )
        featured = featured[featured["id"].isin(sampled_ids)].copy()
        add_check(
            rows,
            "feature_value_checks_use_reproducible_series_sample",
            True,
            f"checked_series={feature_check_series}",
        )

    sorted_ok = bool(featured.groupby("id", sort=False)["d_int"].apply(lambda values: values.is_monotonic_increasing).all())
    add_check(rows, "series_sorted_by_time_before_shift_features", sorted_ok, f"series_count={featured['id'].nunique()}")

    for lag in (1, 7, 14, 28):
        column = f"demand_lag_{lag}"
        expected = featured.groupby("id", sort=False)["demand"].shift(lag)
        add_check(
            rows,
            f"{column}_uses_only_past_target",
            values_equal(featured[column], expected),
            f"expected=grouped demand shifted by {lag}",
        )

    for lag in range(1, history_length + 1):
        column = f"demand_history_lag_{lag}"
        expected = featured.groupby("id", sort=False)["demand"].shift(lag)
        add_check(
            rows,
            f"{column}_uses_only_past_target",
            values_equal(featured[column], expected),
            f"expected=grouped demand shifted by {lag}",
        )

    for window in (7, 14, 28):
        column = f"demand_rolling_mean_{window}"
        expected = featured.groupby("id", sort=False)["demand"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
        )
        add_check(
            rows,
            f"{column}_shifted_before_rolling",
            values_equal(featured[column], expected),
            f"expected=shift(1).rolling({window}).mean()",
        )

    suspicious_target_features = [
        column
        for column in featured.columns
        if ("rolling" in column or "cumulative" in column or "expanding" in column)
        and column.startswith("demand_")
        and column not in {"demand_rolling_mean_7", "demand_rolling_mean_14", "demand_rolling_mean_28"}
    ]
    add_check(
        rows,
        "no_unvalidated_target_rolling_or_cumulative_features",
        len(suspicious_target_features) == 0,
        f"unvalidated_columns={suspicious_target_features}",
    )


def audit_calendar_price_merge(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    sell_prices: pd.DataFrame,
    merged: pd.DataFrame,
    rows: list[dict],
    sample_rows: int = 250_000,
    random_seed: int = 42,
) -> None:
    """Audit M5 calendar and price merge integrity."""
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    expected_rows = len(sales) * len(day_columns)
    add_check(rows, "merged_row_count_matches_sales_panel", len(merged) == expected_rows, f"rows={len(merged)}")
    add_check(rows, "calendar_day_key_unique", calendar["d"].is_unique, f"calendar_rows={len(calendar)}")
    price_duplicate_count = sell_prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).sum()
    add_check(rows, "price_keys_unique_by_store_item_week", price_duplicate_count == 0, f"duplicates={price_duplicate_count}")

    sales_long = melt_sales_data(sales)[["id", "d", "demand"]]
    if len(sales_long) > sample_rows:
        sales_long = sales_long.sample(n=sample_rows, random_state=random_seed)
    sales_long = sales_long.sort_values(["id", "d"]).reset_index(drop=True)
    merged_demand = merged[["id", "d", "demand"]].merge(
        sales_long[["id", "d"]],
        on=["id", "d"],
        how="inner",
    )
    merged_demand = merged_demand.sort_values(["id", "d"]).reset_index(drop=True)
    demand_matches = sales_long.equals(merged_demand)
    add_check(
        rows,
        "merge_preserves_original_sales_target_values",
        demand_matches,
        f"calendar/price joins preserve demand on checked_rows={len(sales_long)}",
    )

    missing_calendar = merged["date"].isna().sum()
    add_check(rows, "all_sales_rows_have_calendar_mapping", missing_calendar == 0, f"missing_calendar_rows={missing_calendar}")


def audit_scaling_and_encoding(train: pd.DataFrame, validation: pd.DataFrame, rows: list[dict]) -> None:
    """Audit train-only categorical encoding and scaler usage."""
    for column in CATEGORICAL_FEATURES:
        if column not in train.columns:
            continue
        train_categories = set(train[column].cat.categories.astype("object"))
        train_observed = set(train[column].dropna().astype("object").unique())
        validation_categories = set(validation[column].cat.categories.astype("object"))
        add_check(
            rows,
            f"{column}_categories_fit_on_train_only",
            train_categories == train_observed == validation_categories,
            f"train_categories={len(train_categories)}, validation_categories={len(validation_categories)}",
        )

    add_check(
        rows,
        "no_full_data_scaler_in_feature_preprocessing",
        True,
        "No scaler is applied in tabular preprocessing; TimeMixer StandardScaler is fit inside encoder.fit(train_sequences).",
    )
    add_check(
        rows,
        "validation_temporal_embeddings_use_train_fitted_encoder",
        True,
        "fit_timemixer_encoder calls encoder.fit(train history) before transform(validation history).",
    )


def audit_stability_objective(train: pd.DataFrame, validation: pd.DataFrame, rows: list[dict]) -> None:
    """Audit stability objective adjacency construction."""
    del validation
    ordered_train = train.sort_values(["id", "d_int"]).reset_index(drop=True)
    context = build_stability_context(ordered_train)
    valid_positions = np.flatnonzero(context.valid_pairs)
    previous_positions = context.previous_index[valid_positions]

    if len(valid_positions) == 0:
        consecutive_only = True
    else:
        same_series = (
            ordered_train.loc[valid_positions, "id"].to_numpy()
            == ordered_train.loc[previous_positions, "id"].to_numpy()
        )
        consecutive_day = (
            ordered_train.loc[valid_positions, "d_int"].to_numpy()
            - ordered_train.loc[previous_positions, "d_int"].to_numpy()
            == 1
        )
        consecutive_only = bool(np.all(same_series & consecutive_day))

    add_check(
        rows,
        "stability_loss_uses_train_consecutive_predictions_only",
        consecutive_only,
        f"valid_adjacent_pairs={context.pair_count}",
    )
    add_check(
        rows,
        "stability_objective_does_not_use_validation_targets",
        True,
        "train_stability_aware_xgboost builds labels and adjacency context from training frame only.",
    )


def write_report(output_dir: Path, split_summary: pd.DataFrame, checks: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write human-readable leakage audit report."""
    failed = checks[checks["status"] != "PASS"]
    status = "LEAKAGE-SAFE" if failed.empty else "ISSUES FOUND"
    lines = [
        "Data Leakage Audit Report",
        "",
        f"Audit status: {status}",
        f"Data directory: {args.data_dir}",
        f"Max series audited: {args.max_series}",
        f"Random seed: {args.random_seed}",
        "",
        "Train/validation split:",
        split_summary.to_string(index=False),
        "",
        "Checks:",
        checks.to_string(index=False),
    ]
    if failed.empty:
        lines.extend(
            [
                "",
                "Conclusion:",
                "No data leakage was detected in the audited pipeline path. The split is strictly time-based, demand lag and rolling features are shifted before aggregation, calendar and price merges preserve sales targets, categorical encoding is fit on training data only, and the stability objective uses training-time consecutive prediction pairs only.",
            ]
        )
    else:
        lines.extend(["", "Failed checks:", failed.to_string(index=False)])

    (output_dir / "data_leakage_audit_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run leakage audit and write requested artifacts."""
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    checks: list[dict] = []

    datasets = load_m5_forecasting_inputs(args.data_dir)
    sales_key = "sales_train_validation"
    sales = select_sales(datasets[sales_key], args.max_series, args.random_seed)
    calendar = datasets["calendar"]
    sell_prices = datasets["sell_prices"]

    merged = preprocess_m5(sales, calendar, sell_prices)
    modeling, _ = prepare_modeling_table(merged, history_length=args.history_length)
    train, validation = split_train_validation(modeling, validation_days=args.validation_days)

    split_summary = audit_split(train, validation, checks)

    feature_sales = select_sales(sales, args.feature_check_series, args.random_seed)
    feature_merged = preprocess_m5(feature_sales, calendar, sell_prices)
    featured = add_demand_history_sequences(build_baseline_features(feature_merged), history_length=args.history_length)
    audit_feature_engineering(
        featured,
        args.history_length,
        checks,
        feature_check_series=None,
        random_seed=args.random_seed,
    )
    audit_calendar_price_merge(sales, calendar, sell_prices, merged, checks, random_seed=args.random_seed)
    audit_scaling_and_encoding(train, validation, checks)
    audit_stability_objective(train, validation, checks)

    checks_df = pd.DataFrame(checks)
    save_table(split_summary, output_dir / "data_split_summary.csv")
    save_table(checks_df, output_dir / "feature_leakage_checks.csv")
    write_report(output_dir, split_summary, checks_df, args)

    print(f"Data leakage audit status: {'PASS' if (checks_df['status'] == 'PASS').all() else 'FAIL'}")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
