"""Feature engineering utilities for demand forecasting baselines."""

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "id",
    target_col: str = "demand",
    lags: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """Add grouped lag features for the target variable."""
    output = df.copy()
    grouped = output.groupby(group_col, sort=False)[target_col]

    for lag in lags:
        output[f"{target_col}_lag_{lag}"] = grouped.shift(lag)

    return output


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str = "id",
    target_col: str = "demand",
    windows: tuple[int, ...] = (7, 28),
) -> pd.DataFrame:
    """Add shifted rolling means to avoid target leakage."""
    output = df.copy()

    for window in windows:
        output[f"{target_col}_rolling_mean_{window}"] = (
            output.groupby(group_col, sort=False)[target_col]
            .transform(lambda series: series.shift(1).rolling(window=window, min_periods=1).mean())
        )

    return output


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar-derived features."""
    output = df.copy()
    if "date" in output.columns:
        date = pd.to_datetime(output["date"])
        output["dayofweek"] = date.dt.dayofweek
        output["month"] = date.dt.month
        output["year"] = date.dt.year
        output["day"] = date.dt.day
        output["weekofyear"] = date.dt.isocalendar().week.astype(int)

    event_columns = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for column in event_columns:
        if column in output.columns:
            output[f"has_{column}"] = output[column].notna().astype(int)

    snap_columns = [column for column in ["snap_CA", "snap_TX", "snap_WI"] if column in output.columns]
    if snap_columns:
        output["snap_any"] = output[snap_columns].max(axis=1)

    return output


def add_price_features(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("store_id", "item_id"),
    price_col: str = "sell_price",
) -> pd.DataFrame:
    """Add price movement features for item-store histories."""
    output = df.copy()
    if price_col not in output.columns:
        return output

    grouped = output.groupby(list(group_cols), sort=False)[price_col]
    output["sell_price_lag_1"] = grouped.shift(1)
    output["sell_price_change_1"] = output[price_col] - output["sell_price_lag_1"]
    output["sell_price_pct_change_1"] = grouped.pct_change(fill_method=None).replace(
        [float("inf"), -float("inf")],
        0,
    )
    output["sell_price_rolling_mean_28"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=28, min_periods=1).mean()
    )
    output["sell_price_relative_to_28d_mean"] = output[price_col] / output["sell_price_rolling_mean_28"]
    return output


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a compact feature set suitable for classical baselines."""
    output = add_calendar_features(df)
    output = add_lag_features(output, lags=(1, 7, 14, 28))
    output = add_rolling_features(output, windows=(7, 14, 28))
    output = add_price_features(output)
    return output
