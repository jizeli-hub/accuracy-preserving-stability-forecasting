"""Preprocessing helpers for M5 demand forecasting experiments."""

import pandas as pd


def melt_sales_data(sales: pd.DataFrame) -> pd.DataFrame:
    """Convert wide daily sales columns into a long item-store-day panel."""
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [column for column in sales.columns if column.startswith("d_")]

    return sales.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="demand",
    )


def attach_calendar(sales_long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Join calendar attributes onto long-format sales data."""
    return sales_long.merge(calendar, on="d", how="left", validate="many_to_one")


def attach_sell_prices(sales_calendar: pd.DataFrame, sell_prices: pd.DataFrame) -> pd.DataFrame:
    """Join sell prices by store, item, and M5 week."""
    return sales_calendar.merge(
        sell_prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )


def preprocess_m5(sales: pd.DataFrame, calendar: pd.DataFrame, sell_prices: pd.DataFrame) -> pd.DataFrame:
    """Create a modeling-ready long-format M5 panel."""
    sales_long = melt_sales_data(sales)
    sales_calendar = attach_calendar(sales_long, calendar)
    output = attach_sell_prices(sales_calendar, sell_prices)
    output["date"] = pd.to_datetime(output["date"])
    output["d_int"] = output["d"].str.replace("d_", "", regex=False).astype(int)
    return output.sort_values(["id", "d_int"]).reset_index(drop=True)
