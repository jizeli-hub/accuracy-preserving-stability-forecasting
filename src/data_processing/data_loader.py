"""Utilities for loading the M5 Forecasting dataset files."""

from pathlib import Path

import pandas as pd


M5_FILES = {
    "calendar": "calendar.csv",
    "sales_train_validation": "sales_train_validation.csv",
    "sales_train_evaluation": "sales_train_evaluation.csv",
    "sell_prices": "sell_prices.csv",
    "sample_submission": "sample_submission.csv",
}


def load_csv(data_dir: str | Path, filename: str, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV file from the project data directory."""
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")
    return pd.read_csv(path, **read_csv_kwargs)


def load_m5_data(data_dir: str | Path, use_evaluation: bool = False) -> dict[str, pd.DataFrame]:
    """Load the core M5 files into a dictionary of data frames."""
    sales_key = "sales_train_evaluation" if use_evaluation else "sales_train_validation"
    file_keys = ["calendar", sales_key, "sell_prices", "sample_submission"]

    return {
        key: load_csv(data_dir, M5_FILES[key])
        for key in file_keys
    }


def load_m5_forecasting_inputs(
    data_dir: str | Path,
    use_evaluation: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load only the sales, calendar, and sell price files needed for modeling."""
    sales_key = "sales_train_evaluation" if use_evaluation else "sales_train_validation"
    file_keys = ["calendar", sales_key, "sell_prices"]

    return {
        key: load_csv(data_dir, M5_FILES[key])
        for key in file_keys
    }
