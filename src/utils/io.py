"""Small input/output helpers for research artifacts."""

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a table to CSV, creating parent folders as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path

