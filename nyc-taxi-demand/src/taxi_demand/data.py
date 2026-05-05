"""Load and save trip or model datasets."""

from pathlib import Path

import pandas as pd


def load_trip_data(path: str | Path) -> pd.DataFrame:
    """Load raw trip-level data from Parquet or CSV."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {suffix}. Use .parquet or .csv.")


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to Parquet or CSV based on file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .parquet or .csv.")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a processed model dataset from Parquet or CSV."""
    return load_trip_data(path)
