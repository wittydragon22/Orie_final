"""Parallel per-file aggregation of taxi trip parquets via multiprocessing."""

from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

from taxi_demand.cleaning import clean_yellow_taxi_trips
from taxi_demand.data import load_trip_data

OUTPUT_COLUMNS = ("PULocationID", "hour_bucket", "pickup_count")


def aggregate_one_file(path: str | Path) -> pd.DataFrame:
    """Read one parquet, clean it, and return per-zone hourly pickup counts."""
    df = load_trip_data(path)
    clean = clean_yellow_taxi_trips(df)
    if clean.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    clean = clean.copy()
    clean["hour_bucket"] = pd.to_datetime(clean["tpep_pickup_datetime"]).dt.floor("h")
    counts = (
        clean.groupby(["PULocationID", "hour_bucket"], as_index=False)
        .size()
        .rename(columns={"size": "pickup_count"})
    )
    return counts[list(OUTPUT_COLUMNS)]


def aggregate_files_parallel(paths, workers: int | None = None) -> pd.DataFrame:
    """Aggregate per-file pickup counts in parallel and reduce to merged totals."""
    paths = list(paths)
    if not paths:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    n_files = len(paths)
    cap = min(n_files, cpu_count())
    if workers is None:
        n_workers = cap
    else:
        n_workers = max(1, min(int(workers), cap))

    if n_workers == 1:
        frames = [aggregate_one_file(p) for p in paths]
    else:
        with Pool(processes=n_workers) as pool:
            frames = pool.map(aggregate_one_file, paths)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    merged = (
        combined.groupby(["PULocationID", "hour_bucket"], as_index=False)["pickup_count"]
        .sum()
    )
    return merged.sort_values(["PULocationID", "hour_bucket"]).reset_index(drop=True)
