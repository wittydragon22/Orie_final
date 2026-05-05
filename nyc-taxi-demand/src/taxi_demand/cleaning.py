"""Clean NYC Yellow Taxi trip records."""

import pandas as pd

REQUIRED_COLUMNS = (
    "tpep_pickup_datetime",
    "PULocationID",
    "trip_distance",
    "fare_amount",
    "passenger_count",
)


def clean_yellow_taxi_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and type-clean Yellow Taxi trips for demand modeling."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[list(REQUIRED_COLUMNS)].copy()

    out["tpep_pickup_datetime"] = pd.to_datetime(
        out["tpep_pickup_datetime"], errors="coerce"
    )

    for col in ("trip_distance", "fare_amount", "passenger_count"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["PULocationID"] = pd.to_numeric(out["PULocationID"], errors="coerce")

    out = out.dropna(subset=["tpep_pickup_datetime", "PULocationID"])

    out = out[
        (out["trip_distance"] > 0)
        & (out["fare_amount"] >= 0)
        & (out["PULocationID"] > 0)
    ]

    out["PULocationID"] = out["PULocationID"].astype(int)

    return out.reset_index(drop=True)
