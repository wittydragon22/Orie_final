"""Feature engineering: hourly aggregation and lags."""

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, and is_weekend from pickup_hour."""
    out = df.copy()
    if "pickup_hour" not in out.columns:
        raise ValueError("DataFrame must contain column 'pickup_hour'.")
    ts = pd.to_datetime(out["pickup_hour"])
    out["hour"] = ts.dt.hour.astype(int)
    out["day_of_week"] = ts.dt.dayofweek.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def make_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned trips to hourly pickup counts per zone."""
    if "tpep_pickup_datetime" not in df.columns or "PULocationID" not in df.columns:
        raise ValueError("Expected columns tpep_pickup_datetime and PULocationID.")

    x = df.copy()
    x["pickup_hour"] = pd.to_datetime(x["tpep_pickup_datetime"]).dt.floor("h")

    counts = (
        x.groupby(["PULocationID", "pickup_hour"], as_index=False)
        .size()
        .rename(columns={"size": "pickup_count"})
    )
    return counts.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag_1h, lag_24h, rolling_mean_24h per zone; fill missing with 0."""
    out = df.sort_values(["PULocationID", "pickup_hour"]).copy()
    by_zone = out.groupby("PULocationID", group_keys=False)["pickup_count"]

    out["lag_1h"] = by_zone.shift(1)
    out["lag_24h"] = by_zone.shift(24)
    out["rolling_mean_24h"] = by_zone.transform(
        lambda s: s.shift(1).rolling(24, min_periods=1).mean()
    )

    out[["lag_1h", "lag_24h", "rolling_mean_24h"]] = out[
        ["lag_1h", "lag_24h", "rolling_mean_24h"]
    ].fillna(0)

    return out.reset_index(drop=True)


def build_model_dataset(clean_trips: pd.DataFrame) -> pd.DataFrame:
    """From cleaned trips, build zone-hour panel with targets and features."""
    hourly = make_hourly_demand(clean_trips)
    hourly = add_time_features(hourly)
    hourly = add_lag_features(hourly)
    return hourly
