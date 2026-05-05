import tempfile
from pathlib import Path

import pandas as pd
import pytest

from taxi_demand.data import load_dataset, load_trip_data, save_dataset
from taxi_demand.features import (
    add_lag_features,
    add_time_features,
    build_model_dataset,
    make_hourly_demand,
)
from taxi_demand.modeling import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_feature_columns,
    predict,
    time_based_split,
    train_random_forest,
)


def test_hourly_demand_aggregation():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2025-01-01 10:15:00",
                    "2025-01-01 10:45:00",
                    "2025-01-01 10:20:00",
                    "2025-01-01 11:00:00",
                ]
            ),
            "PULocationID": [1, 1, 1, 1],
        }
    )
    hourly = make_hourly_demand(df)
    assert len(hourly) == 2
    h10 = hourly[hourly["pickup_hour"] == pd.Timestamp("2025-01-01 10:00:00")]
    assert h10["pickup_count"].iloc[0] == 3
    h11 = hourly[hourly["pickup_hour"] == pd.Timestamp("2025-01-01 11:00:00")]
    assert h11["pickup_count"].iloc[0] == 1


def test_lag_features():
    rows = []
    for h in range(5):
        rows.append(
            {
                "PULocationID": 1,
                "pickup_hour": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=h),
                "pickup_count": float(h + 1),
            }
        )
    df = pd.DataFrame(rows)
    df = add_time_features(df)
    out = add_lag_features(df)
    assert out.loc[out["pickup_hour"].idxmin(), "lag_1h"] == 0.0
    r1 = out[out["pickup_hour"] == pd.Timestamp("2025-01-01 01:00:00")].iloc[0]
    assert r1["lag_1h"] == 1.0
    assert "lag_24h" in out.columns
    assert "rolling_mean_24h" in out.columns


def test_build_model_dataset_end_to_end():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-02 09:00:00"] * 5 + ["2025-01-02 09:30:00"] * 2
            ),
            "PULocationID": [5] * 7,
            "trip_distance": [1.0] * 7,
            "fare_amount": [8.0] * 7,
            "passenger_count": [1] * 7,
        }
    )
    from taxi_demand.cleaning import clean_yellow_taxi_trips

    clean = clean_yellow_taxi_trips(df)
    m = build_model_dataset(clean)
    assert "pickup_count" in m.columns
    assert "hour" in m.columns
    assert "day_of_week" in m.columns
    assert "is_weekend" in m.columns
    assert m["pickup_count"].iloc[0] == 7.0


def test_parquet_csv_roundtrip():
    df = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
    with tempfile.TemporaryDirectory() as td:
        pq = Path(td) / "t.parquet"
        save_dataset(df, pq)
        pd.testing.assert_frame_equal(df, load_trip_data(pq))
        cs = Path(td) / "t.csv"
        save_dataset(df[["b"]], cs)
        pd.testing.assert_frame_equal(df[["b"]], load_dataset(cs))


def test_save_unsupported_extension():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.json"
        with pytest.raises(ValueError, match="Unsupported"):
            save_dataset(pd.DataFrame({"a": [1]}), p)


def test_load_trip_data_unsupported():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "y.bin"
        p.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unsupported"):
            load_trip_data(p)


def test_modeling_split_and_predict_smoke():
    assert get_feature_columns() == FEATURE_COLUMNS
    assert TARGET_COLUMN == "pickup_count"
    ph = pd.date_range("2025-01-01", periods=50, freq="h")
    panel = pd.DataFrame(
        {
            "pickup_hour": ph,
            "pickup_count": list(range(50)),
            "PULocationID": [1] * 50,
            "hour": ph.hour.astype(int),
            "day_of_week": ph.dayofweek.astype(int),
            "is_weekend": (ph.dayofweek >= 5).astype(int),
            "lag_1h": [0.0] + [float(i) for i in range(49)],
            "lag_24h": [0.0] * 50,
            "rolling_mean_24h": [0.0] * 50,
        }
    )
    tr, te = time_based_split(panel, train_frac=0.8)
    model = train_random_forest(tr)
    pred = predict(model, te)
    assert len(pred) == len(te)
