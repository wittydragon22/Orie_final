import pandas as pd
import pytest

from taxi_demand.cleaning import clean_yellow_taxi_trips


def _minimal_trips():
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
                "bad-date",
                "2025-01-01 12:00:00",
                "2025-01-01 13:00:00",
                "2025-01-01 14:00:00",
            ],
            "PULocationID": [1, 2, 3, 0, -1, 4],
            "trip_distance": [1.0, 2.0, 3.0, 0.0, 5.0, 1.0],
            "fare_amount": [10.0, 12.0, 15.0, 8.0, 20.0, -1.0],
            "passenger_count": [1, 1, 2, 1, 1, 1],
        }
    )


def test_cleaning_removes_invalid_rows():
    df = _minimal_trips()
    out = clean_yellow_taxi_trips(df)
    assert len(out) == 2
    assert set(out["PULocationID"].tolist()) == {1, 2}
    assert (out["trip_distance"] > 0).all()
    assert (out["fare_amount"] >= 0).all()


def test_pickup_datetime_conversion():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2025-01-15 08:30:00"],
            "PULocationID": [10],
            "trip_distance": [1.5],
            "fare_amount": [9.0],
            "passenger_count": [1],
        }
    )
    out = clean_yellow_taxi_trips(df)
    assert pd.api.types.is_datetime64_any_dtype(out["tpep_pickup_datetime"])
    assert out["tpep_pickup_datetime"].iloc[0].hour == 8


def test_pulocation_int():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "PULocationID": [100.0, 101.0],
            "trip_distance": [1.0, 2.0],
            "fare_amount": [5.0, 6.0],
            "passenger_count": [1, 2],
        }
    )
    out = clean_yellow_taxi_trips(df)
    assert out["PULocationID"].dtype in (int, "int64", "Int64") or str(
        out["PULocationID"].dtype
    ).startswith("int")


def test_missing_required_columns():
    df = pd.DataFrame({"tpep_pickup_datetime": ["2025-01-01"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_yellow_taxi_trips(df)
