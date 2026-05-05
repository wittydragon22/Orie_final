import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from taxi_demand.parallel import (
    OUTPUT_COLUMNS,
    aggregate_files_parallel,
    aggregate_one_file,
)


def _write_synthetic_parquet(path, n_zone_1: int = 4, n_zone_2: int = 2):
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-01 10:30:00"] * n_zone_1 + ["2025-01-01 11:15:00"] * n_zone_2
            ),
            "PULocationID": [1] * n_zone_1 + [2] * n_zone_2,
            "trip_distance": [1.0] * (n_zone_1 + n_zone_2),
            "fare_amount": [8.0] * (n_zone_1 + n_zone_2),
            "passenger_count": [1] * (n_zone_1 + n_zone_2),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), path)


def test_aggregate_files_parallel_empty_returns_empty():
    result = aggregate_files_parallel([])
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == list(OUTPUT_COLUMNS)


def test_aggregate_one_file_returns_hourly_counts(tmp_path):
    path = tmp_path / "single.parquet"
    _write_synthetic_parquet(path, n_zone_1=4, n_zone_2=2)

    out = aggregate_one_file(path)
    assert list(out.columns) == list(OUTPUT_COLUMNS)
    zone_1 = out[out["PULocationID"] == 1]
    zone_2 = out[out["PULocationID"] == 2]
    assert zone_1["pickup_count"].sum() == 4
    assert zone_2["pickup_count"].sum() == 2


def test_single_worker_path(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    _write_synthetic_parquet(a, n_zone_1=3, n_zone_2=1)
    _write_synthetic_parquet(b, n_zone_1=2, n_zone_2=2)

    out = aggregate_files_parallel([a, b], workers=1)
    total_zone_1 = out.loc[out["PULocationID"] == 1, "pickup_count"].sum()
    total_zone_2 = out.loc[out["PULocationID"] == 2, "pickup_count"].sum()
    assert total_zone_1 == 5
    assert total_zone_2 == 3


def test_multi_worker_path(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    _write_synthetic_parquet(a, n_zone_1=4, n_zone_2=2)
    _write_synthetic_parquet(b, n_zone_1=1, n_zone_2=3)

    out = aggregate_files_parallel([a, b], workers=2)
    total_zone_1 = out.loc[out["PULocationID"] == 1, "pickup_count"].sum()
    total_zone_2 = out.loc[out["PULocationID"] == 2, "pickup_count"].sum()
    assert total_zone_1 == 5
    assert total_zone_2 == 5


def test_two_copies_double_counts(tmp_path):
    same = tmp_path / "same.parquet"
    _write_synthetic_parquet(same, n_zone_1=3, n_zone_2=1)

    one = aggregate_files_parallel([same], workers=1)
    two = aggregate_files_parallel([same, same], workers=1)

    one_total = one["pickup_count"].sum()
    two_total = two["pickup_count"].sum()
    assert two_total == 2 * one_total


def test_workers_cap_when_exceeds_files(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    _write_synthetic_parquet(a, n_zone_1=2, n_zone_2=1)
    _write_synthetic_parquet(b, n_zone_1=1, n_zone_2=2)

    out = aggregate_files_parallel([a, b], workers=99)
    assert out["pickup_count"].sum() == 6
