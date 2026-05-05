import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taxi_demand.streaming import TopK, top_zones_streaming


def test_topk_keeps_only_k_largest():
    top = TopK(3)
    for score, key in [(1, "a"), (5, "b"), (2, "c"), (9, "d"), (4, "e")]:
        top.add(score, key)
    assert top.items() == [(9, "d"), (5, "b"), (4, "e")]
    assert top.keys() == ["d", "b", "e"]
    assert len(top) == 3


def test_topk_add_returns_true_false():
    top = TopK(2)
    assert top.add(1, "a") is True
    assert top.add(2, "b") is True
    assert top.add(3, "c") is True
    assert top.add(0, "d") is False
    assert top.add(2, "e") is False


def test_topk_invalid_k_raises():
    with pytest.raises(ValueError, match="k must be positive"):
        TopK(0)
    with pytest.raises(ValueError, match="k must be positive"):
        TopK(-3)


def test_topk_tie_break_lexicographic():
    top = TopK(2)
    for score, key in [(5, "c"), (5, "a"), (5, "b")]:
        top.add(score, key)
    assert top.keys() == ["a", "b"]
    assert top.add(5, "z") is False
    assert top.keys() == ["a", "b"]


def test_topk_items_does_not_mutate_state():
    top = TopK(3)
    top.update([(10, "x"), (20, "y"), (30, "z")])
    snapshot = top.items()
    _ = top.items()
    _ = top.keys()
    assert top.items() == snapshot
    assert len(top) == 3


def test_top_zones_streaming_synthetic(tmp_path):
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-01 00:00:00"] * 10
            ),
            "PULocationID": [1] * 5 + [2] * 3 + [3] * 2,
            "trip_distance": [1.0] * 10,
            "fare_amount": [5.0] * 10,
            "passenger_count": [1] * 10,
        }
    )
    path = tmp_path / "synthetic.parquet"
    pq.write_table(pa.Table.from_pandas(df), path, row_group_size=4)

    rows = top_zones_streaming(path, k=2)
    assert rows == [(5, 1), (3, 2)]


def test_top_zones_streaming_handles_nulls(tmp_path):
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-01"] * 4),
            "PULocationID": [1, None, 1, 2],
            "trip_distance": [1.0] * 4,
            "fare_amount": [5.0] * 4,
            "passenger_count": [1] * 4,
        }
    )
    path = tmp_path / "with_nulls.parquet"
    pq.write_table(pa.Table.from_pandas(df), path)

    rows = top_zones_streaming(path, k=5)
    assert (2, 1) in rows
    assert (1, 2) in rows
    assert all(zone is not None for _, zone in rows)
