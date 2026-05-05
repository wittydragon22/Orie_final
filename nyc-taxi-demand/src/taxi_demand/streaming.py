"""Streaming top-K and zone counting from large trip parquet files."""

import heapq
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


class _HeapItem:
    """Internal heap entry; min-heap order keeps lex-smaller key on score tie."""

    __slots__ = ("score", "key")

    def __init__(self, score, key) -> None:
        self.score = score
        self.key = key

    def __lt__(self, other: "_HeapItem") -> bool:
        if self.score != other.score:
            return self.score < other.score
        return self.key > other.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _HeapItem):
            return NotImplemented
        return self.score == other.score and self.key == other.key


class TopK:
    """Keep the K largest (score, key) pairs seen, backed by a min-heap."""

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._k = int(k)
        self._heap: list[_HeapItem] = []

    def add(self, score, key) -> bool:
        """Insert a (score, key) pair; return True if it is kept in the top-K."""
        item = _HeapItem(score, key)
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, item)
            return True
        if not (self._heap[0] < item):
            return False
        heapq.heapreplace(self._heap, item)
        return True

    def update(self, iterable) -> None:
        """Add every (score, key) pair from an iterable."""
        for score, key in iterable:
            self.add(score, key)

    def items(self) -> list[tuple]:
        """Return kept items sorted descending by score, ascending by key on ties."""
        return sorted(
            ((it.score, it.key) for it in self._heap),
            key=lambda x: (-x[0], x[1]),
        )

    def keys(self) -> list:
        """Return just the keys of the kept items, in items() order."""
        return [k for _, k in self.items()]

    def __len__(self) -> int:
        return len(self._heap)


def top_zones_streaming(parquet_path: str | Path, k: int = 20) -> list[tuple]:
    """Stream a TLC parquet by row group and return top-K (count, PULocationID) pairs."""
    parquet_path = Path(parquet_path)
    counts: Counter = Counter()

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["PULocationID"]):
        col = batch.column("PULocationID").to_pylist()
        for value in col:
            if value is None:
                continue
            counts[int(value)] += 1

    top = TopK(k)
    for zone, count in counts.items():
        top.add(count, zone)
    return top.items()
