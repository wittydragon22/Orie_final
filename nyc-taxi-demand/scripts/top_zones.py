"""Stream a TLC trip parquet and print the top-K pickup zones."""

import argparse
from pathlib import Path

from taxi_demand.streaming import top_zones_streaming

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "yellow_tripdata_2025-01.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-K pickup zones from a TLC parquet, computed in a streaming pass."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    rows = top_zones_streaming(args.parquet, k=args.k)

    print(f"{'rank':>4}  {'PULocationID':>12}  {'pickup_count':>12}")
    for rank, (count, zone) in enumerate(rows, start=1):
        print(f"{rank:>4}  {zone:>12}  {count:>12}")


if __name__ == "__main__":
    main()
