"""Aggregate multiple TLC trip parquets in parallel and write merged hourly counts."""

import argparse
from pathlib import Path

from taxi_demand.data import save_dataset
from taxi_demand.parallel import aggregate_files_parallel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "processed" / "parallel_aggregated.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-file aggregation in parallel and write merged hourly counts."
    )
    parser.add_argument("--paths", nargs="+", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = aggregate_files_parallel(args.paths, workers=args.workers)
    save_dataset(df, args.out)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
