"""Load raw trips, clean, build hourly model dataset, save processed Parquet."""

from pathlib import Path

from taxi_demand.cleaning import clean_yellow_taxi_trips
from taxi_demand.data import load_trip_data, save_dataset
from taxi_demand.features import build_model_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "yellow_tripdata_2025-01.parquet"
PROCESSED_PATH = ROOT / "data" / "processed" / "hourly_demand_2025_01.parquet"


def main() -> None:
    print(f"Loading {RAW_PATH} ...")
    raw = load_trip_data(RAW_PATH)
    print("Cleaning trips ...")
    clean = clean_yellow_taxi_trips(raw)
    print("Building model dataset ...")
    model_df = build_model_dataset(clean)
    save_dataset(model_df, PROCESSED_PATH)
    print(f"Saved {len(model_df)} rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
