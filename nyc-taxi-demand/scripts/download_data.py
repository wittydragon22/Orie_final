"""Download NYC TLC Yellow Taxi January 2025 Parquet to data/raw/."""

import urllib.request
from pathlib import Path

URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
)
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "yellow_tripdata_2025-01.parquet"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {OUT_PATH} ...")
    urllib.request.urlretrieve(URL, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
