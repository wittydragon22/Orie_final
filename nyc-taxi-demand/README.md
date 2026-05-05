# NYC Yellow Taxi Hourly Demand Prediction

This project predicts **hourly yellow taxi pickup demand by NYC taxi zone** using NYC TLC Yellow Taxi Trip Record Data. We clean the raw trip-level records, aggregate them into a zone-hour demand panel, build time-based and historical-demand features, and train a scikit-learn `RandomForestRegressor` to predict near-future pickup demand. The model is compared against a simple **lag-1 hour baseline**.

## Project overview

Short-term taxi demand forecasting can help transportation planners and mobility operators understand where and when passenger demand is likely to occur. Concretely, the prediction task we're after is:

> Given a taxi zone, hour of day, day of week, weekend indicator, and recent historical pickup demand, predict the number of yellow taxi pickups in that zone during that hour.

Everything lives in a reproducible Python package with runnable scripts, unit tests, a couple of docs, and an exploratory notebook.

## Dataset

- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Dataset used:** Yellow Taxi Trip Record Data
- **File used:** January 2025 Yellow Taxi Parquet file
- **Raw file URL:** `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet`

The raw TLC trip data include pickup and drop-off times, pickup and drop-off location IDs, trip distance, fare amount, passenger count, payment type, and other trip-level fields.

After cleaning and aggregation, the processed model dataset contains:

- **92,178** zone-hour observations
- **259** unique pickup zones
- Target variable: `pickup_count`, the number of pickups in a taxi zone during a given hour

## Research question

Can recent historical demand and time-based features predict short-term NYC yellow taxi pickup demand at the taxi-zone-hour level? Spoiler: yes, and by a comfortable margin over the naive baseline.

## Features and target

**Target.** `pickup_count` — number of yellow taxi pickups in a given pickup zone during a given hour.

**Features:**

- `PULocationID`: pickup taxi zone ID
- `hour`: hour of day
- `day_of_week`: day of week
- `is_weekend`: weekend indicator
- `lag_1h`: pickup demand in the previous observed hour for the same zone
- `lag_24h`: pickup demand 24 observations earlier for the same zone — this one ended up mattering more than we'd guessed
- `rolling_mean_24h`: rolling average of recent pickup demand

We thought about adding weather data but skipped it for this submission — that's on the wishlist for next time.

## Methods

The pipeline:

1. Download January 2025 Yellow Taxi trip record data.
2. Clean invalid trips, including rows with missing pickup time or invalid pickup location.
3. Filter trips with non-positive trip distance or negative fare amount.
4. Aggregate trip-level data into a zone-hour demand table.
5. Create time features and lag-based demand features.
6. Split the data using a time-ordered 80% / 20% train-test split.
7. Train a `RandomForestRegressor`. Standard.
8. Compare against a baseline that just predicts demand using `lag_1h`.
9. Evaluate with MAE, RMSE, and R².
10. Run unit tests to verify data cleaning, feature engineering, modeling, and evaluation.

## Results

The processed dataset was split into:

- **Train rows:** 73,742
- **Test rows:** 18,436

### Test set performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 6.39 | 15.78 | 0.954 |
| Baseline: lag_1h | 9.73 | 23.97 | 0.893 |

The Random Forest beats the lag-1 baseline on all three metrics — combining recent historical demand with time-based features clearly helps short-term taxi pickup demand prediction.

## Key findings

The exploratory analysis shows pretty clear time-of-day patterns:

- Pickup demand is lowest in the early morning hours.
- Demand rises through the day and peaks around the evening commute.
- Thursday and Saturday are the busiest days on average — we'd expected Friday to top the list, so that was a small surprise.
- Top pickup zones include 161, 237, 236, 132, and 230.
- `lag_1h` is the strongest predictor by a wide margin. Very recent demand carries most of the signal.

## What's new in this version

This release adds two pieces inspired by the W7D2 (data streams) and W9 (map-reduce) lectures:

- **Streaming top-K** (`taxi_demand.streaming`): a `TopK` class backed by a `heapq` min-heap, plus a `top_zones_streaming` helper that walks a TLC parquet a row group at a time using `pyarrow.parquet.ParquetFile.iter_batches`. We never load the whole file. Useful when you want the busiest zones for a month without paying RAM for the full trip table.
- **Parallel per-file aggregation** (`taxi_demand.parallel`): `aggregate_one_file` cleans and hour-buckets a single parquet, and `aggregate_files_parallel` fans those out across processes with `multiprocessing.Pool`, then reduces with a groupby-sum. Workers are capped at `min(len(paths), cpu_count())`. The `workers=1` path skips the pool entirely so it's easy to test and easy to debug.

A short write-up of the design choices lives in `docs/design_notes.md`.

## Installation

From the `nyc-taxi-demand` directory:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Download raw data

```bash
python scripts/download_data.py
```

This saves:

```text
data/raw/yellow_tripdata_2025-01.parquet
```

## Build the processed dataset

```bash
python scripts/build_dataset.py
```

This script reads the raw Parquet file, cleans trips, builds the zone-hour model table with lag features, and writes:

```text
data/processed/hourly_demand_2025_01.parquet
```

## Train and evaluate the model

```bash
python scripts/train_model.py
```

Loads the processed dataset, applies the 80/20 time-ordered split, trains the Random Forest, prints MAE, RMSE, and R² for both the model and the lag-1 hour baseline, and reports the worst-performing zones.

## Streaming top-K zones

Skip the full load and stream the raw parquet:

```bash
python scripts/top_zones.py --parquet data/raw/yellow_tripdata_2025-01.parquet --k 20
```

If `--parquet` is omitted, it defaults to the standard raw path. The script prints a small ranked table of `(rank, PULocationID, pickup_count)`.

## Parallel aggregation across files

If you have several monthly parquets and want a merged hourly count table:

```bash
python scripts/parallel_aggregate.py --paths data/raw/yellow_tripdata_2025-01.parquet data/raw/yellow_tripdata_2025-02.parquet --workers 4
```

It writes the merged result to `data/processed/parallel_aggregated.parquet` (override with `--out`).

## Run tests

From the project root:

```bash
pytest --cov=src/taxi_demand
```

If `pytest` is not on your PATH:

```bash
python -m pytest --cov=src/taxi_demand
```

Current test results:

- **28 tests passed**
- **94% package test coverage**

## Project layout

```text
nyc-taxi-demand/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── design_notes.md
├── scripts/
│   ├── download_data.py
│   ├── build_dataset.py
│   ├── train_model.py
│   ├── top_zones.py
│   └── parallel_aggregate.py
├── src/
│   └── taxi_demand/
│       ├── data.py
│       ├── cleaning.py
│       ├── features.py
│       ├── modeling.py
│       ├── evaluation.py
│       ├── streaming.py
│       └── parallel.py
├── tests/
│   ├── test_cleaning.py
│   ├── test_features.py
│   ├── test_evaluation.py
│   ├── test_streaming.py
│   └── test_parallel.py
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Exploratory notebook

`notebooks/exploratory_analysis.ipynb` has the supporting analysis and visualizations:

- hourly pickup demand trends
- average demand by hour of day
- average demand by day of week
- top pickup zones
- model versus baseline comparison
- actual versus predicted demand plots
- zone-level error analysis
- feature importance

## Limitations and future work

Only one month of Yellow Taxi data is used here, so the patterns we learn are basically January-shaped. Future extensions: pull more months, join taxi zone names and boroughs for more readable results, layer in weather, try a couple of additional models (gradient boosted trees, simple temporal CNNs), and check whether the model holds up across seasons or years.

## License

Educational use for ORIE 5270. Data are provided by NYC TLC; refer to NYC TLC terms for data usage and redistribution.
