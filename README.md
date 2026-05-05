# NYC Yellow Taxi Hourly Demand Prediction

This project predicts **hourly yellow taxi pickup demand by NYC taxi zone** using NYC TLC Yellow Taxi Trip Record Data. The project cleans raw trip-level records, aggregates them into a zone-hour demand panel, creates time-based and historical-demand features, and trains a scikit-learn `RandomForestRegressor` to predict future pickup demand. The model is compared against a simple **lag-1 hour baseline**.

## Project Overview

Short-term taxi demand forecasting can help transportation planners and mobility operators understand where and when passenger demand is likely to occur. In this project, the prediction task is:

> Given a taxi zone, hour of day, day of week, weekend indicator, and recent historical pickup demand, predict the number of yellow taxi pickups in that zone during that hour.

The project is organized as a reproducible Python package with runnable scripts, unit tests, documentation, and an exploratory notebook.

## Dataset

- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Dataset used:** Yellow Taxi Trip Record Data
- **File used:** January 2025 Yellow Taxi Parquet file
- **Raw file URL:** `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet`

Raw and processed Parquet files are not committed to GitHub because of file size. The empty `data/raw/` and `data/processed/` folders are kept with `.gitkeep` files. The data can be recreated by running:

```bash
python scripts/download_data.py
python scripts/build_dataset.py
```

The raw TLC trip data include pickup and drop-off times, pickup and drop-off location IDs, trip distance, fare amount, passenger count, payment type, and other trip-level fields.

After cleaning and aggregation, the processed model dataset contains:

- **92,178** zone-hour observations
- **259** unique pickup zones
- Target variable: `pickup_count`, the number of pickups in a taxi zone during a given hour

## Research Question

Can recent historical demand and time-based features predict short-term NYC yellow taxi pickup demand at the taxi-zone-hour level?

## Features and Target

### Target

- `pickup_count`: number of yellow taxi pickups in a given pickup zone during a given hour

### Features

- `PULocationID`: pickup taxi zone ID
- `hour`: hour of day
- `day_of_week`: day of week
- `is_weekend`: weekend indicator
- `lag_1h`: pickup demand in the previous observed hour for the same zone
- `lag_24h`: pickup demand 24 observations earlier for the same zone
- `rolling_mean_24h`: rolling average of recent pickup demand

## Methods

The project pipeline follows these steps:

1. Download January 2025 Yellow Taxi trip record data.
2. Clean invalid trips, including rows with missing pickup time or invalid pickup location.
3. Filter trips with non-positive trip distance or negative fare amount.
4. Aggregate trip-level data into a zone-hour demand table.
5. Create time features and lag-based demand features.
6. Split the data using a time-ordered 80% / 20% train-test split.
7. Train a `RandomForestRegressor`.
8. Compare the model against a baseline that predicts demand using `lag_1h`.
9. Evaluate results using MAE, RMSE, and R².
10. Run unit tests to verify data cleaning, feature engineering, modeling, and evaluation functions.

## Results

The processed dataset was split into:

- **Train rows:** 73,742
- **Test rows:** 18,436

### Test Set Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 6.39 | 15.78 | 0.954 |
| Baseline: lag_1h | 9.73 | 23.97 | 0.893 |

The Random Forest model outperformed the lag-1 hour baseline on all three metrics. This suggests that combining recent historical demand with time-based features improves short-term taxi pickup demand prediction.

## Key Findings

The exploratory analysis shows clear time-based demand patterns:

- Pickup demand is lowest in the early morning hours.
- Demand rises throughout the day and peaks around the evening commute period.
- Thursday and Saturday show relatively high average pickup activity.
- The busiest pickup zones include zones 161, 237, 236, 132, and 230.
- Feature importance shows that `lag_1h` is the strongest predictor, meaning very recent demand is highly informative for short-term forecasting.

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

## Download Raw Data

```bash
python scripts/download_data.py
```

This saves:

```text
data/raw/yellow_tripdata_2025-01.parquet
```

## Build the Processed Dataset

```bash
python scripts/build_dataset.py
```

This script reads the raw Parquet file, cleans trips, builds the zone-hour model table with lag features, and writes:

```text
data/processed/hourly_demand_2025_01.parquet
```

## Train and Evaluate the Model

```bash
python scripts/train_model.py
```

This script loads the processed dataset, applies an 80% / 20% time-ordered train-test split, trains a `RandomForestRegressor`, prints MAE, RMSE, and R² for both the Random Forest model and the lag-1 hour baseline, and reports zone-level error results.

## Run Tests

From the project root:

```bash
pytest --cov=src/taxi_demand
```

If `pytest` is not on your PATH, use:

```bash
python -m pytest --cov=src/taxi_demand
```

Current test results:

- **14 tests passed**
- **95% package test coverage**

## Project Layout

```text
nyc-taxi-demand/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── scripts/
│   ├── download_data.py
│   ├── build_dataset.py
│   └── train_model.py
├── src/
│   └── taxi_demand/
│       ├── data.py
│       ├── cleaning.py
│       ├── features.py
│       ├── modeling.py
│       └── evaluation.py
├── tests/
│   ├── test_cleaning.py
│   ├── test_features.py
│   └── test_evaluation.py
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Exploratory Notebook

The notebook `notebooks/exploratory_analysis.ipynb` provides supporting analysis and visualizations, including:

- hourly pickup demand trends
- average demand by hour of day
- average demand by day of week
- top pickup zones
- model versus baseline comparison
- actual versus predicted demand plots
- zone-level error analysis
- feature importance

## Limitations and Future Work

This project uses only one month of Yellow Taxi data. Future extensions could improve the model by adding more months, joining taxi zone names and boroughs, incorporating weather data, comparing additional models, and testing whether the model generalizes across different seasons or years.

## License

Educational use for ORIE 5270. Data are provided by NYC TLC; refer to NYC TLC terms for data usage and redistribution.
