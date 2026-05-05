# NYC Yellow Taxi Hourly Demand

Predict **hourly pickup demand by taxi zone** using NYC TLC Yellow Taxi trip records. Trip-level data are cleaned, aggregated to zone-hour panels, and modeled with **scikit-learn** `RandomForestRegressor`, compared to a **lag-1h baseline**.

## Dataset

- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (Yellow Taxi).
- **File used in this starter:** January 2025 Parquet:
  - URL: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet`
- **Target:** `pickup_count` (trips per zone per hour).
- **Features:** `PULocationID`, `hour`, `day_of_week`, `is_weekend`, `lag_1h`, `lag_24h`, `rolling_mean_24h`.

## Installation

From the `nyc-taxi-demand` directory:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Download raw data

```bash
python scripts/download_data.py
```

This saves `data/raw/yellow_tripdata_2025-01.parquet`.

## Build the processed dataset

```bash
python scripts/build_dataset.py
```

This reads the raw Parquet, cleans trips, builds the zone-hour model table with lags, and writes `data/processed/hourly_demand_2025_01.parquet`.

## Train and evaluate the model

```bash
python scripts/train_model.py
```

This loads the processed dataset, applies an **80% / 20% time-ordered train/test split**, trains a `RandomForestRegressor`, prints **MAE, RMSE, R²** for the forest and for the **baseline** (predict `pickup_count` with `lag_1h`), and prints the **worst zones by MAE**.

## Run tests

From the project root (with the package installed editable as above):

```bash
pytest --cov=src/taxi_demand
```

If `pytest` is not on your PATH (common on Windows), use:

```bash
python -m pytest --cov=src/taxi_demand
```

## Project layout

- `src/taxi_demand/` — package: loading data, cleaning, features, modeling, evaluation.
- `scripts/` — download, build dataset, train.
- `tests/` — unit tests for cleaning, features, evaluation.
- `notebooks/exploratory_analysis.ipynb` — optional exploration.

## License

Educational use (ORIE 5270). Data © NYC TLC; refer to TLC terms for redistribution.
