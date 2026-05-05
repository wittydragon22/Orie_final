"""Train Random Forest, compare to baseline, report metrics and worst zones."""

from pathlib import Path

from taxi_demand.data import load_dataset
from taxi_demand.evaluation import (
    baseline_last_hour,
    evaluate_by_zone,
    regression_metrics,
)
from taxi_demand.modeling import predict, time_based_split, train_random_forest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "hourly_demand_2025_01.parquet"


def main() -> None:
    print(f"Loading {PROCESSED_PATH} ...")
    df = load_dataset(PROCESSED_PATH)

    train_df, test_df = time_based_split(df, train_frac=0.8)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    print("Training RandomForestRegressor ...")
    model = train_random_forest(train_df)
    y_pred = predict(model, test_df)

    y_true = test_df["pickup_count"].values
    metrics = regression_metrics(y_true, y_pred)
    print("\n=== Random Forest (test) ===")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")

    base_pred = baseline_last_hour(test_df)
    base_metrics = regression_metrics(y_true, base_pred)
    print("\n=== Baseline (lag_1h) ===")
    print(f"  MAE:  {base_metrics['mae']:.4f}")
    print(f"  RMSE: {base_metrics['rmse']:.4f}")
    print(f"  R2:   {base_metrics['r2']:.4f}")

    zone_eval = evaluate_by_zone(test_df, y_pred)
    print("\n=== Worst 10 zones by MAE (Random Forest) ===")
    print(zone_eval.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
