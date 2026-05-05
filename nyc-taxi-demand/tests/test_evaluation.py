import numpy as np
import pandas as pd

from taxi_demand.evaluation import (
    baseline_last_hour,
    evaluate_by_zone,
    regression_metrics,
)


def test_regression_metrics_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    m = regression_metrics(y_true, y_pred)
    assert set(m.keys()) == {"mae", "rmse", "r2"}
    assert m["mae"] >= 0
    assert m["rmse"] >= 0


def test_baseline_last_hour_uses_lag_1h():
    df = pd.DataFrame(
        {
            "pickup_count": [10, 20, 30],
            "lag_1h": [1.0, 2.0, 3.0],
        }
    )
    pred = baseline_last_hour(df)
    assert pred.tolist() == [1.0, 2.0, 3.0]


def test_evaluate_by_zone():
    df = pd.DataFrame(
        {
            "PULocationID": [1, 1, 2, 2],
            "pickup_count": [10.0, 20.0, 5.0, 5.0],
        }
    )
    y_pred = np.array([12.0, 18.0, 4.0, 6.0])
    z = evaluate_by_zone(df, y_pred)
    assert "PULocationID" in z.columns and "mae" in z.columns and "r2" in z.columns
    assert len(z) == 2
    assert not np.isnan(z.loc[z["PULocationID"] == 1, "r2"].iloc[0])


def test_evaluate_by_zone_r2_nan_when_single_row():
    df = pd.DataFrame({"PULocationID": [3], "pickup_count": [1.0]})
    y_pred = np.array([2.0])
    z = evaluate_by_zone(df, y_pred)
    assert z["n_rows"].iloc[0] == 1
    assert np.isnan(z["r2"].iloc[0])
