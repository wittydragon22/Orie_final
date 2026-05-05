"""Regression metrics, baseline, and zone-level evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, and R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def baseline_last_hour(df: pd.DataFrame) -> pd.Series:
    """Naive forecast: predict current pickup_count using lag_1h."""
    if "lag_1h" not in df.columns:
        raise ValueError("DataFrame must contain 'lag_1h'.")
    return pd.Series(df["lag_1h"].values, index=df.index)


def evaluate_by_zone(
    df: pd.DataFrame, y_pred
) -> pd.DataFrame:
    """Per-zone MAE, RMSE, and R2 (R2 is NaN when a zone has fewer than 2 rows)."""
    if "PULocationID" not in df.columns or "pickup_count" not in df.columns:
        raise ValueError("DataFrame must contain PULocationID and pickup_count.")
    y_true = df["pickup_count"].values
    y_pred = np.asarray(y_pred, dtype=float)

    tmp = df.assign(_y_true=y_true, _y_pred=y_pred)
    rows = []
    for zone, grp in tmp.groupby("PULocationID"):
        yt = grp["_y_true"].values
        yp = grp["_y_pred"].values
        n = len(grp)
        if n < 2:
            mae = float(mean_absolute_error(yt, yp))
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            r2 = float(np.nan)
        else:
            m = regression_metrics(yt, yp)
            mae, rmse, r2 = m["mae"], m["rmse"], m["r2"]
        rows.append(
            {
                "PULocationID": zone,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "n_rows": n,
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)
