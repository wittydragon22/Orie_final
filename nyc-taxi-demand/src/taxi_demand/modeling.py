"""Train/test split and Random Forest model."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLUMNS = [
    "PULocationID",
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "rolling_mean_24h",
]
TARGET_COLUMN = "pickup_count"


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names."""
    return list(FEATURE_COLUMNS)


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: earliest train_frac of rows (by pickup_hour) for train."""
    if "pickup_hour" not in df.columns:
        raise ValueError("DataFrame must contain 'pickup_hour'.")
    sorted_df = df.sort_values("pickup_hour").reset_index(drop=True)
    n = len(sorted_df)
    cut = max(1, int(n * train_frac))
    if cut >= n:
        cut = n - 1
    train_df = sorted_df.iloc[:cut].copy()
    test_df = sorted_df.iloc[cut:].copy()
    return train_df, test_df


def train_random_forest(train_df: pd.DataFrame) -> RandomForestRegressor:
    """Fit a RandomForestRegressor on the training frame."""
    X = train_df[FEATURE_COLUMNS]
    y = train_df[TARGET_COLUMN]
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict(model: RandomForestRegressor, df: pd.DataFrame) -> pd.Series:
    """Predict pickup_count for rows in df."""
    X = df[FEATURE_COLUMNS]
    return pd.Series(model.predict(X), index=df.index)
