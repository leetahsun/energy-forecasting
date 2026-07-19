import numpy as np
import pandas as pd

from ml_forecasting.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
)
from ml_forecasting.train import load_model, TARGET_COLUMNS
from shared.models import ForecastRecord


def _build_feature_row(
    series: pd.Series, target_col: str, at_timestamp: pd.Timestamp
) -> pd.DataFrame:
    """Build a single feature row for one target column at one timestamp
    """
    extended = pd.concat([series, pd.Series([np.nan], index=[at_timestamp])])
    df = extended.to_frame(name=target_col)
    df = add_calendar_features(df)
    df = add_lag_features(df, columns=[target_col], lags=DEFAULT_LAGS)
    df = add_rolling_features(df, columns=[target_col], windows=DEFAULT_ROLLING_WINDOWS)
    return df.loc[[at_timestamp]]


def recursive_forecast(
    target_col: str,
    history_series: pd.Series,
    model,
    feature_cols: list[str],
    horizon_hours: int = 24,
) -> pd.Series:
    """Predict horizon hours ahead at only one hour at a time by feeding each
    prediction back into the working series so later hours' rolling/features so that we
    can reference it.
    """
    min_required_history = max(DEFAULT_LAGS)
    if len(history_series) < min_required_history:
        raise ValueError(
            f"Need at least {min_required_history}h of history to forecast "
            f"{target_col}, got {len(history_series)}h."
        )

    working = history_series.copy()
    last_ts = working.index[-1]
    predictions: dict[pd.Timestamp, float] = {}

    for step in range(1, horizon_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=step)
        row = _build_feature_row(working, target_col, next_ts)

        if row[feature_cols].isna().any().any():
            raise ValueError(
                f"Feature row for {target_col} at {next_ts} has missing "
                f"values insufficient history for this forecast step."
            )

        pred = float(model.predict(row[feature_cols])[0])
        predictions[next_ts] = pred
        working.loc[next_ts] = pred  # feed forward so the next step can use it

    return pd.Series(predictions)


def naive_baseline_forecast(
    target_col: str, history_series: pd.Series, horizon_hours: int = 24
) -> pd.Series:
    """Persistence baseline: for each future hour, predict the actual
    value from exactly one week (168h) earlier.
    """
    last_ts = history_series.index[-1]
    predictions: dict[pd.Timestamp, float] = {}
    for step in range(1, horizon_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=step)
        source_ts = next_ts - pd.Timedelta(hours=168)
        if source_ts not in history_series.index:
            raise ValueError(
                f"Naive baseline needs the value from {source_ts} (168h "
                f"before {next_ts}), which isn't in history_series."
            )
        predictions[next_ts] = history_series.loc[source_ts]
    return pd.Series(predictions)


def predict_target(
    target_col: str,
    history_series: pd.Series,
    horizon_hours: int = 24,
    models_dir: str = "ml_forecasting/models",
) -> list[ForecastRecord]:
    """Load the saved XGBoost model for one target and forecast forward."""
    model, feature_cols = load_model(f"{models_dir}/{target_col}_xgboost.joblib")
    predictions = recursive_forecast(
        target_col, history_series, model, feature_cols, horizon_hours
    )

    return [
        ForecastRecord(
            timestamp=ts,
            metric_name=target_col,
            model_name="xgboost_v1",
            predicted_value=val,
        )
        for ts, val in predictions.items()
    ]


def predict_naive_baseline(
    target_col: str, history_series: pd.Series, horizon_hours: int = 24
) -> list[ForecastRecord]:
    predictions = naive_baseline_forecast(target_col, history_series, horizon_hours)
    return [
        ForecastRecord(
            timestamp=ts,
            metric_name=target_col,
            model_name="naive_baseline",
            predicted_value=val,
        )
        for ts, val in predictions.items()
    ]


def generate_all_forecasts(
    history_df: pd.DataFrame,
    horizon_hours: int = 24,
    models_dir: str = "ml_forecasting/models",
) -> list[ForecastRecord]:
    all_records: list[ForecastRecord] = []
    for target_col in TARGET_COLUMNS:
        series = history_df[target_col]
        all_records.extend(predict_target(target_col, series, horizon_hours, models_dir))
        all_records.extend(predict_naive_baseline(target_col, series, horizon_hours))
    return all_records


if __name__ == "__main__":
    import json
    import os

    from shared.smard_client import fetch_all_generation_history, fetch_price_history
    from ml_forecasting.features import build_base_dataframe

    print("Fetching recent SMARD history...")
    generation = fetch_all_generation_history(num_weeks=12)
    price_series = fetch_price_history(num_weeks=12)

    history_df = build_base_dataframe(generation, price_series)

    print("Generating day-ahead forecasts...")
    forecasts = generate_all_forecasts(history_df)

    os.makedirs("reports/ml_forecast", exist_ok=True)
    output = [r.model_dump(mode="json") for r in forecasts]
    with open("reports/ml_forecast/latest_forecast.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(forecasts)} forecast records.")