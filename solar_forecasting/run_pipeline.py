""" solar forecasting pipeline.

Fetches historical solar generation (SMARD) and weather (Open-Meteo),
trains the ML model, evaluates it against the physical clear-sky model on
held-out data, then generates a forward forecast from both approaches
using the weather forecast.

"""

import datetime
import json
import os

import pandas as pd

from shared.smard_client import GENERATION_FILTERS, fetch_history
from solar_forecasting.clearsky_model import (
    calibrate_scale_factor,
    estimate_irradiance_series,
    predict_generation,
)
from solar_forecasting.evaluate import (
    _index_to_epoch_ms,
    compare_models,
    format_comparison_summary,
)
from solar_forecasting.ml_model import (
    DEFAULT_LAGS,
    TARGET_COLUMN,
    WEATHER_FEATURE_COLUMNS,
    build_solar_feature_matrix,
    predict_solar,
    time_based_split,
    train_solar_model,
)
from solar_forecasting.weather import (
    DEFAULT_LAT,
    DEFAULT_LON,
    fetch_forecast_weather,
    fetch_historical_weather,
    weather_to_series,
)


def build_future_solar_features(
    history_df: pd.DataFrame,
    forecast_weather_hourly: dict,
) -> pd.DataFrame:
    """Build feature rows for the forecast horizon from the weather
    forecast,.
    """
    weather_dfs = []
    for var in WEATHER_FEATURE_COLUMNS:
        series = weather_to_series(forecast_weather_hourly, var)
        df = pd.DataFrame(series, columns=["timestamp_ms", var])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        weather_dfs.append(df.set_index("timestamp").drop(columns="timestamp_ms"))
    weather_df = pd.concat(weather_dfs, axis=1)

    # The forecast endpoint returns some hours that overlap with "today"
    future_df = weather_df[weather_df.index > history_df.index.max()].copy()

    future_df["hour"] = future_df.index.hour
    future_df["month"] = future_df.index.month

    for lag in DEFAULT_LAGS:
        lag_timestamps = future_df.index - pd.Timedelta(hours=lag)
        future_df[f"{TARGET_COLUMN}_lag_{lag}h"] = (
            history_df[TARGET_COLUMN].reindex(lag_timestamps).values
        )

    if future_df.isna().any().any():
        raise ValueError(
            "Future feature rows contain NaN:  history_df likely doesn't "
            "cover the lag window needed (need at least "
            f"{max(DEFAULT_LAGS)}h of prior history), or the weather "
            "forecast doesn't align with the expected future timestamps."
        )

    return future_df


def main(
    num_weeks_history: int = 8,
    forecast_days: int = 2,
    out_dir: str | None = None,
) -> str:
    if out_dir is None:
        out_dir = f"reports/solar_forecast/{datetime.date.today().isoformat()}"
    os.makedirs(out_dir, exist_ok=True)

    print("Fetching historical solar generation from SMARD...")
    solar_history = fetch_history(GENERATION_FILTERS["solar"], num_weeks=num_weeks_history)
    start_date = pd.Timestamp(solar_history[0][0], unit="ms", tz="UTC").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(solar_history[-1][0], unit="ms", tz="UTC").strftime("%Y-%m-%d")

    print(f"Fetching historical weather ({start_date} to {end_date})...")
    historical_weather = fetch_historical_weather(start_date=start_date, end_date=end_date)

    print("Building feature matrix...")
    df = build_solar_feature_matrix(solar_history, historical_weather)
    train_df, test_df = time_based_split(df, test_fraction=0.2)

    print(f"Training ML model on {len(train_df)} rows...")
    model, feature_cols = train_solar_model(train_df)

    print("Evaluating physical vs. ML model on held-out test set...")
    comparison = compare_models(
        train_df, test_df, model, feature_cols, lat=DEFAULT_LAT, lon=DEFAULT_LON
    )
    print(format_comparison_summary(comparison))

    print(f"Fetching {forecast_days}-day weather forecast:")
    forecast_weather = fetch_forecast_weather(forecast_days=forecast_days)

    print("Building forecast feature rows:")
    future_df = build_future_solar_features(df, forecast_weather)

    print("Generating ML forecast:")
    ml_forecast = predict_solar(model, feature_cols, future_df)

    print("Generating physical forecast:")

    all_ts_ms = _index_to_epoch_ms(df.index)
    all_irradiance = estimate_irradiance_series(
        all_ts_ms, df["cloud_cover"].tolist(), lat=DEFAULT_LAT, lon=DEFAULT_LON
    )
    scale_factor = calibrate_scale_factor(all_irradiance, df[TARGET_COLUMN].tolist())

    future_ts_ms = _index_to_epoch_ms(future_df.index)
    future_irradiance = estimate_irradiance_series(
        future_ts_ms, future_df["cloud_cover"].tolist(), lat=DEFAULT_LAT, lon=DEFAULT_LON
    )
    physical_forecast = [predict_generation(irr, scale_factor) for irr in future_irradiance]

    output = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation": comparison,
        "forecast": [
            {
                "timestamp": ts.isoformat(),
                "ml_forecast_mw": float(ml_val),
                "physical_forecast_mw": float(phys_val),
            }
            for ts, ml_val, phys_val in zip(future_df.index, ml_forecast, physical_forecast)
        ],
    }

    out_path = f"{out_dir}/forecast.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved forecast + evaluation to {out_path}")
    return out_path


if __name__ == "__main__":
    main()