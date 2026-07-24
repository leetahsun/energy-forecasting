"""End-to-end solar forecasting pipeline.

Fetches historical solar generation (SMARD) and weather (Open-Meteo),
trains the ML model, evaluates it against the physical clear-sky model on
held-out data, then generates a forward forecast from both approaches
using the weather forecast.

IMPORTANT: SMARD's solar generation data is metered/reported with a real
lag behind actual real-time "now" -- observed in practice to run to
several days, well beyond the model's 24h lag feature window. This means
the naive assumption "history ends at basically today, forecast starts
right after" is false. Instead, this pipeline forecasts hour-by-hour,
RECURSIVELY, starting immediately after the last genuinely known solar
generation value, feeding each hour's prediction back in so later hours'
lag_24h feature can reference it -- bridging gaps of any size, not just
gaps shorter than 24h. Hours between "last known real generation" and
actual real-time "now" are effectively a same-day nowcast filling in
not-yet-reported data; hours beyond real "now" are the genuine forecast.

This mirrors the recursive/autoregressive approach in
ml_forecasting/predict.py, which solves the same class of problem for a
different reason (there, hour-2-onward's rolling-window feature needs
hour-1's own prediction; here, the gap itself can exceed the lag window).
"""

import datetime
import json
import math
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
    time_based_split,
    train_solar_model,
)
from solar_forecasting.report import build_report
from solar_forecasting.weather import (
    DEFAULT_LAT,
    DEFAULT_LON,
    fetch_forecast_weather,
    fetch_historical_weather,
    weather_to_series,
)


def _weather_hourly_to_df(weather_hourly: dict) -> pd.DataFrame:
    weather_dfs = []
    for var in WEATHER_FEATURE_COLUMNS:
        series = weather_to_series(weather_hourly, var)
        df = pd.DataFrame(series, columns=["timestamp_ms", var])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        weather_dfs.append(df.set_index("timestamp").drop(columns="timestamp_ms"))
    return pd.concat(weather_dfs, axis=1)


def recursive_solar_forecast(
    history_df: pd.DataFrame,
    weather_hourly: dict,
    model,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Predict solar generation hour-by-hour for every timestamp in
    weather_hourly that comes after history_df's last known real
    observation, feeding each hour's prediction back into a working
    series so later hours' lag_24h feature can reference it.

    Returns a DataFrame indexed by timestamp with the weather columns
    plus the predicted TARGET_COLUMN, covering both the reporting-lag
    gap (a nowcast) and the genuine forward forecast.
    """
    weather_df = _weather_hourly_to_df(weather_hourly)
    future_weather = weather_df[weather_df.index > history_df.index.max()].sort_index()

    if future_weather.empty:
        raise ValueError(
            "No weather timestamps found after history_df's last known "
            "observation -- check that fetch_forecast_weather's past_days "
            "covers the gap between last known generation and today."
        )

    working = history_df[TARGET_COLUMN].copy()
    predictions: dict[pd.Timestamp, float] = {}

    for ts, weather_row in future_weather.iterrows():
        row = {var: weather_row[var] for var in WEATHER_FEATURE_COLUMNS}
        row["hour"] = ts.hour
        row["month"] = ts.month

        for lag in DEFAULT_LAGS:
            lag_ts = ts - pd.Timedelta(hours=lag)
            if lag_ts not in working.index:
                raise ValueError(
                    f"Missing lag_{lag}h value for {ts} (needs {lag_ts}) -- "
                    "the gap between last known generation and this "
                    "timestamp exceeds the available weather/history "
                    "coverage. Increase past_days when fetching the "
                    "weather forecast."
                )
            row[f"{TARGET_COLUMN}_lag_{lag}h"] = working.loc[lag_ts]

        feature_row = pd.DataFrame([row], index=[ts])[feature_cols]
        pred = float(model.predict(feature_row)[0])
        predictions[ts] = pred
        working.loc[ts] = pred  # feed forward so later hours can use it

    result = future_weather.copy()
    result[TARGET_COLUMN] = pd.Series(predictions)
    return result


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
    last_known_ts = pd.Timestamp(solar_history[-1][0], unit="ms", tz="UTC")
    end_date = last_known_ts.strftime("%Y-%m-%d")

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

    # The gap between "last known real generation" and "today" can exceed
    # the model's 24h lag window (observed to run to several days in
    # practice) -- fetch enough past_days of weather to cover that whole
    # gap, not just the forward forecast, so recursive_solar_forecast has
    # a continuous timestamp range to work with.
    now = pd.Timestamp.now(tz="UTC")
    gap_days = max(0, math.ceil((now - last_known_ts).total_seconds() / 86400)) + 1
    past_days = min(gap_days, 92)  # Open-Meteo's documented limit
    print(f"Fetching weather (past_days={past_days}, forecast_days={forecast_days})...")
    forecast_weather = fetch_forecast_weather(forecast_days=forecast_days, past_days=past_days)

    print("Generating ML forecast (recursive, bridging the reporting-lag gap)...")
    future_result = recursive_solar_forecast(df, forecast_weather, model, feature_cols)

    print("Generating physical forecast for the same timestamps...")
    future_ts_ms = _index_to_epoch_ms(future_result.index)
    future_irradiance = estimate_irradiance_series(
        future_ts_ms, future_result["cloud_cover"].tolist(), lat=DEFAULT_LAT, lon=DEFAULT_LON
    )
    all_ts_ms = _index_to_epoch_ms(df.index)
    all_irradiance = estimate_irradiance_series(
        all_ts_ms, df["cloud_cover"].tolist(), lat=DEFAULT_LAT, lon=DEFAULT_LON
    )
    scale_factor = calibrate_scale_factor(all_irradiance, df[TARGET_COLUMN].tolist())
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
            for ts, ml_val, phys_val in zip(
                future_result.index, future_result[TARGET_COLUMN], physical_forecast
            )
        ],
    }

    out_path = f"{out_dir}/forecast.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved forecast + evaluation to {out_path}")

    html_path = f"{out_dir}/forecast.html"
    build_report(output, html_path)
    print(f"Saved dashboard to {html_path}")

    return out_path


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()