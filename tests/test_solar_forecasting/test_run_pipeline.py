"""Tests for solar_forecasting/run_pipeline.py.

Only build_future_solar_features is tested here -- it's the one piece of
run_pipeline.py that's pure logic (no live network calls). main() itself
requires real SMARD + Open-Meteo access and is exercised manually /
via the scheduled GitHub Actions workflow instead.
"""

import pandas as pd
import pytest

from solar_forecasting.ml_model import build_solar_feature_matrix, TARGET_COLUMN
from solar_forecasting.run_pipeline import build_future_solar_features

HOUR_MS = 3_600_000
START_TS = 1_718_928_000_000  # 2024-06-21T00:00:00 UTC


def make_history_df(num_hours: int = 24 * 10):
    """Reuse the same synthetic pattern as test_ml_model.py."""
    import math
    import numpy as np

    rng = np.random.default_rng(5)
    hours = np.arange(num_hours)
    hour_of_day = hours % 24
    daylight = np.clip(np.sin((hour_of_day - 6) / 12 * math.pi), 0, None)
    cloud_cover = rng.uniform(0, 100, num_hours)
    shortwave = daylight * 800 * (1 - 0.6 * cloud_cover / 100)
    temperature = 15 + 10 * daylight
    solar_mw = shortwave * 20

    times_iso = [
        pd.Timestamp(START_TS + i * HOUR_MS, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M")
        for i in range(num_hours)
    ]
    weather_hourly = {
        "time": times_iso,
        "shortwave_radiation": shortwave.tolist(),
        "cloud_cover": cloud_cover.tolist(),
        "temperature_2m": temperature.tolist(),
    }
    solar_series = [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(solar_mw)]

    return build_solar_feature_matrix(solar_series, weather_hourly)


def make_forecast_weather(history_df, forecast_hours: int = 24):
    """Weather 'forecast' that starts a bit BEFORE the end of history
    (simulating Open-Meteo's overlapping 'today' hours) and extends
    forecast_hours past it -- so build_future_solar_features has to
    correctly filter out the overlapping portion.
    """
    overlap_hours = 3
    last_hist_ts_ms = int(history_df.index.max().timestamp() * 1000)
    start_ts_ms = last_hist_ts_ms - overlap_hours * HOUR_MS
    # +1 because the overlap window itself includes last_hist_ts_ms as one
    # of its points (start_ts_ms + overlap_hours*HOUR_MS == last_hist_ts_ms),
    # so forecast_hours genuinely-future points need one more beyond that
    total_hours = overlap_hours + forecast_hours + 1

    times_iso = [
        pd.Timestamp(start_ts_ms + i * HOUR_MS, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M")
        for i in range(total_hours)
    ]
    return {
        "time": times_iso,
        "shortwave_radiation": [300.0] * total_hours,
        "cloud_cover": [40.0] * total_hours,
        "temperature_2m": [18.0] * total_hours,
    }


def test_build_future_solar_features_excludes_overlapping_history_hours():
    history_df = make_history_df()
    forecast_weather = make_forecast_weather(history_df, forecast_hours=24)

    future_df = build_future_solar_features(history_df, forecast_weather)

    # every row must be strictly after the end of history
    assert (future_df.index > history_df.index.max()).all()
    assert len(future_df) == 24  # overlapping hours excluded


def test_build_future_solar_features_has_no_nan():
    history_df = make_history_df()
    forecast_weather = make_forecast_weather(history_df, forecast_hours=24)

    future_df = build_future_solar_features(history_df, forecast_weather)

    assert not future_df.isna().any().any()


def test_build_future_solar_features_lag_24h_matches_real_history():
    history_df = make_history_df()
    forecast_weather = make_forecast_weather(history_df, forecast_hours=24)

    future_df = build_future_solar_features(history_df, forecast_weather)

    # for the first forecast hour, lag_24h should equal the actual
    # historical value from exactly 24h earlier
    first_future_ts = future_df.index[0]
    expected = history_df[TARGET_COLUMN].loc[first_future_ts - pd.Timedelta(hours=24)]
    assert future_df[f"{TARGET_COLUMN}_lag_24h"].iloc[0] == pytest.approx(expected)


def test_build_future_solar_features_raises_with_insufficient_history():

    import math
    import numpy as np

    num_hours = 10
    rng = np.random.default_rng(5)
    hours = np.arange(num_hours)
    hour_of_day = hours % 24
    daylight = np.clip(np.sin((hour_of_day - 6) / 12 * math.pi), 0, None)
    shortwave = daylight * 800
    times_iso = [
        pd.Timestamp(START_TS + i * HOUR_MS, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M")
        for i in range(num_hours)
    ]
    weather_hourly = {
        "time": times_iso,
        "shortwave_radiation": shortwave.tolist(),
        "cloud_cover": [40.0] * num_hours,
        "temperature_2m": [18.0] * num_hours,
    }
    solar_series = [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(shortwave * 20)]

    history_df = build_solar_feature_matrix(solar_series, weather_hourly, drop_na=False)
    forecast_weather = make_forecast_weather(history_df, forecast_hours=24)

    with pytest.raises(ValueError, match="doesn't cover the lag window"):
        build_future_solar_features(history_df, forecast_weather)