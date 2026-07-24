"""Open-Meteo weather client for the solar forecasting module.

Location: a single representative coordinate (central Germany) is used
for the whole country. This is a deliberate simplification SMARD's
solar generation figure is a national aggregate across thousands of
geographically distributed installations, so a single weather station is
an approximation of nationwide conditions, not a precise match. Cloud
cover in Munich and Hamburg can differ substantially on the same day.
"""

import requests

# Roughly central Germany a rough geographic proxy for national-average conditions even though that is not necessicarly precise.
DEFAULT_LAT = 51.3
DEFAULT_LON = 9.5

HOURLY_VARS = ["shortwave_radiation", "cloud_cover", "temperature_2m"]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_historical_weather(
    start_date: str,
    end_date: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:

    """Fetch historical hourly weather for [start_date, end_date]
    (YYYY-MM-DD strings, inclusive). Returns Open-Meteo's raw 'hourly'
    dict: {"time": [...], "shortwave_radiation": [...], ...}.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["hourly"] 

def fetch_forecast_weather(
    forecast_days: int = 2,
    past_days: int = 0,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:
    """Fetch hourly weather covering `past_days` of recent history plus
    `forecast_days` ahead, in one request.

    past_days matters beyond convenience: SMARD's actual solar generation
    reporting lags real-time by an observed multi-day margin (see
    run_pipeline.py's recursive_solar_forecast), so bridging the gap
    between "last known real generation" and "today" requires weather
    data for that whole gap, not just the forward forecast window.
    Open-Meteo documents support for past_days up to 92.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": "UTC",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["hourly"]


def weather_to_series(hourly: dict, variable: str) -> list[tuple[int, float]]:
    """Convert Open-Meteo's hourly dict format into the same
    [(timestamp_ms, value), ...] shape used throughout the rest of this
    project (matching shared/smard_client.py's series format), so solar
    forecasting can reuse the same feature-building patterns as
    ml_forecasting.
    """
    import pandas as pd

    timestamps = pd.to_datetime(hourly["time"], utc=True)
    ts_ms = (
        (timestamps - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    ).tolist()
    values = hourly[variable]
    return list(zip(ts_ms, values))
