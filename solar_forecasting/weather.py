"""Open-Meteo weather client for the solar forecasting module.

Open-Meteo is free and requires no API key for non-commercial use. Two
separate endpoints are used:
  - archive-api.open-meteo.com -- historical weather, for training data
  - api.open-meteo.com -- forecast weather, for actual day-ahead predictions
"""

from shared.http_retry import get_with_retry

# Roughly central Germany near Kassel which is a rough geographic proxy for
# national-average conditions.
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
    """Fetch historical hourly weather
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    resp = get_with_retry(ARCHIVE_URL, params=params, timeout=30)
    return resp.json()["hourly"]


def fetch_forecast_weather(
    forecast_days: int = 2,
    past_days: int = 0,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:
    """Fetch hourly weather covering `past_days` of recent history plus
    `forecast_days` ahead in one request.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": "UTC",
    }
    resp = get_with_retry(FORECAST_URL, params=params, timeout=30)
    return resp.json()["hourly"]


def weather_to_series(hourly: dict, variable: str) -> list[tuple[int, float]]:
    """Convert Open-Meteo's hourly dict format into the same shape used throughout the rest of this
    project so solar forecasting can reuse the same feature building patterns as
    ml_forecasting.
    """
    import pandas as pd

    timestamps = pd.to_datetime(hourly["time"], utc=True)
    ts_ms = (
        (timestamps - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    ).tolist()
    values = hourly[variable]
    return list(zip(ts_ms, values))