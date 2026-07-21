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
    """Fetch historical hourly weather.
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
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:
    """Fetch upcoming hourly weather. Returns the same 'hourly' dict shape as
    fetch_historical_weather, so both can be processed identically.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["hourly"]


def weather_to_series(hourly: dict, variable: str) -> list[tuple[int, float]]:
    """Convert Open-Meteo's hourly dict format into the samemshape used throughout the rest of the project.
    """
    import pandas as pd

    timestamps = pd.to_datetime(hourly["time"], utc=True)
    ts_ms = (
        (timestamps - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    ).tolist()
    values = hourly[variable]
    return list(zip(ts_ms, values))