
import requests

#SMARD is an abbreviation of the German term "Strommarktdaten", which translates to electricity market data. 
#Data that is published on the SMARD website gives an up-to-date overview of what is happening on the electricity market. 
BASE = "https://www.smard.de/app/chart_data"

REGION = "DE"
RESOLUTION = "hour"

GENERATION_FILTERS = {
    "wind_onshore": 4067,
    "wind_offshore": 1225,
    "solar": 4068,
    "lignite": 1223,
    "hard_coal": 4069,
    "gas": 4071,
    "hydro": 1226,
    "biomass": 4066,
}

PRICE_FILTER = 4169  # DE/LU day-ahead price, EUR/MWh

RENEWABLE_SOURCES = {"wind_onshore", "wind_offshore", "solar", "hydro", "biomass"}

def _trim_trailing_unreported(series: list[tuple[int, float | None]]) -> list[tuple[int, float]]:
    """SMARD returns explicit pairs for hours that haven't
    been reported yet. Trim trailing None pairs so callers only see fully
    reported data. None values in the middle of a series are left as-is
    a genuine gap, not a reporting-lag artifact).
    """
    last_valid = -1
    for i in range(len(series) - 1, -1, -1):
        if series[i][1] is not None:
            last_valid = i
            break
    return series[: last_valid + 1]

def get_index_timestamps(filter_id: int) -> list[int]:
    """Get all available weekly bucket timestamps for each filter, starting with the oldest first. -> not
    able to get data between X and Y date directly"""
    url = f"{BASE}/{filter_id}/{REGION}/index_{RESOLUTION}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["timestamps"]


def get_series(filter_id: int, timestamp: int) -> list[tuple[int, float]]:
    """Fetch one weekly bucket's time series for a filter
    Each pair  here is one hourly data point which  is a timestamp and a value"""
    url = f"{BASE}/{filter_id}/{REGION}/{filter_id}_{REGION}_{RESOLUTION}_{timestamp}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["series"]


def get_reference_timestamps(num_weeks: int, anchor_filter: int = GENERATION_FILTERS["wind_onshore"]) -> list[int]:
    """Get the last [N]  weekly bucket timestamps, anchored to one reliable
    filter. Reusing these same timestamps for every other filter/source
    keeps all series aligned to the same calendar window, avoiding the
    per-source drift issue that was encountered earlier where different generation types had different time windows.
    """
    all_ts = get_index_timestamps(anchor_filter)
    return all_ts[-num_weeks:]


def fetch_history(filter_id: int, num_weeks: int = 8) -> list[tuple[int, float]]:
    #currently only 8 weeks 
    """Fetch the last N weekly buckets for based on the filter and loop over the specified time and
    return the combined data for that period.
    """
    timestamps = get_reference_timestamps(num_weeks)
    combined: list[tuple[int, float]] = []
    for ts in timestamps:
        combined.extend(get_series(filter_id, ts))
    return _trim_trailing_unreported(combined)


def fetch_all_generation_history(num_weeks: int = 8) -> dict[str, list[tuple[int, float]]]:
    """Fetch aligned multi week history for every generation source."""
    timestamps = get_reference_timestamps(num_weeks)
    data = {}
    for name, fid in GENERATION_FILTERS.items():
        combined: list[tuple[int, float]] = []
        for ts in timestamps:
            combined.extend(get_series(fid, ts))
        data[name] = _trim_trailing_unreported(combined)
    return data


def fetch_price_history(num_weeks: int = 8) -> list[tuple[int, float]]:
    """Fetch aligned multi-week day-ahead price history."""
    return fetch_history(PRICE_FILTER, num_weeks=num_weeks)