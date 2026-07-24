"""Physical clear-sky solar irradiance model.

Unlike ml_model.py (which learns patterns from data), this module encodes
actual domain knowledge: where the sun is in the sky at a given time and
location determines how much solar radiation is theoretically available,
independent of any historical pattern-matching.

Two stages:
1. Solar position -> clear-sky irradiance (pure astronomy/physics)
2. Cloud cover -> attenuation of that clear-sky irradiance (empirical
approximation, since real atmospheric radiative transfer is far more
complex than this project needs)
"""

import math
from datetime import datetime, timezone

import numpy as np

SOLAR_CONSTANT_W_M2 = 1361.0  # extraterrestrial solar irradiance
EXTINCTION_COEFFICIENT = 0.6   # rough atmospheric attenuation in a clearsky and at sea level-ish
CLOUD_ATTENUATION_FACTOR = 0.75  # fraction of clear-sky irradiance blocked at 100% cloud cover


def solar_elevation_angle(dt: datetime, lat: float, lon: float) -> float:
    """Compute solar elevation angle (degrees above horizon) for a given
    UTC datetime and location, using standard solar position formulas.
    """
    day_of_year = dt.timetuple().tm_yday

    # Solar declination (Cooper's equation)
    declination = 23.45 * math.sin(math.radians(360 / 365 * (284 + day_of_year)))

    # Approximate solar time and hour angle
    solar_time = dt.hour + dt.minute / 60 + lon / 15
    hour_angle = 15 * (solar_time - 12)

    lat_rad = math.radians(lat)
    decl_rad = math.radians(declination)
    hour_rad = math.radians(hour_angle)

    elevation_rad = math.asin(
        math.sin(lat_rad) * math.sin(decl_rad)
        + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_rad)
    )
    return math.degrees(elevation_rad)


def clear_sky_irradiance(dt: datetime, lat: float, lon: float) -> float:
    """Estimate clear sky global horizontal irradiance (W/m^2) using a
    simple exponential extinction approximation. then it returns 0 when the sun
    is below the horizon.
    """
    elevation = solar_elevation_angle(dt, lat, lon)
    if elevation <= 0:
        return 0.0

    elevation_rad = math.radians(elevation)
    # Simple extinction model: irradiance falls off exponentially with
    # air mass, which increases as the sun gets lower in the sky
    # (1 / sin(elevation) approximates air mass at low-to-moderate angles).
    air_mass_factor = 1 / math.sin(elevation_rad)
    irradiance = (
        SOLAR_CONSTANT_W_M2
        * math.sin(elevation_rad)
        * math.exp(-EXTINCTION_COEFFICIENT * air_mass_factor)
    )
    return max(0.0, irradiance)


def apply_cloud_attenuation(clear_sky_w_m2: float, cloud_cover_pct: float) -> float:
    """Reduce clear sky irradiance based on cloud cover (0-100%).
    """
    cloud_fraction = max(0.0, min(100.0, cloud_cover_pct)) / 100
    return clear_sky_w_m2 * (1 - CLOUD_ATTENUATION_FACTOR * cloud_fraction)


def estimate_irradiance_series(
    timestamps_ms: list[int],
    cloud_cover_values: list[float],
    lat: float = 51.3,
    lon: float = 9.5,
) -> list[float]:
    """Estimated cloud-adjusted irradiance for a list of timestamps,
    combining the physical clear-sky model with observed/forecast cloud
    cover.
    """
    results = []
    for ts_ms, cloud_pct in zip(timestamps_ms, cloud_cover_values):
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        clear_sky = clear_sky_irradiance(dt, lat, lon)
        adjusted = apply_cloud_attenuation(clear_sky, cloud_pct)
        results.append(adjusted)
    return results


def calibrate_scale_factor(
    irradiance_w_m2: list[float], actual_generation_mw: list[float]
) -> float:
    """Fit a single linear scale factor (MW per W/m^2) mapping estimated
    irradiance to actual solar generation."""
    irr = np.array(irradiance_w_m2, dtype=float)
    gen = np.array(actual_generation_mw, dtype=float)

    denominator = np.dot(irr, irr)
    if denominator == 0:
        raise ValueError("Cannot calibrate: irradiance values are all zero.")

    return float(np.dot(irr, gen) / denominator)


def predict_generation(irradiance_w_m2: float, scale_factor: float) -> float:
    """Convert estimated irradiance into predicted generation (MW),
    using the calibrated scale factor.
    """
    return max(0.0, irradiance_w_m2 * scale_factor)