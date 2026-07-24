"""Weather-driven ML model for solar generation.

Unlike clearsky_model.py
"""

import pandas as pd
from xgboost import XGBRegressor

from solar_forecasting.weather import weather_to_series

TARGET_COLUMN = "solar_mw"
WEATHER_FEATURE_COLUMNS = ["shortwave_radiation", "cloud_cover", "temperature_2m"]

DEFAULT_LAGS = [24]  # only 1 day backweatherdriven solar output doesn't
                    # have the same weekly seasonality as price/demand does


def _series_to_df(series: list[tuple[int, float]], column_name: str) -> pd.DataFrame:
    df = pd.DataFrame(series, columns=["timestamp_ms", column_name])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df.set_index("timestamp").drop(columns="timestamp_ms")


def build_solar_feature_matrix(
    solar_generation_series: list[tuple[int, float]],
    weather_hourly: dict,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Combine actual solar generation (from SMARD) with weather data
    (from Open-Meteo) into one aligned feature matrix.
    """
    gen_df = _series_to_df(solar_generation_series, TARGET_COLUMN)

    weather_dfs = [
        _series_to_df(weather_to_series(weather_hourly, var), var)
        for var in WEATHER_FEATURE_COLUMNS
    ]
    weather_df = pd.concat(weather_dfs, axis=1)

    combined = gen_df.join(weather_df, how="inner").sort_index()

    combined["hour"] = combined.index.hour
    combined["month"] = combined.index.month

    # A short lag on the target itself helps the model account for
    # persistence effects on top of the weather signal.
    for lag in DEFAULT_LAGS:
        combined[f"{TARGET_COLUMN}_lag_{lag}h"] = combined[TARGET_COLUMN].shift(lag)

    if drop_na:
        combined = combined.dropna()

    return combined


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COLUMN]


def time_based_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split 
    """
    df = df.sort_index()
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_solar_model(train_df: pd.DataFrame, **xgb_params) -> tuple[XGBRegressor, list[str]]:
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols]
    y = train_df[TARGET_COLUMN]

    default_params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    default_params.update(xgb_params)

    model = XGBRegressor(**default_params)
    model.fit(X, y)
    return model, feature_cols


def predict_solar(model: XGBRegressor, feature_cols: list[str], df: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(df[feature_cols]), index=df.index)