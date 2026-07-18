import pandas as pd

from shared.smard_client import RENEWABLE_SOURCES

# Lags chosen to capture daily and weekly seasonality,
DEFAULT_LAGS = [24, 168] #168 hours is a week 
DEFAULT_ROLLING_WINDOWS = [24]


def _series_to_df(series: list[tuple[int, float]], column_name: str) -> pd.DataFrame:
    """Converts the raw series into the a Dataframe
    """
    df = pd.DataFrame(series, columns=["timestamp_ms", column_name])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp").drop(columns="timestamp_ms")
    return df


def build_base_dataframe(
    generation: dict[str, list[tuple[int, float]]],
    price_series: list[tuple[int, float]],
) -> pd.DataFrame:
    """Combine each generation by source series and price series into one
    DataFrame
    """
    gen_dfs = [_series_to_df(series, name) for name, series in generation.items()]
    gen_df = pd.concat(gen_dfs, axis=1)

    renewable_cols = [c for c in gen_df.columns if c in RENEWABLE_SOURCES]
    gen_df["total_generation"] = gen_df.sum(axis=1)
    gen_df["renewable_generation"] = gen_df[renewable_cols].sum(axis=1)
    gen_df["renewable_share_pct"] = (
        gen_df["renewable_generation"] / gen_df["total_generation"] * 100
    )

    price_df = _series_to_df(price_series, "price_eur_mwh")

    combined = gen_df[["renewable_share_pct"]].join(price_df, how="inner")
    return combined.sort_index()


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df.index.month
    return df


def add_lag_features(
    df: pd.DataFrame, columns: list[str], lags: list[int] = None
) -> pd.DataFrame:
    """Add the backdated(lag) versions of the given columns in this case in a value of 24h and 168h
    ago. These will be NaN for the first max rows of lag. It won't drop them automatically so drop if used outside of build matrix function
    """
    lags = lags or DEFAULT_LAGS
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, 
    columns: list[str], 
    windows: list[int] = None
) -> pd.DataFrame:
    """Add rolling mean feature which are computed
    only from past values by shifting one value back before rolling) to avoid leaking the
    current row's own value into its own feature.
    """
    windows = windows or DEFAULT_ROLLING_WINDOWS
    df = df.copy()
    for col in columns:
        for window in windows:
            df[f"{col}_rolling_{window}h_mean"] = (
                df[col].shift(1).rolling(window=window).mean()
            )
    return df


def build_feature_matrix(
    generation: dict[str, list[tuple[int, float]]],
    price_series: list[tuple[int, float]],
    target_columns: list[str] = ("renewable_share_pct", "price_eur_mwh"),
    drop_na: bool = True,
) -> pd.DataFrame:
    """Returns a dataframe ready for train/test splitting.
    """
    df = build_base_dataframe(generation, price_series)
    df = add_calendar_features(df)
    df = add_lag_features(df, columns=list(target_columns))
    df = add_rolling_features(df, columns=list(target_columns))

    if drop_na:
        df = df.dropna()
    # dont fill w zero or average because model will learn on that
    return df