"""Tests for ml_forecasting/features.py.

Uses small, hand-constructed synthetic series so expected values can be
computed by hand -- no network access or real SMARD data needed.
"""

import pytest

from ml_forecasting.features import (
    build_base_dataframe,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
)

HOUR_MS = 3_600_000
START_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC (a Monday)


def make_hourly_series(values: list[float], start_ts: int = START_TS) -> list[tuple[int, float]]:
    return [(start_ts + i * HOUR_MS, v) for i, v in enumerate(values)]


def test_build_base_dataframe_computes_renewable_share():
    # 3 hours: solar + gas, renewable share should be solar / (solar + gas) * 100
    generation = {
        "solar": make_hourly_series([100.0, 200.0, 0.0]),
        "gas": make_hourly_series([900.0, 800.0, 1000.0]),
    }
    price_series = make_hourly_series([40.0, 45.0, 50.0])

    df = build_base_dataframe(generation, price_series)

    assert len(df) == 3
    assert df["renewable_share_pct"].iloc[0] == pytest.approx(10.0)
    assert df["renewable_share_pct"].iloc[1] == pytest.approx(20.0)
    assert df["renewable_share_pct"].iloc[2] == pytest.approx(0.0)
    assert df["price_eur_mwh"].iloc[0] == pytest.approx(40.0)


def test_add_calendar_features_hour_and_weekend():
    generation = {"solar": make_hourly_series([100.0] * 48)}
    price_series = make_hourly_series([40.0] * 48)
    df = build_base_dataframe(generation, price_series)

    df = add_calendar_features(df)

    # START_TS is 2024-01-01 00:00 UTC, a Monday
    assert df["hour"].iloc[0] == 0
    assert df["day_of_week"].iloc[0] == 0  # Monday
    assert df["is_weekend"].iloc[0] == 0
    # 24 hours later is still Monday->Tuesday boundary at hour 0
    assert df["hour"].iloc[24] == 0
    assert df["day_of_week"].iloc[24] == 1  # Tuesday


def test_add_lag_features_shifts_correctly():
    generation = {"solar": make_hourly_series([float(i) for i in range(30)])}
    price_series = make_hourly_series([float(i) for i in range(30)])
    df = build_base_dataframe(generation, price_series)

    df = add_lag_features(df, columns=["price_eur_mwh"], lags=[1, 24])

    # lag_1 at row index 5 should equal the raw value at index 4
    assert df["price_eur_mwh_lag_1h"].iloc[5] == pytest.approx(df["price_eur_mwh"].iloc[4])
    # lag_24 at row index 25 should equal the raw value at index 1
    assert df["price_eur_mwh_lag_24h"].iloc[25] == pytest.approx(df["price_eur_mwh"].iloc[1])
    # rows before the lag window has enough history should be NaN
    assert df["price_eur_mwh_lag_24h"].iloc[0] != df["price_eur_mwh_lag_24h"].iloc[0]  # NaN check


def test_add_rolling_features_excludes_current_row():
    # values 0..9, rolling mean of window=3 shifted by 1 at row index 4
    # should be mean of rows 1,2,3 (NOT including row 4 itself)
    generation = {"solar": make_hourly_series([float(i) for i in range(10)])}
    price_series = make_hourly_series([float(i) for i in range(10)])
    df = build_base_dataframe(generation, price_series)

    df = add_rolling_features(df, columns=["price_eur_mwh"], windows=[3])

    expected = (1.0 + 2.0 + 3.0) / 3
    assert df["price_eur_mwh_rolling_3h_mean"].iloc[4] == pytest.approx(expected)


def test_build_feature_matrix_drops_na_and_has_expected_columns():
    generation = {
        "solar": make_hourly_series([float(i % 24) for i in range(200)]),
        "gas": make_hourly_series([50.0] * 200),
    }
    price_series = make_hourly_series([30.0 + (i % 10) for i in range(200)])

    df = build_feature_matrix(generation, price_series)

    # no NaNs should remain after drop_na
    assert not df.isna().any().any()
    # expected engineered columns exist
    assert "hour" in df.columns
    assert "is_weekend" in df.columns
    assert "renewable_share_pct_lag_24h" in df.columns
    assert "price_eur_mwh_lag_168h" in df.columns
    assert "price_eur_mwh_rolling_24h_mean" in df.columns
    # should have fewer rows than input due to the 168h lag dropping early rows
    assert len(df) == 200 - 168