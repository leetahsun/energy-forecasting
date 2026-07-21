#Tests for ml_forecasting/predict

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ml_forecasting.features import build_base_dataframe, build_feature_matrix
from ml_forecasting.predict import (
    recursive_forecast,
    naive_baseline_forecast,
    predict_target,
    predict_naive_baseline,
    generate_all_forecasts,
)
from ml_forecasting.train import time_based_split, train_xgboost, save_model, TARGET_COLUMNS

HOUR_MS = 3_600_000
START_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 


def make_history_df(num_hours: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    hours = np.arange(num_hours)

    solar = np.clip(100 * np.sin((hours % 24) / 24 * math.pi), 0, None) + rng.normal(0, 2, num_hours)
    gas = 200 + rng.normal(0, 2, num_hours)
    price = 40 + 20 * np.sin(((hours % 24) - 6) / 24 * 2 * math.pi) + rng.normal(0, 1, num_hours)

    generation = {
        "solar": [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(solar)],
        "gas": [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(gas)],
    }
    price_series = [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(price)]

    return build_base_dataframe(generation, price_series)


def train_and_save(feature_df: pd.DataFrame, target_col: str, tmp_dir: str):
    train_df, _ = time_based_split(feature_df, test_fraction=0.2)
    model, feature_cols = train_xgboost(train_df, target_col)
    save_model(model, feature_cols, os.path.join(tmp_dir, f"{target_col}_xgboost.joblib"))
    return model, feature_cols


def test_recursive_forecast_raises_with_insufficient_history():
    history_df = make_history_df(num_hours=50)  # less than 168h needed
    series = history_df["price_eur_mwh"]

    with pytest.raises(ValueError, match="Need at least"):
        recursive_forecast("price_eur_mwh", series, model=None, feature_cols=[], horizon_hours=24)


def test_recursive_forecast_produces_one_prediction_per_hour_with_no_nan():
    history_df = make_history_df(num_hours=24 * 30)
    feature_df = build_feature_matrix(
        {"solar": [(START_TS + i * HOUR_MS, 50.0) for i in range(24 * 30)],
         "gas": [(START_TS + i * HOUR_MS, 200.0) for i in range(24 * 30)]},
        [(START_TS + i * HOUR_MS, 45.0) for i in range(24 * 30)],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        model, feature_cols = train_and_save(feature_df, "price_eur_mwh", tmp_dir)

        predictions = recursive_forecast(
            "price_eur_mwh", history_df["price_eur_mwh"], model, feature_cols, horizon_hours=24
        )

        assert len(predictions) == 24
        assert predictions.index[0] == history_df.index[-1] + pd.Timedelta(hours=1)
        assert all(math.isfinite(v) for v in predictions.values)


def test_recursive_forecast_actually_feeds_predictions_forward():
    history_df = make_history_df(num_hours=24 * 30)
    feature_df = build_feature_matrix(
        {"solar": [(START_TS + i * HOUR_MS, 50.0) for i in range(24 * 30)],
         "gas": [(START_TS + i * HOUR_MS, 200.0) for i in range(24 * 30)]},
        [(START_TS + i * HOUR_MS, 45.0) for i in range(24 * 30)],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        model, feature_cols = train_and_save(feature_df, "price_eur_mwh", tmp_dir)

        # horizon of 3 is enough to prove hour 2 and 3 (which need hour 1's
        # and hour 2's predictions respectively, for the rolling feature)
        # resolve without error
        predictions = recursive_forecast(
            "price_eur_mwh", history_df["price_eur_mwh"], model, feature_cols, horizon_hours=3
        )
        assert len(predictions) == 3


def test_naive_baseline_forecast_uses_value_from_168h_earlier():
    history_df = make_history_df(num_hours=24 * 20)
    series = history_df["price_eur_mwh"]

    predictions = naive_baseline_forecast("price_eur_mwh", series, horizon_hours=24)

    assert len(predictions) == 24
    for ts, val in predictions.items():
        source_ts = ts - pd.Timedelta(hours=168)
        assert val == pytest.approx(series.loc[source_ts])


def test_naive_baseline_forecast_raises_if_not_enough_history():
    history_df = make_history_df(num_hours=50)  # < 168h
    series = history_df["price_eur_mwh"]

    with pytest.raises(ValueError, match="168h before"):
        naive_baseline_forecast("price_eur_mwh", series, horizon_hours=24)


def test_predict_target_returns_forecast_records():
    history_df = make_history_df(num_hours=24 * 30)
    feature_df = build_feature_matrix(
        {"solar": [(START_TS + i * HOUR_MS, 50.0) for i in range(24 * 30)],
         "gas": [(START_TS + i * HOUR_MS, 200.0) for i in range(24 * 30)]},
        [(START_TS + i * HOUR_MS, 45.0) for i in range(24 * 30)],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_and_save(feature_df, "price_eur_mwh", tmp_dir)

        records = predict_target(
            "price_eur_mwh", history_df["price_eur_mwh"], horizon_hours=24, models_dir=tmp_dir
        )

        assert len(records) == 24
        assert all(r.model_name == "xgboost_v1" for r in records)
        assert all(r.actual_value is None for r in records)


def test_predict_naive_baseline_returns_forecast_records():
    history_df = make_history_df(num_hours=24 * 20)

    records = predict_naive_baseline("price_eur_mwh", history_df["price_eur_mwh"], horizon_hours=24)

    assert len(records) == 24
    assert all(r.model_name == "naive_baseline" for r in records)