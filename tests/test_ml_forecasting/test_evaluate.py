#Tests for ml_forecasting/evaluate.py.

import math

import numpy as np
import pandas as pd
import pytest

from ml_forecasting.evaluate import (
    compute_mae,
    compute_rmse,
    evaluate_target,
    evaluate_all,
    format_evaluation_summary,
)
from ml_forecasting.features import build_feature_matrix
from ml_forecasting.train import time_based_split, train_xgboost, train_all_targets

HOUR_MS = 3_600_000
START_TS = 1_704_067_200_000


def make_synthetic_feature_df(num_hours: int = 24 * 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    hours = np.arange(num_hours)

    solar = np.clip(100 * np.sin((hours % 24) / 24 * math.pi), 0, None) + rng.normal(0, 2, num_hours)
    gas = 200 + rng.normal(0, 2, num_hours)
    price = 40 + 20 * np.sin(((hours % 24) - 6) / 24 * 2 * math.pi) + rng.normal(0, 1, num_hours)

    generation = {
        "solar": [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(solar)],
        "gas": [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(gas)],
    }
    price_series = [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(price)]

    return build_feature_matrix(generation, price_series)


def test_compute_mae_known_values():
    predictions = pd.Series([10.0, 20.0, 30.0])
    actuals = pd.Series([12.0, 18.0, 33.0])
    # errors: 2, 2, 3 -> mean = 7/3
    assert compute_mae(predictions, actuals) == pytest.approx(7 / 3)


def test_compute_rmse_known_values():
    predictions = pd.Series([10.0, 20.0])
    actuals = pd.Series([12.0, 16.0])
    # errors: -2, 4 -> squared: 4, 16 -> mean 10 -> sqrt(10)
    assert compute_rmse(predictions, actuals) == pytest.approx(math.sqrt(10))


def test_rmse_greater_than_or_equal_to_mae():
    """RMSE >= MAE always, since squaring
    disproportionately penalizes larger errors. goof sanity check
    """
    rng = np.random.default_rng(1)
    predictions = pd.Series(rng.normal(50, 10, 100))
    actuals = pd.Series(rng.normal(50, 10, 100))

    mae = compute_mae(predictions, actuals)
    rmse = compute_rmse(predictions, actuals)

    assert rmse >= mae


def test_evaluate_target_beats_baseline_on_learnable_pattern():
    df = make_synthetic_feature_df()
    train_df, test_df = time_based_split(df, test_fraction=0.2)
    model, feature_cols = train_xgboost(train_df, "price_eur_mwh")

    result = evaluate_target("price_eur_mwh", test_df, model, feature_cols)

    assert result["target"] == "price_eur_mwh"
    assert result["test_set_size"] == len(test_df)
    assert result["xgboost_beats_baseline"] is True
    assert result["xgboost_improvement_pct"] > 0
    assert result["xgboost"]["mae"] < result["naive_baseline"]["mae"]


def test_evaluate_target_handles_zero_baseline_mae_gracefully():
    """If the naive baseline happens to be perfect (MAE=0), improvement_pct
    would be a divide-by-zero should return None instead of crashing.
    """
    # constant series: baseline (lag_168h of a constant) will exactly
    # match actual, giving baseline_mae == 0
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    test_df = pd.DataFrame({
        "price_eur_mwh": [50.0] * 10,
        "price_eur_mwh_lag_168h": [50.0] * 10,
        "some_feature": range(10),
    }, index=idx)

    class DummyModel:
        def predict(self, X):
            return np.array([50.0] * len(X))

    result = evaluate_target("price_eur_mwh", test_df, DummyModel(), ["some_feature"])

    assert result["naive_baseline"]["mae"] == 0.0
    assert result["xgboost_improvement_pct"] is None


def test_evaluate_all_covers_every_target():
    df = make_synthetic_feature_df()
    train_results = train_all_targets(df, models_dir="/tmp/test_evaluate_models")

    evaluation = evaluate_all(train_results)

    assert set(evaluation.keys()) == {"renewable_share_pct", "price_eur_mwh"}
    for target_col, result in evaluation.items():
        assert result["target"] == target_col


def test_format_evaluation_summary_is_readable_string():
    df = make_synthetic_feature_df()
    train_df, test_df = time_based_split(df, test_fraction=0.2)
    model, feature_cols = train_xgboost(train_df, "price_eur_mwh")
    result = evaluate_target("price_eur_mwh", test_df, model, feature_cols)

    summary = format_evaluation_summary({"price_eur_mwh": result})

    assert "price_eur_mwh" in summary
    assert "BEATS" in summary or "DOES NOT BEAT" in summary