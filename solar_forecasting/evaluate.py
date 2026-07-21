"""Evaluation for the solar forecasting module.
"""

import numpy as np
import pandas as pd

from solar_forecasting.clearsky_model import (
    estimate_irradiance_series,
    calibrate_scale_factor,
    predict_generation,
)
from solar_forecasting.ml_model import TARGET_COLUMN, predict_solar


def compute_mae(predictions: pd.Series, actuals: pd.Series) -> float:
    return float((predictions - actuals).abs().mean())


def compute_rmse(predictions: pd.Series, actuals: pd.Series) -> float:
    return float(np.sqrt(((predictions - actuals) ** 2).mean()))


def _index_to_epoch_ms(index: pd.DatetimeIndex) -> list[int]:
    """Convert a DatetimeIndex to epoch millisecond integers.
    """
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((index - epoch) // pd.Timedelta(milliseconds=1)).tolist()


def evaluate_physical_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lat: float,
    lon: float,
) -> dict:
    """Fit the physical model's one calibration parameter (irradiance ->
    MW scale factor) on the training set, then evaluate on the held-out
    test set.
    """
    train_ts_ms = _index_to_epoch_ms(train_df.index)
    train_irradiance = estimate_irradiance_series(
        train_ts_ms, train_df["cloud_cover"].tolist(), lat=lat, lon=lon
    )
    scale_factor = calibrate_scale_factor(train_irradiance, train_df[TARGET_COLUMN].tolist())

    test_ts_ms = _index_to_epoch_ms(test_df.index)
    test_irradiance = estimate_irradiance_series(
        test_ts_ms, test_df["cloud_cover"].tolist(), lat=lat, lon=lon
    )
    predictions = pd.Series(
        [predict_generation(irr, scale_factor) for irr in test_irradiance],
        index=test_df.index,
    )
    actual = test_df[TARGET_COLUMN]

    return {
        "model_name": "clearsky_physical",
        "scale_factor": scale_factor,
        "mae": compute_mae(predictions, actual),
        "rmse": compute_rmse(predictions, actual),
        "predictions": predictions,
    }


def evaluate_ml_model(model, feature_cols: list[str], test_df: pd.DataFrame) -> dict:
    predictions = predict_solar(model, feature_cols, test_df)
    actual = test_df[TARGET_COLUMN]

    return {
        "model_name": "xgboost_solar",
        "mae": compute_mae(predictions, actual),
        "rmse": compute_rmse(predictions, actual),
        "predictions": predictions,
    }


def compare_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ml_model,
    ml_feature_cols: list[str],
    lat: float,
    lon: float,
) -> dict:
    """Run both evaluations and produce a single comparison result."""
    physical_result = evaluate_physical_model(train_df, test_df, lat, lon)
    ml_result = evaluate_ml_model(ml_model, ml_feature_cols, test_df)

    winner = "xgboost_solar" if ml_result["mae"] < physical_result["mae"] else "clearsky_physical"

    return {
        "test_set_size": len(test_df),
        "clearsky_physical": {"mae": physical_result["mae"], "rmse": physical_result["rmse"]},
        "xgboost_solar": {"mae": ml_result["mae"], "rmse": ml_result["rmse"]},
        "lower_mae_model": winner,
    }


def format_comparison_summary(comparison: dict) -> str:
    phys = comparison["clearsky_physical"]
    ml = comparison["xgboost_solar"]
    return (
        f"Physical (clear-sky): MAE {phys['mae']:.2f} MW, RMSE {phys['rmse']:.2f} MW\n"
        f"ML (XGBoost):         MAE {ml['mae']:.2f} MW, RMSE {ml['rmse']:.2f} MW\n"
        f"Lower MAE: {comparison['lower_mae_model']} -- n={comparison['test_set_size']}"
    )