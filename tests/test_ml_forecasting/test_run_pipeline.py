"""Tests for ml_forecasting/run_pipeline.py.

main orchestrates already-tested functions train, evaluate, predict
with real network calls to SMARD.
"""

import json
import math
import os

import numpy as np
import pytest

from ml_forecasting import run_pipeline

HOUR_MS = 3_600_000
START_TS = 1_704_067_200_000


def make_synthetic_history(num_hours: int = 24 * 60):
    rng = np.random.default_rng(9)
    hours = np.arange(num_hours)

    solar = np.clip(100 * np.sin((hours % 24) / 24 * math.pi), 0, None) + rng.normal(0, 2, num_hours)
    gas = 200 + rng.normal(0, 2, num_hours)
    price = 40 + 20 * np.sin(((hours % 24) - 6) / 24 * 2 * math.pi) + rng.normal(0, 1, num_hours)

    generation = {
        "solar": [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(solar)],
        "gas": [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(gas)],
    }
    price_series = [(START_TS + i * HOUR_MS, float(v)) for i, v in enumerate(price)]
    return generation, price_series


def test_main_produces_valid_report_with_expected_structure(monkeypatch, tmp_path):
    generation, price_series = make_synthetic_history()

    monkeypatch.setattr(run_pipeline, "fetch_all_generation_history", lambda num_weeks: generation)
    monkeypatch.setattr(run_pipeline, "fetch_price_history", lambda num_weeks: price_series)

    out_path = run_pipeline.main(
        num_weeks_history=12,
        forecast_horizon_hours=24,
        out_dir=str(tmp_path / "report"),
        models_dir=str(tmp_path / "models"),
    )

    assert os.path.exists(out_path)
    with open(out_path) as f:
        report = json.load(f)

    assert "generated_at" in report
    assert "evaluation" in report
    assert set(report["evaluation"].keys()) == {"renewable_share_pct", "price_eur_mwh"}
    assert "forecast" in report
    # 2 targets x 2 models (xgboost + naive) x 24 hours = 96 forecast records
    assert len(report["forecast"]) == 2 * 2 * 24


def test_main_forecast_records_have_expected_schema(monkeypatch, tmp_path):
    generation, price_series = make_synthetic_history()
    monkeypatch.setattr(run_pipeline, "fetch_all_generation_history", lambda num_weeks: generation)
    monkeypatch.setattr(run_pipeline, "fetch_price_history", lambda num_weeks: price_series)

    out_path = run_pipeline.main(
        num_weeks_history=12,
        out_dir=str(tmp_path / "report"),
        models_dir=str(tmp_path / "models"),
    )
    with open(out_path) as f:
        report = json.load(f)

    record = report["forecast"][0]
    assert "timestamp" in record
    assert "metric_name" in record
    assert "model_name" in record
    assert "predicted_value" in record
    assert record["actual_value"] is None


def test_main_evaluation_reports_whether_xgboost_beats_baseline(monkeypatch, tmp_path):
    generation, price_series = make_synthetic_history()
    monkeypatch.setattr(run_pipeline, "fetch_all_generation_history", lambda num_weeks: generation)
    monkeypatch.setattr(run_pipeline, "fetch_price_history", lambda num_weeks: price_series)

    out_path = run_pipeline.main(
        num_weeks_history=12,
        out_dir=str(tmp_path / "report"),
        models_dir=str(tmp_path / "models"),
    )
    with open(out_path) as f:
        report = json.load(f)

    price_eval = report["evaluation"]["price_eur_mwh"]
    assert "xgboost_beats_baseline" in price_eval
    assert isinstance(price_eval["xgboost_beats_baseline"], bool)


def test_main_creates_output_directory_if_missing(monkeypatch, tmp_path):
    generation, price_series = make_synthetic_history()
    monkeypatch.setattr(run_pipeline, "fetch_all_generation_history", lambda num_weeks: generation)
    monkeypatch.setattr(run_pipeline, "fetch_price_history", lambda num_weeks: price_series)

    nested_out_dir = str(tmp_path / "does" / "not" / "exist" / "yet")
    out_path = run_pipeline.main(
        num_weeks_history=12,
        out_dir=nested_out_dir,
        models_dir=str(tmp_path / "models"),
    )

    assert os.path.exists(out_path)