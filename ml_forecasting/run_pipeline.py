"""
End-to-end ML forecasting pipeline.

Fetches historical SMARD data, builds features, trains models for both
targets.
"""

import datetime
import json
import os

from ml_forecasting.evaluate import evaluate_all, format_evaluation_summary
from ml_forecasting.features import build_base_dataframe, build_feature_matrix
from ml_forecasting.predict import generate_all_forecasts
from ml_forecasting.train import train_all_targets
from shared.smard_client import fetch_all_generation_history, fetch_price_history


def main(
    num_weeks_history: int = 12,
    forecast_horizon_hours: int = 24,
    out_dir: str | None = None,
    models_dir: str = "ml_forecasting/models",
) -> str:
    if out_dir is None:
        out_dir = f"reports/ml_forecast/{datetime.date.today().isoformat()}"
    os.makedirs(out_dir, exist_ok=True)

    print("Fetching SMARD history:")
    generation = fetch_all_generation_history(num_weeks=num_weeks_history)
    price_series = fetch_price_history(num_weeks=num_weeks_history)

    print("Building feature matrix:")
    feature_df = build_feature_matrix(generation, price_series)

    print(f"Training on {len(feature_df)} rows:")
    train_results = train_all_targets(feature_df, models_dir=models_dir)

    print("Evaluating against naive baseline:")
    evaluation = evaluate_all(train_results)
    print(format_evaluation_summary(evaluation))

    print("Generating forecast:")
    history_df = build_base_dataframe(generation, price_series)
    forecasts = generate_all_forecasts(
        history_df, horizon_hours=forecast_horizon_hours, models_dir=models_dir
    )

    output = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation": evaluation,
        "forecast": [r.model_dump(mode="json") for r in forecasts],
    }

    out_path = f"{out_dir}/report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved evaluation + forecast to {out_path}")
    return out_path


if __name__ == "__main__":
    main()