import numpy as np
import pandas as pd

from ml_forecasting.train import naive_baseline_predict


def compute_mae(predictions: pd.Series, actuals: pd.Series) -> float:
    """Mean Absolute Error.
    """
    return float((predictions - actuals).abs().mean())


def compute_rmse(predictions: pd.Series, actuals: pd.Series) -> float:
    """Root Mean Squared Error
    """
    return float(np.sqrt(((predictions - actuals) ** 2).mean()))


def evaluate_target(
    target_col: str,
    test_df: pd.DataFrame,
    model,
    feature_cols: list[str],
) -> dict:
    """Compare XGBoost vs. naive baseline
    """
    actual = test_df[target_col]

    xgb_predictions = pd.Series(model.predict(test_df[feature_cols]), index=test_df.index)
    baseline_predictions = naive_baseline_predict(test_df, target_col)

    xgb_mae = compute_mae(xgb_predictions, actual)
    xgb_rmse = compute_rmse(xgb_predictions, actual)
    baseline_mae = compute_mae(baseline_predictions, actual)
    baseline_rmse = compute_rmse(baseline_predictions, actual)

    # % improvement over baseline whhere positive means that the XGBoost is better.
    # Guard against baseline_mae == 0 (degenerate case,  a constant
    # test series) to avoid this, divide-by-zero.
    improvement_pct = (
        (baseline_mae - xgb_mae) / baseline_mae * 100 if baseline_mae > 0 else None
    )

    return {
        "target": target_col,
        "test_set_size": len(test_df),
        "xgboost": {"mae": xgb_mae, "rmse": xgb_rmse},
        "naive_baseline": {"mae": baseline_mae, "rmse": baseline_rmse},
        "xgboost_improvement_pct": improvement_pct,
        "xgboost_beats_baseline": xgb_mae < baseline_mae,
    }


def evaluate_all(train_results: dict) -> dict:

    results = {}
    for target_col, info in train_results.items():
        results[target_col] = evaluate_target(
            target_col,
            info["test_df"],
            info["model"],
            info["feature_cols"],
        )
    return results


def format_evaluation_summary(evaluation: dict) -> str:
    
    lines = []
    for target_col, result in evaluation.items():
        verdict = "BEATS" if result["xgboost_beats_baseline"] else "DOES NOT BEAT"
        pct = result["xgboost_improvement_pct"]
        pct_str = f"{pct:+.1f}%" if pct is not None else "n/a"
        lines.append(
            f"{target_col}: XGBoost {verdict} naive baseline "
            f"(MAE {result['xgboost']['mae']:.3f} vs {result['naive_baseline']['mae']:.3f}, "
            f"{pct_str} change) -- n={result['test_set_size']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import os

    from shared.smard_client import fetch_all_generation_history, fetch_price_history
    from ml_forecasting.features import build_feature_matrix
    from ml_forecasting.train import train_all_targets

    print("Fetching SMARD history:")
    generation = fetch_all_generation_history(num_weeks=12)
    price_series = fetch_price_history(num_weeks=12)

    print("Building feature matrix and training:")
    feature_df = build_feature_matrix(generation, price_series)
    train_results = train_all_targets(feature_df)

    print("Evaluating*****")
    evaluation = evaluate_all(train_results)
    print(format_evaluation_summary(evaluation))

    os.makedirs("reports/ml_forecast", exist_ok=True)
    with open("reports/ml_forecast/evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)