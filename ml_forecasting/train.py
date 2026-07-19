import os

import joblib
import pandas as pd
from xgboost import XGBRegressor

TARGET_COLUMNS = ["renewable_share_pct", "price_eur_mwh"]


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """All engineered feature columns for a given target.

    Excludes the raw current-value columns for both targets
    """
    exclude = set(TARGET_COLUMNS)
    return [c for c in df.columns if c not in exclude]


def time_based_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """splitting the data into past data and future data.
    """
    df = df.sort_index()
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def naive_baseline_predict(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Our baseline that the model needs to beat.
    """
    lag_col = f"{target_col}_lag_168h"
    if lag_col not in df.columns:
        raise ValueError(
            f"{lag_col} not found  so build_feature_matrix must include a 168h lag "
            f"for the naive baseline to be computable"
        )
    return df[lag_col]


def train_xgboost(
    train_df: pd.DataFrame,
    target_col: str,
    **xgb_params,
) -> tuple[XGBRegressor, list[str]]:
    """Train an XGBoost regressor for one target column."""
    feature_cols = get_feature_columns(train_df, target_col)
    X = train_df[feature_cols]
    y = train_df[target_col]

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


def save_model(model: XGBRegressor, feature_cols: list[str], path: str) -> None:
    """saving the exact feature column list that the model was trained on.S
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, path) 
    #Saving feature_cols alongside the model avoids a mismatch bug at prediction time if features.py changes what columns it is.


def load_model(path: str) -> tuple[XGBRegressor, list[str]]:
    bundle = joblib.load(path) 
    return bundle["model"], bundle["feature_cols"]


def train_all_targets(
    feature_df: pd.DataFrame, models_dir: str = "ml_forecasting/models"
) -> dict:
    """Train + save an XGBoost model for every target column, using a
    chronological train/test split. Returns models and both splits so
    evaluate.py can compute baseline vs. XGBoost accuracy on the same
    held-out data.
    """
    train_df, test_df = time_based_split(feature_df)

    results = {}
    for target_col in TARGET_COLUMNS:
        model, feature_cols = train_xgboost(train_df, target_col)
        save_model(model, feature_cols, f"{models_dir}/{target_col}_xgboost.joblib")
        results[target_col] = {
            "model": model,
            "feature_cols": feature_cols,
            "train_df": train_df,
            "test_df": test_df,
        }
    return results


if __name__ == "__main__":
    from shared.smard_client import fetch_all_generation_history, fetch_price_history
    from ml_forecasting.features import build_feature_matrix

    print("Fetching SMARD history...")
    generation = fetch_all_generation_history(num_weeks=12)
    price_series = fetch_price_history(num_weeks=12)

    print("Building feature matrix...")
    feature_df = build_feature_matrix(generation, price_series)

    print(f"Training on {len(feature_df)} rows...")
    train_all_targets(feature_df)
    print("Done. Models saved to ml_forecasting/models/")