"""
Debug script: locate NaN gaps in renewable_share_pct that break recursive_forecast.

Run from the repo root:
    python debug_gaps.py
"""

from ml_forecasting.features import build_base_dataframe
from shared.smard_client import fetch_all_generation_history, fetch_price_history


def main(num_weeks: int = 12) -> None:
    print(f"Fetching {num_weeks} weeks of SMARD history...")
    generation = fetch_all_generation_history(num_weeks=num_weeks)
    price_series = fetch_price_history(num_weeks=num_weeks)

    df = build_base_dataframe(generation, price_series)

    print("\n--- Full history NaN check ---")
    nan_rows = df[df["renewable_share_pct"].isna()]
    print(f"{len(nan_rows)} NaN rows out of {len(df)} total rows")
    if not nan_rows.empty:
        print(nan_rows)

    print("\n--- Trailing 168h (last week) NaN check ---")
    tail = df["renewable_share_pct"].tail(168)
    print(f"{tail.isna().sum()} NaNs in the trailing 168 hours")
    print("\nLast 10 rows:")
    print(df.tail(10))

    print("\n--- Index continuity check (looking for missing hourly timestamps) ---")
    full_range = df.index.to_series().diff().value_counts()
    print("Gap sizes between consecutive timestamps (should mostly be 1 hour):")
    print(full_range)


if __name__ == "__main__":
    main()