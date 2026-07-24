"""
Inspect the raw shape of SMARD series data to confirm how unreported
(not-yet-published) hours are represented — as (timestamp, None) pairs,
or missing from the series entirely.

Run from the repo root:
    python inspect_raw_series.py
"""

from shared.smard_client import (
    GENERATION_FILTERS,
    PRICE_FILTER,
    get_reference_timestamps,
    get_series,
)


def inspect(filter_id: int, label: str, num_weeks: int = 1) -> None:
    print(f"\n=== {label} (filter_id={filter_id}) ===")
    timestamps = get_reference_timestamps(num_weeks)
    latest_bucket_ts = timestamps[-1]

    series = get_series(filter_id, latest_bucket_ts)
    print(f"Bucket timestamp: {latest_bucket_ts}")
    print(f"Total points in bucket: {len(series)}")

    print("Last 10 raw (timestamp_ms, value) pairs:")
    for ts_ms, val in series[-10:]:
        print(f"  {ts_ms}  ->  {val!r}")

    none_count = sum(1 for _, v in series if v is None)
    print(f"None values in this bucket: {none_count} / {len(series)}")

    # check for a jump in the interval between consecutive timestamps,
    # which would indicate missing entries rather than explicit None values
    gaps = set()
    for i in range(1, len(series)):
        gaps.add(series[i][0] - series[i - 1][0])
    print(f"Distinct intervals between consecutive timestamps (ms): {gaps}")


def main() -> None:
    inspect(GENERATION_FILTERS["solar"], "solar")
    inspect(GENERATION_FILTERS["wind_onshore"], "wind_onshore")
    inspect(PRICE_FILTER, "price (day-ahead)")


if __name__ == "__main__":
    main()