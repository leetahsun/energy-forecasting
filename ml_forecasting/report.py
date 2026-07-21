"""Plotly dashboard for the ML forecasting module.
"""

from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

TARGET_LABELS = {
    "price_eur_mwh": "Day-Ahead Price (EUR/MWh)",
    "renewable_share_pct": "Renewable Share of Generation (%)",
}

MODEL_COLORS = {
    "xgboost_v1": "#2E7D32",
    "naive_baseline": "#9E9E9E",
}


def _group_forecast_by_target_and_model(forecast_records: list[dict]) -> dict:
    """Group flat forecast records into
    {target: {model_name: [(timestamp, value), ...]}} for easy plotting.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for record in forecast_records:
        target = record["metric_name"]
        model = record["model_name"]
        grouped[target][model].append((record["timestamp"], record["predicted_value"]))
    return grouped


def build_report(report: dict, out_path: str) -> str:
    """Build the full HTML report from the report dict (same structure
    as run_pipeline.py's saved JSON) and write it to out_path.
    """
    targets = list(report["evaluation"].keys())
    grouped_forecast = _group_forecast_by_target_and_model(report["forecast"])

    # One forecast row per target, plus one row for the MAE comparison bars
    fig = make_subplots(
        rows=len(targets) + 1,
        cols=1,
        subplot_titles=[
            f"{TARGET_LABELS.get(t, t)} -- Forecast" for t in targets
        ]
        + ["Model Comparison: MAE (lower is better)"],
        vertical_spacing=0.08,
    )

    for row, target in enumerate(targets, start=1):
        for model_name, points in grouped_forecast[target].items():
            points.sort(key=lambda p: p[0])
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=f"{target} ({model_name})",
                    line=dict(color=MODEL_COLORS.get(model_name, "#1565C0")),
                    legendgroup=model_name,
                ),
                row=row,
                col=1,
            )

    # MAE comparison bar chart, one bar pair per target
    mae_row = len(targets) + 1
    xgb_maes = [report["evaluation"][t]["xgboost"]["mae"] for t in targets]
    baseline_maes = [report["evaluation"][t]["naive_baseline"]["mae"] for t in targets]
    labels = [TARGET_LABELS.get(t, t) for t in targets]

    fig.add_trace(
        go.Bar(x=labels, y=xgb_maes, name="XGBoost", marker_color=MODEL_COLORS["xgboost_v1"]),
        row=mae_row,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels, y=baseline_maes, name="Naive Baseline",
            marker_color=MODEL_COLORS["naive_baseline"],
        ),
        row=mae_row,
        col=1,
    )

    fig.update_layout(
        title=f"ML Forecasting Report -- generated {report['generated_at']}",
        template="plotly_white",
        height=350 * (len(targets) + 1),
        showlegend=True,
        barmode="group",
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


if __name__ == "__main__":
    import json
    import sys

    report_path = sys.argv[1] if len(sys.argv) > 1 else "reports/ml_forecast/report.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else report_path.replace(".json", ".html")

    with open(report_path) as f:
        report = json.load(f)

    build_report(report, out_path)
    print(f"Saved dashboard to {out_path}")