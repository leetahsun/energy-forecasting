"""Plotly dashboard for the solar forecasting module.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODEL_COLORS = {
    "physical": "#F9A825",  # sun-like yellow, for the physical clear-sky model
    "ml": "#1565C0",
}


def build_report(report: dict, out_path: str) -> str:
    forecast = sorted(report["forecast"], key=lambda r: r["timestamp"])
    timestamps = [r["timestamp"] for r in forecast]
    ml_values = [r["ml_forecast_mw"] for r in forecast]
    physical_values = [r["physical_forecast_mw"] for r in forecast]

    evaluation = report["evaluation"]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Solar Generation Forecast (MW)",
            "Model Comparison: MAE (lower is better)",
        ),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps, y=physical_values, mode="lines+markers",
            name="Physical (clear-sky)", line=dict(color=MODEL_COLORS["physical"]),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=ml_values, mode="lines+markers",
            name="ML (XGBoost)", line=dict(color=MODEL_COLORS["ml"]),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=["Physical (clear-sky)", "ML (XGBoost)"],
            y=[evaluation["clearsky_physical"]["mae"], evaluation["xgboost_solar"]["mae"]],
            marker_color=[MODEL_COLORS["physical"], MODEL_COLORS["ml"]],
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"Solar Forecasting Report -- generated {report['generated_at']}",
        template="plotly_white",
        height=700,
        showlegend=True,
    )
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="MAE (MW)", row=2, col=1)

    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


if __name__ == "__main__":
    import json
    import sys

    report_path = sys.argv[1] if len(sys.argv) > 1 else "reports/solar_forecast/forecast.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else report_path.replace(".json", ".html")

    with open(report_path) as f:
        report = json.load(f)

    build_report(report, out_path)
    print(f"Saved dashboard to {out_path}")