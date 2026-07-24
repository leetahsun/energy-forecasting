"""Plotly dashboard for the solar forecasting module.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.dashboard_theme import INK, LINE, MARIGOLD, MEADOW, page_shell, stat_card

MODEL_COLORS = {"physical": MARIGOLD, "ml": MEADOW}
PLOT_CONFIG = {"displaylogo": False, "responsive": True, "displayModeBar": False}


def _build_figure(report: dict) -> go.Figure:
    forecast = sorted(report["forecast"], key=lambda r: r["timestamp"])
    timestamps = [r["timestamp"] for r in forecast]
    ml_values = [r["ml_forecast_mw"] for r in forecast]
    physical_values = [r["physical_forecast_mw"] for r in forecast]
    evaluation = report["evaluation"]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Solar generation forecast (MW)",
            "Model comparison using mean absolute error",
        ),
        row_heights=[0.62, 0.38],
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps, y=physical_values, mode="lines+markers",
            name="Physical (clear-sky)", line=dict(color=MODEL_COLORS["physical"], width=2.5),
            marker=dict(size=5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=ml_values, mode="lines+markers",
            name="XGBoost", line=dict(color=MODEL_COLORS["ml"], width=2.5),
            marker=dict(size=5),
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
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Lexend, sans-serif", color=INK, size=13),
        height=680,
        margin=dict(l=50, r=24, t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, title_text="MW", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, title_text="MAE (MW)", row=2, col=1)
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family="Figtree, sans-serif", size=16, color=INK)

    return fig


def _build_stat_cards(report: dict) -> str:
    ev = report["evaluation"]
    winner = ev["lower_mae_model"]
    winner_label = "XGBoost" if winner == "xgboost_solar" else "Physical (clear-sky)"
    tone = "good" if winner == "xgboost_solar" else "warn"

    cards = [
        stat_card(
            "Physical model MAE",
            f"{ev['clearsky_physical']['mae']:.0f} MW",
            "clear-sky irradiance model",
        ),
        stat_card(
            "ML model MAE",
            f"{ev['xgboost_solar']['mae']:.0f} MW",
            "weather-driven XGBoost",
        ),
        stat_card("Lower error", winner_label, tone=tone),
    ]
    return f'<div class="stat-row">{"".join(cards)}</div>'


def build_report(report: dict, out_path: str) -> str:
    fig = _build_figure(report)
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config=PLOT_CONFIG)

    body = f"""
    <h1 class="page-title">Solar Forecasting</h1>
    <p class="page-sub">
      A physical clear-sky irradiance model against another weather-driven
      XGBoost model which forecasts the national solar generation.
    </p>
    {_build_stat_cards(report)}
    <div class="chart-wrap">{chart_html}</div>
    <p class="note">
      The physical model encodes sun position and cloud cover directly;
      then the ML model learns the relationship from historical data. Neither
      is assumed better going in, meaning that we can compare them. 
    </p>
    """

    html = page_shell(
        title="Solar Forecasting//Energy Forecasting",
        body_html=body,
        generated_at=report.get("generated at", ""),
    )

    with open(out_path, "w") as f:
        f.write(html)
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