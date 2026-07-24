"""Plotly dashboard for the ML forecasting module.
"""

from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.dashboard_theme import (
    CARD, INK, LINE, MARIGOLD, MEADOW, STONE, page_shell, stat_card,
)

TARGET_LABELS = {
    "price_eur_mwh": "Day-Ahead Price (EUR/MWh)",
    "renewable_share_pct": "Renewable Share of Generation (%)",
}

MODEL_COLORS = {
    "xgboost_v1": MEADOW,
    "naive_baseline": STONE,
}

PLOT_CONFIG = {"displaylogo": False, "responsive": True, "displayModeBar": False}


def _group_forecast_by_target_and_model(forecast_records: list[dict]) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for record in forecast_records:
        grouped[record["metric_name"]][record["model_name"]].append(
            (record["timestamp"], record["predicted_value"])
        )
    return grouped


def _build_figure(report: dict) -> go.Figure:
    targets = list(report["evaluation"].keys())
    grouped_forecast = _group_forecast_by_target_and_model(report["forecast"])

    fig = make_subplots(
        rows=len(targets) + 1,
        cols=1,
        subplot_titles=[f"{TARGET_LABELS.get(t, t)}" for t in targets]
        + ["Model comparison — mean absolute error "],
        vertical_spacing=0.1,
    )

    for row, target in enumerate(targets, start=1):
        for model_name, points in grouped_forecast[target].items():
            points.sort(key=lambda p: p[0])
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in points],
                    y=[p[1] for p in points],
                    mode="lines+markers",
                    name=f"{model_name} — {target}",
                    line=dict(color=MODEL_COLORS.get(model_name, INK), width=2.5),
                    marker=dict(size=5),
                    legendgroup=model_name,
                ),
                row=row,
                col=1,
            )

    mae_row = len(targets) + 1
    labels = [TARGET_LABELS.get(t, t) for t in targets]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[report["evaluation"][t]["xgboost"]["mae"] for t in targets],
            name="XGBoost",
            marker_color=MEADOW,
        ),
        row=mae_row,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[report["evaluation"][t]["naive_baseline"]["mae"] for t in targets],
            name="Naive baseline",
            marker_color=STONE,
        ),
        row=mae_row,
        col=1,
    )

    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Lexend, sans-serif", color=INK, size=13),
        height=320 * (len(targets) + 1),
        margin=dict(l=50, r=24, t=50, b=40),
        showlegend=True,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False)
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(family="Figtree, serif", size=16, color=INK)

    return fig


def _build_stat_cards(report: dict) -> str:
    cards = []
    for target, label in TARGET_LABELS.items():
        if target not in report["evaluation"]:
            continue
        ev = report["evaluation"][target]
        pct = ev.get("xgboost_improvement_pct")
        beats = ev.get("xgboost_beats_baseline")
        if pct is not None:
            value = f"{pct:+.0f}%"
            tone = "good" if beats else "warn"
            detail = "lower error than naive baseline" if beats else "naive baseline still wins"
        else:
            value = "—"
            tone = ""
            detail = "baseline had zero error to compare against"
        cards.append(stat_card(label, value, detail, tone=tone))
    return f'<div class="stat-row">{"".join(cards)}</div>'


def build_report(report: dict, out_path: str) -> str:
    fig = _build_figure(report)
    chart_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    body = f"""
    <h1 class="page-title">ML Forecasting</h1>
    <p class="page-sub">
    XGBoost against a naive same-hour-last-week baseline, for day-ahead
    price and renewable generation share.
    </p>
    {_build_stat_cards(report)}
    <div class="chart-wrap">{chart_html}</div>
    <p class="note">
    Improvement is measured on held-out data the model never trained
    on. 
    </p>
    """

    html = page_shell(
        title="ML Forecasting — Energy Forecasting",
        body_html=body,
        generated_at=report.get("generated_at", ""),
    )

    with open(out_path, "w") as f:
        f.write(html)
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