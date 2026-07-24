"""Builds the landing page (index.html) linking to both dashboards.

Previously this HTML lived inline as a heredoc inside publish_dashboard.yml.
Moving it here makes it testable and keeps it visually consistent with
the two report pages via the shared theme.
"""

from shared.dashboard_theme import page_shell

CARDS = [
    {
        "href": "ml_forecast.html",
        "title": "ML Forecasting",
        "desc": "Day-ahead price and renewable share, XGBoost vs. a naive baseline.",
        "cadence": "retrained weekly",
        "icon": (
            '<svg width="28" height="28" viewBox="0 0 28 28" fill="none">'
            '<path d="M4 22 L10 14 L15 18 L24 6" stroke="#3F7D45" stroke-width="2.4" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
            '<circle cx="24" cy="6" r="2.4" fill="#3F7D45"/></svg>'
        ),
    },
    {
        "href": "solar_forecast.html",
        "title": "Solar Forecasting",
        "desc": "A physical clear-sky model against weather-driven XGBoost.",
        "cadence": "updated daily",
        "icon": (
            '<svg width="28" height="28" viewBox="0 0 28 28" fill="none">'
            '<circle cx="14" cy="14" r="5.5" fill="#E8A93E"/>'
            '<g stroke="#E8A93E" stroke-width="2" stroke-linecap="round">'
            '<line x1="14" y1="2" x2="14" y2="5"/><line x1="14" y1="23" x2="14" y2="26"/>'
            '<line x1="2" y1="14" x2="5" y2="14"/><line x1="23" y1="14" x2="26" y2="14"/>'
            '<line x1="5.5" y1="5.5" x2="7.5" y2="7.5"/><line x1="20.5" y1="20.5" x2="22.5" y2="22.5"/>'
            '<line x1="5.5" y1="22.5" x2="7.5" y2="20.5"/><line x1="20.5" y1="7.5" x2="22.5" y2="5.5"/>'
            "</g></svg>"
        ),
    },
]

INDEX_CSS = """
  .card-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin: 28px 0 40px; }
  .module-card {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 26px 24px; text-decoration: none; color: var(--ink);
    display: block; transition: transform 0.15s ease, border-color 0.15s ease;
  }
  .module-card:hover { border-color: var(--meadow); transform: translateY(-2px); }
  .module-card .icon-row { margin-bottom: 16px; }
  .module-card h2 {
    font-family: 'Figtree', serif; font-weight: 600; font-size: 1.3rem; margin: 0 0 8px;
  }
  .module-card p.desc { color: var(--stone); font-size: 0.92rem; margin: 0 0 16px; }
  .module-card .cadence {
    font-family: 'Quicksand', monospace; font-size: 0.72rem; color: var(--meadow);
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .module-card .cta {
    display: inline-block; margin-top: 14px; font-weight: 700; font-size: 0.9rem;
  }
"""


def _card_html(card: dict) -> str:
    return f"""<a class="module-card" href="{card['href']}">
      <div class="icon-row">{card['icon']}</div>
      <h2>{card['title']}</h2>
      <p class="desc">{card['desc']}</p>
      <span class="cadence">{card['cadence']}</span>
      <div class="cta">View dashboard &rarr;</div>
    </a>"""


def build_index(out_path: str, generated_at: str = "") -> str:
    body = f"""
    <style>{INDEX_CSS}</style>
    <h1 class="page-title">Energy Forecasting</h1>
    <p class="page-sub">
      Two ways of forecasting the German electricity market: a
      data-driven model learning from history, and a physical model
      reasoning from first principles.
    </p>
    <div class="card-row">
      {''.join(_card_html(c) for c in CARDS)}
    </div>
    """

    html = page_shell(
        title="Energy Forecasting",
        body_html=body,
        show_back_link=False,
        generated_at=generated_at,
    )

    with open(out_path, "w") as f:
        f.write(html)
    return out_path


if __name__ == "__main__":
    import datetime
    import os
    import sys

    out_path = sys.argv[1] if len(sys.argv) > 1 else "public/index.html"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    build_index(out_path, generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    print(f"Saved landing page to {out_path}")