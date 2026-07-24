"""Shared visual theme for the energy forecasting dashboards.

One token system (colors, type, page shell) used by ml_forecasting/report.py,
solar_forecasting/report.py, and the landing page, so all three pages feel
like one product rather than three independently-styled fragments.

Palette is grounded in the subject, not arbitrary: marigold for solar/sun,
meadow green for renewables and growth, warm ink for editorial structure.
"""

INK = "#1B1B16"
PAPER = "#FAF6EC"
CARD = "#F1ECDD"
MEADOW = "#3F7D45"
MEADOW_SOFT = "#E7F0E3"
MARIGOLD = "#E8A93E"
MARIGOLD_SOFT = "#FBEDD3"
STONE = "#6B655A"
LINE = "#E4DDC8"

FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Figtree:wght@400;600&"
    "family=Lexend:wght@400;600;700&"
    "family=Quicksand:wght@500&display=swap"
)

BASE_CSS = f"""
  :root {{
    --ink: {INK}; --paper: {PAPER}; --card: {CARD};
    --meadow: {MEADOW}; --meadow-soft: {MEADOW_SOFT};
    --marigold: {MARIGOLD}; --marigold-soft: {MARIGOLD_SOFT};
    --stone: {STONE}; --line: {LINE};
  }}
  * {{ box-sizing: border-box; }}
  html {{ -webkit-text-size-adjust: 100%; }}
  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: 'Lexend', -apple-system, sans-serif;
    font-size: 16px;
    line-height: 1.5;
  }}
  @media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
  }}
  a {{ color: inherit; }}
  a:focus-visible, button:focus-visible {{
    outline: 2px solid var(--meadow);
    outline-offset: 3px;
  }}

  .shell {{ max-width: 880px; margin: 0 auto; padding: 0 24px; }}

  .site-header {{
    padding: 28px 0 20px;
    border-bottom: 1px solid var(--line);
  }}
  .site-header .shell {{
    display: flex; align-items: baseline; justify-content: space-between;
    flex-wrap: wrap; gap: 8px;
  }}
  .brand {{
    font-family: 'Figtree', serif;
    font-weight: 600;
    font-size: clamp(1.1rem, 1rem + 1vw, 1.4rem);
    text-decoration: none;
    letter-spacing: -0.01em;
  }}
  .brand .dot {{ color: var(--marigold); }}
  .back-link {{
    font-family: 'Quicksand', monospace;
    font-size: 0.78rem;
    text-decoration: none;
    color: var(--stone);
    border-bottom: 1px solid var(--line);
    padding-bottom: 2px;
  }}
  .back-link:hover {{ color: var(--meadow); border-color: var(--meadow); }}

  h1.page-title {{
    font-family: 'Figtree', serif;
    font-weight: 600;
    font-size: clamp(1.6rem, 1.3rem + 2vw, 2.4rem);
    letter-spacing: -0.01em;
    margin: 40px 0 8px;
  }}
  p.page-sub {{
    color: var(--stone);
    font-size: clamp(0.95rem, 0.9rem + 0.3vw, 1.05rem);
    margin: 0 0 32px;
    max-width: 60ch;
  }}

  .stat-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 36px;
  }}
  .stat-card {{
    background: var(--card);
    border-radius: 18px;
    padding: 20px 22px;
    border: 1px solid var(--line);
  }}
  .stat-card .stat-label {{
    font-family: 'Quicksand', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--stone);
    display: block;
    margin-bottom: 10px;
  }}
  .stat-card .stat-value {{
    font-family: 'Figtree', serif;
    font-weight: 600;
    font-size: 1.9rem;
    line-height: 1;
  }}
  .stat-card .stat-value.good {{ color: var(--meadow); }}
  .stat-card .stat-value.warn {{ color: var(--marigold); }}
  .stat-card .stat-detail {{
    font-size: 0.85rem;
    color: var(--stone);
    margin-top: 8px;
  }}

  .chart-wrap {{
    background: var(--card);
    border-radius: 18px;
    padding: 8px 4px 4px;
    border: 1px solid var(--line);
    margin-bottom: 28px;
  }}

  .note {{
    font-size: 0.85rem;
    color: var(--stone);
    border-left: 3px solid var(--marigold);
    padding: 4px 0 4px 16px;
    margin: 20px 0 40px;
  }}

  .site-footer {{
    padding: 32px 0 60px;
    font-family: 'Quicksand', monospace;
    font-size: 0.75rem;
    color: var(--stone);
  }}

  @media (max-width: 560px) {{
    .shell {{ padding: 0 16px; }}
    .stat-card .stat-value {{ font-size: 1.6rem; }}
  }}
"""


def page_shell(
    *,
    title: str,
    body_html: str,
    show_back_link: bool = True,
    generated_at: str = "",
) -> str:
    """Wrap arbitrary body HTML in the shared page shell: fonts, base
    CSS, header with a link back to the landing page, and a footer.
    """
    back_link = (
        '<a class="back-link" href="index.html">&larr; all dashboards</a>'
        if show_back_link
        else ""
    )
    footer = f"generated {generated_at}" if generated_at else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="{FONT_IMPORT}" rel="stylesheet">
<style>{BASE_CSS}</style>
</head>
<body>
<header class="site-header">
    <div class="shell">
      <a class="brand" href="index.html">Energy Forecasting<span class="dot">.</span></a>
      {back_link}
    </div>
  </header>
  <main class="shell">
    {body_html}
  </main>
  <footer class="site-footer">
    <div class="shell">{footer}</div>
  </footer>
</body>
</html>"""


def stat_card(label: str, value: str, detail: str = "", tone: str = "") -> str:
    """A single stat callout card. tone: 'good' (green), 'warn' (marigold), or '' (ink)."""
    tone_class = f" {tone}" if tone else ""
    detail_html = f'<div class="stat-detail">{detail}</div>' if detail else ""
    return f"""<div class="stat-card">
      <span class="stat-label">{label}</span>
      <div class="stat-value{tone_class}">{value}</div>
      {detail_html}
    </div>"""