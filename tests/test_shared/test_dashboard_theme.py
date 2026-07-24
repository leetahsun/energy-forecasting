"""Tests for shared/dashboard_theme.py."""

from shared.dashboard_theme import page_shell, stat_card, MEADOW, MARIGOLD, INK


def test_page_shell_includes_viewport_meta_for_mobile():
    html = page_shell(title="Test", body_html="<p>hi</p>")
    assert 'name="viewport"' in html
    assert "width=device-width" in html


def test_page_shell_includes_color_tokens():
    html = page_shell(title="Test", body_html="<p>hi</p>")
    assert MEADOW in html
    assert MARIGOLD in html
    assert INK in html


def test_page_shell_includes_reduced_motion_query():
    html = page_shell(title="Test", body_html="<p>hi</p>")
    assert "prefers-reduced-motion" in html


def test_page_shell_includes_mobile_media_query():
    html = page_shell(title="Test", body_html="<p>hi</p>")
    assert "@media (max-width: 560px)" in html


def test_page_shell_back_link_can_be_hidden():
    with_link = page_shell(title="Test", body_html="<p>hi</p>", show_back_link=True)
    without_link = page_shell(title="Test", body_html="<p>hi</p>", show_back_link=False)

    assert "all dashboards" in with_link
    assert "all dashboards" not in without_link


def test_page_shell_embeds_body_content():
    html = page_shell(title="Test", body_html="<p>UNIQUE_MARKER_XYZ</p>")
    assert "UNIQUE_MARKER_XYZ" in html


def test_page_shell_includes_generated_at_when_provided():
    html = page_shell(title="Test", body_html="<p>hi</p>", generated_at="2026-07-24T12:00:00Z")
    assert "2026-07-24T12:00:00Z" in html


def test_stat_card_includes_label_and_value():
    html = stat_card("MAE Improvement", "38.2%", detail="vs. naive baseline")
    assert "MAE Improvement" in html
    assert "38.2%" in html
    assert "vs. naive baseline" in html


def test_stat_card_tone_classes():
    good = stat_card("X", "1", tone="good")
    warn = stat_card("X", "1", tone="warn")
    neutral = stat_card("X", "1")

    assert "stat-value good" in good
    assert "stat-value warn" in warn
    assert "stat-value good" not in neutral
    assert "stat-value warn" not in neutral


def test_stat_card_omits_detail_div_when_not_provided():
    html = stat_card("X", "1")
    assert "stat-detail" not in html