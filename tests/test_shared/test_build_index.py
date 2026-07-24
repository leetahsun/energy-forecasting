"""Tests for shared/build_index.py."""

import os
import tempfile

from shared.build_index import build_index


def test_build_index_produces_valid_html_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "index.html")
        build_index(out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "<html" in content.lower()


def test_build_index_links_to_both_dashboards():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "index.html")
        build_index(out_path)

        with open(out_path) as f:
            content = f.read()

        assert 'href="ml_forecast.html"' in content
        assert 'href="solar_forecast.html"' in content


def test_build_index_has_no_back_link_since_it_is_the_home_page():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "index.html")
        build_index(out_path)

        with open(out_path) as f:
            content = f.read()

        assert "all dashboards" not in content


def test_build_index_includes_mobile_viewport():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "index.html")
        build_index(out_path)

        with open(out_path) as f:
            content = f.read()

        assert "width=device-width" in content


def test_build_index_includes_theme_colors():
    from shared.dashboard_theme import MEADOW, MARIGOLD

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "index.html")
        build_index(out_path)

        with open(out_path) as f:
            content = f.read()

        assert MEADOW in content
        assert MARIGOLD in content


def test_build_index_returns_output_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "index.html")
        result = build_index(out_path)
        assert result == out_path