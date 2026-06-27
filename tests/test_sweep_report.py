"""Unit tests for sweep_report HTML generation."""

from __future__ import annotations

import json
import re

import pytest

from llm_inference_benchmark.sweep_report import (
    SweepPoint,
    _chart_data,
    _sweep_table,
    build_sweep_html,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_points(*, with_ttft: bool = False) -> list[SweepPoint]:
    return [
        SweepPoint(
            concurrency=1,
            tokens_per_second=45.0,
            throughput_rps=1.5,
            p50_latency_ms=600.0,
            p95_latency_ms=700.0,
            p50_ttft_ms=120.0 if with_ttft else None,
            is_knee=False,
        ),
        SweepPoint(
            concurrency=2,
            tokens_per_second=80.0,
            throughput_rps=2.7,
            p50_latency_ms=900.0,
            p95_latency_ms=1100.0,
            p50_ttft_ms=150.0 if with_ttft else None,
            is_knee=True,
        ),
        SweepPoint(
            concurrency=4,
            tokens_per_second=70.0,
            throughput_rps=2.3,
            p50_latency_ms=1800.0,
            p95_latency_ms=2400.0,
            p50_ttft_ms=200.0 if with_ttft else None,
            is_knee=False,
        ),
    ]


# ── Empty input ────────────────────────────────────────────────────────────────


def test_build_sweep_html_empty_returns_placeholder() -> None:
    html = build_sweep_html([])
    assert "No sweep data provided" in html
    assert "<!DOCTYPE html>" in html


def test_build_sweep_html_empty_custom_title() -> None:
    html = build_sweep_html([], title="My Sweep")
    assert "My Sweep" in html


# ── HTML structure ─────────────────────────────────────────────────────────────


def test_build_sweep_html_is_valid_doctype() -> None:
    html = build_sweep_html(_make_points())
    assert html.startswith("<!DOCTYPE html>")


def test_build_sweep_html_contains_title() -> None:
    html = build_sweep_html(_make_points(), title="Test Sweep 123")
    assert "Test Sweep 123" in html


def test_build_sweep_html_contains_plotly_cdn() -> None:
    html = build_sweep_html(_make_points())
    assert "cdn.plot.ly" in html


def test_build_sweep_html_contains_chart_div() -> None:
    html = build_sweep_html(_make_points())
    assert 'id="chart"' in html


def test_build_sweep_html_contains_plotly_newplot() -> None:
    html = build_sweep_html(_make_points())
    assert "Plotly.newPlot" in html


def test_build_sweep_html_level_count_in_subtitle() -> None:
    html = build_sweep_html(_make_points())
    assert "3 level(s)" in html


# ── Knee point ────────────────────────────────────────────────────────────────


def test_build_sweep_html_knee_summary_present() -> None:
    html = build_sweep_html(_make_points())
    assert "Knee point" in html


def test_build_sweep_html_knee_concurrency_in_summary() -> None:
    html = build_sweep_html(_make_points())
    assert "concurrency=2" in html


def test_build_sweep_html_knee_throughput_in_summary() -> None:
    html = build_sweep_html(_make_points())
    assert "80.0 tok/s" in html


def test_build_sweep_html_knee_p95_in_summary() -> None:
    html = build_sweep_html(_make_points())
    assert "1100.0 ms" in html


# ── Table ─────────────────────────────────────────────────────────────────────


def test_sweep_table_contains_all_concurrency_values() -> None:
    table = _sweep_table(_make_points())
    for c in (1, 2, 4):
        assert f"<td>{c}</td>" in table


def test_sweep_table_knee_badge_present() -> None:
    table = _sweep_table(_make_points())
    assert "badge-knee" in table
    assert "knee" in table


def test_sweep_table_knee_row_class() -> None:
    table = _sweep_table(_make_points())
    assert "class='knee-row'" in table


def test_sweep_table_ttft_none_shows_dash() -> None:
    table = _sweep_table(_make_points(with_ttft=False))
    assert "—" in table


def test_sweep_table_ttft_values_shown_when_present() -> None:
    table = _sweep_table(_make_points(with_ttft=True))
    assert "120.0" in table
    assert "150.0" in table


# ── Chart data ────────────────────────────────────────────────────────────────


def test_chart_data_is_valid_json() -> None:
    data = _chart_data(_make_points())
    parsed = json.loads(data)
    assert isinstance(parsed, list)
    assert len(parsed) == 3


def test_chart_data_contains_concurrency_fields() -> None:
    parsed = json.loads(_chart_data(_make_points()))
    assert parsed[0]["c"] == 1
    assert parsed[1]["c"] == 2
    assert parsed[2]["c"] == 4


def test_chart_data_tps_values() -> None:
    parsed = json.loads(_chart_data(_make_points()))
    assert parsed[0]["tps"] == pytest.approx(45.0)
    assert parsed[1]["tps"] == pytest.approx(80.0)


def test_chart_data_knee_flag() -> None:
    parsed = json.loads(_chart_data(_make_points()))
    assert parsed[0]["knee"] is False
    assert parsed[1]["knee"] is True
    assert parsed[2]["knee"] is False


def test_chart_data_p95_values() -> None:
    parsed = json.loads(_chart_data(_make_points()))
    assert parsed[0]["p95"] == pytest.approx(700.0)
    assert parsed[2]["p95"] == pytest.approx(2400.0)


# ── XSS safety ────────────────────────────────────────────────────────────────


def test_build_sweep_html_title_is_escaped() -> None:
    html = build_sweep_html(_make_points(), title="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ── Single-point edge case ────────────────────────────────────────────────────


def test_build_sweep_html_single_point() -> None:
    single = [
        SweepPoint(
            concurrency=1,
            tokens_per_second=50.0,
            throughput_rps=1.0,
            p50_latency_ms=500.0,
            p95_latency_ms=600.0,
            is_knee=True,
        )
    ]
    html = build_sweep_html(single)
    assert "Knee point" in html
    assert "concurrency=1" in html
    assert "50.0 tok/s" in html


# ── Chart JS data is embedded in output ───────────────────────────────────────


def test_chart_json_embedded_in_html() -> None:
    pts = _make_points()
    html = build_sweep_html(pts)
    # The chart JSON array is assigned to a const in the script block
    match = re.search(r"const PTS = (\[.*?\]);", html, re.DOTALL)
    assert match is not None
    parsed = json.loads(match.group(1))
    assert len(parsed) == 3
