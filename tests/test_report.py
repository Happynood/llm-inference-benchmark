"""Tests for the HTML benchmark report generator."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.compare import RunRow
from llm_inference_benchmark.report import _row_label, _short_model, build_report_html, load_runs

FIXTURES = Path(__file__).parent / "fixtures"


def _row(
    backend: str = "mock",
    model: str = "models/test-q4.gguf",
    toks: float = 100.0,
    p95: float = 200.0,
    vram: float | None = None,
    sanity: float | None = None,
    load: float | None = None,
    ttft: float | None = None,
    hw_cpu: str | None = None,
    hw_gpu: str | None = None,
) -> RunRow:
    return RunRow(
        backend=backend,
        model=model,
        request_count=10,
        p50_latency_ms=p95 * 0.8,
        p95_latency_ms=p95,
        tokens_per_second=toks,
        peak_cpu_memory_mb=512.0,
        peak_cuda_memory_mb=None,
        peak_vram_memory_mb=vram,
        sanity_pass_rate=sanity,
        model_load_ms=load,
        p50_ttft_ms=ttft,
        hw_cpu=hw_cpu,
        hw_gpu=hw_gpu,
    )


# ── _short_model ──────────────────────────────────────────────────────────────


def test_short_model_long_path() -> None:
    assert _short_model("/home/user/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf") == (
        "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    )


def test_short_model_hf_id() -> None:
    assert _short_model("HuggingFaceTB/SmolLM2-135M-Instruct") == (
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )


def test_short_model_single_segment() -> None:
    assert _short_model("gpt2") == "gpt2"


# ── _row_label ────────────────────────────────────────────────────────────────


def test_row_label_combines_backend_and_short_model() -> None:
    row = _row(backend="llama-cpp", model="/home/user/models/Llama.gguf")
    assert _row_label(row) == "llama-cpp/models/Llama.gguf"


# ── build_report_html ─────────────────────────────────────────────────────────


def test_empty_rows_returns_placeholder_html() -> None:
    html = build_report_html([], title="Empty")
    assert "<!DOCTYPE html>" in html
    assert "No benchmark runs provided" in html
    assert "Empty" in html


def test_single_run_has_metrics_table() -> None:
    html = build_report_html([_row(toks=120.5, p95=300.0)])
    assert "metrics-table" in html
    assert "120.5" in html
    assert "300.0" in html


def test_title_appears_in_html() -> None:
    html = build_report_html([_row()], title="My Custom Report")
    assert "My Custom Report" in html


def test_plotly_cdn_included() -> None:
    html = build_report_html([_row()])
    assert "plot.ly" in html


def test_pareto_optimal_badge_present_for_single_run() -> None:
    html = build_report_html([_row()])
    assert "badge-pareto" in html
    assert "optimal" in html


def test_pareto_badge_on_dominant_run() -> None:
    fast = _row(toks=200.0, p95=100.0)
    slow = _row(backend="transformers", toks=50.0, p95=800.0)
    html = build_report_html([fast, slow])
    assert "badge-pareto" in html


def test_two_runs_both_appear_in_table() -> None:
    r1 = _row(backend="llama-cpp", toks=150.0)
    r2 = _row(backend="transformers", toks=80.0)
    html = build_report_html([r1, r2])
    assert "llama-cpp" in html
    assert "transformers" in html


def test_runs_sorted_by_throughput_descending() -> None:
    slow = _row(backend="slow", toks=50.0)
    fast = _row(backend="fast", toks=150.0)
    html = build_report_html([slow, fast])
    assert html.index("fast") < html.index("slow")


def test_sanity_rate_formatted_as_percentage() -> None:
    html = build_report_html([_row(sanity=0.95)])
    assert "95.0%" in html


def test_missing_optional_fields_render_dash() -> None:
    html = build_report_html([_row(vram=None, ttft=None)])
    assert "—" in html


def test_hardware_section_shown_when_hw_cpu_set() -> None:
    html = build_report_html([_row(hw_cpu="Intel Core i9", hw_gpu="RTX 3050")])
    assert "Hardware" in html
    assert "Intel Core i9" in html
    assert "RTX 3050" in html


def test_no_hardware_section_when_hw_absent() -> None:
    html = build_report_html([_row(hw_cpu=None, hw_gpu=None)])
    assert "Hardware" not in html


def test_chart_data_embedded_in_html() -> None:
    row = _row(toks=120.0, p95=250.0)
    html = build_report_html([row])
    assert '"x"' in html
    assert '"y"' in html
    assert "120.0" in html


def test_chart_not_rendered_when_metrics_missing() -> None:
    row = RunRow(
        backend="mock",
        model="m",
        request_count=1,
        p50_latency_ms=10.0,
        p95_latency_ms=15.0,
        tokens_per_second=0.0,
        peak_cpu_memory_mb=100.0,
        peak_cuda_memory_mb=None,
    )
    html = build_report_html([row])
    assert "<!DOCTYPE html>" in html


# ── load_runs ─────────────────────────────────────────────────────────────────


def test_load_runs_single_csv(tmp_path: Path) -> None:
    csv_content = FIXTURES / "mock_run.csv"
    rows = load_runs([csv_content])
    assert len(rows) == 1
    assert rows[0].backend == "mock"


def test_load_runs_multiple_csvs(tmp_path: Path) -> None:
    csv = FIXTURES / "mock_run.csv"
    rows = load_runs([csv, csv])
    assert len(rows) == 2


def test_load_runs_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_runs(["/nonexistent/path/run.csv"])


def test_load_runs_malformed_csv_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("not,valid\ncsv,data\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_runs([bad])


# ── CLI command ───────────────────────────────────────────────────────────────


def test_report_cmd_creates_html_file(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "report.html"
    result = runner.invoke(
        main,
        ["report", str(FIXTURES / "mock_run.csv"), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "<!DOCTYPE html>" in out.read_text()


def test_report_cmd_default_output_name(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            main,
            ["report", str(FIXTURES / "mock_run.csv")],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "report.html" in result.output


def test_report_cmd_custom_title(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "r.html"
    result = runner.invoke(
        main,
        ["report", str(FIXTURES / "mock_run.csv"), "--output", str(out), "--title", "Q4 vs Q8"],
    )
    assert result.exit_code == 0
    assert "Q4 vs Q8" in out.read_text()


def test_report_cmd_multiple_csvs(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "multi.html"
    result = runner.invoke(
        main,
        [
            "report",
            str(FIXTURES / "mock_run.csv"),
            str(FIXTURES / "mock_run_with_quality.csv"),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    content = out.read_text()
    assert "2 run(s)" in result.output
    assert "<!DOCTYPE html>" in content


def test_report_cmd_missing_file_shows_error() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["report", "/nonexistent/run.csv"])
    assert result.exit_code != 0
