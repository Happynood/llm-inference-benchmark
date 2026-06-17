"""Tests for diff.py and the llm-bench diff subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.diff import (
    _fmt_change,
    _pct_change,
    build_diff_table,
    find_regressions,
)

FIXTURES = Path(__file__).parent / "fixtures"
MOCK_CSV = FIXTURES / "mock_run.csv"
TRANSFORMERS_CSV = FIXTURES / "transformers_run.csv"
QUALITY_CSV = FIXTURES / "mock_run_with_quality.csv"


def _write_csv(path: Path, **fields: str) -> None:
    """Write a minimal valid benchmark CSV to path."""
    base: dict[str, str] = {
        "request_count": "10",
        "p50_latency_ms": "100.0",
        "p95_latency_ms": "120.0",
        "tokens_per_second": "50.0",
        "total_tokens": "500",
        "backend": "mock",
        "model": "mock-model",
        "peak_cpu_memory_mb": "100.0",
        "peak_cuda_memory_mb": "",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    base.update(fields)
    header = ",".join(base.keys())
    data = ",".join(base.values())
    path.write_text(f"{header}\n{data}\n")


# ---------------------------------------------------------------------------
# _pct_change
# ---------------------------------------------------------------------------


def test_pct_change_decrease() -> None:
    assert _pct_change(100.0, 80.0) == pytest.approx(-20.0)


def test_pct_change_increase() -> None:
    assert _pct_change(100.0, 120.0) == pytest.approx(20.0)


def test_pct_change_zero_baseline_returns_none() -> None:
    assert _pct_change(0.0, 50.0) is None


def test_pct_change_no_change() -> None:
    assert _pct_change(100.0, 100.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _fmt_change
# ---------------------------------------------------------------------------


def test_fmt_change_improvement_lower_is_better() -> None:
    result = _fmt_change(100.0, 80.0, lower_is_better=True)
    assert "✓" in result
    assert "✗" not in result


def test_fmt_change_regression_lower_is_better() -> None:
    result = _fmt_change(100.0, 120.0, lower_is_better=True)
    assert "✗" in result
    assert "✓" not in result


def test_fmt_change_improvement_higher_is_better() -> None:
    result = _fmt_change(50.0, 60.0, lower_is_better=False)
    assert "✓" in result
    assert "✗" not in result


def test_fmt_change_regression_higher_is_better() -> None:
    result = _fmt_change(50.0, 40.0, lower_is_better=False)
    assert "✗" in result
    assert "✓" not in result


def test_fmt_change_no_change_no_annotation() -> None:
    result = _fmt_change(100.0, 100.0, lower_is_better=True)
    assert "✓" not in result
    assert "✗" not in result


def test_fmt_change_none_baseline_returns_na() -> None:
    assert _fmt_change(None, 50.0, lower_is_better=True) == "N/A"


def test_fmt_change_none_current_returns_na() -> None:
    assert _fmt_change(50.0, None, lower_is_better=True) == "N/A"


def test_fmt_change_contains_sign() -> None:
    assert "+" in _fmt_change(100.0, 120.0, lower_is_better=True)
    assert "-" in _fmt_change(100.0, 80.0, lower_is_better=True)


# ---------------------------------------------------------------------------
# build_diff_table — structure and required metrics
# ---------------------------------------------------------------------------


def test_build_diff_table_has_header(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    table = build_diff_table(a, b)
    assert "## Benchmark Diff" in table
    assert "Baseline" in table
    assert "Current" in table


def test_build_diff_table_shows_filenames(tmp_path: Path) -> None:
    a = tmp_path / "before.csv"
    b = tmp_path / "after.csv"
    _write_csv(a)
    _write_csv(b)
    table = build_diff_table(a, b)
    assert "before.csv" in table
    assert "after.csv" in table


def test_build_diff_table_has_required_metrics(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    table = build_diff_table(a, b)
    assert "p50 (ms)" in table
    assert "p95 (ms)" in table
    assert "tok/s" in table
    assert "CPU mem (MB)" in table


def test_build_diff_table_has_legend(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    table = build_diff_table(a, b)
    assert "✓" in table
    assert "✗" in table


def test_build_diff_table_is_markdown_table(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    table = build_diff_table(a, b)
    lines = table.splitlines()
    table_lines = [ln for ln in lines if ln.startswith("|")]
    assert len(table_lines) >= 3  # header, separator, at least one data row


# ---------------------------------------------------------------------------
# Optional metrics — shown or hidden based on data presence
# ---------------------------------------------------------------------------


def test_build_diff_ttft_hidden_when_absent(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    table = build_diff_table(a, b)
    assert "TTFT" not in table


def test_build_diff_ttft_shown_when_present(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p50_ttft_ms="40.0", p95_ttft_ms="80.0")
    _write_csv(b, p50_ttft_ms="35.0", p95_ttft_ms="70.0")
    table = build_diff_table(a, b)
    assert "TTFT p50 (ms)" in table
    assert "TTFT p95 (ms)" in table


def test_build_diff_ttft_one_side_missing(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p50_ttft_ms="40.0", p95_ttft_ms="80.0")
    _write_csv(b)  # no TTFT
    table = build_diff_table(a, b)
    assert "TTFT p50 (ms)" in table
    assert "N/A" in table


def test_build_diff_vram_hidden_when_absent(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    table = build_diff_table(a, b)
    assert "VRAM (MB)" not in table


def test_build_diff_vram_shown_when_present(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, peak_vram_memory_mb="2361.0")
    _write_csv(b, peak_vram_memory_mb="2200.0")
    table = build_diff_table(a, b)
    assert "VRAM (MB)" in table


def test_build_diff_load_ms_shown_when_present(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, model_load_ms="1200.0")
    _write_csv(b, model_load_ms="1100.0")
    table = build_diff_table(a, b)
    assert "Load (ms)" in table


def test_build_diff_sanity_shown_when_present(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, sanity_pass_rate="0.9")
    _write_csv(b, sanity_pass_rate="0.95")
    table = build_diff_table(a, b)
    assert "Sanity %" in table


def test_build_diff_sanity_hidden_when_absent(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    table = build_diff_table(a, b)
    assert "Sanity" not in table


# ---------------------------------------------------------------------------
# Improvement / regression annotations
# ---------------------------------------------------------------------------


def test_build_diff_latency_improvement_marked(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="120.0")
    _write_csv(b, p95_latency_ms="100.0")  # 16.7% faster
    table = build_diff_table(a, b)
    # The p95 row should be marked as improvement
    p95_line = next(ln for ln in table.splitlines() if "p95 (ms)" in ln)
    assert "✓" in p95_line


def test_build_diff_latency_regression_marked(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="100.0")
    _write_csv(b, p95_latency_ms="130.0")  # 30% slower
    table = build_diff_table(a, b)
    p95_line = next(ln for ln in table.splitlines() if "p95 (ms)" in ln)
    assert "✗" in p95_line


def test_build_diff_toks_improvement_marked(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, tokens_per_second="50.0")
    _write_csv(b, tokens_per_second="60.0")  # higher = better
    table = build_diff_table(a, b)
    toks_line = next(ln for ln in table.splitlines() if "tok/s" in ln and "Decode" not in ln)
    assert "✓" in toks_line


def test_build_diff_toks_regression_marked(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, tokens_per_second="60.0")
    _write_csv(b, tokens_per_second="50.0")  # lower = bad
    table = build_diff_table(a, b)
    toks_line = next(ln for ln in table.splitlines() if "tok/s" in ln and "Decode" not in ln)
    assert "✗" in toks_line


def test_build_diff_shows_pct_values(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="100.0")
    _write_csv(b, p95_latency_ms="80.0")
    table = build_diff_table(a, b)
    assert "-20.0%" in table


# ---------------------------------------------------------------------------
# Smoke test with real fixture files
# ---------------------------------------------------------------------------


def test_build_diff_table_two_fixtures() -> None:
    table = build_diff_table(MOCK_CSV, TRANSFORMERS_CSV)
    assert "## Benchmark Diff" in table
    assert "p95 (ms)" in table
    assert "mock" in table
    assert "transformers" in table


def test_build_diff_same_csv_no_change() -> None:
    table = build_diff_table(MOCK_CSV, MOCK_CSV)
    # Data rows (lines starting with "|") should contain no regression markers
    data_rows = [ln for ln in table.splitlines() if ln.startswith("| ") and "Metric" not in ln]
    assert all("✗" not in row for row in data_rows)


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------


def test_diff_subcommand_stdout(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    result = CliRunner().invoke(main, ["diff", str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert "## Benchmark Diff" in result.output
    assert "p95 (ms)" in result.output


def test_diff_subcommand_output_file(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "diff.md"
    _write_csv(a)
    _write_csv(b)
    result = CliRunner().invoke(main, ["diff", str(a), str(b), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "## Benchmark Diff" in out.read_text()


def test_diff_subcommand_two_fixtures() -> None:
    result = CliRunner().invoke(main, ["diff", str(MOCK_CSV), str(TRANSFORMERS_CSV)])
    assert result.exit_code == 0, result.output
    assert "Benchmark Diff" in result.output
    assert "mock" in result.output


def test_diff_subcommand_no_args_fails() -> None:
    result = CliRunner().invoke(main, ["diff"])
    assert result.exit_code != 0


def test_diff_subcommand_one_arg_fails(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    _write_csv(a)
    result = CliRunner().invoke(main, ["diff", str(a)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# find_regressions — unit tests
# ---------------------------------------------------------------------------


def test_find_regressions_none_when_identical(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    _write_csv(a)
    assert find_regressions(a, a) == []


def test_find_regressions_latency_increase(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p50_latency_ms="100.0", p95_latency_ms="120.0")
    _write_csv(b, p50_latency_ms="130.0", p95_latency_ms="150.0")  # slower = regression
    regressions = find_regressions(a, b)
    assert "p50 (ms)" in regressions
    assert "p95 (ms)" in regressions


def test_find_regressions_throughput_decrease(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, tokens_per_second="60.0")
    _write_csv(b, tokens_per_second="50.0")  # lower tok/s = regression
    assert "tok/s" in find_regressions(a, b)


def test_find_regressions_throughput_increase_not_regression(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, tokens_per_second="50.0")
    _write_csv(b, tokens_per_second="60.0")  # higher = improvement
    assert "tok/s" not in find_regressions(a, b)


def test_find_regressions_threshold_filters_small_change(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="100.0")
    _write_csv(b, p95_latency_ms="103.0")  # 3% regression
    assert find_regressions(a, b, threshold_pct=5.0) == []  # within tolerance
    assert "p95 (ms)" in find_regressions(a, b, threshold_pct=0.0)  # caught at 0% threshold


def test_find_regressions_optional_metric_absent_both_skipped(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a)
    _write_csv(b)
    regressions = find_regressions(a, b)
    assert "TTFT p50 (ms)" not in regressions
    assert "VRAM (MB)" not in regressions


def test_find_regressions_optional_metric_present(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p50_ttft_ms="40.0", p95_ttft_ms="80.0")
    _write_csv(b, p50_ttft_ms="55.0", p95_ttft_ms="90.0")  # higher TTFT = worse
    regressions = find_regressions(a, b)
    assert "TTFT p50 (ms)" in regressions
    assert "TTFT p95 (ms)" in regressions


# ---------------------------------------------------------------------------
# CLI --fail-on-regression
# ---------------------------------------------------------------------------


def test_diff_fail_on_regression_exits_1_when_regressed(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="100.0")
    _write_csv(b, p95_latency_ms="130.0")  # 30% regression
    result = CliRunner().invoke(main, ["diff", str(a), str(b), "--fail-on-regression", "0"])
    assert result.exit_code == 1
    assert "✗ Regression check failed" in result.output


def test_diff_fail_on_regression_exits_0_when_no_regression(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="100.0")
    _write_csv(b, p95_latency_ms="90.0")  # improvement
    result = CliRunner().invoke(main, ["diff", str(a), str(b), "--fail-on-regression", "0"])
    assert result.exit_code == 0
    assert "✗ Regression check failed" not in result.output


def test_diff_fail_on_regression_threshold_respected(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="100.0")
    _write_csv(b, p95_latency_ms="104.0")  # 4% regression
    # 5% threshold — 4% is within tolerance
    result = CliRunner().invoke(main, ["diff", str(a), str(b), "--fail-on-regression", "5"])
    assert result.exit_code == 0
    # 3% threshold — 4% exceeds it
    result2 = CliRunner().invoke(main, ["diff", str(a), str(b), "--fail-on-regression", "3"])
    assert result2.exit_code == 1


def test_diff_fail_on_regression_lists_metrics_in_output(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p50_latency_ms="100.0", tokens_per_second="60.0")
    _write_csv(b, p50_latency_ms="120.0", tokens_per_second="50.0")
    result = CliRunner().invoke(main, ["diff", str(a), str(b), "--fail-on-regression", "0"])
    assert result.exit_code == 1
    assert "p50 (ms)" in result.output
    assert "tok/s" in result.output


def test_diff_fail_on_regression_still_prints_table(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="100.0")
    _write_csv(b, p95_latency_ms="130.0")
    result = CliRunner().invoke(main, ["diff", str(a), str(b), "--fail-on-regression", "0"])
    assert result.exit_code == 1
    assert "## Benchmark Diff" in result.output  # table still printed


def test_diff_without_flag_exits_0_even_with_regression(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_csv(a, p95_latency_ms="100.0")
    _write_csv(b, p95_latency_ms="200.0")  # massive regression
    result = CliRunner().invoke(main, ["diff", str(a), str(b)])
    assert result.exit_code == 0  # no --fail-on-regression flag = never exits 1


def test_diff_fail_on_regression_negative_threshold_rejected(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    _write_csv(a)
    result = CliRunner().invoke(main, ["diff", str(a), str(a), "--fail-on-regression", "-1"])
    assert result.exit_code != 0
