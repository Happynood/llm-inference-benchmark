"""Tests for lifecycle and variance metric surfacing in compare, pareto, recommend (v0.23)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.compare import RunRow, load_csv, render_table
from llm_inference_benchmark.pareto import dominates, pareto_classify, render_pareto_table
from llm_inference_benchmark.recommend import (
    Constraints,
    apply_constraints,
    recommend,
    render_recommendation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_FIELDNAMES = [
    "request_count",
    "p50_latency_ms",
    "p95_latency_ms",
    "tokens_per_second",
    "total_tokens",
    "backend",
    "model",
    "peak_cpu_memory_mb",
    "peak_cuda_memory_mb",
    "peak_vram_memory_mb",
    "empty_output_count",
    "min_output_chars",
    "mean_output_chars",
    "repeated_output_count",
    "sanity_pass_rate",
    "perplexity",
    "judge_score",
    "timestamp",
]


def _make_row(
    *,
    backend: str = "mock",
    model: str = "m",
    p95: float = 100.0,
    toks: float = 50.0,
    vram: float | None = None,
    model_load_ms: float | None = None,
    p95_std: float | None = None,
    toks_std: float | None = None,
) -> RunRow:
    return RunRow(
        backend=backend,
        model=model,
        request_count=10,
        p50_latency_ms=p95 * 0.9,
        p95_latency_ms=p95,
        tokens_per_second=toks,
        peak_cpu_memory_mb=400.0,
        peak_cuda_memory_mb=None,
        peak_vram_memory_mb=vram,
        model_load_ms=model_load_ms,
        p95_latency_ms_std=p95_std,
        tokens_per_second_std=toks_std,
    )


def _write_csv(
    path: Path,
    row: RunRow,
    *,
    include_lifecycle: bool = True,
    include_variance: bool = True,
) -> Path:
    """Write a RunRow to a CSV, optionally omitting lifecycle/variance columns."""
    fieldnames = list(_BASE_FIELDNAMES)
    if include_lifecycle:
        fieldnames += ["model_load_ms", "warmup_p50_latency_ms"]
    if include_variance:
        fieldnames += ["repeats", "p95_latency_ms_std", "tokens_per_second_std"]

    def _s(v: object) -> str:
        return "" if v is None else str(v)

    data: dict[str, str] = {
        "request_count": str(row.request_count),
        "p50_latency_ms": str(row.p50_latency_ms),
        "p95_latency_ms": str(row.p95_latency_ms),
        "tokens_per_second": str(row.tokens_per_second),
        "total_tokens": "0",
        "backend": row.backend,
        "model": row.model,
        "peak_cpu_memory_mb": str(row.peak_cpu_memory_mb),
        "peak_cuda_memory_mb": "",
        "peak_vram_memory_mb": _s(row.peak_vram_memory_mb),
        "empty_output_count": "0",
        "min_output_chars": "10",
        "mean_output_chars": "10.0",
        "repeated_output_count": "0",
        "sanity_pass_rate": "1.0",
        "perplexity": "",
        "judge_score": "",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    if include_lifecycle:
        data["model_load_ms"] = _s(row.model_load_ms)
        data["warmup_p50_latency_ms"] = ""
    if include_variance:
        data["repeats"] = ""
        data["p95_latency_ms_std"] = _s(row.p95_latency_ms_std)
        data["tokens_per_second_std"] = _s(row.tokens_per_second_std)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)
    return path


# ---------------------------------------------------------------------------
# compare: Load (ms) column
# ---------------------------------------------------------------------------


def test_compare_table_has_load_column() -> None:
    row = _make_row(model_load_ms=123.4)
    table = render_table([row])
    assert "Load (ms)" in table
    assert "123.4" in table


def test_compare_table_load_na_when_none() -> None:
    row = _make_row(model_load_ms=None)
    table = render_table([row])
    assert "Load (ms)" in table
    assert "N/A" in table


def test_load_csv_reads_model_load_ms(tmp_path: Path) -> None:
    row = _make_row(model_load_ms=456.7)
    p = _write_csv(tmp_path / "r.csv", row)
    loaded = load_csv(p)
    assert loaded.model_load_ms == pytest.approx(456.7)


def test_load_csv_model_load_ms_none_when_column_absent(tmp_path: Path) -> None:
    """Pre-v0.18 CSVs without model_load_ms column → None."""
    row = _make_row()
    p = _write_csv(tmp_path / "r.csv", row, include_lifecycle=False)
    loaded = load_csv(p)
    assert loaded.model_load_ms is None


def test_load_csv_model_load_ms_none_when_blank(tmp_path: Path) -> None:
    """Blank model_load_ms cell → None."""
    row = _make_row(model_load_ms=None)
    p = _write_csv(tmp_path / "r.csv", row)
    loaded = load_csv(p)
    assert loaded.model_load_ms is None


# ---------------------------------------------------------------------------
# compare: variance ± formatting
# ---------------------------------------------------------------------------


def test_compare_table_p95_shows_std_when_present() -> None:
    row = _make_row(p95=100.0, p95_std=2.5)
    table = render_table([row])
    assert "100.00 ± 2.50" in table


def test_compare_table_toks_shows_std_when_present() -> None:
    row = _make_row(toks=50.0, toks_std=1.3)
    table = render_table([row])
    assert "50.0 ± 1.3" in table


def test_compare_table_p95_no_std_plain_when_none() -> None:
    row = _make_row(p95=100.0, p95_std=None)
    table = render_table([row])
    assert "100.00 ±" not in table
    assert "100.00" in table


def test_load_csv_reads_variance_columns(tmp_path: Path) -> None:
    row = _make_row(p95_std=3.1, toks_std=0.8)
    p = _write_csv(tmp_path / "r.csv", row)
    loaded = load_csv(p)
    assert loaded.p95_latency_ms_std == pytest.approx(3.1)
    assert loaded.tokens_per_second_std == pytest.approx(0.8)


def test_load_csv_variance_none_when_columns_absent(tmp_path: Path) -> None:
    """Pre-v0.19 CSVs without variance columns → None."""
    row = _make_row()
    p = _write_csv(tmp_path / "r.csv", row, include_variance=False)
    loaded = load_csv(p)
    assert loaded.p95_latency_ms_std is None
    assert loaded.tokens_per_second_std is None


# ---------------------------------------------------------------------------
# compare: CLI --output smoke test
# ---------------------------------------------------------------------------


def test_cli_compare_shows_load_column(tmp_path: Path) -> None:
    row = _make_row(model_load_ms=77.5)
    p = _write_csv(tmp_path / "r.csv", row)
    result = CliRunner().invoke(main, ["compare", str(p)])
    assert result.exit_code == 0
    assert "Load (ms)" in result.output
    assert "77.5" in result.output


# ---------------------------------------------------------------------------
# pareto: model_load_ms in dominance
# ---------------------------------------------------------------------------


def test_dominates_model_load_ms_considered() -> None:
    fast_load = _make_row(model="fast", p95=100.0, toks=50.0, model_load_ms=200.0)
    slow_load = _make_row(model="slow", p95=100.0, toks=50.0, model_load_ms=500.0)
    assert dominates(fast_load, slow_load)
    assert not dominates(slow_load, fast_load)


def test_dominates_model_load_ms_skipped_when_either_none() -> None:
    """If one row lacks model_load_ms, the metric is excluded from comparison."""
    with_load = _make_row(p95=100.0, toks=50.0, model_load_ms=200.0)
    no_load = _make_row(p95=100.0, toks=50.0, model_load_ms=None)
    # Neither dominates the other when both p95 and tok/s are equal and load is absent for one.
    assert not dominates(with_load, no_load)
    assert not dominates(no_load, with_load)


def test_pareto_table_has_load_column() -> None:
    rows = [_make_row(model="a", model_load_ms=100.0), _make_row(model="b", model_load_ms=200.0)]
    classified = pareto_classify(rows)
    table = render_pareto_table(classified)
    assert "Load (ms)" in table
    assert "100.0" in table
    assert "200.0" in table


def test_pareto_table_load_na_when_none() -> None:
    rows = [_make_row(model_load_ms=None)]
    table = render_pareto_table(pareto_classify(rows))
    assert "Load (ms)" in table
    assert "N/A" in table


# ---------------------------------------------------------------------------
# recommend: --max-load-ms constraint
# ---------------------------------------------------------------------------


def test_constraints_max_load_ms_excludes_too_high() -> None:
    rows = [
        _make_row(model="fast", model_load_ms=100.0),
        _make_row(model="slow", model_load_ms=600.0),
    ]
    candidates, excluded = apply_constraints(rows, Constraints(max_load_ms=500.0))
    assert len(candidates) == 1
    assert candidates[0].model == "fast"
    assert len(excluded) == 1
    assert "load time too high" in excluded[0].reason
    assert "600.0 ms" in excluded[0].reason


def test_constraints_max_load_ms_excludes_unknown() -> None:
    """Runs with model_load_ms=None are excluded when max_load_ms is set."""
    rows = [
        _make_row(model="known", model_load_ms=100.0),
        _make_row(model="unknown", model_load_ms=None),
    ]
    candidates, excluded = apply_constraints(rows, Constraints(max_load_ms=500.0))
    assert len(candidates) == 1
    assert candidates[0].model == "known"
    assert "load time unknown" in excluded[0].reason


def test_constraints_max_load_ms_not_set_allows_none() -> None:
    """Without the constraint, runs with model_load_ms=None are not excluded."""
    rows = [_make_row(model_load_ms=None), _make_row(model_load_ms=1000.0)]
    candidates, excluded = apply_constraints(rows, Constraints())
    assert len(candidates) == 2
    assert excluded == []


def test_recommend_load_shown_in_output() -> None:
    row = _make_row(model="a", model_load_ms=350.5)
    result = recommend([row], Constraints())
    text = render_recommendation(result)
    assert "Load" in text
    assert "350.5 ms" in text


def test_recommend_load_na_when_none() -> None:
    row = _make_row(model="a", model_load_ms=None)
    result = recommend([row], Constraints())
    text = render_recommendation(result)
    assert "Load" in text
    assert "N/A" in text


# ---------------------------------------------------------------------------
# recommend: CLI --max-load-ms smoke test
# ---------------------------------------------------------------------------


def test_cli_recommend_max_load_ms_excludes(tmp_path: Path) -> None:
    fast = _make_row(model="fast", p95=100.0, model_load_ms=100.0)
    slow = _make_row(model="slow", p95=200.0, model_load_ms=2000.0)
    p1 = _write_csv(tmp_path / "fast.csv", fast)
    p2 = _write_csv(tmp_path / "slow.csv", slow)
    result = CliRunner().invoke(main, ["recommend", str(p1), str(p2), "--max-load-ms", "500"])
    assert result.exit_code == 0
    assert "fast" in result.output
    assert "load time too high" in result.output


def test_cli_recommend_max_load_ms_no_winner(tmp_path: Path) -> None:
    row = _make_row(model="a", model_load_ms=5000.0)
    p = _write_csv(tmp_path / "a.csv", row)
    result = CliRunner().invoke(main, ["recommend", str(p), "--max-load-ms", "100"])
    assert result.exit_code == 1
    assert "load time too high" in result.output
