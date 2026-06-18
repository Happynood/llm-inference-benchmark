"""Tests for Pareto dominance analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.compare import RunRow
from llm_inference_benchmark.pareto import (
    build_pareto_json,
    build_pareto_table,
    dominates,
    pareto_classify,
    render_pareto_json,
    render_pareto_table,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _row(
    backend: str = "mock",
    model: str = "m",
    p95: float = 10.0,
    toks: float = 100.0,
    vram: float | None = None,
    sanity: float | None = None,
    ppl: float | None = None,
    judge: float | None = None,
) -> RunRow:
    return RunRow(
        backend=backend,
        model=model,
        request_count=10,
        p50_latency_ms=p95 * 0.95,
        p95_latency_ms=p95,
        tokens_per_second=toks,
        peak_cpu_memory_mb=500.0,
        peak_cuda_memory_mb=None,
        peak_vram_memory_mb=vram,
        sanity_pass_rate=sanity,
        perplexity=ppl,
        judge_score=judge,
    )


# ---------------------------------------------------------------------------
# dominates()
# ---------------------------------------------------------------------------


def test_dominates_better_p95_same_toks() -> None:
    a = _row(p95=9.0, toks=100.0)
    b = _row(p95=10.0, toks=100.0)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_better_toks_same_p95() -> None:
    a = _row(p95=10.0, toks=110.0)
    b = _row(p95=10.0, toks=100.0)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_not_dominates_equal_rows() -> None:
    a = _row(p95=10.0, toks=100.0)
    b = _row(p95=10.0, toks=100.0)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_not_dominates_tradeoff() -> None:
    """Lower p95 but lower tok/s — neither dominates the other."""
    a = _row(p95=9.0, toks=90.0)
    b = _row(p95=11.0, toks=110.0)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_better_on_both_mandatory() -> None:
    a = _row(p95=9.0, toks=110.0)
    b = _row(p95=10.0, toks=100.0)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_includes_vram_when_both_present() -> None:
    a = _row(p95=10.0, toks=100.0, vram=2000.0)
    b = _row(p95=10.0, toks=100.0, vram=3000.0)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_vram_none_in_one_excludes_vram() -> None:
    """When one row has no VRAM data, VRAM must not affect the comparison."""
    # Equal on mandatory metrics, VRAM excluded → neither dominates
    a = _row(p95=10.0, toks=100.0, vram=None)
    b = _row(p95=10.0, toks=100.0, vram=3000.0)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_both_vram_none_uses_mandatory_only() -> None:
    a = _row(p95=9.0, toks=100.0, vram=None)
    b = _row(p95=10.0, toks=100.0, vram=None)
    assert dominates(a, b)


def test_dominates_includes_sanity_when_both_present() -> None:
    a = _row(p95=10.0, toks=100.0, sanity=1.0)
    b = _row(p95=10.0, toks=100.0, sanity=0.8)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_sanity_none_in_one_excludes_sanity() -> None:
    a = _row(p95=10.0, toks=100.0, sanity=None)
    b = _row(p95=10.0, toks=100.0, sanity=0.8)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_all_four_metrics_strict() -> None:
    a = _row(p95=9.0, toks=110.0, vram=2000.0, sanity=1.0)
    b = _row(p95=10.0, toks=100.0, vram=3000.0, sanity=0.9)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_equal_vram_other_strict() -> None:
    """Equal VRAM, strictly better on p95 — should still dominate."""
    a = _row(p95=9.0, toks=100.0, vram=2000.0)
    b = _row(p95=10.0, toks=100.0, vram=2000.0)
    assert dominates(a, b)


def test_dominates_includes_ppl_when_both_present() -> None:
    """Lower perplexity (more confident/fluent) is better."""
    a = _row(p95=10.0, toks=100.0, ppl=8.0)
    b = _row(p95=10.0, toks=100.0, ppl=12.0)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_ppl_none_in_one_excludes_ppl() -> None:
    a = _row(p95=10.0, toks=100.0, ppl=None)
    b = _row(p95=10.0, toks=100.0, ppl=12.0)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_includes_judge_when_both_present() -> None:
    """Higher judge score (more confident in its own relevance) is better."""
    a = _row(p95=10.0, toks=100.0, judge=0.9)
    b = _row(p95=10.0, toks=100.0, judge=0.5)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_judge_none_in_one_excludes_judge() -> None:
    a = _row(p95=10.0, toks=100.0, judge=None)
    b = _row(p95=10.0, toks=100.0, judge=0.9)
    assert not dominates(a, b)
    assert not dominates(b, a)


# ---------------------------------------------------------------------------
# pareto_classify()
# ---------------------------------------------------------------------------


def test_classify_empty_returns_empty() -> None:
    assert pareto_classify([]) == []


def test_classify_single_row_is_optimal() -> None:
    row = _row()
    result = pareto_classify([row])
    assert result == [(row, True)]


def test_classify_all_optimal_tradeoffs() -> None:
    rows = [
        _row(p95=9.0, toks=80.0),
        _row(p95=15.0, toks=120.0),
    ]
    classified = pareto_classify(rows)
    statuses = {r.p95_latency_ms: is_opt for r, is_opt in classified}
    assert statuses[9.0] is True
    assert statuses[15.0] is True


def test_classify_one_dominated() -> None:
    rows = [
        _row(p95=9.0, toks=110.0),
        _row(p95=10.0, toks=100.0),
    ]
    classified = pareto_classify(rows)
    statuses = {r.p95_latency_ms: is_opt for r, is_opt in classified}
    assert statuses[9.0] is True
    assert statuses[10.0] is False


def test_classify_two_dominated_one_optimal() -> None:
    rows = [
        _row(p95=5.0, toks=200.0),
        _row(p95=10.0, toks=100.0),
        _row(p95=12.0, toks=90.0),
    ]
    classified = pareto_classify(rows)
    statuses = {r.p95_latency_ms: is_opt for r, is_opt in classified}
    assert statuses[5.0] is True
    assert statuses[10.0] is False
    assert statuses[12.0] is False


def test_classify_preserves_input_order() -> None:
    rows = [_row(backend=f"b{i}", p95=float(i + 1), toks=float(100 - i * 5)) for i in range(5)]
    classified = pareto_classify(rows)
    assert [r for r, _ in classified] == rows


def test_classify_missing_vram_does_not_crash() -> None:
    rows = [
        _row(p95=10.0, toks=100.0, vram=2000.0),
        _row(p95=12.0, toks=100.0, vram=None),
    ]
    result = pareto_classify(rows)
    assert len(result) == 2


def test_classify_missing_sanity_does_not_crash() -> None:
    rows = [
        _row(p95=10.0, toks=100.0, sanity=1.0),
        _row(p95=12.0, toks=100.0, sanity=None),
    ]
    result = pareto_classify(rows)
    assert len(result) == 2


def test_classify_vram_tiebreaks_when_mandatory_equal() -> None:
    """When p95 and tok/s are equal, VRAM decides dominance."""
    low_vram = _row(p95=10.0, toks=100.0, vram=2000.0)
    high_vram = _row(p95=10.0, toks=100.0, vram=4000.0)
    classified = pareto_classify([low_vram, high_vram])
    statuses = {r.peak_vram_memory_mb: is_opt for r, is_opt in classified}
    assert statuses[2000.0] is True
    assert statuses[4000.0] is False


# ---------------------------------------------------------------------------
# render_pareto_table()
# ---------------------------------------------------------------------------


def test_render_has_pareto_header() -> None:
    classified = pareto_classify([_row()])
    table = render_pareto_table(classified)
    assert "Pareto" in table


def test_render_optimal_marker_in_correct_row() -> None:
    row = _row(backend="fast", p95=9.0, toks=110.0)
    dominated = _row(backend="slow", p95=10.0, toks=100.0)
    classified = [(row, True), (dominated, False)]
    table = render_pareto_table(classified)
    lines = table.splitlines()
    data_lines = lines[2:]  # skip header + separator
    assert "optimal" in data_lines[0]
    assert "optimal" not in data_lines[1]


def test_render_row_count() -> None:
    rows = [_row(backend=f"b{i}", p95=float(i + 1) * 2, toks=float(100 - i * 10)) for i in range(3)]
    classified = pareto_classify(rows)
    lines = render_pareto_table(classified).splitlines()
    assert len(lines) == 3 + 2  # header + separator + 3 data rows


def test_render_vram_na_when_none() -> None:
    classified = [(_row(vram=None), True)]
    table = render_pareto_table(classified)
    assert "N/A" in table


def test_render_vram_value_shown() -> None:
    classified = [(_row(vram=2361.0), True)]
    table = render_pareto_table(classified)
    assert "2361.0" in table


def test_render_sanity_na_when_none() -> None:
    classified = [(_row(sanity=None), True)]
    table = render_pareto_table(classified)
    assert "N/A" in table


def test_render_sanity_percentage() -> None:
    classified = [(_row(sanity=1.0), True)]
    table = render_pareto_table(classified)
    assert "100.0%" in table


def test_render_partial_sanity_percentage() -> None:
    classified = [(_row(sanity=0.8), True)]
    table = render_pareto_table(classified)
    assert "80.0%" in table


def test_render_ppl_na_when_none() -> None:
    classified = [(_row(ppl=None), True)]
    table = render_pareto_table(classified)
    assert "N/A" in table


def test_render_ppl_value_shown() -> None:
    classified = [(_row(ppl=12.34), True)]
    table = render_pareto_table(classified)
    assert "12.34" in table


def test_render_judge_na_when_none() -> None:
    classified = [(_row(judge=None), True)]
    table = render_pareto_table(classified)
    assert "N/A" in table


def test_render_judge_value_shown() -> None:
    classified = [(_row(judge=0.85), True)]
    table = render_pareto_table(classified)
    assert "85.0%" in table


# ---------------------------------------------------------------------------
# build_pareto_table()
# ---------------------------------------------------------------------------


def test_build_pareto_table_from_fixtures() -> None:
    table = build_pareto_table(
        [
            FIXTURES / "mock_run.csv",
            FIXTURES / "transformers_run.csv",
        ]
    )
    assert "Pareto" in table
    assert "mock" in table
    assert "transformers" in table


def test_build_pareto_table_mock_dominates_transformers() -> None:
    """Mock row (p95=5.09, tok/s=9971) dominates transformers (p95=44.67, tok/s=1211)."""
    table = build_pareto_table(
        [
            FIXTURES / "mock_run.csv",
            FIXTURES / "transformers_run.csv",
        ]
    )
    lines = table.splitlines()
    # Find which data line contains 'mock-gpt2' vs 'tiny-gpt2'
    mock_line = next(line for line in lines if "mock-gpt2" in line)
    xfm_line = next(line for line in lines if "tiny-gpt2" in line)
    assert "optimal" in mock_line
    assert "optimal" not in xfm_line


def test_build_pareto_table_empty_raises() -> None:
    with pytest.raises(ValueError, match="At least one CSV"):
        build_pareto_table([])


def test_build_pareto_table_single_csv_is_optimal() -> None:
    table = build_pareto_table([FIXTURES / "mock_run.csv"])
    assert "optimal" in table


def test_build_pareto_table_with_quality_fixture() -> None:
    table = build_pareto_table([FIXTURES / "mock_run_with_quality.csv"])
    assert "Sanity %" in table
    assert "100.0%" in table


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------


def test_pareto_subcommand_stdout() -> None:
    result = CliRunner().invoke(
        main,
        ["pareto", str(FIXTURES / "mock_run.csv"), str(FIXTURES / "transformers_run.csv")],
    )
    assert result.exit_code == 0, result.output
    assert "Pareto" in result.output
    assert "mock" in result.output


def test_pareto_subcommand_output_file(tmp_path: Path) -> None:
    out = tmp_path / "pareto.md"
    result = CliRunner().invoke(
        main,
        ["pareto", str(FIXTURES / "mock_run.csv"), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Pareto" in out.read_text()


def test_pareto_subcommand_no_files_fails() -> None:
    result = CliRunner().invoke(main, ["pareto"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# render_pareto_json()
# ---------------------------------------------------------------------------


def test_render_pareto_json_is_valid_json() -> None:
    classified = pareto_classify([_row(backend="a"), _row(backend="b", p95=20.0, toks=80.0)])
    output = render_pareto_json(classified)
    parsed = json.loads(output)
    assert isinstance(parsed, list)
    assert len(parsed) == 2


def test_render_pareto_json_pareto_field_correct() -> None:
    optimal = _row(backend="fast", p95=9.0, toks=110.0)
    dominated = _row(backend="slow", p95=10.0, toks=100.0)
    classified = [(optimal, True), (dominated, False)]
    parsed = json.loads(render_pareto_json(classified))
    by_backend = {obj["backend"]: obj for obj in parsed}
    assert by_backend["fast"]["pareto"] is True
    assert by_backend["slow"]["pareto"] is False


def test_render_pareto_json_contains_required_fields() -> None:
    classified = pareto_classify([_row()])
    parsed = json.loads(render_pareto_json(classified))
    obj = parsed[0]
    required = (
        "backend",
        "model",
        "request_count",
        "p95_latency_ms",
        "tokens_per_second",
        "pareto",
    )
    for field in required:
        assert field in obj, f"missing field: {field}"


def test_render_pareto_json_optional_fields_are_none_when_absent() -> None:
    classified = pareto_classify([_row(vram=None, sanity=None, ppl=None, judge=None)])
    parsed = json.loads(render_pareto_json(classified))
    obj = parsed[0]
    assert obj["peak_vram_memory_mb"] is None
    assert obj["sanity_pass_rate"] is None
    assert obj["perplexity"] is None
    assert obj["judge_score"] is None


def test_render_pareto_json_optional_fields_present_when_set() -> None:
    classified = pareto_classify([_row(vram=2048.0, sanity=0.9, ppl=8.5, judge=0.75)])
    parsed = json.loads(render_pareto_json(classified))
    obj = parsed[0]
    assert obj["peak_vram_memory_mb"] == pytest.approx(2048.0)
    assert obj["sanity_pass_rate"] == pytest.approx(0.9)
    assert obj["perplexity"] == pytest.approx(8.5)
    assert obj["judge_score"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# build_pareto_json()
# ---------------------------------------------------------------------------


def test_build_pareto_json_from_fixtures() -> None:
    output = build_pareto_json(
        [
            FIXTURES / "mock_run.csv",
            FIXTURES / "transformers_run.csv",
        ]
    )
    parsed = json.loads(output)
    assert isinstance(parsed, list)
    backends = {obj["backend"] for obj in parsed}
    assert "mock" in backends
    assert "transformers" in backends


def test_build_pareto_json_mock_is_optimal() -> None:
    parsed = json.loads(
        build_pareto_json(
            [
                FIXTURES / "mock_run.csv",
                FIXTURES / "transformers_run.csv",
            ]
        )
    )
    by_model = {obj["model"]: obj for obj in parsed}
    assert by_model["mock-gpt2"]["pareto"] is True
    assert by_model["sshleifer/tiny-gpt2"]["pareto"] is False


def test_build_pareto_json_empty_raises() -> None:
    with pytest.raises(ValueError, match="At least one CSV"):
        build_pareto_json([])


# ---------------------------------------------------------------------------
# CLI --format json
# ---------------------------------------------------------------------------


def test_pareto_subcommand_format_json_stdout() -> None:
    result = CliRunner().invoke(
        main,
        ["pareto", str(FIXTURES / "mock_run.csv"), "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
    assert parsed[0]["pareto"] is True


def test_pareto_subcommand_format_json_output_file(tmp_path: Path) -> None:
    out = tmp_path / "pareto.json"
    result = CliRunner().invoke(
        main,
        ["pareto", str(FIXTURES / "mock_run.csv"), "--format", "json", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    parsed = json.loads(out.read_text())
    assert isinstance(parsed, list)
    assert "pareto" in parsed[0]


def test_pareto_subcommand_default_format_is_table() -> None:
    result = CliRunner().invoke(
        main,
        ["pareto", str(FIXTURES / "mock_run.csv")],
    )
    assert result.exit_code == 0, result.output
    assert "Pareto" in result.output
    assert result.output.strip().startswith("|")
