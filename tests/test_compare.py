"""Tests for compare.py and the llm-bench compare subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.compare import (
    RunRow,
    build_comparison_table,
    filter_rows,
    load_csv,
    render_csv,
    render_json,
    render_table,
    sort_rows,
)

FIXTURES = Path(__file__).parent / "fixtures"
MOCK_CSV = FIXTURES / "mock_run.csv"
TRANSFORMERS_CSV = FIXTURES / "transformers_run.csv"
QUALITY_CSV = FIXTURES / "mock_run_with_quality.csv"


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


def test_load_csv_mock_fixture() -> None:
    row = load_csv(MOCK_CSV)
    assert row.backend == "mock"
    assert row.model == "mock-gpt2"
    assert row.request_count == 20
    assert row.p50_latency_ms == pytest.approx(5.01)
    assert row.p95_latency_ms == pytest.approx(5.09)
    assert row.tokens_per_second == pytest.approx(9971.18)
    assert row.peak_cpu_memory_mb == pytest.approx(45.2)
    assert row.peak_cuda_memory_mb is None  # empty string in CSV → None
    assert row.peak_vram_memory_mb is None  # column absent in older CSV → None


def test_load_csv_transformers_fixture() -> None:
    row = load_csv(TRANSFORMERS_CSV)
    assert row.backend == "transformers"
    assert row.peak_cuda_memory_mb == pytest.approx(0.0)


def test_load_csv_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("request_count,backend\n")  # header only, no data rows
    with pytest.raises(ValueError, match="No data rows"):
        load_csv(p)


def test_load_csv_multiple_rows_raises(tmp_path: Path) -> None:
    p = tmp_path / "multi.csv"
    p.write_text(MOCK_CSV.read_text() + "20,5.0,5.1,9000,1000,mock,m2,40.0,,2026-01-01\n")
    with pytest.raises(ValueError, match="Expected 1 data row"):
        load_csv(p)


def test_load_csv_missing_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "missing_col.csv"
    p.write_text("backend,model\nmock,gpt2\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_csv(p)


def test_load_csv_whitespace_cuda_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad_cuda.csv"
    header = "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,timestamp\n"  # noqa: E501
    p.write_text(header + "10,5.0,5.1,9000,500,mock,m,45.0,   ,2026-01-01\n")
    with pytest.raises(ValueError, match="invalid peak_cuda_memory_mb"):
        load_csv(p)


def test_load_csv_with_vram_column(tmp_path: Path) -> None:
    """CSVs that include peak_vram_memory_mb load the value correctly."""
    p = tmp_path / "vram.csv"
    header = "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,peak_vram_memory_mb,timestamp\n"  # noqa: E501
    p.write_text(header + "10,900.0,940.0,53.71,500,llama-cpp,llama3,800.0,,2361.0,2026-01-01\n")
    row = load_csv(p)
    assert row.peak_vram_memory_mb == pytest.approx(2361.0)
    assert row.peak_cuda_memory_mb is None


def test_load_csv_whitespace_vram_raises(tmp_path: Path) -> None:
    """Whitespace-only peak_vram_memory_mb is rejected as invalid."""
    p = tmp_path / "bad_vram.csv"
    header = "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,peak_vram_memory_mb,timestamp\n"  # noqa: E501
    p.write_text(header + "10,5.0,5.1,9000,500,mock,m,45.0,,   ,2026-01-01\n")
    with pytest.raises(ValueError, match="invalid peak_vram_memory_mb"):
        load_csv(p)


def test_load_csv_non_numeric_float_field_raises(tmp_path: Path) -> None:
    """A non-numeric value in an optional float column raises ValueError (not whitespace)."""
    p = tmp_path / "bad_float.csv"
    header = (
        "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,"
        "total_tokens,backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,timestamp\n"
    )
    p.write_text(header + "10,5.0,5.1,9000,500,mock,m,45.0,not-a-number,2026-01-01\n")
    with pytest.raises(ValueError, match="invalid peak_cuda_memory_mb"):
        load_csv(p)


def test_build_comparison_table_empty_paths_raises() -> None:
    with pytest.raises(ValueError, match="At least one CSV path"):
        build_comparison_table([])


def test_load_csv_perplexity_absent_is_none() -> None:
    """Older CSVs without a perplexity column load with perplexity=None."""
    row = load_csv(MOCK_CSV)
    assert row.perplexity is None


def test_load_csv_with_perplexity_column(tmp_path: Path) -> None:
    p = tmp_path / "ppl.csv"
    header = "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,perplexity,timestamp\n"  # noqa: E501
    p.write_text(header + "10,40.0,44.0,1200.0,500,transformers,tiny,720.0,,12.34,2026-01-01\n")
    row = load_csv(p)
    assert row.perplexity == pytest.approx(12.34)


def test_load_csv_perplexity_blank_is_none(tmp_path: Path) -> None:
    p = tmp_path / "ppl_blank.csv"
    header = "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,perplexity,timestamp\n"  # noqa: E501
    p.write_text(header + "10,40.0,44.0,1200.0,500,mock,m,45.0,,,2026-01-01\n")
    row = load_csv(p)
    assert row.perplexity is None


def test_load_csv_judge_score_absent_is_none() -> None:
    """Older CSVs without a judge_score column load with judge_score=None."""
    row = load_csv(MOCK_CSV)
    assert row.judge_score is None


def test_load_csv_with_judge_score_column(tmp_path: Path) -> None:
    p = tmp_path / "judge.csv"
    header = "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,judge_score,timestamp\n"  # noqa: E501
    p.write_text(header + "10,40.0,44.0,1200.0,500,transformers,tiny,720.0,,0.85,2026-01-01\n")
    row = load_csv(p)
    assert row.judge_score == pytest.approx(0.85)


def test_load_csv_judge_score_blank_is_none(tmp_path: Path) -> None:
    p = tmp_path / "judge_blank.csv"
    header = "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,judge_score,timestamp\n"  # noqa: E501
    p.write_text(header + "10,40.0,44.0,1200.0,500,mock,m,45.0,,,2026-01-01\n")
    row = load_csv(p)
    assert row.judge_score is None


# ---------------------------------------------------------------------------
# sort_rows
# ---------------------------------------------------------------------------

_ROWS = [
    RunRow("transformers", "tiny-gpt2", 10, 40.0, 44.0, 1200.0, 720.0, 0.0, 1024.0),
    RunRow("mock", "mock-gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None),
    RunRow("onnx", "bert-base", 15, 10.0, 12.0, 5000.0, 200.0, None, None),
]


def test_sort_by_p95_ascending() -> None:
    result = sort_rows(_ROWS, sort_by="p95")
    p95s = [r.p95_latency_ms for r in result]
    assert p95s == sorted(p95s)


def test_sort_by_backend_alphabetical() -> None:
    result = sort_rows(_ROWS, sort_by="backend")
    backends = [r.backend for r in result]
    assert backends == sorted(backends)


def test_sort_by_model_alphabetical() -> None:
    result = sort_rows(_ROWS, sort_by="model")
    models = [r.model for r in result]
    assert models == sorted(models)


def test_sort_does_not_mutate_input() -> None:
    original_order = [r.backend for r in _ROWS]
    sort_rows(_ROWS, sort_by="p95")
    assert [r.backend for r in _ROWS] == original_order


def test_sort_by_toks_descending() -> None:
    result = sort_rows(_ROWS, sort_by="toks")
    toks = [r.tokens_per_second for r in result]
    assert toks == sorted(toks, reverse=True)


def test_sort_by_load_ascending_none_last() -> None:
    rows_with_load = [
        RunRow("a", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, model_load_ms=300.0),
        RunRow("b", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, model_load_ms=None),
        RunRow("c", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, model_load_ms=100.0),
    ]
    result = sort_rows(rows_with_load, sort_by="load")
    load_times = [r.model_load_ms for r in result]
    assert load_times == [100.0, 300.0, None]


def test_sort_by_load_all_none_preserves_relative_order() -> None:
    rows = [
        RunRow("a", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, model_load_ms=None),
        RunRow("b", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, model_load_ms=None),
    ]
    result = sort_rows(rows, sort_by="load")
    assert [r.backend for r in result] == ["a", "b"]


def test_sort_by_ttft_ascending_none_last() -> None:
    rows = [
        RunRow("a", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, p50_ttft_ms=80.0),
        RunRow("b", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, p50_ttft_ms=None),
        RunRow("c", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, p50_ttft_ms=30.0),
    ]
    result = sort_rows(rows, sort_by="ttft")
    assert [r.backend for r in result] == ["c", "a", "b"]
    assert [r.p50_ttft_ms for r in result] == [30.0, 80.0, None]


def test_sort_by_ttft_all_none_preserves_relative_order() -> None:
    rows = [
        RunRow("a", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, p50_ttft_ms=None),
        RunRow("b", "m", 10, 5.0, 6.0, 100.0, 50.0, None, None, p50_ttft_ms=None),
    ]
    result = sort_rows(rows, sort_by="ttft")
    assert [r.backend for r in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# render_table
# ---------------------------------------------------------------------------


def test_render_table_has_header_and_separator() -> None:
    table = render_table(_ROWS[:1])
    lines = table.splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").replace("-", "").strip()) == set()  # only dashes


def test_render_table_contains_expected_columns() -> None:
    table = render_table(_ROWS[:1])
    for col in [
        "Backend",
        "Model",
        "N",
        "p50 (ms)",
        "p95 (ms)",
        "tok/s",
        "CPU mem (MB)",
        "CUDA mem (MB)",
        "VRAM mem (MB)",
    ]:
        assert col in table


def test_render_table_cuda_none_col_suppressed() -> None:
    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    assert "CUDA mem" not in render_table(rows)


def test_render_table_cuda_zero_shows_value() -> None:
    rows = [RunRow("transformers", "tiny", 10, 40.0, 44.0, 1200.0, 720.0, 0.0, 0.0)]
    table = render_table(rows)
    assert "| 0.0" in table or "0.0 |" in table


def test_render_table_vram_none_col_suppressed() -> None:
    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    assert "VRAM mem" not in render_table(rows)


def test_render_table_vram_value_shown() -> None:
    rows = [RunRow("llama-cpp", "llama3", 10, 900.0, 940.0, 53.0, 800.0, None, 2361.0)]
    table = render_table(rows)
    assert "2361.0" in table


def test_render_table_row_count() -> None:
    table = render_table(_ROWS)
    lines = table.splitlines()
    assert len(lines) == len(_ROWS) + 2  # header + separator + data rows


def test_render_table_mandatory_cols_always_shown() -> None:
    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    table = render_table(rows)
    for col in ("Backend", "Model", "N", "p50 (ms)", "p95 (ms)", "tok/s", "CPU mem (MB)"):
        assert col in table, f"Mandatory column {col!r} missing from table"


def test_render_table_optional_col_shown_when_any_row_has_value() -> None:
    import dataclasses

    row_no_ttft = _ROWS[0]
    row_with_ttft = dataclasses.replace(_ROWS[1], p50_ttft_ms=42.0, p95_ttft_ms=80.0)
    table = render_table([row_no_ttft, row_with_ttft])
    assert "TTFT p50" in table
    assert "N/A" in table  # row_no_ttft shows N/A in the TTFT column


def test_render_table_ppl_col_suppressed_when_all_none() -> None:
    table = render_table(_ROWS[:1])
    assert "PPL" not in table


def test_render_table_ppl_col_shown_when_present() -> None:
    import dataclasses

    row_with_ppl = dataclasses.replace(_ROWS[0], perplexity=12.34)
    table = render_table([row_with_ppl])
    assert "PPL" in table


def test_render_table_ppl_value_shown() -> None:
    rows = [
        RunRow(
            "transformers",
            "tiny-gpt2",
            10,
            40.0,
            44.0,
            1200.0,
            720.0,
            0.0,
            None,
            None,
            None,
            None,
            perplexity=12.34,
        )
    ]
    table = render_table(rows)
    assert "12.34" in table


def test_render_table_judge_col_suppressed_when_all_none() -> None:
    table = render_table(_ROWS[:1])
    assert "Judge" not in table


def test_render_table_judge_col_shown_when_present() -> None:
    import dataclasses

    row_with_judge = dataclasses.replace(_ROWS[0], judge_score=0.75)
    table = render_table([row_with_judge])
    assert "Judge" in table


def test_render_table_judge_value_shown() -> None:
    rows = [
        RunRow(
            "transformers",
            "tiny-gpt2",
            10,
            40.0,
            44.0,
            1200.0,
            720.0,
            0.0,
            None,
            None,
            None,
            None,
            perplexity=None,
            judge_score=0.85,
        )
    ]
    table = render_table(rows)
    assert "85.0%" in table


# ---------------------------------------------------------------------------
# build_comparison_table (integration)
# ---------------------------------------------------------------------------


def test_build_comparison_table_two_fixtures() -> None:
    table = build_comparison_table([MOCK_CSV, TRANSFORMERS_CSV], sort_by="p95")
    assert "mock" in table
    assert "transformers" in table
    # mock p95=5.09 < transformers p95=44.67, so mock appears first
    assert table.index("mock") < table.index("transformers")


def test_build_comparison_table_single_csv() -> None:
    table = build_comparison_table([MOCK_CSV])
    assert "mock-gpt2" in table


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------


def test_compare_subcommand_stdout() -> None:
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--sort", "p95"]
    )
    assert result.exit_code == 0, result.output
    assert "Backend" in result.output
    assert "mock" in result.output
    assert "transformers" in result.output


def test_compare_subcommand_output_file(tmp_path: Path) -> None:
    out = tmp_path / "table.md"
    result = CliRunner().invoke(main, ["compare", str(MOCK_CSV), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    content = out.read_text()
    assert "mock-gpt2" in content


def test_compare_subcommand_sort_backend() -> None:
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--sort", "backend"]
    )
    assert result.exit_code == 0, result.output
    # "mock" < "transformers" alphabetically → mock row appears first in output
    assert result.output.index("mock-gpt2") < result.output.index("transformers")


def test_compare_subcommand_sort_ttft_exits_zero() -> None:
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--sort", "ttft"]
    )
    assert result.exit_code == 0, result.output
    assert "Backend" in result.output


def test_compare_subcommand_no_files_fails() -> None:
    result = CliRunner().invoke(main, ["compare"])
    assert result.exit_code != 0


def test_compare_subcommand_limit_one() -> None:
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--sort", "p95", "--limit", "1"]
    )
    assert result.exit_code == 0, result.output
    # Only one data row should appear; both backends cannot both be present
    has_mock = "mock" in result.output
    has_transformers = "transformers" in result.output
    assert has_mock != has_transformers, "exactly one backend row expected with --limit 1"


def test_compare_subcommand_limit_exceeds_count() -> None:
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--limit", "100"]
    )
    assert result.exit_code == 0, result.output
    assert "mock" in result.output
    assert "transformers" in result.output


def test_compare_subcommand_limit_json() -> None:
    import json as _json

    result = CliRunner().invoke(
        main,
        ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--format", "json", "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    data = _json.loads(result.output)
    assert len(data) == 1


def test_compare_subcommand_limit_zero_rejected() -> None:
    result = CliRunner().invoke(main, ["compare", str(MOCK_CSV), "--limit", "0"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# filter_rows unit tests
# ---------------------------------------------------------------------------

_FILTER_ROWS = [
    RunRow("llama_cpp", "Llama-3.2-3B-Q4_K_M", 10, 20.0, 22.0, 39.0, 800.0, None, 2361.0),
    RunRow("transformers", "tiny-gpt2", 10, 40.0, 44.0, 1200.0, 720.0, 0.0, 1024.0),
    RunRow("mock", "mock-gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None),
]


def test_filter_rows_no_filters_returns_all() -> None:
    assert filter_rows(_FILTER_ROWS, []) == _FILTER_ROWS


def test_filter_rows_by_backend_exact() -> None:
    result = filter_rows(_FILTER_ROWS, ["backend=mock"])
    assert len(result) == 1
    assert result[0].backend == "mock"


def test_filter_rows_by_backend_substring() -> None:
    result = filter_rows(_FILTER_ROWS, ["backend=llama"])
    assert len(result) == 1
    assert result[0].backend == "llama_cpp"


def test_filter_rows_by_model_substring() -> None:
    result = filter_rows(_FILTER_ROWS, ["model=gpt2"])
    assert len(result) == 2
    backends = {r.backend for r in result}
    assert backends == {"transformers", "mock"}


def test_filter_rows_case_insensitive() -> None:
    result = filter_rows(_FILTER_ROWS, ["backend=MOCK"])
    assert len(result) == 1
    assert result[0].backend == "mock"


def test_filter_rows_multiple_and_semantics() -> None:
    result = filter_rows(_FILTER_ROWS, ["model=gpt2", "backend=mock"])
    assert len(result) == 1
    assert result[0].backend == "mock"


def test_filter_rows_no_match_returns_empty() -> None:
    result = filter_rows(_FILTER_ROWS, ["backend=vllm"])
    assert result == []


def test_filter_rows_invalid_field_raises() -> None:
    with pytest.raises(ValueError, match="Unknown filter field"):
        filter_rows(_FILTER_ROWS, ["p95=100"])


def test_filter_rows_missing_equals_raises() -> None:
    with pytest.raises(ValueError, match="expected FIELD=PATTERN"):
        filter_rows(_FILTER_ROWS, ["backendmock"])


# ---------------------------------------------------------------------------
# --filter CLI tests
# ---------------------------------------------------------------------------


def test_compare_filter_by_backend() -> None:
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--filter", "backend=mock"]
    )
    assert result.exit_code == 0
    assert "mock" in result.output
    assert "transformers" not in result.output


def test_compare_filter_by_model_substring() -> None:
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--filter", "model=gpt2"]
    )
    assert result.exit_code == 0
    assert "gpt2" in result.output.lower()


def test_compare_filter_no_match_empty_table() -> None:
    result = CliRunner().invoke(main, ["compare", str(MOCK_CSV), "--filter", "backend=vllm"])
    assert result.exit_code == 0
    # Header line still present; no data rows
    assert "Backend" in result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip() and not ln.startswith("|")]
    assert lines == []


def test_compare_filter_invalid_field_usage_error() -> None:
    result = CliRunner().invoke(main, ["compare", str(MOCK_CSV), "--filter", "p95=100"])
    assert result.exit_code != 0
    assert "Unknown filter field" in result.output


def test_compare_filter_composes_with_limit() -> None:
    result = CliRunner().invoke(
        main,
        ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--filter", "model=gpt2", "--limit", "1"],
    )
    assert result.exit_code == 0
    data_lines = [ln for ln in result.output.splitlines() if ln.startswith("|") and "---" not in ln]
    assert len(data_lines) == 2  # header + 1 data row


def test_compare_filter_composes_with_json() -> None:
    result = CliRunner().invoke(
        main,
        [
            "compare",
            str(MOCK_CSV),
            str(TRANSFORMERS_CSV),
            "--filter",
            "backend=mock",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    import json

    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["backend"] == "mock"


# ---------------------------------------------------------------------------
# Backward-compat: existing llm-bench --config ... still works
# ---------------------------------------------------------------------------


def test_existing_run_behavior_preserved(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config)])
    assert result.exit_code == 0, result.output
    assert "Benchmark Results" in result.output


def test_no_args_exits_nonzero() -> None:
    result = CliRunner().invoke(main, [])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Backward compatibility: old CSVs (no quality columns) → sanity_pass_rate=None
# ---------------------------------------------------------------------------


def test_load_old_csv_sanity_pass_rate_is_none() -> None:
    """Old CSV without sanity_pass_rate column loads cleanly with None."""
    row = load_csv(MOCK_CSV)
    assert row.sanity_pass_rate is None


def test_load_old_transformers_csv_sanity_pass_rate_is_none() -> None:
    row = load_csv(TRANSFORMERS_CSV)
    assert row.sanity_pass_rate is None


def test_render_table_old_csv_sanity_col_suppressed() -> None:
    row = load_csv(MOCK_CSV)
    table = render_table([row])
    assert "Sanity %" not in table


# ---------------------------------------------------------------------------
# New quality fixture
# ---------------------------------------------------------------------------


def test_load_quality_csv_sanity_pass_rate() -> None:
    row = load_csv(QUALITY_CSV)
    assert row.sanity_pass_rate == pytest.approx(1.0)
    assert row.peak_vram_memory_mb is None  # empty string in CSV → None


def test_render_table_quality_shows_percentage() -> None:
    row = load_csv(QUALITY_CSV)
    table = render_table([row])
    assert "100.0%" in table


def test_render_table_sanity_col_suppressed_when_all_none() -> None:
    table = render_table(_ROWS[:1])
    assert "Sanity %" not in table


def test_build_comparison_table_with_quality_csv() -> None:
    table = build_comparison_table([QUALITY_CSV])
    assert "100.0%" in table
    assert "Sanity %" in table


def test_build_comparison_table_mixed_old_and_new() -> None:
    """Mixing old (no quality) and new (with quality) CSVs must not error."""
    table = build_comparison_table([MOCK_CSV, QUALITY_CSV], sort_by="p95")
    assert "mock" in table
    assert "N/A" in table  # old row has no sanity
    assert "100.0%" in table  # new row has sanity=1.0


# ---------------------------------------------------------------------------
# Workload composition fields (v0.22)
# ---------------------------------------------------------------------------


def test_load_csv_old_fixture_decode_fields_are_none() -> None:
    row = load_csv(MOCK_CSV)
    assert row.mean_input_tokens is None
    assert row.mean_output_tokens is None
    assert row.decode_tokens_per_second is None


def test_load_csv_new_row_with_decode_fields(tmp_path: Path) -> None:
    header = (
        "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,"
        "backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,"
        "mean_input_tokens,mean_output_tokens,decode_tokens_per_second,timestamp\n"
    )
    data = "10,5.0,5.1,9000.0,150,mock,m,45.0,,5.0,10.0,9000.0,2026-01-01T00:00:00+00:00\n"
    p = tmp_path / "new.csv"
    p.write_text(header + data)
    row = load_csv(p)
    assert row.mean_input_tokens == pytest.approx(5.0)
    assert row.mean_output_tokens == pytest.approx(10.0)
    assert row.decode_tokens_per_second == pytest.approx(9000.0)


def test_load_csv_blank_decode_tokens_per_second(tmp_path: Path) -> None:
    header = (
        "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,"
        "backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,"
        "mean_input_tokens,mean_output_tokens,decode_tokens_per_second,timestamp\n"
    )
    data = "5,5.0,5.1,0.0,25,mock,m,45.0,,5.0,0.0,,2026-01-01T00:00:00+00:00\n"
    p = tmp_path / "noop.csv"
    p.write_text(header + data)
    row = load_csv(p)
    assert row.decode_tokens_per_second is None


def test_render_table_out_tok_s_suppressed_when_all_none() -> None:
    table = render_table(_ROWS[:1])
    assert "Out tok/s" not in table
    assert "In tok" not in table
    assert "Out tok" not in table


def test_render_table_decode_tps_cols_suppressed_for_old_csv() -> None:
    row = load_csv(MOCK_CSV)
    table = render_table([row])
    assert "Out tok/s" not in table


def test_load_csv_ttft_columns_blank_for_old_csv() -> None:
    row = load_csv(MOCK_CSV)
    assert row.p50_ttft_ms is None
    assert row.p95_ttft_ms is None


def test_load_csv_parses_ttft_columns(tmp_path: Path) -> None:
    header = (
        "request_count,p50_latency_ms,p95_latency_ms,tokens_per_second,total_tokens,"
        "backend,model,peak_cpu_memory_mb,peak_cuda_memory_mb,peak_vram_memory_mb,"
        "empty_output_count,min_output_chars,mean_output_chars,repeated_output_count,"
        "sanity_pass_rate,task_quality_pass_rate,task_quality_checked_count,"
        "model_load_ms,warmup_p50_latency_ms,p50_ttft_ms,p95_ttft_ms,"
        "repeats,p95_latency_ms_std,tokens_per_second_std,perplexity,judge_score,"
        "mean_input_tokens,mean_output_tokens,decode_tokens_per_second,timestamp\n"
    )
    data = (
        "10,12.5,18.0,80.0,100,openai,my-model,200.0,,,"
        "0,5,20.0,0,1.0,,,,"
        ",8.3,15.1,"
        ",,,,,"
        "5.0,10.0,,2026-01-01T00:00:00+00:00\n"
    )
    p = tmp_path / "ttft.csv"
    p.write_text(header + data)
    row = load_csv(p)
    assert row.p50_ttft_ms == pytest.approx(8.3)
    assert row.p95_ttft_ms == pytest.approx(15.1)


def test_render_table_ttft_cols_suppressed_when_all_none() -> None:
    table = render_table(_ROWS[:1])
    assert "TTFT p50" not in table
    assert "TTFT p95" not in table


def test_render_table_ttft_cols_suppressed_for_old_csv() -> None:
    row = load_csv(MOCK_CSV)
    table = render_table([row])
    assert "TTFT p50" not in table


# ---------------------------------------------------------------------------
# render_json
# ---------------------------------------------------------------------------


def test_render_json_returns_valid_json() -> None:
    import json

    out = render_json(_ROWS)
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == len(_ROWS)


def test_render_json_required_fields_present() -> None:
    import json

    parsed = json.loads(render_json(_ROWS[:1]))
    record = parsed[0]
    for key in (
        "backend",
        "model",
        "request_count",
        "p50_latency_ms",
        "p95_latency_ms",
        "tokens_per_second",
    ):
        assert key in record, f"Missing key: {key}"


def test_render_json_optional_fields_null_when_absent() -> None:
    import json

    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    parsed = json.loads(render_json(rows))
    record = parsed[0]
    assert record["peak_cuda_memory_mb"] is None
    assert record["peak_vram_memory_mb"] is None
    assert record["p50_ttft_ms"] is None
    assert record["model_load_ms"] is None
    assert record["perplexity"] is None
    assert record["judge_score"] is None


def test_render_json_optional_fields_present_when_set() -> None:
    import dataclasses
    import json

    row_with_opts = dataclasses.replace(
        _ROWS[0],
        p50_ttft_ms=42.0,
        p95_ttft_ms=80.0,
        model_load_ms=1234.5,
        perplexity=12.34,
        judge_score=0.85,
        peak_vram_memory_mb=2048.0,
    )
    parsed = json.loads(render_json([row_with_opts]))
    record = parsed[0]
    assert record["p50_ttft_ms"] == pytest.approx(42.0)
    assert record["p95_ttft_ms"] == pytest.approx(80.0)
    assert record["model_load_ms"] == pytest.approx(1234.5)
    assert record["perplexity"] == pytest.approx(12.34)
    assert record["judge_score"] == pytest.approx(0.85)
    assert record["peak_vram_memory_mb"] == pytest.approx(2048.0)


def test_render_json_values_match_row() -> None:
    import json

    row = _ROWS[0]
    parsed = json.loads(render_json([row]))
    record = parsed[0]
    assert record["backend"] == row.backend
    assert record["model"] == row.model
    assert record["request_count"] == row.request_count
    assert record["p50_latency_ms"] == pytest.approx(row.p50_latency_ms)
    assert record["tokens_per_second"] == pytest.approx(row.tokens_per_second)


def test_render_json_respects_sort_order() -> None:
    import json

    rows = sort_rows(_ROWS, sort_by="toks")
    parsed = json.loads(render_json(rows))
    toks = [r["tokens_per_second"] for r in parsed]
    assert toks == sorted(toks, reverse=True)


# ---------------------------------------------------------------------------
# CLI --format json
# ---------------------------------------------------------------------------


def test_compare_subcommand_format_json_stdout() -> None:
    import json

    result = CliRunner().invoke(main, ["compare", str(MOCK_CSV), "--format", "json"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["backend"] == "mock"
    assert parsed[0]["model"] == "mock-gpt2"


def test_compare_subcommand_format_json_two_files() -> None:
    import json

    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert len(parsed) == 2
    backends = {r["backend"] for r in parsed}
    assert backends == {"mock", "transformers"}


def test_compare_subcommand_format_json_file_output(tmp_path: Path) -> None:
    import json

    out = tmp_path / "results.json"
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), "--format", "json", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    parsed = json.loads(out.read_text())
    assert isinstance(parsed, list)
    assert parsed[0]["backend"] == "mock"


def test_compare_subcommand_format_table_is_default() -> None:
    result = CliRunner().invoke(main, ["compare", str(MOCK_CSV)])
    assert result.exit_code == 0, result.output
    assert "Backend" in result.output  # Markdown table header


def test_compare_subcommand_format_json_sorted_by_toks() -> None:
    import json

    result = CliRunner().invoke(
        main,
        ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--format", "json", "--sort", "toks"],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    toks = [r["tokens_per_second"] for r in parsed]
    assert toks == sorted(toks, reverse=True)


# ---------------------------------------------------------------------------
# render_csv unit tests
# ---------------------------------------------------------------------------


def test_render_csv_header_row() -> None:
    import csv as _csv
    import io

    out = render_csv(_ROWS[:1])
    reader = _csv.DictReader(io.StringIO(out))
    assert reader.fieldnames is not None
    for field in (
        "backend",
        "model",
        "request_count",
        "p50_latency_ms",
        "p95_latency_ms",
        "tokens_per_second",
        "peak_cpu_memory_mb",
    ):
        assert field in reader.fieldnames, f"Missing field: {field}"


def test_render_csv_row_count() -> None:
    out = render_csv(_ROWS)
    lines = out.splitlines()
    assert len(lines) == len(_ROWS) + 1  # header + data rows


def test_render_csv_empty_rows_header_only() -> None:
    out = render_csv([])
    lines = out.splitlines()
    assert len(lines) == 1
    assert "backend" in lines[0]


def test_render_csv_optional_fields_empty_when_absent() -> None:
    import csv as _csv
    import io

    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    out = render_csv(rows)
    record = next(_csv.DictReader(io.StringIO(out)))
    assert record["peak_cuda_memory_mb"] == ""
    assert record["peak_vram_memory_mb"] == ""
    assert record["p50_ttft_ms"] == ""
    assert record["model_load_ms"] == ""
    assert record["perplexity"] == ""
    assert record["judge_score"] == ""


def test_render_csv_optional_fields_present_when_set() -> None:
    import csv as _csv
    import dataclasses
    import io

    row = dataclasses.replace(
        _ROWS[0],
        p50_ttft_ms=42.0,
        p95_ttft_ms=80.0,
        model_load_ms=1234.5,
        perplexity=12.34,
        judge_score=0.85,
        peak_vram_memory_mb=2048.0,
    )
    out = render_csv([row])
    record = next(_csv.DictReader(io.StringIO(out)))
    assert float(record["p50_ttft_ms"]) == pytest.approx(42.0)
    assert float(record["p95_ttft_ms"]) == pytest.approx(80.0)
    assert float(record["model_load_ms"]) == pytest.approx(1234.5)
    assert float(record["perplexity"]) == pytest.approx(12.34)
    assert float(record["judge_score"]) == pytest.approx(0.85)
    assert float(record["peak_vram_memory_mb"]) == pytest.approx(2048.0)


def test_render_csv_values_match_row() -> None:
    import csv as _csv
    import io

    row = _ROWS[0]
    out = render_csv([row])
    record = next(_csv.DictReader(io.StringIO(out)))
    assert record["backend"] == row.backend
    assert record["model"] == row.model
    assert int(record["request_count"]) == row.request_count
    assert float(record["p50_latency_ms"]) == pytest.approx(row.p50_latency_ms)
    assert float(record["tokens_per_second"]) == pytest.approx(row.tokens_per_second)


def test_render_csv_parseable_round_trip() -> None:
    import csv as _csv
    import io

    out = render_csv(_ROWS)
    records = list(_csv.DictReader(io.StringIO(out)))
    assert len(records) == len(_ROWS)
    backends = [r["backend"] for r in records]
    assert set(backends) == {r.backend for r in _ROWS}


# ---------------------------------------------------------------------------
# CLI --format csv
# ---------------------------------------------------------------------------


def test_compare_subcommand_format_csv_stdout() -> None:
    result = CliRunner().invoke(main, ["compare", str(MOCK_CSV), "--format", "csv"])
    assert result.exit_code == 0, result.output
    assert "backend" in result.output
    assert "mock" in result.output
    assert "mock-gpt2" in result.output


def test_compare_subcommand_format_csv_two_files() -> None:
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--format", "csv"]
    )
    assert result.exit_code == 0, result.output
    import csv as _csv
    import io

    records = list(_csv.DictReader(io.StringIO(result.output)))
    assert len(records) == 2
    backends = {r["backend"] for r in records}
    assert backends == {"mock", "transformers"}


def test_compare_subcommand_format_csv_file_output(tmp_path: Path) -> None:
    out = tmp_path / "summary.csv"
    result = CliRunner().invoke(
        main, ["compare", str(MOCK_CSV), "--format", "csv", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    import csv as _csv

    records = list(_csv.DictReader(out.open()))
    assert len(records) == 1
    assert records[0]["backend"] == "mock"


def test_compare_subcommand_format_csv_composes_with_limit() -> None:
    result = CliRunner().invoke(
        main,
        ["compare", str(MOCK_CSV), str(TRANSFORMERS_CSV), "--format", "csv", "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    import csv as _csv
    import io

    records = list(_csv.DictReader(io.StringIO(result.output)))
    assert len(records) == 1


def test_compare_subcommand_format_csv_composes_with_filter() -> None:
    result = CliRunner().invoke(
        main,
        [
            "compare",
            str(MOCK_CSV),
            str(TRANSFORMERS_CSV),
            "--format",
            "csv",
            "--filter",
            "backend=mock",
        ],
    )
    assert result.exit_code == 0, result.output
    import csv as _csv
    import io

    records = list(_csv.DictReader(io.StringIO(result.output)))
    assert len(records) == 1
    assert records[0]["backend"] == "mock"
