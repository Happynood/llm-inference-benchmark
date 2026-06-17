"""Tests for compare.py and the llm-bench compare subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.compare import (
    RunRow,
    build_comparison_table,
    load_csv,
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


def test_render_table_cuda_none_shows_na() -> None:
    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    assert "N/A" in render_table(rows)


def test_render_table_cuda_zero_shows_value() -> None:
    rows = [RunRow("transformers", "tiny", 10, 40.0, 44.0, 1200.0, 720.0, 0.0, 0.0)]
    table = render_table(rows)
    # cuda=0.0 must appear as "0.0", not "N/A"; the Sanity % column may still show N/A
    assert "| 0.0" in table or "0.0 |" in table


def test_render_table_vram_none_shows_na() -> None:
    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    assert "N/A" in render_table(rows)


def test_render_table_vram_value_shown() -> None:
    rows = [RunRow("llama-cpp", "llama3", 10, 900.0, 940.0, 53.0, 800.0, None, 2361.0)]
    table = render_table(rows)
    assert "2361.0" in table


def test_render_table_row_count() -> None:
    table = render_table(_ROWS)
    lines = table.splitlines()
    assert len(lines) == len(_ROWS) + 2  # header + separator + data rows


def test_render_table_ppl_header() -> None:
    table = render_table(_ROWS[:1])
    assert "PPL" in table


def test_render_table_ppl_na_when_none() -> None:
    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    assert "N/A" in render_table(rows)


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


def test_render_table_judge_header() -> None:
    table = render_table(_ROWS[:1])
    assert "Judge" in table


def test_render_table_judge_na_when_none() -> None:
    rows = [RunRow("mock", "gpt2", 20, 5.0, 5.1, 9000.0, 45.0, None, None)]
    assert "N/A" in render_table(rows)


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


def test_compare_subcommand_no_files_fails() -> None:
    result = CliRunner().invoke(main, ["compare"])
    assert result.exit_code != 0


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


def test_render_table_old_csv_sanity_shows_na() -> None:
    row = load_csv(MOCK_CSV)
    table = render_table([row])
    assert "N/A" in table
    assert "Sanity %" in table


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


def test_render_table_sanity_header_present() -> None:
    table = render_table(_ROWS[:1])
    assert "Sanity %" in table


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


def test_render_table_has_out_tok_s_header() -> None:
    table = render_table(_ROWS[:1])
    assert "Out tok/s" in table
    assert "In tok" in table
    assert "Out tok" in table


def test_render_table_decode_tps_na_for_old_rows() -> None:
    row = load_csv(MOCK_CSV)
    table = render_table([row])
    assert "Out tok/s" in table
    assert "N/A" in table  # decode_tokens_per_second is None for old CSV


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


def test_render_table_shows_ttft_headers() -> None:
    table = render_table(_ROWS[:1])
    assert "TTFT p50 (ms)" in table
    assert "TTFT p95 (ms)" in table


def test_render_table_shows_na_for_missing_ttft() -> None:
    row = load_csv(MOCK_CSV)
    table = render_table([row])
    assert "TTFT p50 (ms)" in table
