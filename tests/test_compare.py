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
    assert "0.0" in table
    assert "N/A" not in table


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
