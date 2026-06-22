"""Tests for ``llm-bench sweep`` — concurrency ramp command."""

from __future__ import annotations

import csv
from pathlib import Path

from click.testing import CliRunner

from llm_inference_benchmark.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mock_config(path: Path, prompts: Path, requests: int = 4) -> None:
    path.write_text(
        f"backend: mock\n"
        f"model: test-model\n"
        f"requests: {requests}\n"
        f"warmup_requests: 0\n"
        f"prompts_file: {prompts}\n"
        f"mock:\n"
        f"  latency_ms: 1\n"
        f"  tokens_per_response: 10\n"
    )


# ---------------------------------------------------------------------------
# Basic sweep
# ---------------------------------------------------------------------------


def test_sweep_two_levels(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts)
    out = tmp_path / "sweep.csv"

    result = CliRunner().invoke(
        main,
        ["sweep", "--config", str(cfg), "--concurrency-range", "1,2", "--output", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert len(rows) == 2
    assert rows[0]["concurrency"] == "1"
    assert rows[1]["concurrency"] == "2"
    for row in rows:
        assert "throughput_rps" in row
        assert "p50_latency_ms" in row
        assert "p95_latency_ms" in row
        assert "tokens_per_second" in row
        assert float(row["throughput_rps"]) > 0


def test_sweep_single_level(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts)
    out = tmp_path / "sweep.csv"

    result = CliRunner().invoke(
        main,
        ["sweep", "--config", str(cfg), "--concurrency-range", "4", "--output", str(out)],
    )

    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["concurrency"] == "4"


def test_sweep_output_default_name(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts)
    default_out = Path("sweep_results.csv")

    try:
        result = CliRunner().invoke(
            main,
            ["sweep", "--config", str(cfg), "--concurrency-range", "1"],
        )
        assert result.exit_code == 0, result.output
        assert default_out.exists()
    finally:
        default_out.unlink(missing_ok=True)


def test_sweep_requests_override(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts, requests=10)
    out = tmp_path / "sweep.csv"

    result = CliRunner().invoke(
        main,
        [
            "sweep",
            "--config",
            str(cfg),
            "--concurrency-range",
            "1",
            "--requests",
            "2",
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert rows[0]["request_count"] == "2"


def test_sweep_summary_printed(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts)
    out = tmp_path / "sweep.csv"

    result = CliRunner().invoke(
        main,
        ["sweep", "--config", str(cfg), "--concurrency-range", "1,2", "--output", str(out)],
    )

    assert "Sweep Summary" in result.output
    assert "Knee point" in result.output
    assert "concurrency" in result.output.lower()


# ---------------------------------------------------------------------------
# Early stop
# ---------------------------------------------------------------------------


def test_sweep_early_stop_on_p95(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts)
    out = tmp_path / "sweep.csv"

    # max-p95-ms=0 forces a stop after the first level
    result = CliRunner().invoke(
        main,
        [
            "sweep",
            "--config",
            str(cfg),
            "--concurrency-range",
            "1,2,4",
            "--max-p95-ms",
            "0",
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 1
    rows = list(csv.DictReader(out.read_text().splitlines()))
    # Only the first level completes before the threshold is breached
    assert len(rows) == 1


def test_sweep_no_early_stop_when_under_threshold(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts)
    out = tmp_path / "sweep.csv"

    result = CliRunner().invoke(
        main,
        [
            "sweep",
            "--config",
            str(cfg),
            "--concurrency-range",
            "1,2",
            "--max-p95-ms",
            "99999",
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_sweep_bad_concurrency_range(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts)

    result = CliRunner().invoke(
        main,
        ["sweep", "--config", str(cfg), "--concurrency-range", "1,two,3"],
    )

    assert result.exit_code != 0
    assert "integer" in result.output.lower() or "usage" in result.output.lower()


def test_sweep_zero_concurrency_rejected(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_mock_config(cfg, tmp_prompts)

    result = CliRunner().invoke(
        main,
        ["sweep", "--config", str(cfg), "--concurrency-range", "0,1"],
    )

    assert result.exit_code != 0


def test_sweep_missing_config_rejected(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["sweep", "--config", str(tmp_path / "no_such.yaml"), "--concurrency-range", "1"],
    )

    assert result.exit_code != 0
