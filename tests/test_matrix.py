"""Tests for run matrix config — parsing, CLI execution, and backward compatibility."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.matrix import MatrixConfig, MatrixRunConfig, load_matrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mock_config(path: Path, prompts: Path, requests: int = 2) -> None:
    path.write_text(
        f"backend: mock\n"
        f"model: test-model\n"
        f"requests: {requests}\n"
        f"warmup_requests: 0\n"
        f"prompts_file: {prompts}\n"
        f"mock:\n"
        f"  latency_ms: 0\n"
        f"  tokens_per_response: 5\n"
    )


def _write_matrix(path: Path, results_dir: str, runs: list[dict]) -> None:
    import yaml

    path.write_text(yaml.dump({"results_dir": results_dir, "runs": runs}))


# ---------------------------------------------------------------------------
# MatrixRunConfig
# ---------------------------------------------------------------------------


def test_matrix_run_config_minimal() -> None:
    r = MatrixRunConfig(name="my-run", config="configs/example.yaml")
    assert r.name == "my-run"
    assert r.config == "configs/example.yaml"
    assert r.workload_profile is None


def test_matrix_run_config_with_profile() -> None:
    r = MatrixRunConfig(name="chat", config="configs/example.yaml", workload_profile="short_chat")
    assert r.workload_profile == "short_chat"


def test_matrix_run_config_invalid_profile_fails() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Unknown workload profile"):
        MatrixRunConfig(name="bad", config="configs/example.yaml", workload_profile="no_such")


# ---------------------------------------------------------------------------
# Name validation (path traversal prevention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escaped",
        "foo/bar",
        "foo\\bar",
        ".hidden",
        "",
        "run name",
        "run!name",
        "run\x00name",
    ],
)
def test_matrix_run_config_rejects_bad_names(bad_name: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MatrixRunConfig(name=bad_name, config="configs/example.yaml")


@pytest.mark.parametrize(
    "good_name",
    ["run-a", "run_b", "run.c", "Run1", "mock-short-chat", "v2.0-cpu"],
)
def test_matrix_run_config_accepts_valid_names(good_name: str) -> None:
    r = MatrixRunConfig(name=good_name, config="configs/example.yaml")
    assert r.name == good_name


# ---------------------------------------------------------------------------
# MatrixConfig
# ---------------------------------------------------------------------------


def test_matrix_config_default_results_dir() -> None:
    mc = MatrixConfig(runs=[MatrixRunConfig(name="r", config="c.yaml")])
    assert mc.results_dir == "results"


def test_matrix_config_custom_results_dir() -> None:
    mc = MatrixConfig(
        results_dir="my-experiments",
        runs=[MatrixRunConfig(name="r", config="c.yaml")],
    )
    assert mc.results_dir == "my-experiments"


def test_matrix_config_empty_runs_fails() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MatrixConfig(runs=[])


def test_matrix_config_duplicate_names_fails() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Duplicate run name"):
        MatrixConfig(
            runs=[
                MatrixRunConfig(name="same", config="a.yaml"),
                MatrixRunConfig(name="same", config="b.yaml"),
            ]
        )


def test_matrix_config_unique_names_ok() -> None:
    mc = MatrixConfig(
        runs=[
            MatrixRunConfig(name="a", config="a.yaml"),
            MatrixRunConfig(name="b", config="b.yaml"),
        ]
    )
    assert len(mc.runs) == 2


# ---------------------------------------------------------------------------
# load_matrix
# ---------------------------------------------------------------------------


def test_load_matrix_minimal(tmp_path: Path) -> None:
    f = tmp_path / "matrix.yaml"
    f.write_text("runs:\n  - name: r1\n    config: configs/example.yaml\n")
    mc = load_matrix(f)
    assert len(mc.runs) == 1
    assert mc.runs[0].name == "r1"
    assert mc.results_dir == "results"


def test_load_matrix_with_results_dir(tmp_path: Path) -> None:
    f = tmp_path / "matrix.yaml"
    f.write_text(
        "results_dir: my-results\n"
        "runs:\n"
        "  - name: r1\n"
        "    config: c.yaml\n"
        "    workload_profile: short_chat\n"
    )
    mc = load_matrix(f)
    assert mc.results_dir == "my-results"
    assert mc.runs[0].workload_profile == "short_chat"


def test_load_matrix_invalid_profile_fails(tmp_path: Path) -> None:
    from pydantic import ValidationError

    f = tmp_path / "matrix.yaml"
    f.write_text("runs:\n  - name: r1\n    config: c.yaml\n    workload_profile: bad\n")
    with pytest.raises(ValidationError, match="Unknown workload profile"):
        load_matrix(f)


def test_load_matrix_duplicate_names_fails(tmp_path: Path) -> None:
    from pydantic import ValidationError

    f = tmp_path / "matrix.yaml"
    f.write_text("runs:\n  - name: same\n    config: a.yaml\n  - name: same\n    config: b.yaml\n")
    with pytest.raises(ValidationError, match="Duplicate run name"):
        load_matrix(f)


# ---------------------------------------------------------------------------
# CLI: dry-run
# ---------------------------------------------------------------------------


def test_cli_matrix_dry_run_lists_runs(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(tmp_path / "results"),
        [
            {"name": "run-a", "config": str(cfg)},
            {"name": "run-b", "config": str(cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "run-a" in result.output
    assert "run-b" in result.output
    assert "2 run(s)" in result.output


def test_cli_matrix_dry_run_creates_no_files(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "run-a", "config": str(cfg)}])

    CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--dry-run"])
    assert not results_dir.exists()


def test_cli_matrix_dry_run_shows_profile(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(tmp_path / "results"),
        [{"name": "chat", "config": str(cfg), "workload_profile": "short_chat"}],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--dry-run"])
    assert "short_chat" in result.output


# ---------------------------------------------------------------------------
# CLI: execute
# ---------------------------------------------------------------------------


def test_cli_matrix_creates_csv_per_run(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\nWorld\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [
            {"name": "run-a", "config": str(cfg)},
            {"name": "run-b", "config": str(cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    assert result.exit_code == 0, result.output
    assert (results_dir / "run-a.csv").exists()
    assert (results_dir / "run-b.csv").exists()


def test_cli_matrix_creates_manifest_per_run(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "my-run", "config": str(cfg)}])

    CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    assert (results_dir / "my-run.manifest.json").exists()


def test_cli_matrix_csv_has_correct_columns(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "run-a", "config": str(cfg)}])

    CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    rows = list(csv.DictReader((results_dir / "run-a.csv").read_text().splitlines()))
    assert len(rows) == 1
    assert "p50_latency_ms" in rows[0]
    assert "backend" in rows[0]


def test_cli_matrix_manifest_is_valid_json(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "run-a", "config": str(cfg)}])

    CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    data = json.loads((results_dir / "run-a.manifest.json").read_text())
    assert "backend" in data
    assert "timestamp" in data


def test_cli_matrix_creates_results_dir(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "deep" / "nested" / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "r", "config": str(cfg)}])

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    assert result.exit_code == 0, result.output
    assert results_dir.is_dir()


def test_cli_matrix_output_mentions_compare(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(tmp_path / "results"), [{"name": "r", "config": str(cfg)}])

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    assert "compare" in result.output


def test_cli_matrix_workload_profile_override(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg_file = tmp_path / "config.yaml"
    _write_mock_config(cfg_file, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    # Override the profile's file path so it points to our tmp prompts
    from unittest.mock import patch

    _write_matrix(
        matrix,
        str(results_dir),
        [{"name": "chat", "config": str(cfg_file), "workload_profile": "short_chat"}],
    )
    from llm_inference_benchmark.config import BenchmarkConfig

    with patch.object(BenchmarkConfig, "resolve_prompts_file", return_value=str(prompts)):
        result = CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    assert result.exit_code == 0, result.output
    assert (results_dir / "chat.csv").exists()


# ---------------------------------------------------------------------------
# Backward compatibility — single-run CLI still works alongside matrix command
# ---------------------------------------------------------------------------


def test_single_run_cli_still_works_after_matrix_added(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config)])
    assert result.exit_code == 0, result.output
    assert "Benchmark Results" in result.output


def test_compare_subcommand_still_works() -> None:
    fixtures = Path(__file__).parent / "fixtures"
    result = CliRunner().invoke(
        main, ["compare", str(fixtures / "mock_run.csv"), str(fixtures / "transformers_run.csv")]
    )
    assert result.exit_code == 0, result.output
    assert "mock" in result.output
