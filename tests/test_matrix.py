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


# ---------------------------------------------------------------------------
# --continue-on-error
# ---------------------------------------------------------------------------


def test_matrix_continue_on_error_runs_all_when_one_fails(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    good_cfg = tmp_path / "good.yaml"
    _write_mock_config(good_cfg, prompts)
    bad_cfg = tmp_path / "nonexistent_config.yaml"  # does not exist → FileNotFoundError

    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [
            {"name": "will-fail", "config": str(bad_cfg)},
            {"name": "will-pass", "config": str(good_cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--continue-on-error"])
    assert result.exit_code == 1
    assert (results_dir / "will-pass.csv").exists()
    assert not (results_dir / "will-fail.csv").exists()
    assert "will-fail" in result.output
    assert "1 failed" in result.output


def test_matrix_continue_on_error_exit_zero_when_all_pass(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}, {"name": "run-b", "config": str(cfg)}],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--continue-on-error"])
    assert result.exit_code == 0, result.output
    assert (results_dir / "run-a.csv").exists()
    assert (results_dir / "run-b.csv").exists()


def test_matrix_without_continue_on_error_stops_on_first_failure(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    good_cfg = tmp_path / "good.yaml"
    _write_mock_config(good_cfg, prompts)
    bad_cfg = tmp_path / "nonexistent_config.yaml"

    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [
            {"name": "will-fail", "config": str(bad_cfg)},
            {"name": "should-not-run", "config": str(good_cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    assert result.exit_code != 0
    assert not (results_dir / "should-not-run.csv").exists()


# ---------------------------------------------------------------------------
# --format json  (dry-run)
# ---------------------------------------------------------------------------


def test_matrix_dry_run_json_exits_zero(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(tmp_path / "results"), [{"name": "run-a", "config": str(cfg)}])

    result = CliRunner().invoke(
        main, ["matrix", "--config", str(matrix), "--dry-run", "--format", "json"]
    )
    assert result.exit_code == 0, result.output


def test_matrix_dry_run_json_emits_valid_json(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(tmp_path / "results"),
        [{"name": "run-a", "config": str(cfg)}, {"name": "run-b", "config": str(cfg)}],
    )

    result = CliRunner().invoke(
        main, ["matrix", "--config", str(matrix), "--dry-run", "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total"] == 2
    assert len(data["runs"]) == 2


def test_matrix_dry_run_json_run_names(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(tmp_path / "results"),
        [{"name": "alpha", "config": str(cfg)}, {"name": "beta", "config": str(cfg)}],
    )

    result = CliRunner().invoke(
        main, ["matrix", "--config", str(matrix), "--dry-run", "--format", "json"]
    )
    data = json.loads(result.output)
    names = [r["name"] for r in data["runs"]]
    assert names == ["alpha", "beta"]


def test_matrix_dry_run_json_run_fields(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "run-a", "config": str(cfg)}])

    result = CliRunner().invoke(
        main, ["matrix", "--config", str(matrix), "--dry-run", "--format", "json"]
    )
    data = json.loads(result.output)
    run = data["runs"][0]
    assert run["index"] == 1
    assert run["name"] == "run-a"
    assert run["config"] == str(cfg)
    assert run["output"].endswith("run-a.csv")
    assert run["manifest"].endswith("run-a.manifest.json")


def test_matrix_dry_run_json_no_table_text(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(tmp_path / "results"), [{"name": "run-a", "config": str(cfg)}])

    result = CliRunner().invoke(
        main, ["matrix", "--config", str(matrix), "--dry-run", "--format", "json"]
    )
    assert "run(s)" not in result.output
    assert "config:" not in result.output


def test_matrix_dry_run_json_creates_no_files(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "run-a", "config": str(cfg)}])

    CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--dry-run", "--format", "json"])
    assert not results_dir.exists()


# ---------------------------------------------------------------------------
# --format json  (execution)
# ---------------------------------------------------------------------------


def test_matrix_json_mode_exits_zero_on_success(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\nWorld\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}, {"name": "run-b", "config": str(cfg)}],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--format", "json"])
    assert result.exit_code == 0, result.output


def _parse_json_output(output: str) -> dict:
    """Extract the JSON object from the last line of mixed stdout/stderr output."""
    return json.loads(output.strip().rsplit("\n", 1)[-1])


def test_matrix_json_mode_emits_valid_json(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\nWorld\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}, {"name": "run-b", "config": str(cfg)}],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = _parse_json_output(result.output)
    assert data["total"] == 2
    assert data["succeeded"] == 2
    assert data["failed"] == 0
    assert len(data["runs"]) == 2


def test_matrix_json_mode_run_status_ok(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "run-a", "config": str(cfg)}])

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--format", "json"])
    data = _parse_json_output(result.output)
    run = data["runs"][0]
    assert run["status"] == "ok"
    assert run["output"].endswith("run-a.csv")
    assert run["manifest"].endswith("run-a.manifest.json")


def test_matrix_json_mode_writes_files(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "run-a", "config": str(cfg)}])

    CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--format", "json"])
    assert (results_dir / "run-a.csv").exists()
    assert (results_dir / "run-a.manifest.json").exists()


def test_matrix_json_mode_no_human_text_in_json_line(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(matrix, str(results_dir), [{"name": "run-a", "config": str(cfg)}])

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--format", "json"])
    json_line = result.output.strip().rsplit("\n", 1)[-1]
    assert json_line.startswith("{")
    assert "Backend:" not in json_line
    assert "Matrix:" not in json_line


def test_matrix_json_mode_failed_run_exits_nonzero(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    good_cfg = tmp_path / "good.yaml"
    _write_mock_config(good_cfg, prompts)
    bad_cfg = tmp_path / "nonexistent.yaml"
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [
            {"name": "will-fail", "config": str(bad_cfg)},
            {"name": "will-pass", "config": str(good_cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--format", "json"])
    assert result.exit_code != 0


def test_matrix_json_mode_failure_captured_in_output(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    good_cfg = tmp_path / "good.yaml"
    _write_mock_config(good_cfg, prompts)
    bad_cfg = tmp_path / "nonexistent.yaml"
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [
            {"name": "will-fail", "config": str(bad_cfg)},
            {"name": "will-pass", "config": str(good_cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--format", "json"])
    data = _parse_json_output(result.output)
    assert data["total"] == 2
    assert data["failed"] == 1
    assert data["succeeded"] == 1
    statuses = {r["name"]: r["status"] for r in data["runs"]}
    assert statuses["will-fail"] == "failed"
    assert statuses["will-pass"] == "ok"
    assert "error" in next(r for r in data["runs"] if r["name"] == "will-fail")


def test_matrix_config_non_dict_input_raises_validation_error() -> None:
    """_expand_sweep returns non-dict data unchanged; Pydantic then raises ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MatrixConfig.model_validate("not-a-dict")


# ---------------------------------------------------------------------------
# dataset field
# ---------------------------------------------------------------------------


def test_matrix_run_config_accepts_known_dataset() -> None:
    r = MatrixRunConfig(name="r", config="c.yaml", dataset="wildchat")
    assert r.dataset == "wildchat"


def test_matrix_run_config_dataset_none_by_default() -> None:
    r = MatrixRunConfig(name="r", config="c.yaml")
    assert r.dataset is None


def test_matrix_run_config_rejects_unknown_dataset() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Unknown dataset"):
        MatrixRunConfig(name="r", config="c.yaml", dataset="no-such-dataset")


def test_load_matrix_with_dataset_field(tmp_path: Path) -> None:
    f = tmp_path / "matrix.yaml"
    f.write_text("runs:\n  - name: r1\n    config: configs/example.yaml\n    dataset: wildchat\n")
    mc = load_matrix(f)
    assert mc.runs[0].dataset == "wildchat"


def test_load_matrix_invalid_dataset_fails(tmp_path: Path) -> None:
    from pydantic import ValidationError

    f = tmp_path / "matrix.yaml"
    f.write_text("runs:\n  - name: r1\n    config: c.yaml\n    dataset: bogus\n")
    with pytest.raises(ValidationError, match="Unknown dataset"):
        load_matrix(f)


def test_cli_matrix_dry_run_shows_dataset(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(tmp_path / "results"),
        [{"name": "chat", "config": str(cfg), "dataset": "wildchat"}],
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "wildchat" in result.output


def test_cli_matrix_dry_run_json_includes_dataset(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(tmp_path / "results"),
        [{"name": "chat", "config": str(cfg), "dataset": "wildchat"}],
    )

    result = CliRunner().invoke(
        main, ["matrix", "--config", str(matrix), "--dry-run", "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["runs"][0]["dataset"] == "wildchat"


def test_cli_matrix_uses_dataset_prompts(tmp_path: Path) -> None:
    """When dataset is set, matrix runner loads prompts from the cached JSONL."""
    from unittest.mock import patch

    prompts_file = tmp_path / "p.txt"
    prompts_file.write_text("fallback prompt\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts_file, requests=2)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [{"name": "ds-run", "config": str(cfg), "dataset": "wildchat"}],
    )

    dataset_prompts = ["Dataset prompt one", "Dataset prompt two"]
    with patch(
        "llm_inference_benchmark.datasets.load_prompts", return_value=dataset_prompts
    ) as mock_load:
        result = CliRunner().invoke(main, ["matrix", "--config", str(matrix)])

    assert result.exit_code == 0, result.output
    mock_load.assert_called_once()
    assert mock_load.call_args[0][0] == "wildchat"
    assert (results_dir / "ds-run.csv").exists()


def test_cli_matrix_dataset_not_cached_fails(tmp_path: Path) -> None:
    """A dataset that is registered but not yet cached gives a clear ClickException."""
    from unittest.mock import patch

    prompts_file = tmp_path / "p.txt"
    prompts_file.write_text("fallback\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts_file)
    results_dir = tmp_path / "results"
    matrix = tmp_path / "matrix.yaml"
    _write_matrix(
        matrix,
        str(results_dir),
        [{"name": "ds-run", "config": str(cfg), "dataset": "wildchat"}],
    )

    with patch(
        "llm_inference_benchmark.datasets.load_prompts",
        side_effect=FileNotFoundError("Dataset 'wildchat' is not cached"),
    ):
        result = CliRunner().invoke(main, ["matrix", "--config", str(matrix)])

    assert result.exit_code != 0
    assert "not cached" in result.output.lower() or "wildchat" in result.output
