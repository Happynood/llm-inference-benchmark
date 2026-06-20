"""Tests for the llm-bench pipeline subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from llm_inference_benchmark.cli import main

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


def _write_pipeline(
    path: Path,
    results_dir: str,
    runs: list[dict],
    pipeline: dict | None = None,
) -> None:
    data: dict = {"results_dir": results_dir, "runs": runs}
    if pipeline is not None:
        data["pipeline"] = pipeline
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# Happy path: all outputs produced
# ---------------------------------------------------------------------------


def test_pipeline_happy_path_compare_files_written(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\nWorld\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [
            {"name": "run-a", "config": str(cfg)},
            {"name": "run-b", "config": str(cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    assert result.exit_code == 0, result.output
    assert (results_dir / "compare.md").exists()
    assert (results_dir / "compare.json").exists()


def test_pipeline_happy_path_pareto_files_written(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\nWorld\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [
            {"name": "run-a", "config": str(cfg)},
            {"name": "run-b", "config": str(cfg)},
        ],
        pipeline={"pareto": True},
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    assert result.exit_code == 0, result.output
    assert (results_dir / "pareto.md").exists()
    assert (results_dir / "pareto.json").exists()


def test_pipeline_pareto_false_no_pareto_files(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
        pipeline={"pareto": False},
    )

    CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    assert not (results_dir / "pareto.md").exists()
    assert not (results_dir / "pareto.json").exists()


def test_pipeline_compare_json_is_valid_json(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
    )

    CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    data = json.loads((results_dir / "compare.json").read_text())
    assert isinstance(data, list)
    assert len(data) == 1


def test_pipeline_recommend_files_written_when_winner(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    # latency_ms=0 → p95 near 0; max_p95_ms=5000 will easily pass
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
        pipeline={"recommend": {"max_p95_ms": 5000}},
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    assert result.exit_code == 0, result.output
    assert (results_dir / "recommend.md").exists()
    assert (results_dir / "recommend.json").exists()


def test_pipeline_recommend_json_has_winner(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
        pipeline={"recommend": {"max_p95_ms": 5000}},
    )

    CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    data = json.loads((results_dir / "recommend.json").read_text())
    assert data["winner"] is not None


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_pipeline_dry_run_exits_zero(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(tmp_path / "results"),
        [{"name": "run-a", "config": str(cfg)}],
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline), "--dry-run"])
    assert result.exit_code == 0, result.output


def test_pipeline_dry_run_creates_no_files(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
    )

    CliRunner().invoke(main, ["pipeline", "--config", str(pipeline), "--dry-run"])
    assert not results_dir.exists()


def test_pipeline_dry_run_shows_run_names(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(tmp_path / "results"),
        [
            {"name": "alpha", "config": str(cfg)},
            {"name": "beta", "config": str(cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline), "--dry-run"])
    assert "alpha" in result.output
    assert "beta" in result.output


def test_pipeline_dry_run_shows_compare_step(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(tmp_path / "results"),
        [{"name": "run-a", "config": str(cfg)}],
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline), "--dry-run"])
    assert "compare" in result.output


def test_pipeline_dry_run_shows_pareto_step_when_enabled(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(tmp_path / "results"),
        [{"name": "run-a", "config": str(cfg)}],
        pipeline={"pareto": True},
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline), "--dry-run"])
    assert "pareto" in result.output


def test_pipeline_dry_run_hides_pareto_step_when_disabled(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(tmp_path / "results"),
        [{"name": "run-a", "config": str(cfg)}],
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline), "--dry-run"])
    assert "pareto" not in result.output


def test_pipeline_dry_run_shows_recommend_step_when_configured(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(tmp_path / "results"),
        [{"name": "run-a", "config": str(cfg)}],
        pipeline={"recommend": {"max_p95_ms": 5000}},
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline), "--dry-run"])
    assert "recommend" in result.output


# ---------------------------------------------------------------------------
# results_dir creation
# ---------------------------------------------------------------------------


def test_pipeline_creates_missing_results_dir(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "deep" / "nested" / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    assert result.exit_code == 0, result.output
    assert results_dir.is_dir()


# ---------------------------------------------------------------------------
# recommend exits 1 when no winner
# ---------------------------------------------------------------------------


def test_pipeline_recommend_exits_1_when_no_winner(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    # max_vram_mb with mock backend → VRAM unknown → all runs excluded → no winner
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
        pipeline={"recommend": {"max_vram_mb": 1}},
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    assert result.exit_code == 1
    # Output files still written even when no winner
    assert (results_dir / "recommend.md").exists()
    assert (results_dir / "recommend.json").exists()


def test_pipeline_recommend_json_null_winner_when_no_winner(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
        pipeline={"recommend": {"max_vram_mb": 1}},
    )

    CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    data = json.loads((results_dir / "recommend.json").read_text())
    assert data["winner"] is None


# ---------------------------------------------------------------------------
# --continue-on-error
# ---------------------------------------------------------------------------


def test_pipeline_continue_on_error_keeps_going_after_failed_cell(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    good_cfg = tmp_path / "good.yaml"
    _write_mock_config(good_cfg, prompts)
    bad_cfg = tmp_path / "nonexistent_config.yaml"
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [
            {"name": "will-fail", "config": str(bad_cfg)},
            {"name": "will-pass", "config": str(good_cfg)},
        ],
    )

    result = CliRunner().invoke(
        main, ["pipeline", "--config", str(pipeline), "--continue-on-error"]
    )
    assert result.exit_code == 1
    assert (results_dir / "will-pass.csv").exists()
    assert not (results_dir / "will-fail.csv").exists()


def test_pipeline_continue_on_error_writes_compare_from_successful_csvs(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    good_cfg = tmp_path / "good.yaml"
    _write_mock_config(good_cfg, prompts)
    bad_cfg = tmp_path / "nonexistent_config.yaml"
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [
            {"name": "will-fail", "config": str(bad_cfg)},
            {"name": "will-pass", "config": str(good_cfg)},
        ],
    )

    CliRunner().invoke(main, ["pipeline", "--config", str(pipeline), "--continue-on-error"])
    assert (results_dir / "compare.md").exists()
    data = json.loads((results_dir / "compare.json").read_text())
    assert len(data) == 1  # only the successful run


def test_pipeline_without_continue_on_error_stops_on_first_failure(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    good_cfg = tmp_path / "good.yaml"
    _write_mock_config(good_cfg, prompts)
    bad_cfg = tmp_path / "nonexistent_config.yaml"
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    _write_pipeline(
        pipeline,
        str(results_dir),
        [
            {"name": "will-fail", "config": str(bad_cfg)},
            {"name": "should-not-run", "config": str(good_cfg)},
        ],
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    assert result.exit_code != 0
    assert not (results_dir / "should-not-run.csv").exists()


# ---------------------------------------------------------------------------
# Bare matrix YAML (no pipeline: block) is valid pipeline input
# ---------------------------------------------------------------------------


def test_pipeline_bare_matrix_config_is_valid(tmp_path: Path) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg = tmp_path / "config.yaml"
    _write_mock_config(cfg, prompts)
    results_dir = tmp_path / "results"
    pipeline = tmp_path / "pipeline.yaml"
    # No "pipeline:" block — just a plain matrix config
    _write_pipeline(
        pipeline,
        str(results_dir),
        [{"name": "run-a", "config": str(cfg)}],
    )

    result = CliRunner().invoke(main, ["pipeline", "--config", str(pipeline)])
    assert result.exit_code == 0, result.output
    assert (results_dir / "compare.md").exists()
