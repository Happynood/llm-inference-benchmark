"""Tests for the sweep expansion module (v0.17)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_inference_benchmark.sweep import (
    apply_overrides,
    expand_sweep,
    validate_override_path,
)

# ---------------------------------------------------------------------------
# validate_override_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "backend",
        "model",
        "requests",
        "warmup_requests",
        "prompts_file",
        "quality_file",
        "llama_cpp.n_gpu_layers",
        "llama_cpp.max_tokens",
        "llama_cpp.n_ctx",
        "llama_cpp.temperature",
        "llama_cpp.n_threads",
        "llama_cpp.verbose",
        "hf.max_new_tokens",
        "hf.device",
        "hf.torch_dtype",
        "hf.do_sample",
        "mock.latency_ms",
        "mock.tokens_per_response",
        "openai.base_url",
        "openai.max_tokens",
        "openai.temperature",
        "openai.timeout_s",
        "openai.stream",
        "onnx.max_new_tokens",
        "onnx.device",
        "onnx.do_sample",
        "onnx.export",
        "vllm.max_new_tokens",
        "vllm.temperature",
        "vllm.tensor_parallel_size",
        "vllm.gpu_memory_utilization",
        "vllm.dtype",
    ],
)
def test_valid_override_paths(path: str) -> None:
    validate_override_path(path)  # must not raise


@pytest.mark.parametrize(
    "bad_path,expected_fragment",
    [
        ("nonexistent_field", "Unknown override path"),
        ("llama_cpp.nonexistent", "Unknown field"),
        ("hf.nonexistent", "Unknown field"),
        ("mock.nonexistent", "Unknown field"),
        ("openai.nonexistent", "Unknown field"),
        ("onnx.nonexistent", "Unknown field"),
        ("vllm.nonexistent", "Unknown field"),
        ("llama_cpp.n_gpu_layers.extra", "too deep"),
        ("requests.nested", "does not support nested overrides"),
    ],
)
def test_invalid_override_paths(bad_path: str, expected_fragment: str) -> None:
    with pytest.raises(ValueError, match=expected_fragment):
        validate_override_path(bad_path)


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


def test_apply_top_level_override() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig()
    result = apply_overrides(cfg, {"requests": 5})
    assert result.requests == 5
    assert cfg.requests == 20  # original unchanged (immutable)


def test_apply_nested_override_llama_cpp() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig()
    result = apply_overrides(cfg, {"llama_cpp.n_gpu_layers": 99})
    assert result.llama_cpp.n_gpu_layers == 99
    assert cfg.llama_cpp.n_gpu_layers == 0  # original unchanged


def test_apply_nested_override_mock() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig()
    result = apply_overrides(cfg, {"mock.latency_ms": 42.0})
    assert result.mock.latency_ms == 42.0


def test_apply_multiple_overrides() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig()
    result = apply_overrides(cfg, {"requests": 3, "llama_cpp.max_tokens": 128})
    assert result.requests == 3
    assert result.llama_cpp.max_tokens == 128


def test_apply_overrides_empty_is_noop() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig()
    result = apply_overrides(cfg, {})
    assert result == cfg


# ---------------------------------------------------------------------------
# expand_sweep
# ---------------------------------------------------------------------------


def test_expand_sweep_single_param() -> None:
    expanded = expand_sweep("configs/example.yaml", {"mock.latency_ms": [5, 10, 20]})
    assert len(expanded) == 3
    names = [n for n, _, _ in expanded]
    assert names == [
        "sweep-latency_ms-5",
        "sweep-latency_ms-10",
        "sweep-latency_ms-20",
    ]


def test_expand_sweep_cartesian_product() -> None:
    expanded = expand_sweep(
        "configs/example.yaml",
        {"mock.latency_ms": [5, 10], "mock.tokens_per_response": [25, 50]},
    )
    assert len(expanded) == 4
    names = [n for n, _, _ in expanded]
    assert names == [
        "sweep-latency_ms-5-tokens_per_response-25",
        "sweep-latency_ms-5-tokens_per_response-50",
        "sweep-latency_ms-10-tokens_per_response-25",
        "sweep-latency_ms-10-tokens_per_response-50",
    ]


def test_expand_sweep_preserves_base_config() -> None:
    base = "configs/example.yaml"
    expanded = expand_sweep(base, {"mock.latency_ms": [5]})
    _, config, _ = expanded[0]
    assert config == base


def test_expand_sweep_overrides_correct() -> None:
    expanded = expand_sweep(
        "configs/example.yaml",
        {"llama_cpp.n_gpu_layers": [0, 99]},
    )
    assert expanded[0][2] == {"llama_cpp.n_gpu_layers": 0}
    assert expanded[1][2] == {"llama_cpp.n_gpu_layers": 99}


def test_expand_sweep_deterministic_order() -> None:
    """Same input always produces same output order."""
    e1 = expand_sweep("c.yaml", {"mock.latency_ms": [1, 2, 3]})
    e2 = expand_sweep("c.yaml", {"mock.latency_ms": [1, 2, 3]})
    assert [n for n, _, _ in e1] == [n for n, _, _ in e2]


def test_expand_sweep_float_value_in_name() -> None:
    expanded = expand_sweep("c.yaml", {"llama_cpp.temperature": [0.0, 0.5, 1.0]})
    names = [n for n, _, _ in expanded]
    assert "sweep-temperature-0.0" in names
    assert "sweep-temperature-0.5" in names
    assert "sweep-temperature-1.0" in names


def test_expand_sweep_rejects_unknown_path() -> None:
    with pytest.raises(ValueError, match="Unknown override path"):
        expand_sweep("c.yaml", {"bad_field": [1, 2]})


def test_expand_sweep_rejects_empty_value_list() -> None:
    with pytest.raises(ValueError, match="empty value list"):
        expand_sweep("c.yaml", {"mock.latency_ms": []})


def test_expand_sweep_rejects_empty_sweep_dict() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        expand_sweep("c.yaml", {})


# ---------------------------------------------------------------------------
# MatrixConfig sweep integration
# ---------------------------------------------------------------------------


def test_matrix_config_sweep_expands_runs() -> None:
    from llm_inference_benchmark.matrix import MatrixConfig

    mc = MatrixConfig.model_validate(
        {
            "base_config": "configs/example.yaml",
            "sweep": {"mock.latency_ms": [5, 10, 20]},
        }
    )
    assert len(mc.runs) == 3
    assert mc.runs[0].name == "sweep-latency_ms-5"
    assert mc.runs[0].overrides == {"mock.latency_ms": 5}


def test_matrix_config_sweep_requires_base_config() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="base_config"):
        from llm_inference_benchmark.matrix import MatrixConfig

        MatrixConfig.model_validate({"sweep": {"mock.latency_ms": [5]}})


def test_matrix_config_sweep_and_runs_conflict() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Cannot combine"):
        from llm_inference_benchmark.matrix import MatrixConfig

        MatrixConfig.model_validate(
            {
                "base_config": "c.yaml",
                "sweep": {"mock.latency_ms": [5]},
                "runs": [{"name": "r", "config": "c.yaml"}],
            }
        )


def test_matrix_config_sweep_invalid_path_errors() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Unknown override path"):
        from llm_inference_benchmark.matrix import MatrixConfig

        MatrixConfig.model_validate(
            {
                "base_config": "c.yaml",
                "sweep": {"nonexistent.field": [1, 2]},
            }
        )


def test_matrix_config_sweep_three_by_two() -> None:
    from llm_inference_benchmark.matrix import MatrixConfig

    mc = MatrixConfig.model_validate(
        {
            "base_config": "c.yaml",
            "sweep": {
                "mock.latency_ms": [5, 10, 20],
                "mock.tokens_per_response": [25, 50],
            },
        }
    )
    assert len(mc.runs) == 6


def test_load_matrix_sweep_yaml(tmp_path: Path) -> None:
    import yaml as _yaml

    from llm_inference_benchmark.matrix import load_matrix

    f = tmp_path / "sweep.yaml"
    f.write_text(
        _yaml.dump(
            {
                "base_config": "configs/example.yaml",
                "sweep": {"mock.latency_ms": [5, 10]},
            }
        )
    )
    mc = load_matrix(f)
    assert len(mc.runs) == 2
    assert mc.runs[0].config == "configs/example.yaml"


# ---------------------------------------------------------------------------
# CLI dry-run shows overrides
# ---------------------------------------------------------------------------


def test_cli_matrix_sweep_dry_run(tmp_path: Path) -> None:
    import yaml as _yaml
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    matrix = tmp_path / "sweep.yaml"
    matrix.write_text(
        _yaml.dump(
            {
                "base_config": "configs/example.yaml",
                "results_dir": str(tmp_path / "results"),
                "sweep": {"mock.latency_ms": [5, 10]},
            }
        )
    )
    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "sweep-latency_ms-5" in result.output
    assert "sweep-latency_ms-10" in result.output
    assert "overrides" in result.output
    assert "latency_ms=5" in result.output


def test_cli_matrix_sweep_dry_run_no_files(tmp_path: Path) -> None:
    import yaml as _yaml
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    results_dir = tmp_path / "results"
    matrix = tmp_path / "sweep.yaml"
    matrix.write_text(
        _yaml.dump(
            {
                "base_config": "configs/example.yaml",
                "results_dir": str(results_dir),
                "sweep": {"mock.latency_ms": [5]},
            }
        )
    )
    CliRunner().invoke(main, ["matrix", "--config", str(matrix), "--dry-run"])
    assert not results_dir.exists()


# ---------------------------------------------------------------------------
# CLI execute: sweep with mock backend
# ---------------------------------------------------------------------------


def _write_mock_config(path: Path, prompts: Path, requests: int = 2) -> None:
    path.write_text(
        f"backend: mock\n"
        f"model: test-model\n"
        f"requests: {requests}\n"
        f"warmup_requests: 0\n"
        f"prompts_file: {prompts}\n"
        f"mock:\n"
        f"  latency_ms: 1\n"
        f"  tokens_per_response: 5\n"
    )


def test_cli_matrix_sweep_executes_all_runs(tmp_path: Path) -> None:
    import yaml as _yaml
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    base_cfg = tmp_path / "base.yaml"
    _write_mock_config(base_cfg, prompts)
    results_dir = tmp_path / "results"

    matrix = tmp_path / "sweep.yaml"
    matrix.write_text(
        _yaml.dump(
            {
                "base_config": str(base_cfg),
                "results_dir": str(results_dir),
                "sweep": {"mock.latency_ms": [1, 2, 3]},
            }
        )
    )

    result = CliRunner().invoke(main, ["matrix", "--config", str(matrix)])
    assert result.exit_code == 0, result.output
    assert (results_dir / "sweep-latency_ms-1.csv").exists()
    assert (results_dir / "sweep-latency_ms-2.csv").exists()
    assert (results_dir / "sweep-latency_ms-3.csv").exists()


def test_cli_matrix_sweep_overrides_applied(tmp_path: Path) -> None:
    """The override values must reach the CSV output."""
    import csv

    import yaml as _yaml
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    base_cfg = tmp_path / "base.yaml"
    _write_mock_config(base_cfg, prompts, requests=1)
    results_dir = tmp_path / "results"

    matrix = tmp_path / "sweep.yaml"
    matrix.write_text(
        _yaml.dump(
            {
                "base_config": str(base_cfg),
                "results_dir": str(results_dir),
                "sweep": {"mock.tokens_per_response": [10, 100]},
            }
        )
    )

    CliRunner().invoke(main, ["matrix", "--config", str(matrix)])

    row10 = list(
        csv.DictReader((results_dir / "sweep-tokens_per_response-10.csv").read_text().splitlines())
    )[0]
    row100 = list(
        csv.DictReader((results_dir / "sweep-tokens_per_response-100.csv").read_text().splitlines())
    )[0]
    # Mock backend: total_tokens = tokens_per_response × request_count
    assert int(row10["total_tokens"]) < int(row100["total_tokens"])


# ---------------------------------------------------------------------------
# Backward compatibility — existing explicit runs still work alongside sweep
# ---------------------------------------------------------------------------


def test_explicit_runs_unaffected_by_sweep_code() -> None:
    from llm_inference_benchmark.matrix import MatrixConfig, MatrixRunConfig

    mc = MatrixConfig(runs=[MatrixRunConfig(name="r", config="c.yaml")])
    assert len(mc.runs) == 1
    assert mc.runs[0].overrides == {}


def test_explicit_runs_no_overrides_field_in_yaml() -> None:
    import tempfile

    from llm_inference_benchmark.matrix import load_matrix

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("runs:\n  - name: r1\n    config: c.yaml\n")
        fname = f.name
    mc = load_matrix(fname)
    assert mc.runs[0].overrides == {}


def test_expand_sweep_duplicate_names_raises() -> None:
    """Values with ambiguous string representations that collapse to the same name fragment
    should raise ValueError rather than silently overwriting a run."""
    with pytest.raises(ValueError, match="duplicate run name"):
        expand_sweep("c.yaml", {"mock.latency_ms": ["a/b", "a-b"]})
