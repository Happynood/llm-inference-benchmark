"""Unit tests for the vLLM backend.

All tests mock vllm.LLM and vllm.SamplingParams — no GPU or vllm package
required for CI. Integration tests (require vllm and a CUDA GPU) are marked
with pytest.mark.integration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import llm_inference_benchmark.backends.vllm_backend as vllm_mod

_HAS_VLLM = importlib.util.find_spec("vllm") is not None

_FAKE_MODEL = "fake/vllm-model"
_INPUT_TOKEN_IDS = [1, 2, 3]
_OUTPUT_TOKEN_IDS = [4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(text: str = "generated text", token_ids: list[int] | None = None) -> object:
    comp = MagicMock()
    comp.text = text
    comp.token_ids = token_ids if token_ids is not None else _OUTPUT_TOKEN_IDS
    return comp


def _make_request_output(
    prompt_token_ids: list[int] | None = None,
    completions: list[object] | None = None,
    metrics: object | None = None,
    prompt_logprobs: list | None = None,
) -> MagicMock:
    out = MagicMock()
    out.prompt_token_ids = prompt_token_ids if prompt_token_ids is not None else _INPUT_TOKEN_IDS
    out.outputs = completions if completions is not None else [_make_completion()]
    out.metrics = metrics
    out.prompt_logprobs = prompt_logprobs
    return out


def _make_mock_llm(outputs: list | None = None) -> tuple[MagicMock, MagicMock]:
    """Return (LLM class mock, instance mock)."""
    default_output = _make_request_output()
    instance = MagicMock()
    instance.generate.return_value = outputs if outputs is not None else [default_output]
    cls = MagicMock(return_value=instance)
    return cls, instance


def _make_mock_sampling_params() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Import-error path
# ---------------------------------------------------------------------------


def test_import_error_raises_with_helpful_message() -> None:
    with (
        patch.object(vllm_mod, "_AVAILABLE", False),
        patch.object(vllm_mod, "LLM", None),
        patch.object(vllm_mod, "SamplingParams", None),
    ):
        with pytest.raises(ImportError, match="uv sync --extra vllm"):
            vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)


# ---------------------------------------------------------------------------
# Backend name
# ---------------------------------------------------------------------------


def test_backend_name_is_vllm() -> None:
    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", _make_mock_sampling_params()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        assert backend.name == "vllm"


# ---------------------------------------------------------------------------
# Constructor — LLM instantiation
# ---------------------------------------------------------------------------


def test_constructor_passes_model_id() -> None:
    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        call_kwargs = llm_cls.call_args.kwargs
        assert call_kwargs["model"] == _FAKE_MODEL


def test_constructor_passes_tensor_parallel_size() -> None:
    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        vllm_mod.VLLMBackend(model_id=_FAKE_MODEL, tensor_parallel_size=2)
        assert llm_cls.call_args.kwargs["tensor_parallel_size"] == 2


def test_constructor_passes_gpu_memory_utilization() -> None:
    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        vllm_mod.VLLMBackend(model_id=_FAKE_MODEL, gpu_memory_utilization=0.8)
        assert llm_cls.call_args.kwargs["gpu_memory_utilization"] == 0.8


def test_constructor_passes_dtype() -> None:
    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        vllm_mod.VLLMBackend(model_id=_FAKE_MODEL, dtype="float16")
        assert llm_cls.call_args.kwargs["dtype"] == "float16"


def test_constructor_uses_seed_zero_when_no_seed_given() -> None:
    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        vllm_mod.VLLMBackend(model_id=_FAKE_MODEL, seed=None)
        assert llm_cls.call_args.kwargs["seed"] == 0


def test_constructor_passes_explicit_seed() -> None:
    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        vllm_mod.VLLMBackend(model_id=_FAKE_MODEL, seed=42)
        assert llm_cls.call_args.kwargs["seed"] == 42


# ---------------------------------------------------------------------------
# generate() — return type and fields
# ---------------------------------------------------------------------------


def test_generate_returns_generation_result() -> None:
    from llm_inference_benchmark.backends.base import GenerationResult

    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.generate("hello")
        assert isinstance(result, GenerationResult)


def test_generate_latency_is_positive() -> None:
    llm_cls, _ = _make_mock_llm()
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.generate("hello")
        assert result.latency_ms > 0


def test_generate_text_from_completion() -> None:
    completion = _make_completion(text="hello there")
    out = _make_request_output(completions=[completion])
    llm_cls, instance = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.text == "hello there"


def test_generate_input_token_count() -> None:
    out = _make_request_output(prompt_token_ids=[10, 20, 30, 40])
    llm_cls, _ = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.input_tokens == 4


def test_generate_output_token_count() -> None:
    completion = _make_completion(token_ids=[1, 2, 3, 4, 5, 6])
    out = _make_request_output(completions=[completion])
    llm_cls, _ = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.output_tokens == 6


def test_generate_ttft_none_when_metrics_absent() -> None:
    out = _make_request_output(metrics=None)
    llm_cls, _ = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.ttft_ms is None


def test_generate_ttft_none_when_first_token_time_missing() -> None:
    metrics = SimpleNamespace(first_token_time=None, arrival_time=1000.0)
    out = _make_request_output(metrics=metrics)
    llm_cls, _ = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.ttft_ms is None


def test_generate_ttft_computed_from_metrics() -> None:
    metrics = SimpleNamespace(arrival_time=1000.0, first_token_time=1000.05)
    out = _make_request_output(metrics=metrics)
    llm_cls, _ = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.ttft_ms == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# compute_perplexity()
# ---------------------------------------------------------------------------


def _make_logprob(logprob: float) -> object:
    lp = MagicMock()
    lp.logprob = logprob
    return lp


def test_compute_perplexity_returns_none_for_empty_logprobs() -> None:
    out = _make_request_output(prompt_logprobs=None)
    llm_cls, instance = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.compute_perplexity(["text"])
        assert result is None


def test_compute_perplexity_returns_none_for_short_sequence() -> None:
    # Sequence with only 1 token — nothing to score
    out = _make_request_output(
        prompt_token_ids=[42],
        prompt_logprobs=[None],
    )
    llm_cls, instance = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.compute_perplexity(["x"])
        assert result is None


def test_compute_perplexity_uses_actual_token_logprob() -> None:
    import math

    token_ids = [10, 20, 30]
    # logprob[0] is None (no context), logprob[1] for token_id=20, logprob[2] for token_id=30
    logprob_20 = _make_logprob(-1.0)
    logprob_30 = _make_logprob(-2.0)
    prompt_logprobs = [
        None,
        {20: logprob_20},
        {30: logprob_30},
    ]
    out = _make_request_output(prompt_token_ids=token_ids, prompt_logprobs=prompt_logprobs)
    llm_cls, instance = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.compute_perplexity(["dummy"])
    # mean NLL = (1.0 + 2.0) / 2 = 1.5 → perplexity = exp(1.5)
    assert result == pytest.approx(math.exp(1.5), rel=1e-6)


def test_compute_perplexity_skips_missing_token_id() -> None:
    import math

    token_ids = [10, 20, 30]
    # token 20 is missing from its position dict — only token 30 contributes
    logprob_30 = _make_logprob(-1.0)
    prompt_logprobs = [
        None,
        {99: _make_logprob(-5.0)},  # wrong token id
        {30: logprob_30},
    ]
    out = _make_request_output(prompt_token_ids=token_ids, prompt_logprobs=prompt_logprobs)
    llm_cls, _ = _make_mock_llm(outputs=[out])
    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL)
        result = backend.compute_perplexity(["dummy"])
    assert result == pytest.approx(math.exp(1.0), rel=1e-6)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_vllm_backend_config_defaults() -> None:
    from llm_inference_benchmark.config import VLLMBackendConfig

    cfg = VLLMBackendConfig()
    assert cfg.max_new_tokens == 50
    assert cfg.temperature == 0.0
    assert cfg.tensor_parallel_size == 1
    assert cfg.gpu_memory_utilization == 0.9
    assert cfg.dtype == "auto"


def test_vllm_backend_config_custom() -> None:
    from llm_inference_benchmark.config import VLLMBackendConfig

    cfg = VLLMBackendConfig(
        max_new_tokens=100,
        temperature=0.7,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
    )
    assert cfg.max_new_tokens == 100
    assert cfg.temperature == 0.7
    assert cfg.tensor_parallel_size == 2
    assert cfg.gpu_memory_utilization == 0.8
    assert cfg.dtype == "bfloat16"


def test_benchmark_config_accepts_vllm_backend() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig(backend="vllm", model="meta-llama/Llama-3.2-3B")
    assert cfg.backend == "vllm"
    assert cfg.model == "meta-llama/Llama-3.2-3B"


def test_benchmark_config_vllm_sub_config() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig, VLLMBackendConfig

    cfg = BenchmarkConfig(
        backend="vllm",
        model="meta-llama/Llama-3.2-3B",
        vllm=VLLMBackendConfig(max_new_tokens=200, tensor_parallel_size=2),
    )
    assert cfg.vllm.max_new_tokens == 200
    assert cfg.vllm.tensor_parallel_size == 2


def test_benchmark_config_vllm_from_yaml(tmp_path: Path) -> None:
    from llm_inference_benchmark.config import load_config

    cfg_file = tmp_path / "vllm.yaml"
    cfg_file.write_text(
        "backend: vllm\n"
        "model: meta-llama/Llama-3.2-3B\n"
        "requests: 5\n"
        "warmup_requests: 1\n"
        "vllm:\n"
        "  max_new_tokens: 128\n"
        "  tensor_parallel_size: 1\n"
        "  gpu_memory_utilization: 0.85\n"
        "  dtype: bfloat16\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.backend == "vllm"
    assert cfg.vllm.max_new_tokens == 128
    assert cfg.vllm.gpu_memory_utilization == 0.85
    assert cfg.vllm.dtype == "bfloat16"


# ---------------------------------------------------------------------------
# validate-config CLI
# ---------------------------------------------------------------------------


def test_validate_config_shows_vllm_params(tmp_path: Path, tmp_prompts: Path) -> None:
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    cfg_file = tmp_path / "vllm.yaml"
    cfg_file.write_text(
        f"backend: vllm\n"
        f"model: meta-llama/Llama-3.2-3B\n"
        f"requests: 5\n"
        f"warmup_requests: 1\n"
        f"prompts_file: {tmp_prompts}\n"
        f"vllm:\n"
        f"  max_new_tokens: 128\n"
        f"  tensor_parallel_size: 1\n"
        f"  gpu_memory_utilization: 0.85\n"
        f"  dtype: bfloat16\n"
    )
    runner = CliRunner()
    result = runner.invoke(main, ["validate-config", "--config", str(cfg_file)])
    assert result.exit_code == 0, result.output
    assert "vllm.max_new_tokens" in result.output
    assert "128" in result.output
    assert "bfloat16" in result.output
    assert "OK" in result.output


def test_validate_config_vllm_json_output(tmp_path: Path, tmp_prompts: Path) -> None:
    import json

    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    cfg_file = tmp_path / "vllm.yaml"
    cfg_file.write_text(
        f"backend: vllm\n"
        f"model: meta-llama/Llama-3.2-3B\n"
        f"requests: 5\n"
        f"warmup_requests: 1\n"
        f"prompts_file: {tmp_prompts}\n"
        f"vllm:\n"
        f"  max_new_tokens: 64\n"
        f"  dtype: float16\n"
    )
    runner = CliRunner()
    result = runner.invoke(main, ["validate-config", "--config", str(cfg_file), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["backend"] == "vllm"
    assert data["valid"] is True
    assert data["backend_config"]["max_new_tokens"] == 64
    assert data["backend_config"]["dtype"] == "float16"
    assert data["backend_config"]["tensor_parallel_size"] == 1


# ---------------------------------------------------------------------------
# _build_backend dispatch
# ---------------------------------------------------------------------------


def test_build_backend_dispatches_to_vllm(tmp_path: Path, tmp_prompts: Path) -> None:
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    cfg_file = tmp_path / "vllm.yaml"
    out_csv = tmp_path / "out.csv"
    cfg_file.write_text(
        f"backend: vllm\n"
        f"model: {_FAKE_MODEL}\n"
        f"requests: 2\n"
        f"warmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"vllm:\n"
        f"  max_new_tokens: 5\n"
    )

    llm_cls, _ = _make_mock_llm()

    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(cfg_file), "--output", str(out_csv)])

    assert result.exit_code == 0, result.output
    assert out_csv.exists()
    assert "vllm" in out_csv.read_text()


# ---------------------------------------------------------------------------
# Full run_benchmark integration (mocked)
# ---------------------------------------------------------------------------


def test_full_run_benchmark_with_mocked_vllm(tmp_prompts: Path) -> None:
    from llm_inference_benchmark.config import BenchmarkConfig
    from llm_inference_benchmark.runner import load_prompts, run_benchmark

    out = _make_request_output(
        prompt_token_ids=[1, 2, 3, 4],
        completions=[_make_completion(text="response", token_ids=[5, 6, 7, 8, 9, 10])],
    )
    llm_cls, instance = _make_mock_llm(outputs=[out])
    instance.generate.return_value = [out]

    with (
        patch.object(vllm_mod, "_AVAILABLE", True),
        patch.object(vllm_mod, "LLM", llm_cls),
        patch.object(vllm_mod, "SamplingParams", MagicMock()),
    ):
        backend = vllm_mod.VLLMBackend(model_id=_FAKE_MODEL, max_new_tokens=6)
        cfg = BenchmarkConfig(
            backend="vllm",
            model=_FAKE_MODEL,
            requests=3,
            warmup_requests=1,
            prompts_file=str(tmp_prompts),
        )
        report = run_benchmark(backend, cfg, load_prompts(str(tmp_prompts)))

    assert report.request_count == 3
    assert report.backend == "vllm"
    assert report.model == _FAKE_MODEL
    assert report.p50_latency_ms > 0


# ---------------------------------------------------------------------------
# Integration tests (require vllm and a CUDA GPU)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_vllm_backend() -> None:
    """Real inference — requires: uv sync --extra vllm and a CUDA GPU."""
    import os

    if not _HAS_VLLM:
        pytest.skip("vllm not installed — run: uv sync --extra vllm")

    model_id = os.environ.get("VLLM_MODEL_ID", "facebook/opt-125m")
    backend = vllm_mod.VLLMBackend(model_id=model_id, max_new_tokens=10)
    result = backend.generate("The capital of France is")
    assert result.latency_ms > 0
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert isinstance(result.text, str)
