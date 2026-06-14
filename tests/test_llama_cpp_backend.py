"""Unit tests for the llama-cpp-python backend.

All tests use mocked Llama instances — no GGUF model file or llama-cpp-python
package is required for CI. Integration tests (require a real GGUF file and
llama-cpp-python installed) are marked with pytest.mark.integration and skipped
automatically in standard CI runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llm_inference_benchmark.backends.llama_cpp as llama_cpp_mod

_HAS_LLAMA_CPP = importlib.util.find_spec("llama_cpp") is not None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_MODEL = "/fake/model.gguf"


def _make_llama_response(
    text: str = "mocked completion",
    prompt_tokens: int = 5,
    completion_tokens: int = 10,
) -> dict:
    return {
        "choices": [{"text": text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_mock_llama(
    text: str = "mocked completion",
    prompt_tokens: int = 5,
    completion_tokens: int = 10,
) -> tuple[MagicMock, MagicMock]:
    """Return (MockLlamaCls, mock_instance).

    MockLlamaCls(...) returns mock_instance.
    mock_instance(prompt, ...) returns the fake completion dict.
    """
    instance = MagicMock()
    instance.return_value = _make_llama_response(text, prompt_tokens, completion_tokens)
    cls = MagicMock(return_value=instance)
    return cls, instance


# ---------------------------------------------------------------------------
# Import-error path (llama_cpp not installed)
# ---------------------------------------------------------------------------


def test_import_error_raises_with_helpful_message() -> None:
    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", False),
        patch.object(llama_cpp_mod, "Llama", None),
    ):
        with pytest.raises(ImportError, match="uv sync --extra llama-cpp"):
            llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)


def test_import_error_message_mentions_cuda() -> None:
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", False),
        patch.object(llama_cpp_mod, "Llama", None),
    ):
        with pytest.raises(ImportError, match="CUDA"):
            llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)


# ---------------------------------------------------------------------------
# Backend name
# ---------------------------------------------------------------------------


def test_backend_name_is_llama_cpp() -> None:
    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        assert backend.name == "llama-cpp"


# ---------------------------------------------------------------------------
# Constructor passes correct kwargs to Llama
# ---------------------------------------------------------------------------


def test_constructor_passes_model_path() -> None:
    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        call_kwargs = cls.call_args.kwargs
        assert call_kwargs["model_path"] == _FAKE_MODEL


def test_constructor_defaults() -> None:
    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        kw = cls.call_args.kwargs
        assert kw["n_ctx"] == 2048
        assert kw["n_gpu_layers"] == 0
        assert kw["verbose"] is False
        assert "n_threads" not in kw


def test_constructor_n_threads_omitted_when_none() -> None:
    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL, n_threads=None)
        assert "n_threads" not in cls.call_args.kwargs


def test_constructor_n_threads_passed_when_set() -> None:
    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL, n_threads=8)
        assert cls.call_args.kwargs["n_threads"] == 8


def test_constructor_n_gpu_layers_forwarded() -> None:
    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL, n_gpu_layers=20)
        assert cls.call_args.kwargs["n_gpu_layers"] == 20


# ---------------------------------------------------------------------------
# generate() output
# ---------------------------------------------------------------------------


def test_generate_returns_generation_result() -> None:
    from llm_inference_benchmark.backends.base import GenerationResult

    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        result = backend.generate("hello")
        assert isinstance(result, GenerationResult)


def test_generate_text_from_completion() -> None:
    cls, _ = _make_mock_llama(text="world")
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        result = backend.generate("hello")
        assert result.text == "world"


def test_generate_input_tokens() -> None:
    cls, _ = _make_mock_llama(prompt_tokens=7)
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.input_tokens == 7


def test_generate_output_tokens() -> None:
    cls, _ = _make_mock_llama(completion_tokens=13)
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.output_tokens == 13


def test_generate_latency_is_positive() -> None:
    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        result = backend.generate("hello")
        assert result.latency_ms > 0


def test_generate_passes_echo_false() -> None:
    cls, instance = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL)
        backend.generate("prompt text")
        call_kwargs = instance.call_args.kwargs
        assert call_kwargs.get("echo") is False


def test_generate_passes_max_tokens() -> None:
    cls, instance = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL, max_tokens=30)
        backend.generate("prompt")
        assert instance.call_args.kwargs.get("max_tokens") == 30


def test_generate_passes_temperature() -> None:
    cls, instance = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL, temperature=0.7)
        backend.generate("prompt")
        assert instance.call_args.kwargs.get("temperature") == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_llama_cpp_backend_config_defaults() -> None:
    from llm_inference_benchmark.config import LlamaCppBackendConfig

    cfg = LlamaCppBackendConfig()
    assert cfg.n_ctx == 2048
    assert cfg.n_gpu_layers == 0
    assert cfg.max_tokens == 50
    assert cfg.temperature == 0.0
    assert cfg.n_threads is None
    assert cfg.verbose is False


def test_llama_cpp_backend_config_custom() -> None:
    from llm_inference_benchmark.config import LlamaCppBackendConfig

    cfg = LlamaCppBackendConfig(
        n_ctx=4096,
        n_gpu_layers=20,
        max_tokens=100,
        temperature=0.5,
        n_threads=8,
        verbose=True,
    )
    assert cfg.n_ctx == 4096
    assert cfg.n_gpu_layers == 20
    assert cfg.max_tokens == 100
    assert cfg.n_threads == 8


def test_benchmark_config_accepts_llama_cpp_backend() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig(backend="llama-cpp", model="/path/to/model.gguf")
    assert cfg.backend == "llama-cpp"
    assert cfg.model == "/path/to/model.gguf"


def test_benchmark_config_llama_cpp_sub_config() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig, LlamaCppBackendConfig

    cfg = BenchmarkConfig(
        backend="llama-cpp",
        model="/path/to/model.gguf",
        llama_cpp=LlamaCppBackendConfig(n_ctx=4096, n_gpu_layers=20),
    )
    assert cfg.llama_cpp.n_ctx == 4096
    assert cfg.llama_cpp.n_gpu_layers == 20


def test_benchmark_config_llama_cpp_from_yaml(tmp_path: Path) -> None:
    from llm_inference_benchmark.config import load_config

    cfg_file = tmp_path / "llama-cpp.yaml"
    cfg_file.write_text(
        "backend: llama-cpp\n"
        "model: /path/to/model.gguf\n"
        "requests: 5\n"
        "warmup_requests: 1\n"
        "llama_cpp:\n"
        "  n_ctx: 1024\n"
        "  n_gpu_layers: 0\n"
        "  max_tokens: 20\n"
        "  temperature: 0.0\n"
        "  verbose: false\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.backend == "llama-cpp"
    assert cfg.llama_cpp.n_ctx == 1024
    assert cfg.llama_cpp.n_gpu_layers == 0


# ---------------------------------------------------------------------------
# CLI / _build_backend integration
# ---------------------------------------------------------------------------


def test_build_backend_dispatches_to_llama_cpp(tmp_path: Path, tmp_prompts: Path) -> None:
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    cfg_file = tmp_path / "llama-cpp.yaml"
    out_csv = tmp_path / "out.csv"
    cfg_file.write_text(
        f"backend: llama-cpp\n"
        f"model: /fake/model.gguf\n"
        f"requests: 2\n"
        f"warmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"llama_cpp:\n"
        f"  n_ctx: 512\n"
        f"  n_gpu_layers: 0\n"
        f"  max_tokens: 5\n"
    )

    cls, _ = _make_mock_llama()
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(cfg_file), "--output", str(out_csv)])

    assert result.exit_code == 0, result.output
    assert out_csv.exists()
    content = out_csv.read_text()
    assert "llama-cpp" in content


# ---------------------------------------------------------------------------
# Full run_benchmark integration (mocked)
# ---------------------------------------------------------------------------


def test_full_run_benchmark_with_mock(tmp_prompts: Path) -> None:
    from llm_inference_benchmark.config import BenchmarkConfig
    from llm_inference_benchmark.runner import load_prompts, run_benchmark

    cls, _ = _make_mock_llama(prompt_tokens=8, completion_tokens=12)
    with (
        patch.object(llama_cpp_mod, "_AVAILABLE", True),
        patch.object(llama_cpp_mod, "Llama", cls),
    ):
        backend = llama_cpp_mod.LlamaCppBackend(model_path=_FAKE_MODEL, max_tokens=12)
        cfg = BenchmarkConfig(
            backend="llama-cpp",
            model=_FAKE_MODEL,
            requests=5,
            warmup_requests=1,
            prompts_file=str(tmp_prompts),
        )
        report = run_benchmark(backend, cfg, load_prompts(str(tmp_prompts)))

    assert report.request_count == 5
    assert report.backend == "llama-cpp"
    assert report.model == _FAKE_MODEL
    assert report.p50_latency_ms > 0
    assert report.tokens_per_second > 0
    assert report.total_tokens == (8 + 12) * 5


# ---------------------------------------------------------------------------
# Integration tests (require real llama_cpp install + GGUF file)
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.integration
_skip_without_llama_cpp = pytest.mark.skipif(
    not _HAS_LLAMA_CPP,
    reason="llama-cpp-python not installed — run: uv sync --extra llama-cpp",
)


@pytest.mark.integration
@_skip_without_llama_cpp
def test_integration_real_model(tmp_path: Path, tmp_prompts: Path) -> None:
    """Real GGUF inference — set LLAMA_MODEL_PATH env var to a local GGUF file."""
    import os

    model_path = os.environ.get("LLAMA_MODEL_PATH")
    if not model_path:
        pytest.skip("LLAMA_MODEL_PATH env var not set — skipping real-model test")

    from llm_inference_benchmark.backends.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend(
        model_path=model_path,
        n_ctx=256,
        n_gpu_layers=0,
        max_tokens=10,
        verbose=False,
    )
    result = backend.generate("The capital of France is")
    assert result.latency_ms > 0
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert isinstance(result.text, str)
