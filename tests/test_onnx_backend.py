"""Unit tests for the ONNX Runtime backend.

All tests mock ORTModelForCausalLM and AutoTokenizer — no real model or optimum
package is required for CI. Integration tests (require optimum[onnxruntime] and a
pre-exported ONNX model) are marked with pytest.mark.integration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llm_inference_benchmark.backends.onnx as onnx_mod

_HAS_OPTIMUM = (
    importlib.util.find_spec("optimum") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)

_FAKE_MODEL = "fake/onnx-model"
_INPUT_LEN = 3
_OUTPUT_LEN = 5
_PAD_TOKEN_ID = 50256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_torch() -> MagicMock:
    mock = MagicMock()
    mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return mock


def _make_mock_tokenizer(input_len: int = _INPUT_LEN) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.pad_token_id = _PAD_TOKEN_ID

    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, input_len)
    tokenizer.return_value = {"input_ids": mock_input_ids}
    tokenizer.decode.return_value = "mocked output text"

    tokenizer_cls = MagicMock()
    tokenizer_cls.from_pretrained.return_value = tokenizer
    return tokenizer_cls


def _make_mock_ort_cls(
    input_len: int = _INPUT_LEN,
    output_len: int = _OUTPUT_LEN,
    call_logits_processor: bool = False,
) -> tuple[MagicMock, MagicMock]:
    """Return (ORTModelForCausalLM mock cls, model instance mock).

    When call_logits_processor=True, model.generate calls each logits_processor
    once so that TTFT is recorded by _FirstTokenTimer.
    """
    total_len = input_len + output_len

    mock_output_ids = MagicMock()
    mock_output_ids.shape = (1, total_len)
    mock_output_ids.__getitem__ = MagicMock(return_value=MagicMock())

    model_instance = MagicMock()

    if call_logits_processor:

        def _fake_generate(**kwargs: object) -> MagicMock:
            for proc in kwargs.get("logits_processor", []):  # type: ignore[union-attr]
                proc(MagicMock(), MagicMock())
            return mock_output_ids

        model_instance.generate.side_effect = _fake_generate
    else:
        model_instance.generate.return_value = mock_output_ids

    ort_cls = MagicMock()
    ort_cls.from_pretrained.return_value = model_instance
    return ort_cls, model_instance


# ---------------------------------------------------------------------------
# Import-error path
# ---------------------------------------------------------------------------


def test_import_error_raises_with_helpful_message() -> None:
    with (
        patch.object(onnx_mod, "_AVAILABLE", False),
        patch.object(onnx_mod, "ORTModelForCausalLM", None),
        patch.object(onnx_mod, "AutoTokenizer", None),
    ):
        with pytest.raises(ImportError, match="uv sync --extra onnx"):
            onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)


# ---------------------------------------------------------------------------
# Backend name
# ---------------------------------------------------------------------------


def test_backend_name_is_onnx() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        assert backend.name == "onnx"


# ---------------------------------------------------------------------------
# Constructor — from_pretrained arguments
# ---------------------------------------------------------------------------


def test_constructor_calls_from_pretrained_with_model_id() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        ort_cls.from_pretrained.assert_called_once()
        # from_pretrained is called with model_id as the first positional argument
        assert ort_cls.from_pretrained.call_args.args[0] == _FAKE_MODEL


def test_constructor_passes_export_false_by_default() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        call_kwargs = ort_cls.from_pretrained.call_args.kwargs
        assert call_kwargs.get("export") is False


def test_constructor_passes_export_true_when_requested() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        onnx_mod.OnnxBackend(model_id=_FAKE_MODEL, export=True)
        call_kwargs = ort_cls.from_pretrained.call_args.kwargs
        assert call_kwargs.get("export") is True


def test_constructor_uses_cpu_provider_by_default() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        onnx_mod.OnnxBackend(model_id=_FAKE_MODEL, device="cpu")
        call_kwargs = ort_cls.from_pretrained.call_args.kwargs
        assert call_kwargs.get("provider") == "CPUExecutionProvider"


def test_constructor_uses_cuda_provider_when_device_is_cuda() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        onnx_mod.OnnxBackend(model_id=_FAKE_MODEL, device="cuda")
        call_kwargs = ort_cls.from_pretrained.call_args.kwargs
        assert call_kwargs.get("provider") == "CUDAExecutionProvider"


def test_constructor_uses_cuda_provider_for_indexed_device() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        onnx_mod.OnnxBackend(model_id=_FAKE_MODEL, device="cuda:0")
        call_kwargs = ort_cls.from_pretrained.call_args.kwargs
        assert call_kwargs.get("provider") == "CUDAExecutionProvider"


def test_constructor_sets_pad_token_when_none() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    tok_cls.from_pretrained.return_value.pad_token_id = None
    tok_cls.from_pretrained.return_value.eos_token_id = 0

    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        assert backend._tokenizer.pad_token_id == 0


# ---------------------------------------------------------------------------
# generate() — return type and fields
# ---------------------------------------------------------------------------


def test_generate_returns_generation_result() -> None:
    from llm_inference_benchmark.backends.base import GenerationResult

    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        result = backend.generate("hello world")
        assert isinstance(result, GenerationResult)


def test_generate_latency_is_positive() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        result = backend.generate("hello world")
        assert result.latency_ms > 0


def test_generate_text_from_tokenizer_decode() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    tok_cls.from_pretrained.return_value.decode.return_value = "decoded output"
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.text == "decoded output"


def test_generate_input_token_count() -> None:
    ort_cls, _ = _make_mock_ort_cls(input_len=7)
    tok_cls = _make_mock_tokenizer(input_len=7)
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.input_tokens == 7


def test_generate_output_token_count() -> None:
    ort_cls, _ = _make_mock_ort_cls(input_len=3, output_len=8)
    tok_cls = _make_mock_tokenizer(input_len=3)
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.output_tokens == 8


def test_generate_ttft_none_when_logits_processor_not_called() -> None:
    ort_cls, _ = _make_mock_ort_cls(call_logits_processor=False)
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.ttft_ms is None


def test_generate_ttft_recorded_when_logits_processor_called() -> None:
    ort_cls, _ = _make_mock_ort_cls(call_logits_processor=True)
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        result = backend.generate("prompt")
        assert result.ttft_ms is not None
        assert result.ttft_ms > 0


def test_generate_calls_model_with_max_new_tokens() -> None:
    ort_cls, model_instance = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL, max_new_tokens=42)
        backend.generate("prompt")
        call_kwargs = model_instance.generate.call_args.kwargs
        assert call_kwargs.get("max_new_tokens") == 42


def test_generate_calls_model_with_do_sample_false_by_default() -> None:
    ort_cls, model_instance = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        backend.generate("prompt")
        call_kwargs = model_instance.generate.call_args.kwargs
        assert call_kwargs.get("do_sample") is False


def test_generate_passes_logits_processor_list() -> None:
    ort_cls, model_instance = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", _make_mock_torch()),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL)
        backend.generate("prompt")
        call_kwargs = model_instance.generate.call_args.kwargs
        lp = call_kwargs.get("logits_processor", [])
        assert len(lp) == 1
        assert isinstance(lp[0], onnx_mod._FirstTokenTimer)


# ---------------------------------------------------------------------------
# Seed handling
# ---------------------------------------------------------------------------


def test_seed_calls_torch_manual_seed() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    mock_torch = _make_mock_torch()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", mock_torch),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL, seed=42)
        backend.generate("prompt")
        mock_torch.manual_seed.assert_called_once_with(42)


def test_no_seed_does_not_call_torch_manual_seed() -> None:
    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    mock_torch = _make_mock_torch()
    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", mock_torch),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL, seed=None)
        backend.generate("prompt")
        mock_torch.manual_seed.assert_not_called()


# ---------------------------------------------------------------------------
# _FirstTokenTimer unit tests
# ---------------------------------------------------------------------------


def test_first_token_timer_records_on_first_call() -> None:
    import time

    start = time.perf_counter()
    timer = onnx_mod._FirstTokenTimer(start)
    assert timer.ttft_ms is None
    timer(MagicMock(), MagicMock())
    assert timer.ttft_ms is not None
    assert timer.ttft_ms >= 0


def test_first_token_timer_does_not_overwrite_on_second_call() -> None:
    import time

    start = time.perf_counter()
    timer = onnx_mod._FirstTokenTimer(start)
    timer(MagicMock(), MagicMock())
    first_ttft = timer.ttft_ms
    timer(MagicMock(), MagicMock())
    assert timer.ttft_ms == first_ttft


def test_first_token_timer_returns_scores_unchanged() -> None:
    import time

    start = time.perf_counter()
    timer = onnx_mod._FirstTokenTimer(start)
    scores = MagicMock()
    result = timer(MagicMock(), scores)
    assert result is scores


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_onnx_backend_config_defaults() -> None:
    from llm_inference_benchmark.config import OnnxBackendConfig

    cfg = OnnxBackendConfig()
    assert cfg.max_new_tokens == 50
    assert cfg.device == "cpu"
    assert cfg.do_sample is False
    assert cfg.export is False


def test_onnx_backend_config_custom() -> None:
    from llm_inference_benchmark.config import OnnxBackendConfig

    cfg = OnnxBackendConfig(max_new_tokens=100, device="cuda", do_sample=True, export=True)
    assert cfg.max_new_tokens == 100
    assert cfg.device == "cuda"
    assert cfg.do_sample is True
    assert cfg.export is True


def test_benchmark_config_accepts_onnx_backend() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig(backend="onnx", model="path/to/onnx-model")
    assert cfg.backend == "onnx"
    assert cfg.model == "path/to/onnx-model"


def test_benchmark_config_onnx_sub_config() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig, OnnxBackendConfig

    cfg = BenchmarkConfig(
        backend="onnx",
        model="path/to/onnx-model",
        onnx=OnnxBackendConfig(max_new_tokens=20, device="cuda"),
    )
    assert cfg.onnx.max_new_tokens == 20
    assert cfg.onnx.device == "cuda"


def test_benchmark_config_onnx_from_yaml(tmp_path: Path) -> None:
    from llm_inference_benchmark.config import load_config

    cfg_file = tmp_path / "onnx.yaml"
    cfg_file.write_text(
        "backend: onnx\n"
        "model: path/to/onnx-model\n"
        "requests: 5\n"
        "warmup_requests: 1\n"
        "onnx:\n"
        "  max_new_tokens: 20\n"
        "  device: cpu\n"
        "  export: false\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.backend == "onnx"
    assert cfg.onnx.max_new_tokens == 20
    assert cfg.onnx.export is False


# ---------------------------------------------------------------------------
# validate-config CLI
# ---------------------------------------------------------------------------


def test_validate_config_shows_onnx_params(tmp_path: Path, tmp_prompts: Path) -> None:
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    cfg_file = tmp_path / "onnx.yaml"
    cfg_file.write_text(
        f"backend: onnx\n"
        f"model: path/to/onnx-model\n"
        f"requests: 5\n"
        f"warmup_requests: 1\n"
        f"prompts_file: {tmp_prompts}\n"
        f"onnx:\n"
        f"  max_new_tokens: 30\n"
        f"  device: cpu\n"
        f"  export: false\n"
    )
    runner = CliRunner()
    result = runner.invoke(main, ["validate-config", "--config", str(cfg_file)])
    assert result.exit_code == 0, result.output
    assert "onnx.max_new_tokens" in result.output
    assert "30" in result.output
    assert "OK" in result.output


# ---------------------------------------------------------------------------
# _build_backend integration
# ---------------------------------------------------------------------------


def test_build_backend_dispatches_to_onnx(tmp_path: Path, tmp_prompts: Path) -> None:
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    cfg_file = tmp_path / "onnx.yaml"
    out_csv = tmp_path / "out.csv"
    cfg_file.write_text(
        f"backend: onnx\n"
        f"model: fake/onnx-model\n"
        f"requests: 2\n"
        f"warmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"onnx:\n"
        f"  max_new_tokens: 5\n"
    )

    ort_cls, _ = _make_mock_ort_cls()
    tok_cls = _make_mock_tokenizer()
    mock_torch = _make_mock_torch()

    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", mock_torch),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(cfg_file), "--output", str(out_csv)])

    assert result.exit_code == 0, result.output
    assert out_csv.exists()
    assert "onnx" in out_csv.read_text()


# ---------------------------------------------------------------------------
# Full run_benchmark integration (mocked)
# ---------------------------------------------------------------------------


def test_full_run_benchmark_with_mocked_onnx(tmp_prompts: Path) -> None:
    from llm_inference_benchmark.config import BenchmarkConfig
    from llm_inference_benchmark.runner import load_prompts, run_benchmark

    ort_cls, _ = _make_mock_ort_cls(input_len=4, output_len=6)
    tok_cls = _make_mock_tokenizer(input_len=4)
    mock_torch = _make_mock_torch()

    with (
        patch.object(onnx_mod, "_AVAILABLE", True),
        patch.object(onnx_mod, "ORTModelForCausalLM", ort_cls),
        patch.object(onnx_mod, "AutoTokenizer", tok_cls),
        patch.object(onnx_mod, "torch", mock_torch),
    ):
        backend = onnx_mod.OnnxBackend(model_id=_FAKE_MODEL, max_new_tokens=6)
        cfg = BenchmarkConfig(
            backend="onnx",
            model=_FAKE_MODEL,
            requests=3,
            warmup_requests=1,
            prompts_file=str(tmp_prompts),
        )
        report = run_benchmark(backend, cfg, load_prompts(str(tmp_prompts)))

    assert report.request_count == 3
    assert report.backend == "onnx"
    assert report.model == _FAKE_MODEL
    assert report.p50_latency_ms > 0


# ---------------------------------------------------------------------------
# Integration tests (require optimum[onnxruntime] and a real model)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_onnx_backend() -> None:
    """Real inference — requires: uv sync --extra onnx and a pre-exported ONNX model."""
    import os

    if not _HAS_OPTIMUM:
        pytest.skip("optimum[onnxruntime] not installed — run: uv sync --extra onnx")

    model_id = os.environ.get("ONNX_MODEL_ID", "optimum-internal-testing/tiny-random-GPT2Model")
    backend = onnx_mod.OnnxBackend(model_id=model_id, max_new_tokens=10, export=True)
    result = backend.generate("The capital of France is")
    assert result.latency_ms > 0
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert isinstance(result.text, str)
