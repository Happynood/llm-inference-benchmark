"""Unit tests for the OpenAI-compatible endpoint backend.

All tests mock urllib's urlopen — no real network call is made. Integration
tests against a live local server are marked with pytest.mark.integration and
skipped automatically unless OPENAI_ENDPOINT_BASE_URL is set.
"""

from __future__ import annotations

import json
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llm_inference_benchmark.backends.openai_endpoint as openai_endpoint_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_URL = "http://localhost:8080/v1"
_MODEL = "test-model"


def _make_chat_response(
    content: str = "mocked completion",
    prompt_tokens: int | None = 5,
    completion_tokens: int | None = 10,
) -> dict:
    body: dict = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if prompt_tokens is not None and completion_tokens is not None:
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return body


def _make_mock_urlopen(body: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(body).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return MagicMock(return_value=response)


# ---------------------------------------------------------------------------
# Backend name
# ---------------------------------------------------------------------------


def test_backend_name_is_openai() -> None:
    backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
    assert backend.name == "openai"


# ---------------------------------------------------------------------------
# generate() success path
# ---------------------------------------------------------------------------


def test_generate_returns_generation_result() -> None:
    from llm_inference_benchmark.backends.base import GenerationResult

    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate("hello")
        assert isinstance(result, GenerationResult)


def test_generate_text_from_completion() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response(content="world"))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate("hello")
        assert result.text == "world"


def test_generate_uses_usage_tokens_when_present() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response(prompt_tokens=7, completion_tokens=13))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate("prompt")
        assert result.input_tokens == 7
        assert result.output_tokens == 13


def test_generate_falls_back_to_word_count_when_usage_missing() -> None:
    body = _make_chat_response(
        content="four word reply here", prompt_tokens=None, completion_tokens=None
    )
    mock_urlopen = _make_mock_urlopen(body)
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate("two words")
        assert result.input_tokens == 2
        assert result.output_tokens == 4


def test_generate_latency_is_positive() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate("hello")
        assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def test_generate_posts_to_chat_completions_path() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        backend.generate("hello")
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://localhost:8080/v1/chat/completions"


def test_generate_strips_trailing_slash_from_base_url() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url="http://localhost:8080/v1/", model=_MODEL
        )
        backend.generate("hello")
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://localhost:8080/v1/chat/completions"


def test_generate_sends_model_and_max_tokens_in_payload() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, max_tokens=42, temperature=0.5
        )
        backend.generate("hello there")
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == _MODEL
        assert payload["max_tokens"] == 42
        assert payload["temperature"] == pytest.approx(0.5)
        assert payload["messages"] == [{"role": "user", "content": "hello there"}]


def test_generate_passes_configured_timeout() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, timeout_s=5.0
        )
        backend.generate("hello")
        assert mock_urlopen.call_args.kwargs["timeout"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# API key handling (must come from an environment variable only)
# ---------------------------------------------------------------------------


def test_generate_omits_authorization_header_when_api_key_env_unset() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        backend.generate("hello")
        request = mock_urlopen.call_args.args[0]
        assert request.get_header("Authorization") is None


def test_generate_omits_authorization_header_when_env_var_not_set() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with (
        patch.object(openai_endpoint_mod, "urlopen", mock_urlopen),
        patch.dict(os.environ, {}, clear=False),
    ):
        os.environ.pop("OPENAI_TEST_KEY_DOES_NOT_EXIST", None)
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, api_key_env="OPENAI_TEST_KEY_DOES_NOT_EXIST"
        )
        backend.generate("hello")
        request = mock_urlopen.call_args.args[0]
        assert request.get_header("Authorization") is None


def test_generate_sends_authorization_header_when_api_key_env_set() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with (
        patch.object(openai_endpoint_mod, "urlopen", mock_urlopen),
        patch.dict(os.environ, {"OPENAI_TEST_KEY": "secret-value-123"}),
    ):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, api_key_env="OPENAI_TEST_KEY"
        )
        backend.generate("hello")
        request = mock_urlopen.call_args.args[0]
        assert request.get_header("Authorization") == "Bearer secret-value-123"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_generate_raises_runtime_error_on_http_error() -> None:
    mock_urlopen = MagicMock(
        side_effect=urllib.error.HTTPError(
            url=_BASE_URL, code=500, msg="Internal Server Error", hdrs=MagicMock(), fp=None
        )
    )
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        with pytest.raises(RuntimeError, match="OpenAI-compatible endpoint request failed"):
            backend.generate("hello")


def test_generate_raises_runtime_error_on_connection_error() -> None:
    mock_urlopen = MagicMock(side_effect=urllib.error.URLError("connection refused"))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        with pytest.raises(RuntimeError, match="OpenAI-compatible endpoint request failed"):
            backend.generate("hello")


def test_generate_raises_runtime_error_on_timeout() -> None:
    mock_urlopen = MagicMock(side_effect=TimeoutError("timed out"))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        with pytest.raises(RuntimeError, match="OpenAI-compatible endpoint request failed"):
            backend.generate("hello")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_openai_endpoint_config_defaults() -> None:
    from llm_inference_benchmark.config import OpenAIEndpointConfig

    cfg = OpenAIEndpointConfig()
    assert cfg.base_url == "http://localhost:8080/v1"
    assert cfg.api_key_env is None
    assert cfg.max_tokens == 50
    assert cfg.temperature == 0.0
    assert cfg.timeout_s == 60.0


def test_openai_endpoint_config_custom() -> None:
    from llm_inference_benchmark.config import OpenAIEndpointConfig

    cfg = OpenAIEndpointConfig(
        base_url="http://example.com/v1",
        api_key_env="MY_API_KEY",
        max_tokens=100,
        temperature=0.7,
        timeout_s=30.0,
    )
    assert cfg.base_url == "http://example.com/v1"
    assert cfg.api_key_env == "MY_API_KEY"
    assert cfg.max_tokens == 100
    assert cfg.timeout_s == 30.0


def test_benchmark_config_accepts_openai_backend() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig(backend="openai", model="gpt-test")
    assert cfg.backend == "openai"
    assert cfg.model == "gpt-test"


def test_benchmark_config_openai_sub_config() -> None:
    from llm_inference_benchmark.config import BenchmarkConfig, OpenAIEndpointConfig

    cfg = BenchmarkConfig(
        backend="openai",
        model="gpt-test",
        openai=OpenAIEndpointConfig(base_url="http://example.com/v1", max_tokens=20),
    )
    assert cfg.openai.base_url == "http://example.com/v1"
    assert cfg.openai.max_tokens == 20


def test_benchmark_config_openai_from_yaml(tmp_path: Path) -> None:
    from llm_inference_benchmark.config import load_config

    cfg_file = tmp_path / "openai.yaml"
    cfg_file.write_text(
        "backend: openai\n"
        "model: gpt-test\n"
        "requests: 5\n"
        "warmup_requests: 1\n"
        "openai:\n"
        "  base_url: http://localhost:8080/v1\n"
        "  max_tokens: 20\n"
        "  temperature: 0.0\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.backend == "openai"
    assert cfg.openai.base_url == "http://localhost:8080/v1"
    assert cfg.openai.max_tokens == 20


# ---------------------------------------------------------------------------
# CLI / _build_backend integration
# ---------------------------------------------------------------------------


def test_build_backend_dispatches_to_openai(tmp_path: Path, tmp_prompts: Path) -> None:
    from click.testing import CliRunner

    from llm_inference_benchmark.cli import main

    cfg_file = tmp_path / "openai.yaml"
    out_csv = tmp_path / "out.csv"
    cfg_file.write_text(
        f"backend: openai\n"
        f"model: gpt-test\n"
        f"requests: 2\n"
        f"warmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"openai:\n"
        f"  base_url: {_BASE_URL}\n"
        f"  max_tokens: 5\n"
    )

    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(cfg_file), "--output", str(out_csv)])

    assert result.exit_code == 0, result.output
    assert out_csv.exists()
    content = out_csv.read_text()
    assert "openai" in content


# ---------------------------------------------------------------------------
# Full run_benchmark integration (mocked)
# ---------------------------------------------------------------------------


def test_full_run_benchmark_with_mock(tmp_prompts: Path) -> None:
    from llm_inference_benchmark.config import BenchmarkConfig
    from llm_inference_benchmark.runner import load_prompts, run_benchmark

    mock_urlopen = _make_mock_urlopen(_make_chat_response(prompt_tokens=8, completion_tokens=12))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, max_tokens=12
        )
        cfg = BenchmarkConfig(
            backend="openai",
            model=_MODEL,
            requests=5,
            warmup_requests=1,
            prompts_file=str(tmp_prompts),
        )
        report = run_benchmark(backend, cfg, load_prompts(str(tmp_prompts)))

    assert report.request_count == 5
    assert report.backend == "openai"
    assert report.model == _MODEL
    assert report.p50_latency_ms > 0
    assert report.tokens_per_second > 0
    assert report.total_tokens == (8 + 12) * 5


# ---------------------------------------------------------------------------
# Integration test (requires a real running OpenAI-compatible server)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Streaming (stream=True) — SSE path
# ---------------------------------------------------------------------------


def _make_streaming_mock_urlopen(
    chunks: list[str],
    usage: dict | None = None,
) -> MagicMock:
    """Build a urlopen mock whose response yields SSE lines via readline()."""
    lines: list[bytes] = []
    for content in chunks:
        data: dict = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
        lines.append(f"data: {json.dumps(data)}\n".encode())
    if usage is not None:
        final: dict = {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage}
        lines.append(f"data: {json.dumps(final)}\n".encode())
    lines.append(b"data: [DONE]\n")
    lines.append(b"")  # EOF sentinel

    line_iter = iter(lines)
    response = MagicMock()
    response.readline.side_effect = lambda: next(line_iter, b"")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return MagicMock(return_value=response)


def test_streaming_returns_ttft() -> None:
    mock_urlopen = _make_streaming_mock_urlopen(["Hello", " world"])
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, stream=True
        )
        result = backend.generate("hi")
    assert result.ttft_ms is not None
    assert result.ttft_ms > 0


def test_streaming_assembles_full_text() -> None:
    mock_urlopen = _make_streaming_mock_urlopen(["Hello", ",", " world", "!"])
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, stream=True
        )
        result = backend.generate("hi")
    assert result.text == "Hello, world!"


def test_streaming_uses_usage_tokens_when_present() -> None:
    usage = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    mock_urlopen = _make_streaming_mock_urlopen(["one two three"], usage=usage)
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, stream=True
        )
        result = backend.generate("hello there")
    assert result.input_tokens == 4
    assert result.output_tokens == 6


def test_streaming_falls_back_to_word_count_when_no_usage() -> None:
    mock_urlopen = _make_streaming_mock_urlopen(["one two three four"])
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, stream=True
        )
        result = backend.generate("two words")
    assert result.input_tokens == 2
    assert result.output_tokens == 4


def test_streaming_sends_stream_true_in_payload() -> None:
    mock_urlopen = _make_streaming_mock_urlopen(["ok"])
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, stream=True
        )
        backend.generate("hi")
    request = mock_urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert payload.get("stream") is True


def test_non_streaming_does_not_send_stream_in_payload() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        backend.generate("hi")
    request = mock_urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert "stream" not in payload


def test_non_streaming_ttft_is_none() -> None:
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate("hi")
    assert result.ttft_ms is None


def test_streaming_ttft_is_none_when_no_content_chunks() -> None:
    mock_urlopen = _make_streaming_mock_urlopen([])
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, stream=True
        )
        result = backend.generate("hi")
    assert result.ttft_ms is None


def test_streaming_skips_empty_deltas_for_ttft() -> None:
    """First delta is empty string; TTFT must come from the second non-empty one."""
    mock_urlopen = _make_streaming_mock_urlopen(["", "actual content"])
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = openai_endpoint_mod.OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, stream=True
        )
        result = backend.generate("hi")
    assert result.ttft_ms is not None
    assert result.text == "actual content"


def test_streaming_default_is_off() -> None:
    backend = openai_endpoint_mod.OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
    assert backend._stream is False


def test_openai_endpoint_config_stream_defaults_false() -> None:
    from llm_inference_benchmark.config import OpenAIEndpointConfig

    cfg = OpenAIEndpointConfig()
    assert cfg.stream is False


def test_openai_endpoint_config_stream_can_be_enabled() -> None:
    from llm_inference_benchmark.config import OpenAIEndpointConfig

    cfg = OpenAIEndpointConfig(stream=True)
    assert cfg.stream is True


# ---------------------------------------------------------------------------
# Integration test (requires a real running OpenAI-compatible server)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_real_endpoint() -> None:
    """Real HTTP call — set OPENAI_ENDPOINT_BASE_URL to a running server's base URL."""
    base_url = os.environ.get("OPENAI_ENDPOINT_BASE_URL")
    if not base_url:
        pytest.skip("OPENAI_ENDPOINT_BASE_URL env var not set — skipping live endpoint test")

    model = os.environ.get("OPENAI_ENDPOINT_MODEL", "default")
    backend = openai_endpoint_mod.OpenAIEndpointBackend(
        base_url=base_url, model=model, max_tokens=10
    )
    result = backend.generate("The capital of France is")
    assert result.latency_ms > 0
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert isinstance(result.text, str)
