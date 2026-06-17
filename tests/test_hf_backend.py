"""Integration tests for the HuggingFace Transformers backend.

Requires: uv sync --extra transformers
First run downloads sshleifer/tiny-gpt2 (~4 MB) from HuggingFace Hub.
Tests are skipped automatically when transformers is not installed.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_TRANSFORMERS = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)

pytestmark = pytest.mark.integration

skip_without_transformers = pytest.mark.skipif(
    not _HAS_TRANSFORMERS,
    reason="transformers extra not installed — run: uv sync --extra transformers",
)

_TINY_MODEL = "sshleifer/tiny-gpt2"
_MAX_NEW = 10  # keep tests fast on CPU


@pytest.fixture(scope="module")
def hf_backend():  # type: ignore[no-untyped-def]
    if not _HAS_TRANSFORMERS:
        pytest.skip("transformers extra not installed — run: uv sync --extra transformers")
    from llm_inference_benchmark.backends.hf import HFBackend

    return HFBackend(
        model_id=_TINY_MODEL,
        max_new_tokens=_MAX_NEW,
        device="cpu",
        torch_dtype="float32",
        do_sample=False,
    )


@skip_without_transformers
def test_backend_name(hf_backend) -> None:  # type: ignore[no-untyped-def]
    assert hf_backend.name == "transformers"


@skip_without_transformers
def test_generate_returns_positive_latency(hf_backend) -> None:  # type: ignore[no-untyped-def]
    result = hf_backend.generate("The capital of France is")
    assert result.latency_ms > 0


@skip_without_transformers
def test_generate_returns_ttft_ms(hf_backend) -> None:  # type: ignore[no-untyped-def]
    result = hf_backend.generate("The capital of France is")
    assert result.ttft_ms is not None
    assert result.ttft_ms > 0
    assert result.ttft_ms <= result.latency_ms


@skip_without_transformers
def test_generate_token_counts(hf_backend) -> None:  # type: ignore[no-untyped-def]
    result = hf_backend.generate("Hello world")
    assert result.input_tokens > 0
    assert 0 < result.output_tokens <= _MAX_NEW


@skip_without_transformers
def test_generate_text_is_string(hf_backend) -> None:  # type: ignore[no-untyped-def]
    result = hf_backend.generate("Once upon a time")
    assert isinstance(result.text, str)


@skip_without_transformers
def test_generate_is_deterministic(hf_backend) -> None:  # type: ignore[no-untyped-def]
    """Greedy decoding (do_sample=False) produces identical output each run."""
    r1 = hf_backend.generate("The sky is")
    r2 = hf_backend.generate("The sky is")
    assert r1.output_tokens == r2.output_tokens
    assert r1.text == r2.text


@skip_without_transformers
def test_compute_perplexity_returns_positive_float(hf_backend) -> None:  # type: ignore[no-untyped-def]
    ppl = hf_backend.compute_perplexity(["The capital of France is Paris."])
    assert ppl is not None
    assert ppl > 0


@skip_without_transformers
def test_compute_perplexity_none_for_empty_texts(hf_backend) -> None:  # type: ignore[no-untyped-def]
    assert hf_backend.compute_perplexity([]) is None


@skip_without_transformers
def test_compute_perplexity_skips_single_token_texts(hf_backend) -> None:  # type: ignore[no-untyped-def]
    """A text that tokenizes to <2 tokens has no next-token target to score."""
    assert hf_backend.compute_perplexity([""]) is None


@skip_without_transformers
def test_compute_judge_score_returns_value_in_unit_interval(hf_backend) -> None:  # type: ignore[no-untyped-def]
    score = hf_backend.compute_judge_score(
        ["What is the capital of France?"], ["The capital of France is Paris."]
    )
    assert score is not None
    assert 0.0 <= score <= 1.0


@skip_without_transformers
def test_compute_judge_score_none_for_empty_lists(hf_backend) -> None:  # type: ignore[no-untyped-def]
    assert hf_backend.compute_judge_score([], []) is None


@skip_without_transformers
def test_run_benchmark_with_hf_backend(tmp_prompts: pytest.FixtureRequest) -> None:
    from llm_inference_benchmark.backends.hf import HFBackend
    from llm_inference_benchmark.config import BenchmarkConfig
    from llm_inference_benchmark.runner import load_prompts, run_benchmark

    backend = HFBackend(model_id=_TINY_MODEL, max_new_tokens=5, device="cpu")
    cfg = BenchmarkConfig(
        backend="transformers",
        model=_TINY_MODEL,
        requests=3,
        warmup_requests=1,
        prompts_file=str(tmp_prompts),
    )
    report = run_benchmark(backend, cfg, load_prompts(str(tmp_prompts)))

    assert report.request_count == 3
    assert report.backend == "transformers"
    assert report.model == _TINY_MODEL
    assert report.p50_latency_ms > 0
    assert report.tokens_per_second > 0
