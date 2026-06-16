"""Tests for perplexity computation (v0.20)."""

from __future__ import annotations

import math

import pytest

from llm_inference_benchmark.backends.base import Backend, GenerationResult
from llm_inference_benchmark.perplexity import perplexity_from_nll

# ---------------------------------------------------------------------------
# perplexity_from_nll() — pure math, hand-checkable
# ---------------------------------------------------------------------------


def test_perplexity_from_nll_known_value() -> None:
    # total_nll = log(2) over 2 tokens → mean_nll = log(2)/2 → ppl = sqrt(2)
    ppl = perplexity_from_nll(total_nll=math.log(2), total_tokens=2)
    assert ppl == pytest.approx(math.sqrt(2))


def test_perplexity_from_nll_zero_nll_is_one() -> None:
    # Perfectly confident predictions (log-prob 0 for every token) → ppl == 1.0
    ppl = perplexity_from_nll(total_nll=0.0, total_tokens=5)
    assert ppl == pytest.approx(1.0)


def test_perplexity_from_nll_higher_nll_is_higher_perplexity() -> None:
    low = perplexity_from_nll(total_nll=1.0, total_tokens=10)
    high = perplexity_from_nll(total_nll=5.0, total_tokens=10)
    assert high > low


def test_perplexity_from_nll_raises_on_zero_tokens() -> None:
    with pytest.raises(ValueError, match="total_tokens must be positive"):
        perplexity_from_nll(total_nll=1.0, total_tokens=0)


def test_perplexity_from_nll_raises_on_negative_tokens() -> None:
    with pytest.raises(ValueError, match="total_tokens must be positive"):
        perplexity_from_nll(total_nll=1.0, total_tokens=-3)


# ---------------------------------------------------------------------------
# Backend default — backends without logit access return None
# ---------------------------------------------------------------------------


class _DummyBackend(Backend):
    @property
    def name(self) -> str:
        return "dummy"

    def generate(self, prompt: str) -> GenerationResult:
        return GenerationResult(text="x", input_tokens=1, output_tokens=1, latency_ms=1.0)


def test_default_backend_compute_perplexity_returns_none() -> None:
    backend = _DummyBackend()
    assert backend.compute_perplexity(["hello world"]) is None


def test_mock_backend_compute_perplexity_returns_none() -> None:
    from llm_inference_benchmark.backends.mock import MockBackend

    backend = MockBackend(model="mock-gpt2")
    assert backend.compute_perplexity(["hello world"]) is None
