"""Tests for LLM-as-judge scoring (v0.21)."""

from __future__ import annotations

import math

import pytest

from llm_inference_benchmark.backends.base import Backend, GenerationResult
from llm_inference_benchmark.judge import (
    judge_score_from_probabilities,
    probability_from_yes_no_logits,
)

# ---------------------------------------------------------------------------
# probability_from_yes_no_logits() — pure math, hand-checkable
# ---------------------------------------------------------------------------


def test_probability_from_yes_no_logits_equal_logits_is_half() -> None:
    p = probability_from_yes_no_logits(yes_logit=1.5, no_logit=1.5)
    assert p == pytest.approx(0.5)


def test_probability_from_yes_no_logits_higher_yes_is_higher_probability() -> None:
    low = probability_from_yes_no_logits(yes_logit=0.0, no_logit=2.0)
    high = probability_from_yes_no_logits(yes_logit=2.0, no_logit=0.0)
    assert high > low


def test_probability_from_yes_no_logits_extreme_difference_is_near_bounds() -> None:
    near_one = probability_from_yes_no_logits(yes_logit=50.0, no_logit=0.0)
    near_zero = probability_from_yes_no_logits(yes_logit=0.0, no_logit=50.0)
    assert near_one == pytest.approx(1.0, abs=1e-6)
    assert near_zero == pytest.approx(0.0, abs=1e-6)


def test_probability_from_yes_no_logits_matches_sigmoid_of_difference() -> None:
    p = probability_from_yes_no_logits(yes_logit=3.0, no_logit=1.0)
    expected = 1.0 / (1.0 + math.exp(-(3.0 - 1.0)))
    assert p == pytest.approx(expected)


# ---------------------------------------------------------------------------
# judge_score_from_probabilities() — pure math, hand-checkable
# ---------------------------------------------------------------------------


def test_judge_score_from_probabilities_returns_mean() -> None:
    score = judge_score_from_probabilities([0.2, 0.4, 0.6, 0.8])
    assert score == pytest.approx(0.5)


def test_judge_score_from_probabilities_single_value() -> None:
    score = judge_score_from_probabilities([0.75])
    assert score == pytest.approx(0.75)


def test_judge_score_from_probabilities_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="probabilities must be non-empty"):
        judge_score_from_probabilities([])


# ---------------------------------------------------------------------------
# Backend default — backends without logit access return None
# ---------------------------------------------------------------------------


class _DummyBackend(Backend):
    @property
    def name(self) -> str:
        return "dummy"

    def generate(self, prompt: str) -> GenerationResult:
        return GenerationResult(text="x", input_tokens=1, output_tokens=1, latency_ms=1.0)


def test_default_backend_compute_judge_score_returns_none() -> None:
    backend = _DummyBackend()
    assert backend.compute_judge_score(["prompt"], ["hello world"]) is None


def test_mock_backend_compute_judge_score_returns_none() -> None:
    from llm_inference_benchmark.backends.mock import MockBackend

    backend = MockBackend(model="mock-gpt2")
    assert backend.compute_judge_score(["prompt"], ["hello world"]) is None
