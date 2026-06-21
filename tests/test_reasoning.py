"""Tests for the reasoning token parser."""

from __future__ import annotations

import pytest

from llm_inference_benchmark.reasoning import parse_reasoning, reasoning_stats  # noqa: E402

# ── parse_reasoning ────────────────────────────────────────────────────────────


def test_no_start_tag_returns_empty_reasoning_and_full_answer() -> None:
    text = "Hello, world!"
    r, a = parse_reasoning(text, "<think>", "</think>")
    assert r == ""
    assert a == text


def test_basic_split() -> None:
    text = "<think>This is my thinking.</think>Here is my answer."
    r, a = parse_reasoning(text, "<think>", "</think>")
    assert r == "This is my thinking."
    assert a == "Here is my answer."


def test_missing_close_tag_rest_is_reasoning() -> None:
    text = "<think>Still thinking..."
    r, a = parse_reasoning(text, "<think>", "</think>")
    assert r == "Still thinking..."
    assert a == ""


def test_answer_text_before_think_tag() -> None:
    text = "Preamble. <think>Thinking here.</think>Answer."
    r, a = parse_reasoning(text, "<think>", "</think>")
    assert r == "Thinking here."
    assert a == "Preamble. Answer."


def test_multiple_think_blocks() -> None:
    text = "<think>Step 1.</think>Middle.<think>Step 2.</think>End."
    r, a = parse_reasoning(text, "<think>", "</think>")
    assert r == "Step 1.Step 2."
    assert a == "Middle.End."


def test_empty_think_block() -> None:
    text = "<think></think>Answer only."
    r, a = parse_reasoning(text, "<think>", "</think>")
    assert r == ""
    assert a == "Answer only."


def test_empty_input_text() -> None:
    r, a = parse_reasoning("", "<think>", "</think>")
    assert r == ""
    assert a == ""


def test_only_think_block_no_answer() -> None:
    text = "<think>Pure reasoning.</think>"
    r, a = parse_reasoning(text, "<think>", "</think>")
    assert r == "Pure reasoning."
    assert a == ""


def test_custom_tags() -> None:
    text = "[THINK]Reasoning here.[/THINK]Final answer."
    r, a = parse_reasoning(text, "[THINK]", "[/THINK]")
    assert r == "Reasoning here."
    assert a == "Final answer."


def test_empty_start_tag_returns_empty_reasoning() -> None:
    text = "Hello."
    r, a = parse_reasoning(text, "", "</think>")
    assert r == ""
    assert a == text


def test_whitespace_only_think_block() -> None:
    text = "<think>   </think>Answer."
    r, a = parse_reasoning(text, "<think>", "</think>")
    assert r == "   "
    assert a == "Answer."


def test_nested_tag_text_not_treated_as_nesting() -> None:
    # Tags are plain strings, not XML — inner occurrences of the start tag are
    # just content until the first close tag.
    text = "<think>outer <think>inner</think>after</think>done"
    r, a = parse_reasoning(text, "<think>", "</think>")
    # First </think> closes the first <think>; "outer <think>inner" is reasoning.
    assert r == "outer <think>inner"
    assert a == "after</think>done"


# ── reasoning_stats ────────────────────────────────────────────────────────────


def test_reasoning_stats_basic() -> None:
    texts = ["<think>AAAA</think>BBBB"]
    output_tokens = [8]
    mean_r, mean_a, frac = reasoning_stats(texts, output_tokens, "<think>", "</think>")
    # reasoning="AAAA" (4 chars), answer="BBBB" (4 chars) → content fraction = 4/8 = 0.5
    assert frac == pytest.approx(0.5)
    assert mean_r == pytest.approx(4.0)
    assert mean_a == pytest.approx(4.0)


def test_reasoning_stats_no_think_block() -> None:
    texts = ["No thinking here."]
    output_tokens = [10]
    mean_r, mean_a, frac = reasoning_stats(texts, output_tokens, "<think>", "</think>")
    assert frac == pytest.approx(0.0)
    assert mean_r == pytest.approx(0.0)
    assert mean_a == pytest.approx(10.0)


def test_reasoning_stats_full_reasoning() -> None:
    # reasoning="All reasoning." (14 chars), answer="" (0 chars) → fraction = 14/14 = 1.0
    texts = ["<think>All reasoning.</think>"]
    output_tokens = [10]
    mean_r, mean_a, frac = reasoning_stats(texts, output_tokens, "<think>", "</think>")
    assert frac == pytest.approx(1.0)
    assert mean_r == pytest.approx(10.0)
    assert mean_a == pytest.approx(0.0)


def test_reasoning_stats_multiple_requests() -> None:
    # First text: reasoning="AB"(2), answer="AB"(2) → frac=0.5; 10 tok → 5 r, 5 a
    # Second text: reasoning=""(0), answer="No thinking."(12) → frac=0.0; 20 tok → 0 r, 20 a
    texts = ["<think>AB</think>AB", "No thinking."]
    output_tokens = [10, 20]
    mean_r, mean_a, frac = reasoning_stats(texts, output_tokens, "<think>", "</think>")
    assert frac == pytest.approx(0.25)  # mean of 0.5 and 0.0
    assert mean_r == pytest.approx(2.5)  # mean of 5 and 0
    assert mean_a == pytest.approx(12.5)  # mean of 5 and 20


def test_reasoning_stats_empty_inputs() -> None:
    mean_r, mean_a, frac = reasoning_stats([], [], "<think>", "</think>")
    assert mean_r == 0.0
    assert mean_a == 0.0
    assert frac == 0.0


def test_reasoning_stats_mismatched_lengths() -> None:
    mean_r, mean_a, frac = reasoning_stats(["text"], [1, 2], "<think>", "</think>")
    assert mean_r == 0.0
    assert mean_a == 0.0
    assert frac == 0.0


def test_reasoning_stats_zero_output_tokens_text_present() -> None:
    # reasoning="ABC"(3), answer="DEF"(3) → content fraction = 3/6 = 0.5; tokens = 0
    texts = ["<think>ABC</think>DEF"]
    output_tokens = [0]
    mean_r, mean_a, frac = reasoning_stats(texts, output_tokens, "<think>", "</think>")
    assert frac == pytest.approx(0.5)
    assert mean_r == pytest.approx(0.0)
    assert mean_a == pytest.approx(0.0)
