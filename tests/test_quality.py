"""Tests for the output sanity / quality module."""

from __future__ import annotations

import pytest

from llm_inference_benchmark.quality import QualityReport, compute_quality  # noqa: E402

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_list_returns_defaults() -> None:
    report = compute_quality([])
    assert report.empty_output_count == 0
    assert report.min_output_chars == 0
    assert report.mean_output_chars == 0.0
    assert report.repeated_output_count == 0
    assert report.sanity_pass_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Normal outputs
# ---------------------------------------------------------------------------


def test_all_normal_outputs() -> None:
    texts = ["The sky is blue.", "Paris is in France.", "Water boils at 100°C."]
    report = compute_quality(texts)
    assert report.empty_output_count == 0
    assert report.min_output_chars == len("The sky is blue.")
    assert report.mean_output_chars == pytest.approx(
        sum(len(t) for t in texts) / len(texts), rel=1e-6
    )
    assert report.repeated_output_count == 0
    assert report.sanity_pass_rate == pytest.approx(1.0)


def test_single_normal_output() -> None:
    report = compute_quality(["hello world"])
    assert report.empty_output_count == 0
    assert report.min_output_chars == 11
    assert report.mean_output_chars == pytest.approx(11.0)
    assert report.repeated_output_count == 0
    assert report.sanity_pass_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Empty outputs
# ---------------------------------------------------------------------------


def test_single_empty_output() -> None:
    report = compute_quality([""])
    assert report.empty_output_count == 1
    assert report.min_output_chars == 0
    assert report.sanity_pass_rate == pytest.approx(0.0)


def test_whitespace_only_counts_as_empty() -> None:
    report = compute_quality(["   ", "\t\n", ""])
    assert report.empty_output_count == 3
    assert report.min_output_chars == 0
    assert report.sanity_pass_rate == pytest.approx(0.0)


def test_mixed_empty_and_normal() -> None:
    # 2 normal, 1 empty out of 3
    report = compute_quality(["hello", "", "world"])
    assert report.empty_output_count == 1
    assert report.min_output_chars == 0
    assert report.mean_output_chars == pytest.approx((5 + 0 + 5) / 3)
    assert report.sanity_pass_rate == pytest.approx(2 / 3)


def test_all_empty_outputs() -> None:
    report = compute_quality(["", "", ""])
    assert report.empty_output_count == 3
    assert report.min_output_chars == 0
    assert report.mean_output_chars == pytest.approx(0.0)
    assert report.sanity_pass_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Repeated outputs
# ---------------------------------------------------------------------------


def test_all_same_outputs_fully_repeated() -> None:
    texts = ["same text"] * 5
    report = compute_quality(texts)
    # All 5 are duplicates of each other
    assert report.repeated_output_count == 5
    assert report.empty_output_count == 0
    assert report.sanity_pass_rate == pytest.approx(1.0)  # non-empty, but repeated


def test_no_repeats_repeated_count_zero() -> None:
    texts = ["alpha", "beta", "gamma", "delta"]
    report = compute_quality(texts)
    assert report.repeated_output_count == 0


def test_partial_repeats() -> None:
    # "a" appears 3 times, "b" appears once → 3 repeated
    texts = ["a", "b", "a", "a"]
    report = compute_quality(texts)
    assert report.repeated_output_count == 3


def test_two_groups_repeated() -> None:
    # "x" ×2, "y" ×2 → all 4 are repeated
    texts = ["x", "y", "x", "y"]
    report = compute_quality(texts)
    assert report.repeated_output_count == 4


# ---------------------------------------------------------------------------
# Sanity pass rate
# ---------------------------------------------------------------------------


def test_sanity_pass_rate_is_fraction_non_empty() -> None:
    # 3 non-empty, 2 empty out of 5
    texts = ["ok", "", "fine", "", "good"]
    report = compute_quality(texts)
    assert report.sanity_pass_rate == pytest.approx(3 / 5)


def test_sanity_pass_rate_not_affected_by_repeats() -> None:
    # Repeated outputs count as passing (non-empty)
    texts = ["same"] * 10
    report = compute_quality(texts)
    assert report.sanity_pass_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# char metrics strip whitespace
# ---------------------------------------------------------------------------


def test_char_counts_use_stripped_text() -> None:
    # "  hi  " → stripped = "hi" → 2 chars
    report = compute_quality(["  hi  ", "world"])
    assert report.min_output_chars == 2
    assert report.mean_output_chars == pytest.approx((2 + 5) / 2)


def test_min_output_chars_includes_empty() -> None:
    report = compute_quality(["hello", "", "world"])
    assert report.min_output_chars == 0


# ---------------------------------------------------------------------------
# Type check: return type is QualityReport
# ---------------------------------------------------------------------------


def test_returns_quality_report_instance() -> None:
    result = compute_quality(["test"])
    assert isinstance(result, QualityReport)
