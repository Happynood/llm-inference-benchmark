"""Lightweight output sanity checks for benchmark runs.

These metrics detect structural problems (empty or duplicate completions)
and are not a measure of semantic quality or task accuracy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityReport:
    """Per-run aggregate output sanity metrics.

    All metrics are computed over the benchmark requests (warmup excluded).
    Char counts use stripped text; whitespace-only outputs count as empty.
    """

    empty_output_count: int
    """Number of completions whose stripped text is empty."""

    min_output_chars: int
    """Minimum stripped character count across all completions."""

    mean_output_chars: float
    """Mean stripped character count across all completions."""

    repeated_output_count: int
    """Number of completions whose text is identical to at least one other completion
    in this run. With temperature=0.0 and cycling prompts, some repetition is
    mathematically expected; use this as a degeneration signal, not an absolute check.
    """

    sanity_pass_rate: float
    """Fraction of completions that are non-empty, in [0.0, 1.0].
    1.0 means every output contained at least one non-whitespace character.
    Does not penalise repetition — a repeated but non-empty output is considered passing.
    """


def compute_quality(texts: list[str]) -> QualityReport:
    """Compute per-run output sanity metrics from completion texts.

    Args:
        texts: Raw completion strings from the benchmark loop (warmup excluded).

    Returns:
        QualityReport with aggregate sanity counts and rates.
    """
    if not texts:
        return QualityReport(
            empty_output_count=0,
            min_output_chars=0,
            mean_output_chars=0.0,
            repeated_output_count=0,
            sanity_pass_rate=1.0,
        )

    stripped = [t.strip() for t in texts]
    char_counts = [len(s) for s in stripped]
    empty_count = sum(1 for s in stripped if not s)

    counts = Counter(stripped)
    repeated_count = sum(1 for s in stripped if counts[s] > 1)

    return QualityReport(
        empty_output_count=empty_count,
        min_output_chars=min(char_counts),
        mean_output_chars=sum(char_counts) / len(char_counts),
        repeated_output_count=repeated_count,
        sanity_pass_rate=(len(texts) - empty_count) / len(texts),
    )
