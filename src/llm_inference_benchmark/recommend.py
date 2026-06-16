"""Constraint-based configuration recommender.

Filters benchmark CSV runs against explicit user constraints, then selects
the best candidate using Pareto dominance and lowest p95 latency as a
tiebreaker.

Constraint semantics:
  - max_p95_ms    : exclude runs where p95_latency_ms > threshold
  - max_vram_mb   : exclude runs where peak_vram_memory_mb > threshold;
                    also excludes runs with a missing VRAM reading when
                    the constraint is active
  - min_sanity    : exclude runs where sanity_pass_rate < threshold;
                    also excludes runs with a missing sanity reading when
                    the constraint is active
  - min_quality   : exclude runs where task_quality_pass_rate < threshold;
                    also excludes runs with a missing task quality reading
                    when the constraint is active
  - max_perplexity: exclude runs where perplexity > threshold;
                    also excludes runs with a missing perplexity reading
                    when the constraint is active
  - min_judge     : exclude runs where judge_score < threshold;
                    also excludes runs with a missing judge_score reading
                    when the constraint is active
  - No constraint set for an optional metric → missing value is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_inference_benchmark.compare import RunRow, load_csv
from llm_inference_benchmark.pareto import pareto_classify


@dataclass(frozen=True)
class Constraints:
    max_vram_mb: float | None = None
    max_p95_ms: float | None = None
    min_sanity: float | None = None
    min_quality: float | None = None
    max_perplexity: float | None = None
    min_judge: float | None = None
    max_load_ms: float | None = None


@dataclass(frozen=True)
class Exclusion:
    row: RunRow
    reason: str


@dataclass(frozen=True)
class RecommendationResult:
    winner: RunRow | None
    is_pareto_optimal: bool
    candidates: list[RunRow]
    excluded: list[Exclusion]


def _check_row(row: RunRow, constraints: Constraints) -> str | None:
    """Return an exclusion reason, or None if the row satisfies all constraints."""
    if constraints.max_p95_ms is not None and row.p95_latency_ms > constraints.max_p95_ms:
        return (
            f"p95 latency too high ({row.p95_latency_ms:.1f} ms > {constraints.max_p95_ms:.1f} ms)"
        )

    if constraints.max_vram_mb is not None:
        if row.peak_vram_memory_mb is None:
            return f"VRAM unknown (constraint requires ≤ {constraints.max_vram_mb:.0f} MB)"
        if row.peak_vram_memory_mb > constraints.max_vram_mb:
            return (
                f"VRAM too high "
                f"({row.peak_vram_memory_mb:.1f} MB > {constraints.max_vram_mb:.1f} MB)"
            )

    if constraints.min_sanity is not None:
        if row.sanity_pass_rate is None:
            return f"sanity unknown (constraint requires ≥ {constraints.min_sanity:.2f})"
        if row.sanity_pass_rate < constraints.min_sanity:
            return f"sanity too low ({row.sanity_pass_rate:.2f} < {constraints.min_sanity:.2f})"

    if constraints.min_quality is not None:
        if row.task_quality_pass_rate is None:
            return f"task quality unknown (constraint requires ≥ {constraints.min_quality:.2f})"
        if row.task_quality_pass_rate < constraints.min_quality:
            return (
                f"task quality too low "
                f"({row.task_quality_pass_rate:.2f} < {constraints.min_quality:.2f})"
            )

    if constraints.max_perplexity is not None:
        if row.perplexity is None:
            return f"perplexity unknown (constraint requires ≤ {constraints.max_perplexity:.2f})"
        if row.perplexity > constraints.max_perplexity:
            return f"perplexity too high ({row.perplexity:.2f} > {constraints.max_perplexity:.2f})"

    if constraints.min_judge is not None:
        if row.judge_score is None:
            return f"judge score unknown (constraint requires ≥ {constraints.min_judge:.2f})"
        if row.judge_score < constraints.min_judge:
            return f"judge score too low ({row.judge_score:.2f} < {constraints.min_judge:.2f})"

    if constraints.max_load_ms is not None:
        if row.model_load_ms is None:
            return f"load time unknown (constraint requires ≤ {constraints.max_load_ms:.0f} ms)"
        if row.model_load_ms > constraints.max_load_ms:
            return (
                f"load time too high "
                f"({row.model_load_ms:.1f} ms > {constraints.max_load_ms:.1f} ms)"
            )

    return None


def apply_constraints(
    rows: list[RunRow], constraints: Constraints
) -> tuple[list[RunRow], list[Exclusion]]:
    """Partition rows into (candidates_passing_all_constraints, exclusions)."""
    candidates: list[RunRow] = []
    excluded: list[Exclusion] = []
    for row in rows:
        reason = _check_row(row, constraints)
        if reason is None:
            candidates.append(row)
        else:
            excluded.append(Exclusion(row=row, reason=reason))
    return candidates, excluded


def _lowest_p95(rows: list[RunRow]) -> RunRow:
    return min(rows, key=lambda r: r.p95_latency_ms)


def recommend(rows: list[RunRow], constraints: Constraints) -> RecommendationResult:
    """Apply constraints, then select the best candidate."""
    if not rows:
        return RecommendationResult(
            winner=None, is_pareto_optimal=False, candidates=[], excluded=[]
        )

    candidates, excluded = apply_constraints(rows, constraints)
    if not candidates:
        return RecommendationResult(
            winner=None, is_pareto_optimal=False, candidates=[], excluded=excluded
        )

    classified = pareto_classify(candidates)
    optimal = [r for r, is_opt in classified if is_opt]

    if optimal:
        winner = _lowest_p95(optimal)
        is_pareto_optimal = True
    else:
        # All candidates are mutually non-dominating; pick the fastest.
        winner = _lowest_p95(candidates)
        is_pareto_optimal = False

    return RecommendationResult(
        winner=winner,
        is_pareto_optimal=is_pareto_optimal,
        candidates=candidates,
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_DIVIDER = "─" * 42


def _fmt_vram(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.1f} MB"


def _fmt_ms(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.1f} ms"


def _fmt_rate(v: float | None) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _fmt_ppl(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def _fmt_judge(v: float | None) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _display_model(model: str) -> str:
    """Return a short display name: filename for absolute paths, full string otherwise."""
    if "/" in model:
        return Path(model).name
    return model


def render_recommendation(result: RecommendationResult) -> str:
    lines: list[str] = []

    if result.winner is not None:
        w = result.winner
        lines += [
            "Recommendation",
            _DIVIDER,
            f"  Backend  : {w.backend}",
            f"  Model    : {_display_model(w.model)}",
            f"  N        : {w.request_count}",
            f"  p95      : {w.p95_latency_ms:.2f} ms",
            f"  tok/s    : {w.tokens_per_second:.1f}",
            f"  Load     : {_fmt_ms(w.model_load_ms)}",
            f"  VRAM     : {_fmt_vram(w.peak_vram_memory_mb)}",
            f"  Sanity   : {_fmt_rate(w.sanity_pass_rate)}",
            f"  Task Q   : {_fmt_rate(w.task_quality_pass_rate)}",
            f"  PPL      : {_fmt_ppl(w.perplexity)}",
            f"  Judge    : {_fmt_judge(w.judge_score)}",
            "",
        ]
        n = len(result.candidates)
        pareto_note = "; Pareto-optimal" if result.is_pareto_optimal else ""
        lines.append(
            f"Why: lowest p95 among {n} candidate(s) passing all constraints{pareto_note}."
        )
    else:
        lines.append("No recommendation: no runs satisfy all constraints.")

    if result.excluded:
        lines += [
            "",
            f"Excluded ({len(result.excluded)})",
            _DIVIDER,
        ]
        for ex in result.excluded:
            lines.append(f"  {ex.row.backend}  {_display_model(ex.row.model)}  →  {ex.reason}")

    return "\n".join(lines)


def build_recommendation(paths: list[str | Path], constraints: Constraints) -> tuple[str, bool]:
    """Load CSVs, apply constraints, render recommendation. Returns (text, has_winner)."""
    if not paths:
        raise ValueError("At least one CSV path is required")
    rows = [load_csv(p) for p in paths]
    result = recommend(rows, constraints)
    return render_recommendation(result), result.winner is not None
