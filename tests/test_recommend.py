"""Tests for the constraint-based recommender (v0.15)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.compare import RunRow
from llm_inference_benchmark.recommend import (
    Constraints,
    Exclusion,
    RecommendationResult,
    apply_constraints,
    build_recommendation,
    recommend,
    render_recommendation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    backend: str = "mock",
    model: str = "model-a",
    p95: float = 100.0,
    toks: float = 50.0,
    vram: float | None = 2000.0,
    sanity: float | None = 1.0,
    cpu: float = 500.0,
    ppl: float | None = None,
) -> RunRow:
    return RunRow(
        backend=backend,
        model=model,
        request_count=10,
        p50_latency_ms=p95 * 0.95,
        p95_latency_ms=p95,
        tokens_per_second=toks,
        peak_cpu_memory_mb=cpu,
        peak_cuda_memory_mb=None,
        peak_vram_memory_mb=vram,
        sanity_pass_rate=sanity,
        perplexity=ppl,
    )


_CSV_FIELDNAMES = [
    "request_count",
    "p50_latency_ms",
    "p95_latency_ms",
    "tokens_per_second",
    "total_tokens",
    "backend",
    "model",
    "peak_cpu_memory_mb",
    "peak_cuda_memory_mb",
    "peak_vram_memory_mb",
    "empty_output_count",
    "min_output_chars",
    "mean_output_chars",
    "repeated_output_count",
    "sanity_pass_rate",
    "perplexity",
    "timestamp",
]


def _write_csv(path: Path, row: RunRow) -> Path:
    def _s(v: float | None) -> str:
        return "" if v is None else str(v)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "request_count": row.request_count,
                "p50_latency_ms": row.p50_latency_ms,
                "p95_latency_ms": row.p95_latency_ms,
                "tokens_per_second": row.tokens_per_second,
                "total_tokens": 0,
                "backend": row.backend,
                "model": row.model,
                "peak_cpu_memory_mb": row.peak_cpu_memory_mb,
                "peak_cuda_memory_mb": _s(row.peak_cuda_memory_mb),
                "peak_vram_memory_mb": _s(row.peak_vram_memory_mb),
                "empty_output_count": 0,
                "min_output_chars": 50,
                "mean_output_chars": 50.0,
                "repeated_output_count": 0,
                "sanity_pass_rate": _s(row.sanity_pass_rate),
                "perplexity": _s(row.perplexity),
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        )
    return path


# ---------------------------------------------------------------------------
# apply_constraints()
# ---------------------------------------------------------------------------


def test_apply_constraints_no_constraints_all_pass() -> None:
    rows = [_row(p95=100.0), _row(p95=200.0)]
    candidates, excluded = apply_constraints(rows, Constraints())
    assert len(candidates) == 2
    assert excluded == []


def test_apply_constraints_max_p95_filters() -> None:
    rows = [_row(model="fast", p95=80.0), _row(model="slow", p95=200.0)]
    candidates, excluded = apply_constraints(rows, Constraints(max_p95_ms=100.0))
    assert len(candidates) == 1
    assert candidates[0].model == "fast"
    assert len(excluded) == 1
    assert "p95 latency too high" in excluded[0].reason
    assert "200.0 ms" in excluded[0].reason


def test_apply_constraints_max_vram_filters() -> None:
    rows = [_row(model="small", vram=2000.0), _row(model="big", vram=5000.0)]
    candidates, excluded = apply_constraints(rows, Constraints(max_vram_mb=4000.0))
    assert len(candidates) == 1
    assert candidates[0].model == "small"
    assert "VRAM too high" in excluded[0].reason


def test_apply_constraints_min_sanity_filters() -> None:
    rows = [_row(model="ok", sanity=1.0), _row(model="bad", sanity=0.5)]
    candidates, excluded = apply_constraints(rows, Constraints(min_sanity=0.9))
    assert len(candidates) == 1
    assert candidates[0].model == "ok"
    assert "sanity too low" in excluded[0].reason


def test_apply_constraints_missing_vram_excluded_when_constrained() -> None:
    row = _row(vram=None)
    candidates, excluded = apply_constraints([row], Constraints(max_vram_mb=4000.0))
    assert candidates == []
    assert "VRAM unknown" in excluded[0].reason


def test_apply_constraints_missing_vram_ok_when_unconstrained() -> None:
    row = _row(vram=None)
    candidates, excluded = apply_constraints([row], Constraints())
    assert len(candidates) == 1
    assert excluded == []


def test_apply_constraints_missing_sanity_excluded_when_constrained() -> None:
    row = _row(sanity=None)
    candidates, excluded = apply_constraints([row], Constraints(min_sanity=1.0))
    assert candidates == []
    assert "sanity unknown" in excluded[0].reason


def test_apply_constraints_missing_sanity_ok_when_unconstrained() -> None:
    row = _row(sanity=None)
    candidates, excluded = apply_constraints([row], Constraints())
    assert len(candidates) == 1
    assert excluded == []


def test_apply_constraints_max_perplexity_filters() -> None:
    rows = [_row(model="fluent", ppl=8.0), _row(model="degraded", ppl=20.0)]
    candidates, excluded = apply_constraints(rows, Constraints(max_perplexity=10.0))
    assert len(candidates) == 1
    assert candidates[0].model == "fluent"
    assert "perplexity too high" in excluded[0].reason


def test_apply_constraints_missing_perplexity_excluded_when_constrained() -> None:
    row = _row(ppl=None)
    candidates, excluded = apply_constraints([row], Constraints(max_perplexity=10.0))
    assert candidates == []
    assert "perplexity unknown" in excluded[0].reason


def test_apply_constraints_missing_perplexity_ok_when_unconstrained() -> None:
    row = _row(ppl=None)
    candidates, excluded = apply_constraints([row], Constraints())
    assert len(candidates) == 1
    assert excluded == []


# ---------------------------------------------------------------------------
# recommend()
# ---------------------------------------------------------------------------


def test_recommend_one_clear_winner() -> None:
    rows = [
        _row(model="fast", p95=100.0, toks=60.0, vram=2000.0),
        _row(model="slow", p95=500.0, toks=30.0, vram=2000.0),
    ]
    result = recommend(rows, Constraints(max_p95_ms=300.0))
    assert result.winner is not None
    assert result.winner.model == "fast"
    assert len(result.candidates) == 1
    assert len(result.excluded) == 1


def test_recommend_prefers_pareto_optimal_over_dominated() -> None:
    # A dominates B: lower p95, higher tok/s, same sanity.
    a = _row(model="optimal", p95=100.0, toks=80.0, sanity=1.0)
    b = _row(model="dominated", p95=200.0, toks=40.0, sanity=1.0)
    result = recommend([a, b], Constraints())
    assert result.winner is not None
    assert result.winner.model == "optimal"
    assert result.is_pareto_optimal is True


def test_recommend_no_candidates_returns_no_winner() -> None:
    rows = [_row(p95=900.0), _row(p95=950.0)]
    result = recommend(rows, Constraints(max_p95_ms=500.0))
    assert result.winner is None
    assert result.is_pareto_optimal is False
    assert len(result.excluded) == 2


def test_recommend_empty_input_returns_no_winner() -> None:
    result = recommend([], Constraints())
    assert result.winner is None
    assert result.candidates == []
    assert result.excluded == []


def test_recommend_single_candidate_is_optimal() -> None:
    rows = [_row(model="only", p95=100.0)]
    result = recommend(rows, Constraints())
    assert result.winner is not None
    assert result.winner.model == "only"
    assert result.is_pareto_optimal is True


def test_recommend_tiebreak_by_p95() -> None:
    # Two Pareto-optimal rows (no dominance): pick the lower p95.
    a = _row(model="faster", p95=100.0, toks=40.0)
    b = _row(model="throughput", p95=200.0, toks=80.0)
    result = recommend([a, b], Constraints())
    # Neither dominates: a has lower p95, b has higher tok/s.
    assert result.winner is not None
    assert result.winner.model == "faster"


def test_recommend_multiple_pass_constraints_picks_fastest() -> None:
    rows = [
        _row(model="m1", p95=150.0, toks=50.0, vram=2000.0),
        _row(model="m2", p95=120.0, toks=50.0, vram=2000.0),
        _row(model="m3", p95=200.0, toks=50.0, vram=5000.0),
    ]
    result = recommend(rows, Constraints(max_vram_mb=3000.0, max_p95_ms=400.0))
    assert result.winner is not None
    assert result.winner.model == "m2"
    assert len(result.candidates) == 2
    assert len(result.excluded) == 1


# ---------------------------------------------------------------------------
# render_recommendation()
# ---------------------------------------------------------------------------


def test_render_recommendation_contains_key_fields() -> None:
    w = _row(model="my-model", p95=100.0, toks=55.3, vram=2361.0, sanity=1.0)
    result = RecommendationResult(winner=w, is_pareto_optimal=True, candidates=[w], excluded=[])
    text = render_recommendation(result)
    assert "Recommendation" in text
    assert "mock" in text  # backend
    assert "my-model" in text
    assert "100.00 ms" in text
    assert "55.3" in text
    assert "2361.0 MB" in text
    assert "100.0%" in text
    assert "Pareto-optimal" in text


def test_render_recommendation_no_winner_message() -> None:
    result = RecommendationResult(winner=None, is_pareto_optimal=False, candidates=[], excluded=[])
    text = render_recommendation(result)
    assert "No recommendation" in text


def test_render_recommendation_shows_excluded_reasons() -> None:
    w = _row(model="ok", p95=100.0)
    bad = _row(model="bad", p95=900.0)
    result = RecommendationResult(
        winner=w,
        is_pareto_optimal=True,
        candidates=[w],
        excluded=[Exclusion(row=bad, reason="p95 latency too high (900.0 ms > 500.0 ms)")],
    )
    text = render_recommendation(result)
    assert "Excluded (1)" in text
    assert "p95 latency too high" in text


def test_render_recommendation_na_for_missing_vram() -> None:
    w = _row(vram=None, sanity=None)
    result = RecommendationResult(winner=w, is_pareto_optimal=False, candidates=[w], excluded=[])
    text = render_recommendation(result)
    assert "N/A" in text


def test_render_recommendation_short_model_name_for_absolute_path() -> None:
    w = _row(model="/home/user/models/Llama-3.2-3B-Q4_K_M.gguf")
    result = RecommendationResult(winner=w, is_pareto_optimal=True, candidates=[w], excluded=[])
    text = render_recommendation(result)
    assert "Llama-3.2-3B-Q4_K_M.gguf" in text
    assert "/home/user" not in text


def test_render_recommendation_candidate_count_in_why() -> None:
    rows = [_row(model="a", p95=100.0), _row(model="b", p95=200.0)]
    result = recommend(rows, Constraints())
    text = render_recommendation(result)
    assert "2 candidate(s)" in text


# ---------------------------------------------------------------------------
# build_recommendation()
# ---------------------------------------------------------------------------


def test_build_recommendation_requires_at_least_one_path() -> None:
    with pytest.raises(ValueError, match="[Aa]t least one"):
        build_recommendation([], Constraints())


def test_build_recommendation_returns_text_and_has_winner(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(p95=100.0, vram=2000.0))
    text, has_winner = build_recommendation([p], Constraints(max_vram_mb=3000.0))
    assert has_winner is True
    assert "Recommendation" in text


def test_build_recommendation_no_winner_when_all_excluded(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(p95=900.0))
    text, has_winner = build_recommendation([p], Constraints(max_p95_ms=500.0))
    assert has_winner is False
    assert "No recommendation" in text


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_recommend_winner_exits_zero(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(p95=100.0, vram=2000.0, sanity=1.0))
    result = CliRunner().invoke(
        main,
        ["recommend", str(p), "--max-vram-mb", "4000", "--max-p95-ms", "500"],
    )
    assert result.exit_code == 0, result.output
    assert "Recommendation" in result.output


def test_cli_recommend_no_winner_exits_nonzero(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(p95=900.0))
    result = CliRunner().invoke(
        main,
        ["recommend", str(p), "--max-p95-ms", "100"],
    )
    assert result.exit_code != 0
    assert "No recommendation" in result.output


def test_cli_recommend_no_constraints_exits_zero(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(p95=100.0))
    result = CliRunner().invoke(main, ["recommend", str(p)])
    assert result.exit_code == 0, result.output


def test_cli_recommend_output_file(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(p95=100.0))
    out = tmp_path / "rec.txt"
    result = CliRunner().invoke(main, ["recommend", str(p), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Recommendation" in out.read_text()


def test_cli_recommend_two_csvs_picks_best(tmp_path: Path) -> None:
    fast = _write_csv(tmp_path / "fast.csv", _row(model="fast", p95=100.0, vram=2000.0))
    slow = _write_csv(tmp_path / "slow.csv", _row(model="slow", p95=800.0, vram=2000.0))
    result = CliRunner().invoke(
        main,
        ["recommend", str(fast), str(slow), "--max-p95-ms", "500"],
    )
    assert result.exit_code == 0, result.output
    assert "fast" in result.output
    assert "Excluded (1)" in result.output


def test_cli_recommend_missing_vram_excluded_with_vram_constraint(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(vram=None))
    result = CliRunner().invoke(main, ["recommend", str(p), "--max-vram-mb", "4000"])
    assert result.exit_code != 0
    assert "VRAM unknown" in result.output


def test_cli_recommend_min_sanity_constraint(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(sanity=0.5))
    result = CliRunner().invoke(main, ["recommend", str(p), "--min-sanity", "1.0"])
    assert result.exit_code != 0
    assert "sanity too low" in result.output


def test_cli_recommend_max_perplexity_constraint(tmp_path: Path) -> None:
    p = _write_csv(tmp_path / "run.csv", _row(ppl=20.0))
    result = CliRunner().invoke(main, ["recommend", str(p), "--max-perplexity", "10.0"])
    assert result.exit_code != 0
    assert "perplexity too high" in result.output
