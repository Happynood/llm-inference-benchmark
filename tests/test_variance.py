"""Tests for repeated-trial variance reporting (v0.19)."""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from llm_inference_benchmark.backends.mock import MockBackend
from llm_inference_benchmark.config import BenchmarkConfig
from llm_inference_benchmark.metrics import MetricsReport, aggregate_repeat_reports
from llm_inference_benchmark.runner import load_prompts, run_repeated

# ---------------------------------------------------------------------------
# MetricsReport: new field defaults
# ---------------------------------------------------------------------------


def _make_report(**kwargs: object) -> MetricsReport:
    defaults: dict[str, object] = dict(
        request_count=10,
        p50_latency_ms=5.0,
        p95_latency_ms=5.5,
        tokens_per_second=100.0,
        total_tokens=150,
        backend="mock",
        model="mock-gpt2",
        peak_cpu_memory_mb=50.0,
        peak_cuda_memory_mb=None,
        peak_vram_memory_mb=None,
    )
    defaults.update(kwargs)
    return MetricsReport(**defaults)  # type: ignore[arg-type]


def test_metrics_report_variance_fields_default_none() -> None:
    report = _make_report()
    assert report.repeats is None
    assert report.p95_latency_ms_std is None
    assert report.tokens_per_second_std is None


# ---------------------------------------------------------------------------
# aggregate_repeat_reports: single report
# ---------------------------------------------------------------------------


def test_aggregate_single_report_returns_none_std() -> None:
    r = _make_report(p95_latency_ms=6.0, tokens_per_second=120.0)
    agg = aggregate_repeat_reports([r])
    assert agg.p95_latency_ms == pytest.approx(6.0)
    assert agg.tokens_per_second == pytest.approx(120.0)
    assert agg.repeats is None
    assert agg.p95_latency_ms_std is None
    assert agg.tokens_per_second_std is None


def test_aggregate_single_report_preserves_other_fields() -> None:
    r = _make_report(backend="transformers", model="tiny-gpt2", peak_cpu_memory_mb=200.0)
    agg = aggregate_repeat_reports([r])
    assert agg.backend == "transformers"
    assert agg.model == "tiny-gpt2"
    assert agg.peak_cpu_memory_mb == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# aggregate_repeat_reports: multiple reports — median
# ---------------------------------------------------------------------------


def test_aggregate_median_p95_three_reports() -> None:
    reports = [
        _make_report(p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(p95_latency_ms=7.0, tokens_per_second=90.0),
        _make_report(p95_latency_ms=6.0, tokens_per_second=95.0),
    ]
    agg = aggregate_repeat_reports(reports)
    # median of [5.0, 7.0, 6.0] = 6.0
    assert agg.p95_latency_ms == pytest.approx(6.0)
    # median of [100.0, 90.0, 95.0] = 95.0
    assert agg.tokens_per_second == pytest.approx(95.0)
    assert agg.repeats == 3


def test_aggregate_median_p50_three_reports() -> None:
    reports = [
        _make_report(p50_latency_ms=4.0, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(p50_latency_ms=6.0, p95_latency_ms=7.0, tokens_per_second=90.0),
        _make_report(p50_latency_ms=5.0, p95_latency_ms=6.0, tokens_per_second=95.0),
    ]
    agg = aggregate_repeat_reports(reports)
    # median of [4.0, 6.0, 5.0] = 5.0
    assert agg.p50_latency_ms == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# aggregate_repeat_reports: std dev
# ---------------------------------------------------------------------------


def test_aggregate_std_zero_for_constant_p95() -> None:
    reports = [_make_report(p95_latency_ms=5.0, tokens_per_second=100.0) for _ in range(3)]
    agg = aggregate_repeat_reports(reports)
    assert agg.p95_latency_ms_std == pytest.approx(0.0, abs=1e-10)
    assert agg.tokens_per_second_std == pytest.approx(0.0, abs=1e-10)


def test_aggregate_std_matches_statistics_stdev() -> None:
    """p95 std dev is sample std dev (n-1) via statistics.stdev."""
    p95_values = [5.0, 6.0, 8.0]
    tok_values = [100.0, 80.0, 90.0]
    reports = [
        _make_report(p95_latency_ms=p, tokens_per_second=t)
        for p, t in zip(p95_values, tok_values, strict=True)
    ]
    agg = aggregate_repeat_reports(reports)
    assert agg.p95_latency_ms_std == pytest.approx(statistics.stdev(p95_values))
    assert agg.tokens_per_second_std == pytest.approx(statistics.stdev(tok_values))


def test_aggregate_std_two_reports() -> None:
    reports = [
        _make_report(p95_latency_ms=4.0, tokens_per_second=100.0),
        _make_report(p95_latency_ms=6.0, tokens_per_second=80.0),
    ]
    agg = aggregate_repeat_reports(reports)
    assert agg.p95_latency_ms_std == pytest.approx(statistics.stdev([4.0, 6.0]))
    assert agg.repeats == 2


# ---------------------------------------------------------------------------
# aggregate_repeat_reports: memory peaks
# ---------------------------------------------------------------------------


def test_aggregate_takes_max_cpu_memory() -> None:
    reports = [
        _make_report(peak_cpu_memory_mb=50.0, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(peak_cpu_memory_mb=80.0, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(peak_cpu_memory_mb=60.0, p95_latency_ms=5.0, tokens_per_second=100.0),
    ]
    agg = aggregate_repeat_reports(reports)
    assert agg.peak_cpu_memory_mb == pytest.approx(80.0)


def test_aggregate_takes_max_optional_vram() -> None:
    reports = [
        _make_report(peak_vram_memory_mb=1000.0, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(peak_vram_memory_mb=1200.0, p95_latency_ms=5.0, tokens_per_second=100.0),
    ]
    agg = aggregate_repeat_reports(reports)
    assert agg.peak_vram_memory_mb == pytest.approx(1200.0)


def test_aggregate_optional_vram_none_when_all_none() -> None:
    reports = [
        _make_report(peak_vram_memory_mb=None, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(peak_vram_memory_mb=None, p95_latency_ms=5.0, tokens_per_second=100.0),
    ]
    agg = aggregate_repeat_reports(reports)
    assert agg.peak_vram_memory_mb is None


# ---------------------------------------------------------------------------
# aggregate_repeat_reports: non-aggregated fields come from last report
# ---------------------------------------------------------------------------


def test_aggregate_model_load_ms_from_first_report() -> None:
    reports = [
        _make_report(model_load_ms=150.0, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(model_load_ms=None, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(model_load_ms=None, p95_latency_ms=5.0, tokens_per_second=100.0),
    ]
    agg = aggregate_repeat_reports(reports)
    # model_load_ms was measured before the first repeat; subsequent ones get None
    assert agg.model_load_ms == pytest.approx(150.0)


def test_aggregate_perplexity_from_last_report() -> None:
    reports = [
        _make_report(perplexity=10.0, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(perplexity=12.0, p95_latency_ms=5.0, tokens_per_second=100.0),
    ]
    agg = aggregate_repeat_reports(reports)
    assert agg.perplexity == pytest.approx(12.0)


def test_aggregate_judge_score_from_last_report() -> None:
    reports = [
        _make_report(judge_score=0.6, p95_latency_ms=5.0, tokens_per_second=100.0),
        _make_report(judge_score=0.9, p95_latency_ms=5.0, tokens_per_second=100.0),
    ]
    agg = aggregate_repeat_reports(reports)
    assert agg.judge_score == pytest.approx(0.9)


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError, match="No reports"):
        aggregate_repeat_reports([])


# ---------------------------------------------------------------------------
# BenchmarkConfig: repeats field
# ---------------------------------------------------------------------------


def test_config_repeats_default_one() -> None:
    cfg = BenchmarkConfig()
    assert cfg.repeats == 1


def test_config_repeats_set() -> None:
    cfg = BenchmarkConfig(repeats=3)
    assert cfg.repeats == 3


def test_config_repeats_zero_invalid() -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(repeats=0)


def test_config_repeats_negative_invalid() -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(repeats=-1)


# ---------------------------------------------------------------------------
# run_repeated: single repeat behaves like run_benchmark
# ---------------------------------------------------------------------------


def test_run_repeated_single_matches_run_benchmark(tmp_prompts: Path) -> None:
    backend = MockBackend(model="test", latency_ms=0, tokens_per_response=10)
    cfg = BenchmarkConfig(requests=5, warmup_requests=0, repeats=1)
    report = run_repeated(backend, cfg, load_prompts(tmp_prompts))
    assert report.request_count == 5
    assert report.repeats is None
    assert report.p95_latency_ms_std is None
    assert report.tokens_per_second_std is None


def test_run_repeated_multiple_gives_aggregated_report(tmp_prompts: Path) -> None:
    backend = MockBackend(model="test", latency_ms=5, tokens_per_response=10)
    cfg = BenchmarkConfig(requests=5, warmup_requests=0, repeats=3)
    report = run_repeated(backend, cfg, load_prompts(tmp_prompts))
    assert report.repeats == 3
    assert report.p95_latency_ms_std is not None
    assert report.tokens_per_second_std is not None
    assert report.request_count == 5


def test_run_repeated_model_load_ms_attached(tmp_prompts: Path) -> None:
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(requests=3, warmup_requests=0, repeats=2)
    report = run_repeated(backend, cfg, load_prompts(tmp_prompts), model_load_ms=42.0)
    assert report.model_load_ms == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# compare.load_csv: tolerates extra variance columns
# ---------------------------------------------------------------------------


def test_load_csv_tolerates_variance_columns(tmp_path: Path) -> None:
    """Old compare loader must accept CSVs that contain the new variance columns."""
    from llm_inference_benchmark.compare import load_csv

    csv_content = (
        "request_count,backend,model,p50_latency_ms,p95_latency_ms,"
        "tokens_per_second,peak_cpu_memory_mb,peak_cuda_memory_mb,"
        "repeats,p95_latency_ms_std,tokens_per_second_std\n"
        "10,mock,mock-gpt2,5.0,5.5,100.0,50.0,,3,0.2,1.5\n"
    )
    p = tmp_path / "run.csv"
    p.write_text(csv_content)
    row = load_csv(p)
    assert row.backend == "mock"
    assert row.p95_latency_ms == pytest.approx(5.5)


def test_load_csv_without_variance_columns_still_works(tmp_path: Path) -> None:
    """Old CSVs without variance columns load without error (backward compat)."""
    from llm_inference_benchmark.compare import load_csv

    csv_content = (
        "request_count,backend,model,p50_latency_ms,p95_latency_ms,"
        "tokens_per_second,peak_cpu_memory_mb,peak_cuda_memory_mb\n"
        "10,mock,mock-gpt2,5.0,5.5,100.0,50.0,\n"
    )
    p = tmp_path / "old.csv"
    p.write_text(csv_content)
    row = load_csv(p)
    assert row.backend == "mock"
