import pytest

from llm_inference_benchmark.metrics import RequestMetrics, compute_metrics
from llm_inference_benchmark.quality import QualityReport


def _results(latencies: list[float]) -> list[RequestMetrics]:
    return [RequestMetrics(latency_ms=lat, input_tokens=5, output_tokens=10) for lat in latencies]


def test_p50_latency() -> None:
    report = compute_metrics(_results([10.0, 20.0, 30.0, 40.0, 50.0]), backend="mock", model="t")
    assert report.p50_latency_ms == 30.0


def test_p95_latency() -> None:
    latencies = [float(i) for i in range(1, 101)]
    report = compute_metrics(_results(latencies), backend="mock", model="t")
    # idx = 0.95*99 = 94.05 → sorted[94]=95, sorted[95]=96 → 95 + 0.05 = 95.05
    assert report.p95_latency_ms == pytest.approx(95.05, abs=0.01)


def test_total_tokens() -> None:
    # Each request: 5 input + 10 output = 15 total; 5 requests → 75
    report = compute_metrics(_results([10.0] * 5), backend="mock", model="t")
    assert report.total_tokens == 75
    assert report.request_count == 5


def test_tokens_per_second_positive() -> None:
    report = compute_metrics(_results([100.0] * 10), backend="mock", model="t")
    # 10 requests * 10 output tokens / (10 * 0.1s) = 10 tok / 1s = 10
    assert report.tokens_per_second == pytest.approx(10.0 * 10 / 1.0, rel=1e-6)


def test_empty_results_raises() -> None:
    with pytest.raises(ValueError, match="No results"):
        compute_metrics([], backend="mock", model="t")


def test_report_has_timestamp() -> None:
    report = compute_metrics(_results([5.0]), backend="mock", model="t")
    assert report.timestamp  # non-empty ISO string
    assert "T" in report.timestamp


def test_memory_fields_passed_through() -> None:
    report = compute_metrics(
        _results([10.0]),
        backend="mock",
        model="t",
        peak_cpu_memory_mb=42.5,
        peak_cuda_memory_mb=128.0,
    )
    assert report.peak_cpu_memory_mb == pytest.approx(42.5)
    assert report.peak_cuda_memory_mb == pytest.approx(128.0)


def test_cuda_memory_defaults_to_none() -> None:
    report = compute_metrics(_results([10.0]), backend="mock", model="t", peak_cpu_memory_mb=1.0)
    assert report.peak_cuda_memory_mb is None


def test_cpu_memory_defaults_to_zero() -> None:
    report = compute_metrics(_results([10.0]), backend="mock", model="t")
    assert report.peak_cpu_memory_mb == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Quality fields
# ---------------------------------------------------------------------------


def test_quality_fields_pass_through_from_report() -> None:
    q = QualityReport(
        empty_output_count=1,
        min_output_chars=0,
        mean_output_chars=25.0,
        repeated_output_count=3,
        sanity_pass_rate=0.8,
    )
    report = compute_metrics(_results([10.0] * 5), backend="mock", model="t", quality=q)
    assert report.empty_output_count == 1
    assert report.min_output_chars == 0
    assert report.mean_output_chars == pytest.approx(25.0)
    assert report.repeated_output_count == 3
    assert report.sanity_pass_rate == pytest.approx(0.8)


def test_quality_defaults_when_none() -> None:
    report = compute_metrics(_results([10.0]), backend="mock", model="t", quality=None)
    assert report.empty_output_count == 0
    assert report.min_output_chars == 0
    assert report.mean_output_chars == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Perplexity field (v0.20)
# ---------------------------------------------------------------------------


def test_perplexity_defaults_to_none() -> None:
    report = compute_metrics(_results([10.0]), backend="mock", model="t")
    assert report.perplexity is None


def test_perplexity_passed_through() -> None:
    report = compute_metrics(_results([10.0]), backend="mock", model="t", perplexity=12.34)
    assert report.perplexity == pytest.approx(12.34)
    assert report.repeated_output_count == 0
    assert report.sanity_pass_rate == pytest.approx(1.0)


def test_quality_defaults_when_not_provided() -> None:
    report = compute_metrics(_results([10.0]), backend="mock", model="t")
    assert report.sanity_pass_rate == pytest.approx(1.0)
    assert report.empty_output_count == 0


# ---------------------------------------------------------------------------
# Judge score field (v0.21)
# ---------------------------------------------------------------------------


def test_judge_score_defaults_to_none() -> None:
    report = compute_metrics(_results([10.0]), backend="mock", model="t")
    assert report.judge_score is None


def test_judge_score_passed_through() -> None:
    report = compute_metrics(_results([10.0]), backend="mock", model="t", judge_score=0.85)
    assert report.judge_score == pytest.approx(0.85)
    assert report.repeated_output_count == 0
    assert report.sanity_pass_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Workload composition fields (v0.22)
# ---------------------------------------------------------------------------


def test_mean_token_counts() -> None:
    results = [
        RequestMetrics(latency_ms=10.0, input_tokens=4, output_tokens=8),
        RequestMetrics(latency_ms=10.0, input_tokens=6, output_tokens=12),
    ]
    report = compute_metrics(results, backend="mock", model="t")
    assert report.mean_input_tokens == pytest.approx(5.0)
    assert report.mean_output_tokens == pytest.approx(10.0)


def test_decode_tokens_per_second_equals_tokens_per_second() -> None:
    results = [RequestMetrics(latency_ms=100.0, input_tokens=5, output_tokens=10)] * 10
    report = compute_metrics(results, backend="mock", model="t")
    assert report.decode_tokens_per_second is not None
    assert report.decode_tokens_per_second == pytest.approx(report.tokens_per_second, rel=1e-9)


def test_decode_tokens_per_second_none_when_no_output() -> None:
    results = [RequestMetrics(latency_ms=10.0, input_tokens=5, output_tokens=0)]
    report = compute_metrics(results, backend="mock", model="t")
    assert report.decode_tokens_per_second is None
    assert report.tokens_per_second == pytest.approx(0.0)


def test_wall_clock_elapsed_overrides_sum_of_latencies() -> None:
    # 4 requests each 100 ms → sequential sum = 400 ms = 0.4 s → 4*10/0.4 = 100 tok/s
    # but with wall_clock_elapsed_s=0.1 (all in parallel) → 4*10/0.1 = 400 tok/s
    results = [RequestMetrics(latency_ms=100.0, input_tokens=5, output_tokens=10)] * 4
    seq = compute_metrics(results, backend="mock", model="t")
    concurrent = compute_metrics(results, backend="mock", model="t", wall_clock_elapsed_s=0.1)
    assert seq.tokens_per_second == pytest.approx(100.0)
    assert concurrent.tokens_per_second == pytest.approx(400.0)


def test_compute_metrics_with_ttft_values() -> None:
    results = [RequestMetrics(latency_ms=100.0, input_tokens=5, output_tokens=10)] * 4
    ttft = [20.0, 30.0, 25.0, 35.0]
    report = compute_metrics(results, backend="mock", model="t", ttft_values=ttft)
    assert report.p50_ttft_ms is not None
    assert report.p95_ttft_ms is not None
    assert report.p50_ttft_ms == pytest.approx(27.5)
    assert report.p95_ttft_ms == pytest.approx(34.25)


def test_compute_metrics_without_ttft_returns_none() -> None:
    results = [RequestMetrics(latency_ms=100.0, input_tokens=5, output_tokens=10)]
    report = compute_metrics(results, backend="mock", model="t")
    assert report.p50_ttft_ms is None
    assert report.p95_ttft_ms is None


def test_compute_metrics_empty_ttft_list_returns_none() -> None:
    results = [RequestMetrics(latency_ms=100.0, input_tokens=5, output_tokens=10)]
    report = compute_metrics(results, backend="mock", model="t", ttft_values=[])
    assert report.p50_ttft_ms is None
    assert report.p95_ttft_ms is None


def test_percentile_sorted_empty_list_returns_zero() -> None:
    from llm_inference_benchmark.metrics import _percentile_sorted

    assert _percentile_sorted([], 50) == 0.0
    assert _percentile_sorted([], 95) == 0.0
