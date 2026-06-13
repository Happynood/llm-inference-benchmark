import pytest

from llm_inference_benchmark.metrics import RequestMetrics, compute_metrics


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
