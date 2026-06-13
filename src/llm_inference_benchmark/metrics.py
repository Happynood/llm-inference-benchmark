from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class RequestMetrics:
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class MetricsReport:
    request_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float
    total_tokens: int
    backend: str
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def compute_metrics(results: list[RequestMetrics], backend: str, model: str) -> MetricsReport:
    """Aggregate raw per-request results into a MetricsReport."""
    if not results:
        raise ValueError("No results to compute metrics from")

    latencies = sorted(r.latency_ms for r in results)
    total_output_tokens = sum(r.output_tokens for r in results)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in results)
    # For sequential execution, sum-of-latencies equals wall-clock time.
    # When concurrent execution is added, switch to measured wall-clock elapsed.
    total_latency_s = sum(latencies) / 1000.0

    return MetricsReport(
        request_count=len(results),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=_percentile_sorted(latencies, 95),
        tokens_per_second=total_output_tokens / total_latency_s if total_latency_s > 0 else 0.0,
        total_tokens=total_tokens,
        backend=backend,
        model=model,
    )


def _percentile_sorted(sorted_data: list[float], p: int) -> float:
    """Linear interpolation percentile on a pre-sorted list (C1 method, matches NumPy default)."""
    if not sorted_data:
        return 0.0
    idx = (p / 100) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    fraction = idx - lower
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])
