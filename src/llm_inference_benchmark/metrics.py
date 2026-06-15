from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from llm_inference_benchmark.quality import QualityReport
from llm_inference_benchmark.task_quality import TaskQualityReport


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
    peak_cpu_memory_mb: float
    peak_cuda_memory_mb: float | None
    peak_vram_memory_mb: float | None
    # Output sanity fields — always populated for new runs; default to "no issues" so
    # MetricsReport can be constructed in tests without providing a QualityReport.
    empty_output_count: int = 0
    min_output_chars: int = 0
    mean_output_chars: float = 0.0
    repeated_output_count: int = 0
    sanity_pass_rate: float = 1.0
    task_quality_pass_rate: float | None = None
    task_quality_checked_count: int | None = None
    # Lifecycle metrics — absent in runs that predate v0.18 (blank in CSV, None here)
    model_load_ms: float | None = None
    warmup_p50_latency_ms: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def compute_metrics(
    results: list[RequestMetrics],
    backend: str,
    model: str,
    peak_cpu_memory_mb: float = 0.0,
    peak_cuda_memory_mb: float | None = None,
    peak_vram_memory_mb: float | None = None,
    quality: QualityReport | None = None,
    task_quality: TaskQualityReport | None = None,
    model_load_ms: float | None = None,
    warmup_p50_latency_ms: float | None = None,
) -> MetricsReport:
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
        p50_latency_ms=_percentile_sorted(latencies, 50),
        p95_latency_ms=_percentile_sorted(latencies, 95),
        tokens_per_second=total_output_tokens / total_latency_s if total_latency_s > 0 else 0.0,
        total_tokens=total_tokens,
        backend=backend,
        model=model,
        peak_cpu_memory_mb=peak_cpu_memory_mb,
        peak_cuda_memory_mb=peak_cuda_memory_mb,
        peak_vram_memory_mb=peak_vram_memory_mb,
        empty_output_count=quality.empty_output_count if quality is not None else 0,
        min_output_chars=quality.min_output_chars if quality is not None else 0,
        mean_output_chars=quality.mean_output_chars if quality is not None else 0.0,
        repeated_output_count=quality.repeated_output_count if quality is not None else 0,
        sanity_pass_rate=quality.sanity_pass_rate if quality is not None else 1.0,
        task_quality_pass_rate=(
            task_quality.task_quality_pass_rate if task_quality is not None else None
        ),
        task_quality_checked_count=(
            task_quality.task_quality_checked_count if task_quality is not None else None
        ),
        model_load_ms=model_load_ms,
        warmup_p50_latency_ms=warmup_p50_latency_ms,
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
