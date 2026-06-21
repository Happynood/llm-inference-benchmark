from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime

from llm_inference_benchmark.hardware import HardwareProfile
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
    # Repeated-trial variance (v0.19) — None for single runs (config.repeats == 1)
    # p95 and tok/s become median-across-repeats; std fields hold sample std dev (n-1)
    repeats: int | None = None
    p95_latency_ms_std: float | None = None
    tokens_per_second_std: float | None = None
    # TTFT (v0.23) — None unless the backend was run in streaming mode (currently
    # openai_endpoint with stream=True only).
    p50_ttft_ms: float | None = None
    p95_ttft_ms: float | None = None
    # TPOT (v0.24) — time per output token in the decode phase: (latency - ttft) / output_tokens.
    # None when TTFT data is unavailable or no requests produced output tokens.
    p50_tpot_ms: float | None = None
    tpot_stddev_ms: float | None = None
    # Self-perplexity (v0.20) — None unless config.measure_perplexity is set and the
    # backend exposes token-level log-probabilities (currently transformers only).
    perplexity: float | None = None
    # LLM-as-judge score (v0.21) — None unless config.measure_judge is set and the
    # backend exposes token-level log-probabilities (currently transformers only).
    judge_score: float | None = None
    # Workload composition (v0.22) — mean token counts per request and decode
    # throughput.  decode_tokens_per_second == tokens_per_second for sequential
    # execution; they diverge when concurrent execution tracks prefill separately.
    # None when no output tokens were produced in the run.
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    decode_tokens_per_second: float | None = None
    # Hardware profile (v0.25) — compact snapshot of the machine that ran the benchmark.
    # All fields default to None so existing test code that omits them still compiles.
    hw_cpu: str | None = None
    hw_cpu_cores: int | None = None
    hw_ram_gb: float | None = None
    hw_gpu: str | None = None
    hw_vram_gb: float | None = None
    hw_os: str | None = None
    # Reasoning token parser (v0.26) — populated when reasoning_start_tag /
    # reasoning_end_tag are set in the config.  Token counts are estimated from
    # the char-length fraction of backend-reported output_tokens.
    mean_reasoning_tokens: float | None = None
    mean_answer_tokens: float | None = None
    reasoning_fraction: float | None = None
    # Energy efficiency (v0.27) — populated when GPU (nvidia-smi) or CPU (RAPL)
    # power measurement is available.  energy_joules covers the benchmark window
    # only (warmup excluded).
    energy_joules: float | None = None
    tokens_per_joule: float | None = None
    # Thermal throttling index (v0.28) — percentage drop in tok/s from the first
    # 25 % of elapsed time to the last 25 %.  Positive values indicate frequency
    # scaling / thermal throttling.  None for concurrent runs, runs shorter than
    # 10 s, or runs with fewer than 8 requests.
    thermal_throttle_pct: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _compute_thermal_throttle(
    results: list[RequestMetrics],
    *,
    is_sequential: bool,
) -> float | None:
    """Compare tok/s in first vs last 25 % of sequential wall time.

    Returns the percentage drop (positive = throttling, 0 = stable/faster).
    Returns None when: concurrent run, fewer than 8 requests, or total
    elapsed time under 10 s.
    """
    if not is_sequential or len(results) < 8:
        return None

    cum_s: list[float] = []
    t = 0.0
    for r in results:
        t += r.latency_ms / 1000.0
        cum_s.append(t)

    total_s = cum_s[-1]
    if total_s < 10.0:
        return None

    boundary_early = total_s * 0.25
    boundary_late = total_s * 0.75

    early_tokens = sum(
        r.output_tokens for r, end in zip(results, cum_s) if end <= boundary_early
    )
    late_tokens = sum(
        r.output_tokens for r, end in zip(results, cum_s) if end >= boundary_late
    )

    if boundary_early <= 0 or early_tokens == 0:
        return None
    late_time = total_s - boundary_late
    if late_time <= 0:
        return None

    tps_early = early_tokens / boundary_early
    tps_late = late_tokens / late_time
    pct = (tps_early - tps_late) / tps_early * 100.0
    return round(max(0.0, pct), 2)


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
    perplexity: float | None = None,
    judge_score: float | None = None,
    wall_clock_elapsed_s: float | None = None,
    ttft_values: list[float] | None = None,
    tpot_values: list[float] | None = None,
    hardware: HardwareProfile | None = None,
    mean_reasoning_tokens: float | None = None,
    mean_answer_tokens: float | None = None,
    reasoning_fraction: float | None = None,
    energy_joules: float | None = None,
    is_sequential: bool = True,
) -> MetricsReport:
    """Aggregate raw per-request results into a MetricsReport."""
    if not results:
        raise ValueError("No results to compute metrics from")

    latencies = sorted(r.latency_ms for r in results)
    n = len(results)
    total_output_tokens = sum(r.output_tokens for r in results)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in results)
    # Sequential: sum-of-latencies == wall-clock time.
    # Concurrent: caller passes measured wall-clock elapsed so throughput reflects
    # actual parallelism rather than summed per-request times.
    total_latency_s = (
        wall_clock_elapsed_s if wall_clock_elapsed_s is not None else sum(latencies) / 1000.0
    )

    sorted_ttft = sorted(ttft_values) if ttft_values else None

    return MetricsReport(
        request_count=n,
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
        p50_ttft_ms=_percentile_sorted(sorted_ttft, 50) if sorted_ttft else None,
        p95_ttft_ms=_percentile_sorted(sorted_ttft, 95) if sorted_ttft else None,
        p50_tpot_ms=(_percentile_sorted(sorted(tpot_values), 50) if tpot_values else None),
        tpot_stddev_ms=(
            statistics.stdev(tpot_values) if tpot_values and len(tpot_values) >= 2 else None
        ),
        perplexity=perplexity,
        judge_score=judge_score,
        mean_input_tokens=sum(r.input_tokens for r in results) / n,
        mean_output_tokens=sum(r.output_tokens for r in results) / n,
        decode_tokens_per_second=(
            total_output_tokens / total_latency_s
            if total_output_tokens > 0 and total_latency_s > 0
            else None
        ),
        hw_cpu=hardware.cpu if hardware is not None else None,
        hw_cpu_cores=hardware.cpu_cores if hardware is not None else None,
        hw_ram_gb=hardware.ram_gb if hardware is not None else None,
        hw_gpu=hardware.gpu if hardware is not None else None,
        hw_vram_gb=hardware.vram_gb if hardware is not None else None,
        hw_os=hardware.os if hardware is not None else None,
        mean_reasoning_tokens=mean_reasoning_tokens,
        mean_answer_tokens=mean_answer_tokens,
        reasoning_fraction=reasoning_fraction,
        energy_joules=energy_joules,
        tokens_per_joule=(
            total_output_tokens / energy_joules
            if energy_joules is not None and energy_joules > 0 and total_output_tokens > 0
            else None
        ),
        thermal_throttle_pct=_compute_thermal_throttle(results, is_sequential=is_sequential),
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


def _max_optional(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _median_optional(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def aggregate_repeat_reports(reports: list[MetricsReport]) -> MetricsReport:
    """Return a single MetricsReport summarising N repeated benchmark runs.

    When len(reports) == 1 the report is returned unchanged with variance fields as None
    (backward-compatible: the CSV looks identical to a non-repeated run).

    For len(reports) >= 2:
    - p50/p95 latency and tokens_per_second become the median across repeats.
    - p95_latency_ms_std and tokens_per_second_std are the sample standard deviation
      (n-1 denominator, via statistics.stdev) across repeats.
    - Memory peaks (CPU, CUDA, VRAM) take the maximum across repeats.
    - Non-aggregated fields (quality, task quality, warmup, perplexity, judge_score) come
      from the last repeat.
    - model_load_ms comes from the first repeat (backend was constructed once before repeats).

    Capture scope: repeats share one process and warm cache state, so variance reflects
    in-process loop jitter, not cold-start or cross-machine variance.
    """
    if not reports:
        raise ValueError("No reports to aggregate")

    if len(reports) == 1:
        return reports[0]

    n = len(reports)
    first = reports[0]
    last = reports[-1]

    p50s = [r.p50_latency_ms for r in reports]
    p95s = [r.p95_latency_ms for r in reports]
    toks = [r.tokens_per_second for r in reports]

    return MetricsReport(
        request_count=last.request_count,
        p50_latency_ms=statistics.median(p50s),
        p95_latency_ms=statistics.median(p95s),
        tokens_per_second=statistics.median(toks),
        total_tokens=last.total_tokens,
        backend=last.backend,
        model=last.model,
        peak_cpu_memory_mb=max(r.peak_cpu_memory_mb for r in reports),
        peak_cuda_memory_mb=_max_optional([r.peak_cuda_memory_mb for r in reports]),
        peak_vram_memory_mb=_max_optional([r.peak_vram_memory_mb for r in reports]),
        empty_output_count=last.empty_output_count,
        min_output_chars=last.min_output_chars,
        mean_output_chars=last.mean_output_chars,
        repeated_output_count=last.repeated_output_count,
        sanity_pass_rate=last.sanity_pass_rate,
        task_quality_pass_rate=last.task_quality_pass_rate,
        task_quality_checked_count=last.task_quality_checked_count,
        model_load_ms=first.model_load_ms,
        warmup_p50_latency_ms=last.warmup_p50_latency_ms,
        p50_ttft_ms=_median_optional([r.p50_ttft_ms for r in reports]),
        p95_ttft_ms=_median_optional([r.p95_ttft_ms for r in reports]),
        p50_tpot_ms=_median_optional([r.p50_tpot_ms for r in reports]),
        tpot_stddev_ms=_median_optional([r.tpot_stddev_ms for r in reports]),
        perplexity=last.perplexity,
        judge_score=last.judge_score,
        mean_input_tokens=last.mean_input_tokens,
        mean_output_tokens=last.mean_output_tokens,
        decode_tokens_per_second=_median_optional([r.decode_tokens_per_second for r in reports]),
        repeats=n,
        p95_latency_ms_std=statistics.stdev(p95s),
        tokens_per_second_std=statistics.stdev(toks),
        hw_cpu=first.hw_cpu,
        hw_cpu_cores=first.hw_cpu_cores,
        hw_ram_gb=first.hw_ram_gb,
        hw_gpu=first.hw_gpu,
        hw_vram_gb=first.hw_vram_gb,
        hw_os=first.hw_os,
        mean_reasoning_tokens=_median_optional([r.mean_reasoning_tokens for r in reports]),
        mean_answer_tokens=_median_optional([r.mean_answer_tokens for r in reports]),
        reasoning_fraction=_median_optional([r.reasoning_fraction for r in reports]),
        energy_joules=_median_optional([r.energy_joules for r in reports]),
        tokens_per_joule=_median_optional([r.tokens_per_joule for r in reports]),
        thermal_throttle_pct=_median_optional([r.thermal_throttle_pct for r in reports]),
        timestamp=last.timestamp,
    )
