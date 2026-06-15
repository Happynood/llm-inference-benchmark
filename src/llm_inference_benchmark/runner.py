from __future__ import annotations

import statistics
import time
from pathlib import Path

from llm_inference_benchmark.backends.base import Backend
from llm_inference_benchmark.config import BenchmarkConfig
from llm_inference_benchmark.memory import (
    MemorySampler,
    NvidiaSmiSampler,
    cuda_peak_mb,
    reset_cuda_peak,
)
from llm_inference_benchmark.metrics import MetricsReport, RequestMetrics, compute_metrics
from llm_inference_benchmark.quality import compute_quality
from llm_inference_benchmark.task_quality import (
    TaskQualityReport,
    compute_task_quality,
    load_task_rubrics,
)


def load_prompts(path: str | Path) -> list[str]:
    """Read non-blank lines from a prompt file and return them as a list."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No prompts found in {path}")
    return lines


def run_benchmark(
    backend: Backend,
    config: BenchmarkConfig,
    prompts: list[str],
    model_load_ms: float | None = None,
) -> MetricsReport:
    """Run warmup + benchmark loop and return aggregated metrics including peak memory.

    Memory measurement window covers only the benchmark loop (warmup excluded) so
    first-use allocation spikes do not inflate the reported peak. reset_cuda_peak() is
    called inside the MemorySampler context so CPU and CUDA windows are co-incident.

    model_load_ms: elapsed time (ms) for backend construction, measured at the call site
    before run_benchmark is invoked. Pass None when not measured.
    """
    warmup_latencies: list[float] = []
    for i in range(config.warmup_requests):
        t0 = time.perf_counter()
        backend.generate(prompts[i % len(prompts)])
        warmup_latencies.append((time.perf_counter() - t0) * 1000.0)

    warmup_p50 = statistics.median(warmup_latencies) if warmup_latencies else None

    results: list[RequestMetrics] = []
    texts: list[str] = []
    with MemorySampler() as mem, NvidiaSmiSampler() as vram:
        reset_cuda_peak()
        for i in range(config.requests):
            result = backend.generate(prompts[i % len(prompts)])
            results.append(
                RequestMetrics(
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
            )
            texts.append(result.text)

    task_qual: TaskQualityReport | None = None
    if config.quality_file is not None:
        rubrics = load_task_rubrics(config.quality_file)
        task_qual = compute_task_quality(texts, len(prompts), rubrics)

    return compute_metrics(
        results,
        backend=backend.name,
        model=config.model,
        peak_cpu_memory_mb=mem.peak_cpu_mb,
        peak_cuda_memory_mb=cuda_peak_mb(),
        peak_vram_memory_mb=vram.peak_vram_mb,
        quality=compute_quality(texts),
        task_quality=task_qual,
        model_load_ms=model_load_ms,
        warmup_p50_latency_ms=warmup_p50,
    )
