from __future__ import annotations

import asyncio
import random
import statistics
import time
from pathlib import Path

from llm_inference_benchmark.backends.base import Backend
from llm_inference_benchmark.config import BenchmarkConfig
from llm_inference_benchmark.hardware import HardwareProfile
from llm_inference_benchmark.hardware import detect as detect_hardware
from llm_inference_benchmark.memory import (
    MemorySampler,
    NvidiaSmiSampler,
    cuda_peak_mb,
    reset_cuda_peak,
)
from llm_inference_benchmark.metrics import (
    MetricsReport,
    RequestMetrics,
    aggregate_repeat_reports,
    compute_metrics,
)
from llm_inference_benchmark.quality import compute_quality
from llm_inference_benchmark.reasoning import reasoning_stats
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


def _resolve_prompt_sequence(prompts: list[str], n: int, seed: int | None) -> list[str]:
    """Return the ordered list of prompts to use for n benchmark requests.

    When seed is None the existing cycling behaviour is preserved.
    When seed is set the sequence is drawn with replacement using random.Random(seed),
    making both prompt selection and model-side sampling fully reproducible from the
    same config.
    """
    if seed is None:
        return [prompts[i % len(prompts)] for i in range(n)]
    return random.Random(seed).choices(prompts, k=n)


async def _run_concurrent(
    backend: Backend, config: BenchmarkConfig, prompts: list[str]
) -> tuple[list[RequestMetrics], list[str], list[str], float, list[float], list[float]]:
    """Fire config.requests requests with up to config.concurrency in flight at once.

    Returns (results, texts, prompts_used, wall_clock_elapsed_s, ttft_values, tpot_values).
    Uses asyncio.to_thread so sync Backend.generate() calls run in the default
    thread-pool executor without blocking the event loop.
    """
    sem = asyncio.Semaphore(config.concurrency)

    async def _one(i: int) -> tuple[str, RequestMetrics, str, float | None]:
        async with sem:
            prompt = prompts[i % len(prompts)]
            r = await asyncio.to_thread(backend.generate, prompt)
            return (
                prompt,
                RequestMetrics(
                    latency_ms=r.latency_ms,
                    input_tokens=r.input_tokens,
                    output_tokens=r.output_tokens,
                ),
                r.text,
                r.ttft_ms,
            )

    wall_t0 = time.perf_counter()
    quads = await asyncio.gather(*[_one(i) for i in range(config.requests)])
    wall_elapsed_s = time.perf_counter() - wall_t0

    prompts_used = [q[0] for q in quads]
    results = [q[1] for q in quads]
    texts = [q[2] for q in quads]
    ttft_values = [q[3] for q in quads if q[3] is not None]
    tpot_values = [
        (q[1].latency_ms - q[3]) / q[1].output_tokens
        for q in quads
        if q[3] is not None and q[1].output_tokens > 0 and q[1].latency_ms > q[3]
    ]
    return results, texts, prompts_used, wall_elapsed_s, ttft_values, tpot_values


def run_benchmark(
    backend: Backend,
    config: BenchmarkConfig,
    prompts: list[str],
    model_load_ms: float | None = None,
    hardware: HardwareProfile | None = None,
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

    sequence = _resolve_prompt_sequence(prompts, config.requests, config.seed)

    results: list[RequestMetrics] = []
    texts: list[str] = []
    prompts_used: list[str] = []
    wall_clock_elapsed_s: float | None = None
    ttft_values: list[float] = []
    tpot_values: list[float] = []

    with MemorySampler() as mem, NvidiaSmiSampler() as vram:
        reset_cuda_peak()
        if config.concurrency > 1:
            results, texts, prompts_used, wall_clock_elapsed_s, ttft_values, tpot_values = (
                asyncio.run(_run_concurrent(backend, config, sequence))
            )
        else:
            for i in range(config.requests):
                prompt = sequence[i]
                result = backend.generate(prompt)
                results.append(
                    RequestMetrics(
                        latency_ms=result.latency_ms,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                    )
                )
                texts.append(result.text)
                prompts_used.append(prompt)
                if result.ttft_ms is not None:
                    ttft_values.append(result.ttft_ms)
                    if result.output_tokens > 0 and result.latency_ms > result.ttft_ms:
                        tpot_values.append(
                            (result.latency_ms - result.ttft_ms) / result.output_tokens
                        )

    task_qual: TaskQualityReport | None = None
    if config.quality_file is not None:
        rubrics = load_task_rubrics(config.quality_file)
        task_qual = compute_task_quality(texts, len(prompts), rubrics)

    perplexity: float | None = None
    if config.measure_perplexity:
        perplexity = backend.compute_perplexity(texts)

    judge_score: float | None = None
    if config.measure_judge:
        judge_score = backend.compute_judge_score(prompts_used, texts)

    mean_r_tokens: float | None = None
    mean_a_tokens: float | None = None
    r_fraction: float | None = None
    if config.reasoning_start_tag and config.reasoning_end_tag:
        output_token_counts = [r.output_tokens for r in results]
        mean_r_tokens, mean_a_tokens, r_fraction = reasoning_stats(
            texts,
            output_token_counts,
            config.reasoning_start_tag,
            config.reasoning_end_tag,
        )

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
        perplexity=perplexity,
        judge_score=judge_score,
        wall_clock_elapsed_s=wall_clock_elapsed_s,
        ttft_values=ttft_values or None,
        tpot_values=tpot_values or None,
        hardware=hardware,
        mean_reasoning_tokens=mean_r_tokens,
        mean_answer_tokens=mean_a_tokens,
        reasoning_fraction=r_fraction,
    )


def run_repeated(
    backend: Backend,
    config: BenchmarkConfig,
    prompts: list[str],
    model_load_ms: float | None = None,
) -> MetricsReport:
    """Run the benchmark config.repeats times and aggregate into one MetricsReport.

    When config.repeats == 1 (the default), delegates directly to run_benchmark with no
    overhead and returns a report with variance fields set to None (backward-compatible).

    When config.repeats > 1, each repeat executes the full warmup + benchmark loop with
    its own memory measurement window. model_load_ms is attached to the first repeat only
    (the backend was already constructed before run_repeated is called). The returned
    report's p50/p95/tok/s are the median across repeats; p95_latency_ms_std and
    tokens_per_second_std are sample standard deviations (n-1 denominator).
    """
    hw = detect_hardware()
    if config.repeats == 1:
        return run_benchmark(backend, config, prompts, model_load_ms=model_load_ms, hardware=hw)

    single_reports: list[MetricsReport] = []
    for i in range(config.repeats):
        r = run_benchmark(
            backend,
            config,
            prompts,
            model_load_ms=model_load_ms if i == 0 else None,
            hardware=hw,
        )
        single_reports.append(r)
    return aggregate_repeat_reports(single_reports)
