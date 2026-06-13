from __future__ import annotations

from pathlib import Path

from llm_inference_benchmark.backends.base import Backend
from llm_inference_benchmark.config import BenchmarkConfig
from llm_inference_benchmark.metrics import MetricsReport, RequestMetrics, compute_metrics


def load_prompts(path: str | Path) -> list[str]:
    """Read non-blank lines from a prompt file and return them as a list."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No prompts found in {path}")
    return lines


def run_benchmark(backend: Backend, config: BenchmarkConfig, prompts: list[str]) -> MetricsReport:
    """Run warmup + benchmark loop and return aggregated metrics."""
    for i in range(config.warmup_requests):
        backend.generate(prompts[i % len(prompts)])

    results: list[RequestMetrics] = []
    for i in range(config.requests):
        result = backend.generate(prompts[i % len(prompts)])
        results.append(
            RequestMetrics(
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        )

    return compute_metrics(results, backend=backend.name, model=config.model)
