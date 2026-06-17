from pathlib import Path
from unittest.mock import patch

import pytest

from llm_inference_benchmark.backends.mock import MockBackend
from llm_inference_benchmark.config import BenchmarkConfig
from llm_inference_benchmark.runner import load_prompts, run_benchmark


def test_load_prompts(tmp_prompts: Path) -> None:
    prompts = load_prompts(tmp_prompts)
    assert len(prompts) == 3
    assert "France" in prompts[0]


def test_load_prompts_ignores_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text("\n\nHello\n\nWorld\n\n")
    assert load_prompts(p) == ["Hello", "World"]


def test_load_prompts_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(ValueError, match="No prompts"):
        load_prompts(p)


def test_run_benchmark_request_count(tmp_prompts: Path) -> None:
    backend = MockBackend(model="test", latency_ms=0, tokens_per_response=10)
    cfg = BenchmarkConfig(requests=5, warmup_requests=1, prompts_file=str(tmp_prompts))
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.request_count == 5


def test_run_benchmark_backend_and_model(tmp_prompts: Path) -> None:
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(
        model="my-model", requests=3, warmup_requests=0, prompts_file=str(tmp_prompts)
    )
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.backend == "mock"
    assert report.model == "my-model"


def test_run_benchmark_wraps_prompts(tmp_prompts: Path) -> None:
    """Requests > len(prompts) cycles through prompts without error."""
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(requests=10, warmup_requests=0, prompts_file=str(tmp_prompts))
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.request_count == 10


def test_run_benchmark_has_cpu_memory(tmp_prompts: Path) -> None:
    backend = MockBackend(model="test", latency_ms=0, tokens_per_response=10)
    cfg = BenchmarkConfig(requests=5, warmup_requests=0, prompts_file=str(tmp_prompts))
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.peak_cpu_memory_mb > 0


def test_run_benchmark_cuda_memory_absent_without_gpu(tmp_prompts: Path) -> None:
    """On a CPU-only machine (CI), CUDA memory should be None."""
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            return  # skip assertion on real GPU machines
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(requests=3, warmup_requests=0, prompts_file=str(tmp_prompts))
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.peak_cuda_memory_mb is None


def test_run_benchmark_vram_none_without_nvidia_smi(tmp_prompts: Path) -> None:
    """peak_vram_memory_mb is None when nvidia-smi is not available."""
    with patch("llm_inference_benchmark.memory.subprocess.run", side_effect=FileNotFoundError):
        backend = MockBackend(model="test", latency_ms=0)
        cfg = BenchmarkConfig(requests=3, warmup_requests=0, prompts_file=str(tmp_prompts))
        report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.peak_vram_memory_mb is None


def test_run_benchmark_quality_sanity_pass_rate_full(tmp_prompts: Path) -> None:
    """MockBackend always returns non-empty text, so sanity_pass_rate should be 1.0."""
    backend = MockBackend(model="test", latency_ms=0, tokens_per_response=5)
    cfg = BenchmarkConfig(requests=5, warmup_requests=1, prompts_file=str(tmp_prompts))
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.sanity_pass_rate == pytest.approx(1.0)
    assert report.empty_output_count == 0


def test_run_benchmark_quality_fields_populated(tmp_prompts: Path) -> None:
    """After a benchmark run, all quality fields must be present."""
    backend = MockBackend(model="test", latency_ms=0, tokens_per_response=3)
    cfg = BenchmarkConfig(requests=4, warmup_requests=0, prompts_file=str(tmp_prompts))
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.min_output_chars > 0  # mock produces non-empty text
    assert report.mean_output_chars > 0.0
    assert report.repeated_output_count >= 0


# ---------------------------------------------------------------------------
# Perplexity wiring (v0.20)
# ---------------------------------------------------------------------------


class _PerplexityBackend(MockBackend):
    """MockBackend that reports a fixed perplexity, for wiring tests only."""

    def compute_perplexity(self, texts: list[str]) -> float | None:
        return 7.5


def test_run_benchmark_perplexity_none_when_not_measured(tmp_prompts: Path) -> None:
    backend = _PerplexityBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(requests=3, warmup_requests=0, prompts_file=str(tmp_prompts))
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.perplexity is None


def test_run_benchmark_perplexity_populated_when_measured(tmp_prompts: Path) -> None:
    backend = _PerplexityBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(
        requests=3, warmup_requests=0, prompts_file=str(tmp_prompts), measure_perplexity=True
    )
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.perplexity == pytest.approx(7.5)


def test_run_benchmark_perplexity_none_for_backend_without_support(tmp_prompts: Path) -> None:
    """Backends that return None from compute_perplexity propagate None even when measured."""
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(
        requests=3, warmup_requests=0, prompts_file=str(tmp_prompts), measure_perplexity=True
    )
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.perplexity is None


# ---------------------------------------------------------------------------
# Judge score wiring (v0.21)
# ---------------------------------------------------------------------------


class _JudgeBackend(MockBackend):
    """MockBackend that reports a fixed judge score, for wiring tests only."""

    def compute_judge_score(self, prompts: list[str], texts: list[str]) -> float | None:
        return 0.9


def test_run_benchmark_judge_score_none_when_not_measured(tmp_prompts: Path) -> None:
    backend = _JudgeBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(requests=3, warmup_requests=0, prompts_file=str(tmp_prompts))
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.judge_score is None


def test_run_benchmark_judge_score_populated_when_measured(tmp_prompts: Path) -> None:
    backend = _JudgeBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(
        requests=3, warmup_requests=0, prompts_file=str(tmp_prompts), measure_judge=True
    )
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.judge_score == pytest.approx(0.9)


def test_run_benchmark_judge_score_none_for_backend_without_support(tmp_prompts: Path) -> None:
    """Backends that return None from compute_judge_score propagate None even when measured."""
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(
        requests=3, warmup_requests=0, prompts_file=str(tmp_prompts), measure_judge=True
    )
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.judge_score is None


def test_run_benchmark_concurrent_request_count(tmp_prompts: Path) -> None:
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(
        requests=6, concurrency=3, warmup_requests=0, prompts_file=str(tmp_prompts)
    )
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    assert report.request_count == 6


def test_run_benchmark_concurrent_throughput_uses_wall_clock(tmp_prompts: Path) -> None:
    """With concurrency > 1 throughput is total_output_tokens / wall_clock, not sum-of-latencies."""
    import time

    backend = MockBackend(model="test", latency_ms=50, tokens_per_response=10)
    cfg = BenchmarkConfig(
        requests=4, concurrency=4, warmup_requests=0, prompts_file=str(tmp_prompts)
    )
    wall_t0 = time.perf_counter()
    report = run_benchmark(backend, cfg, load_prompts(tmp_prompts))
    _ = time.perf_counter() - wall_t0

    # Sequential sum-of-latencies would be 4 * 50 ms = 200 ms → throughput = 200 tok/s.
    # With concurrency=4 all requests fire at once; wall clock is ~50 ms → throughput
    # should be considerably higher than the sequential baseline.
    sequential_tps = (4 * 10) / (4 * 0.05)  # 200 tok/s
    assert report.tokens_per_second > sequential_tps * 1.5
