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
