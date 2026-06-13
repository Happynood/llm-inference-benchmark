"""Unit tests for memory measurement utilities — CPU only, no CUDA required."""

from __future__ import annotations

import time

from llm_inference_benchmark.memory import MemorySampler, cuda_peak_mb, reset_cuda_peak


def test_sampler_peak_is_positive() -> None:
    with MemorySampler(interval_s=0.01) as m:
        _ = bytearray(10 * 1024 * 1024)  # 10 MB to ensure RSS delta is visible
        time.sleep(0.05)  # let at least one poll cycle complete
    assert m.peak_cpu_mb > 0


def test_sampler_peak_is_float() -> None:
    with MemorySampler(interval_s=0.01) as m:
        pass
    assert isinstance(m.peak_cpu_mb, float)


def test_sampler_stops_cleanly_on_exception() -> None:
    """MemorySampler exits cleanly even when the body raises."""
    try:
        with MemorySampler(interval_s=0.01) as m:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert isinstance(m.peak_cpu_mb, float)


def test_sampler_peak_non_decreasing() -> None:
    """Peak reported after exit must be >= peak read immediately after enter."""
    with MemorySampler(interval_s=0.01) as m:
        initial_mb = m.peak_cpu_mb
        time.sleep(0.03)
    assert m.peak_cpu_mb >= initial_mb


def test_cuda_peak_mb_none_or_float() -> None:
    """Returns None (no GPU) or a non-negative float (GPU present)."""
    result = cuda_peak_mb()
    assert result is None or (isinstance(result, float) and result >= 0)


def test_reset_cuda_peak_no_crash() -> None:
    """reset_cuda_peak() must not raise regardless of CUDA availability."""
    reset_cuda_peak()  # should be a no-op on CPU-only machines
