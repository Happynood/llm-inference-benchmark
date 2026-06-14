"""Unit tests for memory measurement utilities — CPU only, no CUDA required."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from llm_inference_benchmark.memory import (
    MemorySampler,
    NvidiaSmiSampler,
    _query_vram_mib,
    cuda_peak_mb,
    reset_cuda_peak,
)


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


# ---------------------------------------------------------------------------
# NvidiaSmiSampler
# ---------------------------------------------------------------------------


def test_nvidia_smi_sampler_returns_none_when_unavailable() -> None:
    """peak_vram_mb is None when nvidia-smi cannot be found."""
    with patch("llm_inference_benchmark.memory.subprocess.run", side_effect=FileNotFoundError):
        with NvidiaSmiSampler() as s:
            pass
    assert s.peak_vram_mb is None


def test_nvidia_smi_sampler_returns_float_when_available() -> None:
    """peak_vram_mb is a non-negative float when nvidia-smi succeeds."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "1234\n"
    with patch("llm_inference_benchmark.memory.subprocess.run", return_value=mock_result):
        with NvidiaSmiSampler(interval_s=0.01) as s:
            time.sleep(0.05)
    assert isinstance(s.peak_vram_mb, float)
    assert s.peak_vram_mb == pytest.approx(1234.0)


def test_nvidia_smi_sampler_peak_non_decreasing() -> None:
    """Peak VRAM must not decrease once observed."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "500\n"
    with patch("llm_inference_benchmark.memory.subprocess.run", return_value=mock_result):
        with NvidiaSmiSampler(interval_s=0.01) as s:
            initial = s.peak_vram_mb
            time.sleep(0.05)
    assert s.peak_vram_mb is not None
    assert s.peak_vram_mb >= (initial or 0)


def test_nvidia_smi_sampler_exits_cleanly_on_exception() -> None:
    """NvidiaSmiSampler __exit__ is called even when the body raises."""
    with patch("llm_inference_benchmark.memory.subprocess.run", side_effect=FileNotFoundError):
        try:
            with NvidiaSmiSampler() as s:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    assert s.peak_vram_mb is None


def test_nvidia_smi_sampler_exits_cleanly_with_gpu() -> None:
    """NvidiaSmiSampler __exit__ stops the thread cleanly when GPU is present."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "2048\n"
    with patch("llm_inference_benchmark.memory.subprocess.run", return_value=mock_result):
        with NvidiaSmiSampler(interval_s=0.01) as s:
            pass
    assert s.peak_vram_mb == pytest.approx(2048.0)


# ---------------------------------------------------------------------------
# _query_vram_mib
# ---------------------------------------------------------------------------


def test_query_vram_mib_returns_none_on_file_not_found() -> None:
    with patch("llm_inference_benchmark.memory.subprocess.run", side_effect=FileNotFoundError):
        assert _query_vram_mib() is None


def test_query_vram_mib_returns_none_on_nonzero_returncode() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    with patch("llm_inference_benchmark.memory.subprocess.run", return_value=mock_result):
        assert _query_vram_mib() is None


def test_query_vram_mib_parses_integer_output() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "2048\n"
    with patch("llm_inference_benchmark.memory.subprocess.run", return_value=mock_result):
        assert _query_vram_mib() == pytest.approx(2048.0)


def test_query_vram_mib_returns_none_on_exception() -> None:
    with patch("llm_inference_benchmark.memory.subprocess.run", side_effect=OSError("no device")):
        assert _query_vram_mib() is None
