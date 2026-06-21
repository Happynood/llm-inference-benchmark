"""Tests for the energy measurement (power) module."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_inference_benchmark.power import PowerSampler, _query_gpu_power_w, _read_rapl_uj

# ── _query_gpu_power_w ────────────────────────────────────────────────────────


def test_query_gpu_power_returns_float_on_success() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "85.50\n"
    with patch("subprocess.run", return_value=mock_result):
        result = _query_gpu_power_w()
    assert result == pytest.approx(85.5)


def test_query_gpu_power_returns_none_on_nonzero_returncode() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    with patch("subprocess.run", return_value=mock_result):
        result = _query_gpu_power_w()
    assert result is None


def test_query_gpu_power_returns_none_on_na_output() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "N/A\n"
    with patch("subprocess.run", return_value=mock_result):
        result = _query_gpu_power_w()
    assert result is None


def test_query_gpu_power_returns_none_on_exception() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        result = _query_gpu_power_w()
    assert result is None


def test_query_gpu_power_returns_none_on_timeout() -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
        result = _query_gpu_power_w()
    assert result is None


# ── _read_rapl_uj ─────────────────────────────────────────────────────────────


def test_read_rapl_uj_parses_file(tmp_path: Path) -> None:
    energy_file = tmp_path / "energy_uj"
    energy_file.write_text("1234567890\n")
    with patch("llm_inference_benchmark.power._RAPL_ENERGY_PATH", energy_file):
        result = _read_rapl_uj()
    assert result == 1234567890


def test_read_rapl_uj_returns_none_when_file_missing() -> None:
    with patch(
        "llm_inference_benchmark.power._RAPL_ENERGY_PATH",
        Path("/nonexistent/energy_uj"),
    ):
        result = _read_rapl_uj()
    assert result is None


# ── PowerSampler — GPU path ───────────────────────────────────────────────────


def test_power_sampler_gpu_energy_calculation() -> None:
    """GPU: mean power * duration = energy."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "100.0\n"

    with patch("subprocess.run", return_value=mock_result):
        with patch("llm_inference_benchmark.power._read_rapl_uj", return_value=None):
            sampler = PowerSampler(gpu_interval_s=0.01)
            with sampler:
                time.sleep(0.05)

    energy = sampler.energy_joules
    assert energy is not None
    assert energy > 0


def test_power_sampler_gpu_not_available_falls_back_to_rapl() -> None:
    call_count = 0

    def fake_read_rapl() -> int | None:
        nonlocal call_count
        call_count += 1
        return 1_000_000 if call_count == 1 else 3_000_000

    with patch("subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
        with patch("llm_inference_benchmark.power._read_rapl_uj", side_effect=fake_read_rapl):
            with patch("llm_inference_benchmark.power._read_rapl_max_uj", return_value=None):
                sampler = PowerSampler()
                with sampler:
                    time.sleep(0.01)

    energy = sampler.energy_joules
    assert energy is not None
    # delta = 2_000_000 µJ = 2 J
    assert energy == pytest.approx(2.0)


def test_power_sampler_rapl_counter_wrap(tmp_path: Path) -> None:
    """RAPL counter wraps: end < start but max_energy_range_uj handles it."""
    call_count = 0
    max_uj = 10_000_000

    def fake_read_rapl() -> int | None:
        nonlocal call_count
        call_count += 1
        return 9_500_000 if call_count == 1 else 500_000  # wrapped

    with patch("subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
        with patch("llm_inference_benchmark.power._read_rapl_uj", side_effect=fake_read_rapl):
            with patch("llm_inference_benchmark.power._read_rapl_max_uj", return_value=max_uj):
                sampler = PowerSampler()
                with sampler:
                    time.sleep(0.01)

    energy = sampler.energy_joules
    assert energy is not None
    # delta = 500_000 - 9_500_000 + 10_000_000 = 1_000_000 µJ = 1 J
    assert energy == pytest.approx(1.0)


def test_power_sampler_returns_none_when_neither_source_available() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
        with patch("llm_inference_benchmark.power._read_rapl_uj", return_value=None):
            with patch("llm_inference_benchmark.power._read_rapl_max_uj", return_value=None):
                sampler = PowerSampler()
                with sampler:
                    time.sleep(0.01)

    assert sampler.energy_joules is None


def test_power_sampler_is_context_manager_safe_on_exception() -> None:
    """Sampler __exit__ must run even when the body raises."""
    with patch("subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
        with patch("llm_inference_benchmark.power._read_rapl_uj", return_value=None):
            sampler = PowerSampler()
            try:
                with sampler:
                    raise RuntimeError("benchmark failed")
            except RuntimeError:
                pass

    # No assertion — just verifying no secondary exception from __exit__.
    assert sampler.energy_joules is None
