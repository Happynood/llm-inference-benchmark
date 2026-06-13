"""Tests for GPU fingerprint collection in manifest.py.

All GPU-dependent paths are mocked so tests pass on CPU-only CI machines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.config import load_config
from llm_inference_benchmark.manifest import (
    GpuInfo,
    _collect_gpu_info,
    collect_manifest,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NVIDIA_SMI_OUTPUT = "NVIDIA GeForce RTX 3050 Laptop GPU, 535.183.01, 12.2, 4096\n"

_MOCK_SUBPROCESS_OK = MagicMock(returncode=0, stdout=_NVIDIA_SMI_OUTPUT)
_MOCK_SUBPROCESS_FAIL = MagicMock(returncode=1, stdout="")


def _make_torch_mock(cuda_available: bool = True) -> MagicMock:
    m = MagicMock()
    m.cuda.is_available.return_value = cuda_available
    m.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3050 Laptop GPU"
    return m


# ---------------------------------------------------------------------------
# GpuInfo dataclass
# ---------------------------------------------------------------------------


def test_gpu_info_all_none_fields() -> None:
    gpu = GpuInfo(
        name=None,
        driver_version=None,
        cuda_version=None,
        vram_total_mb=None,
        torch_cuda_available=None,
        torch_cuda_device_name=None,
    )
    assert gpu.name is None
    assert gpu.vram_total_mb is None


def test_gpu_info_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    gpu = GpuInfo(
        name="RTX 3050",
        driver_version="535.0",
        cuda_version="12.2",
        vram_total_mb=4096,
        torch_cuda_available=True,
        torch_cuda_device_name="RTX 3050",
    )
    with pytest.raises(FrozenInstanceError):
        gpu.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# nvidia-smi parsing
# ---------------------------------------------------------------------------


def test_collect_gpu_info_nvidia_smi_success() -> None:
    with patch("llm_inference_benchmark.manifest.subprocess.run", return_value=_MOCK_SUBPROCESS_OK):
        with patch.dict(sys.modules, {"torch": None}):
            gpu = _collect_gpu_info()
    assert gpu is not None
    assert gpu.name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert gpu.driver_version == "535.183.01"
    assert gpu.cuda_version == "12.2"
    assert gpu.vram_total_mb == 4096


def test_collect_gpu_info_nvidia_smi_nonzero_return() -> None:
    with patch(
        "llm_inference_benchmark.manifest.subprocess.run", return_value=_MOCK_SUBPROCESS_FAIL
    ):
        with patch.dict(sys.modules, {"torch": None}):
            gpu = _collect_gpu_info()
    # nvidia-smi failed, torch absent → no GPU info at all
    assert gpu is None


def test_collect_gpu_info_nvidia_smi_not_found() -> None:
    with patch("llm_inference_benchmark.manifest.subprocess.run", side_effect=FileNotFoundError):
        with patch.dict(sys.modules, {"torch": None}):
            gpu = _collect_gpu_info()
    assert gpu is None


def test_collect_gpu_info_bad_vram_value() -> None:
    bad_output = MagicMock(returncode=0, stdout="RTX 3050, 535.0, 12.2, [N/A]\n")
    with patch("llm_inference_benchmark.manifest.subprocess.run", return_value=bad_output):
        with patch.dict(sys.modules, {"torch": None}):
            gpu = _collect_gpu_info()
    assert gpu is not None
    assert gpu.vram_total_mb is None  # failed int() parse
    assert gpu.name == "RTX 3050"


# ---------------------------------------------------------------------------
# torch.cuda paths
# ---------------------------------------------------------------------------


def test_collect_gpu_info_torch_cuda_available() -> None:
    mock_torch = _make_torch_mock(cuda_available=True)
    with patch("llm_inference_benchmark.manifest.subprocess.run", side_effect=FileNotFoundError):
        with patch.dict(sys.modules, {"torch": mock_torch}):
            gpu = _collect_gpu_info()
    assert gpu is not None
    assert gpu.torch_cuda_available is True
    assert gpu.torch_cuda_device_name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    # nvidia-smi unavailable → smi fields are None
    assert gpu.name is None


def test_collect_gpu_info_torch_cuda_not_available() -> None:
    mock_torch = _make_torch_mock(cuda_available=False)
    with patch("llm_inference_benchmark.manifest.subprocess.run", side_effect=FileNotFoundError):
        with patch.dict(sys.modules, {"torch": mock_torch}):
            gpu = _collect_gpu_info()
    assert gpu is not None
    assert gpu.torch_cuda_available is False
    assert gpu.torch_cuda_device_name is None  # not called when cuda not available


def test_collect_gpu_info_both_sources_available() -> None:
    mock_torch = _make_torch_mock(cuda_available=True)
    with patch("llm_inference_benchmark.manifest.subprocess.run", return_value=_MOCK_SUBPROCESS_OK):
        with patch.dict(sys.modules, {"torch": mock_torch}):
            gpu = _collect_gpu_info()
    assert gpu is not None
    assert gpu.name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert gpu.torch_cuda_available is True


def test_collect_gpu_info_all_unavailable_returns_none() -> None:
    with patch("llm_inference_benchmark.manifest.subprocess.run", side_effect=FileNotFoundError):
        with patch.dict(sys.modules, {"torch": None}):
            gpu = _collect_gpu_info()
    assert gpu is None


# ---------------------------------------------------------------------------
# RunManifest integration
# ---------------------------------------------------------------------------


def test_collect_manifest_has_gpu_field(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    m = collect_manifest(tmp_config, cfg)
    assert hasattr(m, "gpu")
    # gpu is either None or a GpuInfo instance
    assert m.gpu is None or isinstance(m.gpu, GpuInfo)


def test_write_manifest_gpu_none_serializes(tmp_config: Path, tmp_path: Path) -> None:
    from dataclasses import replace

    cfg = load_config(tmp_config)
    m = replace(collect_manifest(tmp_config, cfg), gpu=None)
    out = tmp_path / "manifest.json"
    write_manifest(m, out)
    data = json.loads(out.read_text())
    assert "gpu" in data
    assert data["gpu"] is None


def test_write_manifest_gpu_present_serializes(tmp_config: Path, tmp_path: Path) -> None:
    from dataclasses import replace

    cfg = load_config(tmp_config)
    gpu = GpuInfo(
        name="RTX 3050",
        driver_version="535.0",
        cuda_version="12.2",
        vram_total_mb=4096,
        torch_cuda_available=True,
        torch_cuda_device_name="RTX 3050",
    )
    m = replace(collect_manifest(tmp_config, cfg), gpu=gpu)
    out = tmp_path / "manifest.json"
    write_manifest(m, out)
    data = json.loads(out.read_text())
    assert isinstance(data["gpu"], dict)
    assert data["gpu"]["name"] == "RTX 3050"
    assert data["gpu"]["vram_total_mb"] == 4096
    assert data["gpu"]["torch_cuda_available"] is True


def test_cli_manifest_json_has_gpu_key(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--manifest", str(out)])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert "gpu" in data
