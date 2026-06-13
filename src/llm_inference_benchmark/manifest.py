"""Collect and write a reproducibility manifest for every benchmark run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_inference_benchmark.config import BenchmarkConfig


@dataclass(frozen=True)
class GpuInfo:
    """NVIDIA GPU fingerprint collected from nvidia-smi and torch.cuda.

    All fields are None when the relevant source is unavailable — the manifest
    is always written successfully even on CPU-only machines.
    """

    name: str | None
    driver_version: str | None
    cuda_version: str | None
    vram_total_mb: int | None
    torch_cuda_available: bool | None
    torch_cuda_device_name: str | None


@dataclass(frozen=True)
class RunManifest:
    timestamp: str
    backend: str
    model: str
    git_commit: str | None
    git_dirty: bool | None
    config_sha256: str
    prompts_sha256: str
    python_version: str
    platform_info: str
    cpu_model: str
    cpu_count: int | None
    package_version: str
    torch_version: str | None
    transformers_version: str | None
    psutil_version: str | None
    gpu: GpuInfo | None


def collect_manifest(config_path: str | Path, cfg: BenchmarkConfig) -> RunManifest:
    """Snapshot the full environment needed to reproduce this benchmark run."""
    return RunManifest(
        timestamp=datetime.now(UTC).isoformat(),
        backend=cfg.backend,
        model=cfg.model,
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        config_sha256=_file_sha256(config_path),
        prompts_sha256=_file_sha256(cfg.prompts_file),
        python_version=sys.version,
        platform_info=platform.platform(),
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count(),
        package_version=_pkg_version("llm-inference-benchmark") or "unknown",
        torch_version=_pkg_version("torch"),
        transformers_version=_pkg_version("transformers"),
        psutil_version=_pkg_version("psutil"),
        gpu=_collect_gpu_info(),
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> None:
    """Write manifest as pretty-printed JSON."""
    Path(path).write_text(json.dumps(asdict(manifest), indent=2) + "\n")


# ---------------------------------------------------------------------------
# Private helpers — all return None / empty on failure, never raise
# ---------------------------------------------------------------------------


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        return bool(r.stdout.strip())
    except Exception:
        return None


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _collect_gpu_info() -> GpuInfo | None:
    """Try nvidia-smi and torch.cuda; return None when neither is available.

    Takes GPU 0 only when multiple GPUs are present. All subprocess and
    import calls are guarded — this function never raises.
    """
    name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    vram_total_mb: int | None = None
    torch_cuda_available: bool | None = None
    torch_cuda_device_name: str | None = None

    # --- nvidia-smi ---
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,cuda_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            first_line = r.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 4:
                name = parts[0]
                driver_version = parts[1]
                cuda_version = parts[2]
                try:
                    vram_total_mb = int(parts[3])
                except ValueError:
                    pass
    except Exception:
        pass

    # --- torch.cuda ---
    try:
        import torch  # noqa: PLC0415

        torch_cuda_available = torch.cuda.is_available()
        if torch_cuda_available:
            torch_cuda_device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Return None only when we obtained nothing at all.
    if all(
        v is None
        for v in (
            name,
            driver_version,
            cuda_version,
            vram_total_mb,
            torch_cuda_available,
            torch_cuda_device_name,
        )
    ):
        return None

    return GpuInfo(
        name=name,
        driver_version=driver_version,
        cuda_version=cuda_version,
        vram_total_mb=vram_total_mb,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_device_name=torch_cuda_device_name,
    )
