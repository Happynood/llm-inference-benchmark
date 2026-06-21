"""Compact hardware profile — CPU, RAM, GPU — attached to every benchmark result."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HardwareProfile:
    cpu: str
    cpu_cores: int | None
    ram_gb: float
    gpu: str | None
    vram_gb: float | None
    os: str


def detect() -> HardwareProfile:
    """Return a hardware snapshot. Never raises — all fields fall back gracefully."""
    gpu_name, vram_gb = _gpu_info()
    return HardwareProfile(
        cpu=_cpu_model(),
        cpu_cores=_cpu_cores(),
        ram_gb=_ram_gb(),
        gpu=gpu_name,
        vram_gb=vram_gb,
        os=platform.platform(),
    )


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _cpu_cores() -> int | None:
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        return int(n) if n is not None else None
    except Exception:
        return None


def _ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        return 0.0


def _gpu_info() -> tuple[str | None, float | None]:
    """Return (gpu_name, vram_gb) from nvidia-smi in a single call, or (None, None)."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 2:
                name = parts[0] or None
                try:
                    vram_gb = round(int(parts[1]) / 1024, 1)
                except ValueError:
                    vram_gb = None
                return name, vram_gb
    except Exception:
        pass
    return None, None
