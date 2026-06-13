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
