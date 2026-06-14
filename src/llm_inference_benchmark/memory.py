"""Peak memory measurement utilities.

CPU peak: background thread polls psutil RSS every 50 ms.
CUDA peak: torch.cuda.max_memory_allocated(), lazily imported, None when unavailable.
VRAM peak: background thread polls nvidia-smi every 500 ms, None when unavailable.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import types

import psutil


class MemorySampler:
    """Context manager that polls process RSS to capture peak CPU memory.

    Only covers the body of the `with` block. Start it *after* warmup so
    first-use allocation spikes (KV-cache, JIT compile) are excluded.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self._proc = psutil.Process(os.getpid())
        self._interval = interval_s
        self._lock = threading.Lock()
        self._peak_bytes: int = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> MemorySampler:
        with self._lock:
            self._peak_bytes = self._proc.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _poll(self) -> None:
        while self._running:
            rss = self._proc.memory_info().rss
            with self._lock:
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            time.sleep(self._interval)

    @property
    def peak_cpu_mb(self) -> float:
        with self._lock:
            return self._peak_bytes / (1024 * 1024)


def _query_vram_mib() -> float | None:
    """Query current GPU VRAM usage from nvidia-smi. Returns None if unavailable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


class NvidiaSmiSampler:
    """Context manager that polls nvidia-smi for driver-level peak VRAM usage.

    Polls every interval_s seconds in a background daemon thread.  Returns None
    when nvidia-smi is absent or the first probe fails — CI-safe on machines
    without a GPU or the nvidia driver tools.
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self._interval = interval_s
        self._lock = threading.Lock()
        self._peak_mib: float = 0.0
        self._available = False
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> NvidiaSmiSampler:
        initial = _query_vram_mib()
        if initial is not None:
            self._available = True
            with self._lock:
                self._peak_mib = initial
            self._running = True
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll(self) -> None:
        while self._running:
            mib = _query_vram_mib()
            if mib is not None:
                with self._lock:
                    if mib > self._peak_mib:
                        self._peak_mib = mib
            time.sleep(self._interval)

    @property
    def peak_vram_mb(self) -> float | None:
        """Peak VRAM usage in MiB observed during the context, or None if unavailable."""
        if not self._available:
            return None
        with self._lock:
            return self._peak_mib


def reset_cuda_peak() -> None:
    """Reset CUDA peak-memory counter before a benchmark run. No-op when CUDA absent."""
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def cuda_peak_mb() -> float | None:
    """Return peak CUDA memory allocated (MB) since last reset, or None if unavailable."""
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None
