"""Peak memory measurement utilities.

CPU peak: background thread polls psutil RSS every 50 ms.
CUDA peak: torch.cuda.max_memory_allocated(), lazily imported, None when unavailable.
"""

from __future__ import annotations

import os
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
