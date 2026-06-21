"""Energy measurement utilities for tokens-per-joule benchmarking.

Two measurement strategies, tried in order:

1. **GPU (nvidia-smi)** — polls ``power.draw`` in a background thread and
   multiplies mean wattage by elapsed seconds to get joules.  Available on
   NVIDIA GPUs with the driver tools installed.

2. **CPU (Intel RAPL)** — reads the ``energy_uj`` counter in
   ``/sys/class/powercap/intel-rapl:0/`` at the start and end of the window
   and returns the delta in joules.  Available on most Linux/Intel systems
   without root (if the kernel exposes the sysfs interface).

When neither source is readable ``energy_joules`` is ``None``.

The preferred source is GPU power; RAPL is the fallback.  On machines that
have both, GPU power more accurately reflects the inference workload.
"""

from __future__ import annotations

import subprocess
import threading
import time
import types
from pathlib import Path

_RAPL_ENERGY_PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
_RAPL_MAX_PATH = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")


def _query_gpu_power_w() -> float | None:
    """Return current GPU power draw in watts, or None if unavailable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            raw = r.stdout.strip().splitlines()[0].strip()
            if raw.lower() not in ("n/a", "[n/a]", ""):
                return float(raw)
    except Exception:
        pass
    return None


def _read_rapl_uj() -> int | None:
    """Read the Intel RAPL package energy counter in micro-joules, or None."""
    try:
        return int(_RAPL_ENERGY_PATH.read_text().strip())
    except Exception:
        return None


def _read_rapl_max_uj() -> int | None:
    """Read the maximum value of the RAPL counter before it wraps, or None."""
    try:
        return int(_RAPL_MAX_PATH.read_text().strip())
    except Exception:
        return None


class PowerSampler:
    """Context manager that measures energy consumed during the benchmark window.

    On ``__enter__``:
    - Starts a background thread that polls ``nvidia-smi power.draw`` every
      *gpu_interval_s* seconds (if a GPU is available).
    - Records the RAPL energy counter as a fallback.

    On ``__exit__``:
    - Stops the GPU polling thread and records elapsed time.
    - Records the final RAPL counter.

    ``energy_joules`` returns:
    - GPU energy (mean_power_W × duration_s) when GPU sampling succeeded.
    - RAPL delta in joules when GPU is unavailable but RAPL is readable.
    - ``None`` when neither source is available.
    """

    def __init__(self, gpu_interval_s: float = 0.5) -> None:
        self._gpu_interval = gpu_interval_s
        self._lock = threading.Lock()
        self._gpu_samples: list[float] = []
        self._gpu_available = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self._rapl_start_uj: int | None = None
        self._rapl_end_uj: int | None = None
        self._rapl_max_uj: int | None = None

    def __enter__(self) -> PowerSampler:
        self._t_start = time.perf_counter()
        # Try GPU first.
        initial = _query_gpu_power_w()
        if initial is not None:
            self._gpu_available = True
            with self._lock:
                self._gpu_samples = [initial]
            self._running = True
            self._thread = threading.Thread(target=self._poll_gpu, daemon=True)
            self._thread.start()
        # Record RAPL regardless; used as fallback.
        self._rapl_start_uj = _read_rapl_uj()
        self._rapl_max_uj = _read_rapl_max_uj()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        self._t_end = time.perf_counter()
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._rapl_end_uj = _read_rapl_uj()

    def _poll_gpu(self) -> None:
        while self._running:
            w = _query_gpu_power_w()
            if w is not None:
                with self._lock:
                    self._gpu_samples.append(w)
            time.sleep(self._gpu_interval)

    @property
    def energy_joules(self) -> float | None:
        """Total energy consumed during the context window, or None if unmeasurable."""
        duration_s = self._t_end - self._t_start
        if duration_s <= 0:
            return None

        # Prefer GPU measurement.
        if self._gpu_available:
            with self._lock:
                samples = list(self._gpu_samples)
            if samples:
                mean_w = sum(samples) / len(samples)
                return mean_w * duration_s

        # Fall back to RAPL.
        if self._rapl_start_uj is not None and self._rapl_end_uj is not None:
            delta_uj = self._rapl_end_uj - self._rapl_start_uj
            if delta_uj < 0 and self._rapl_max_uj is not None:
                # Counter wrapped around.
                delta_uj += self._rapl_max_uj
            if delta_uj >= 0:
                return delta_uj / 1_000_000.0

        return None
