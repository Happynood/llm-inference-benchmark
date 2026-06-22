from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal


@dataclass
class BackendProbe:
    backend: str
    status: Literal["OK", "SKIP", "FAIL"]
    latency_ms: float | None
    reason: str


def _probe_mock() -> BackendProbe:
    from llm_inference_benchmark.backends.mock import MockBackend

    backend = MockBackend(model="verify", latency_ms=1, tokens_per_response=5)
    try:
        t0 = time.perf_counter()
        backend.generate("Hello")
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return BackendProbe(backend="mock", status="OK", latency_ms=latency_ms, reason="")
    except Exception as exc:  # noqa: BLE001
        return BackendProbe(backend="mock", status="FAIL", latency_ms=None, reason=str(exc))


_INSTALL_HINT: dict[str, str] = {
    "transformers": "uv sync --extra transformers",
    "llama-cpp": "uv sync --extra llama-cpp",
    "onnx": "uv sync --extra onnx",
    "vllm": "uv sync --extra vllm",
}


def _probe_import(backend_name: str, modules: list[str]) -> BackendProbe:
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            hint = _INSTALL_HINT.get(backend_name, "")
            reason = f"missing: {mod}" + (f" — install: {hint}" if hint else "")
            return BackendProbe(
                backend=backend_name,
                status="SKIP",
                latency_ms=None,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001
            return BackendProbe(
                backend=backend_name,
                status="FAIL",
                latency_ms=None,
                reason=str(exc),
            )
    return BackendProbe(backend=backend_name, status="OK", latency_ms=None, reason="deps installed")


def run_probes() -> list[BackendProbe]:
    """Return one probe result per backend in a fixed order."""
    return [
        _probe_mock(),
        _probe_import("transformers", ["transformers", "torch"]),
        _probe_import("llama-cpp", ["llama_cpp"]),
        BackendProbe(backend="openai", status="OK", latency_ms=None, reason="stdlib only"),
        _probe_import("onnx", ["optimum", "onnxruntime"]),
        _probe_import("vllm", ["vllm"]),
    ]
