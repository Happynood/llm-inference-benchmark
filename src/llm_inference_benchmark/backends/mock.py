from __future__ import annotations

import time

from llm_inference_benchmark.backends.base import Backend, GenerationResult


class MockBackend(Backend):
    """Deterministic mock backend for CI and architecture validation."""

    def __init__(
        self,
        model: str,
        latency_ms: float = 10.0,
        tokens_per_response: int = 50,
        ttft_ms: float | None = None,
        seed: int | None = None,  # accepted for config compatibility; mock output is deterministic
    ) -> None:
        self._model = model
        self._latency_s = latency_ms / 1000.0
        self._tokens_per_response = tokens_per_response
        self._ttft_ms = ttft_ms

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str) -> GenerationResult:
        start = time.perf_counter()
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        input_tokens = len(prompt.split())
        return GenerationResult(
            text="mock " * self._tokens_per_response,
            input_tokens=input_tokens,
            output_tokens=self._tokens_per_response,
            latency_ms=elapsed_ms,
            ttft_ms=self._ttft_ms,
        )
