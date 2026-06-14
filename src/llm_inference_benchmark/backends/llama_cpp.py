"""llama-cpp-python backend (GGUF quantized inference).

Optional dependency — install with:  uv sync --extra llama-cpp

For GPU (CUDA) support, build with CUDA enabled:
    CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp

This backend requires a local GGUF model file. No download is performed here.
Obtain a model separately (e.g. from Hugging Face) and set 'model:' to the local path:
    model: /path/to/llama-3-8b-q4_k_m.gguf
"""

from __future__ import annotations

import time

from llm_inference_benchmark.backends.base import Backend, GenerationResult

try:
    from llama_cpp import Llama  # type: ignore[import-untyped]

    _AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[assignment, misc]
    _AVAILABLE = False


class LlamaCppBackend(Backend):
    """Inference backend using llama-cpp-python (GGUF quantized models)."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        max_tokens: int = 50,
        temperature: float = 0.0,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        if not _AVAILABLE or Llama is None:
            raise ImportError(
                "llama-cpp backend requires optional deps:\n"
                "  uv sync --extra llama-cpp\n"
                "For GPU (CUDA):\n"
                '  CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp'
            )

        kwargs: dict[str, object] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads

        self._llm = Llama(**kwargs)  # type: ignore[misc]
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def name(self) -> str:
        return "llama-cpp"

    def generate(self, prompt: str) -> GenerationResult:
        start = time.perf_counter()
        output = self._llm(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            echo=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        text: str = output["choices"][0]["text"]
        input_tokens: int = output["usage"]["prompt_tokens"]
        output_tokens: int = output["usage"]["completion_tokens"]

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )
