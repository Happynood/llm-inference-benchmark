"""ONNX Runtime backend via Hugging Face Optimum.

Optional dependency — install with:  uv sync --extra onnx

Requires a model exported to ONNX format. Two ways to obtain one:
  - Pre-exported: point 'model:' to a local directory or HF Hub ID with ONNX files.
  - Auto-export: set 'onnx.export: true' to convert a HF CausalLM model on first run
    (requires transformers + torch; takes time on the first call).

Example models with pre-built ONNX files on HF Hub:
  optimum-internal-testing/tiny-random-GPT2Model
"""

from __future__ import annotations

import os
import time

from llm_inference_benchmark.backends.base import Backend, GenerationResult


class _FirstTokenTimer:
    """Records wall-clock time when the first output token's logits are ready."""

    def __init__(self, start: float) -> None:
        self._start = start
        self.ttft_ms: float | None = None

    def __call__(self, input_ids: object, scores: object) -> object:
        if self.ttft_ms is None:
            self.ttft_ms = (time.perf_counter() - self._start) * 1000.0
        return scores


try:
    import torch  # type: ignore[import-untyped]
    from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore[import-untyped]
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    _AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    ORTModelForCausalLM = None  # type: ignore[assignment, misc]
    AutoTokenizer = None  # type: ignore[assignment, misc]
    _AVAILABLE = False


_PROVIDER_MAP: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


class OnnxBackend(Backend):
    """Inference backend using ONNX Runtime via Hugging Face Optimum."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 50,
        device: str = "cpu",
        do_sample: bool = False,
        export: bool = False,
        seed: int | None = None,
    ) -> None:
        if not _AVAILABLE:
            raise ImportError("onnx backend requires optional deps:\n  uv sync --extra onnx")

        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

        provider = _PROVIDER_MAP.get(device.split(":")[0], "CPUExecutionProvider")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = ORTModelForCausalLM.from_pretrained(
            model_id,
            export=export,
            provider=provider,
        )

        self._device = device
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample
        self._seed = seed

    @property
    def name(self) -> str:
        return "onnx"

    def generate(self, prompt: str) -> GenerationResult:
        if self._seed is not None:
            torch.manual_seed(self._seed)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_len: int = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        timer = _FirstTokenTimer(start)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                logits_processor=[timer],
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        output_len: int = output_ids.shape[1] - input_len
        text: str = self._tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)

        return GenerationResult(
            text=text,
            input_tokens=input_len,
            output_tokens=output_len,
            latency_ms=elapsed_ms,
            ttft_ms=timer.ttft_ms,
        )
