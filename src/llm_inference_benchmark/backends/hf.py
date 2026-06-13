"""HuggingFace Transformers backend (AutoModelForCausalLM).

Optional dependency — install with:  uv sync --extra transformers
First run downloads the model from HuggingFace Hub into ~/.cache/huggingface/.
"""

from __future__ import annotations

import os
import time

from llm_inference_benchmark.backends.base import Backend, GenerationResult

try:
    import torch  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class HFBackend(Backend):
    """Inference backend using HuggingFace Transformers AutoModelForCausalLM."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 50,
        device: str = "cpu",
        torch_dtype: str = "float32",
        do_sample: bool = False,
    ) -> None:
        if not _AVAILABLE:
            raise ImportError(
                "transformers backend requires optional deps:\n  uv sync --extra transformers"
            )

        # httpx (used by huggingface_hub) rejects the bare socks:// proxy scheme.
        # Clear ALL_PROXY/all_proxy so HF Hub falls through to http_proxy/https_proxy.
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(torch_dtype, torch.float32)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        # GPT-2 and similar models have no dedicated pad token; fall back to eos.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self._model.eval()

        self._device = device
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample

    @property
    def name(self) -> str:
        return "transformers"

    def generate(self, prompt: str) -> GenerationResult:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        input_len: int = inputs["input_ids"].shape[1]

        with torch.no_grad():
            start = time.perf_counter()
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

        output_len: int = output_ids.shape[1] - input_len
        text: str = self._tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)

        return GenerationResult(
            text=text,
            input_tokens=input_len,
            output_tokens=output_len,
            latency_ms=elapsed_ms,
        )
