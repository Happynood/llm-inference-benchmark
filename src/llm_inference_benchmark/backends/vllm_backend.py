"""vLLM backend for high-throughput GPU inference.

Optional dependency — install with:  uv sync --extra vllm

Uses vLLM's offline inference API (vllm.LLM) for direct in-process generation
without a separate HTTP server. This gives access to token-level logprobs for
perplexity computation and accurate TTFT from request metrics.

Requires a CUDA-capable GPU. CPU inference is not supported by vLLM.
"""

from __future__ import annotations

import time

from llm_inference_benchmark.backends.base import Backend, GenerationResult
from llm_inference_benchmark.perplexity import perplexity_from_nll

try:
    from vllm import LLM, SamplingParams  # type: ignore[import-untyped]

    _AVAILABLE = True
except ImportError:
    LLM = None  # type: ignore[assignment, misc]
    SamplingParams = None  # type: ignore[assignment, misc]
    _AVAILABLE = False


class VLLMBackend(Backend):
    """Inference backend using vLLM's offline LLM API."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        seed: int | None = None,
    ) -> None:
        if not _AVAILABLE:
            raise ImportError("vllm backend requires optional deps:\n  uv sync --extra vllm")

        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._seed = seed

        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed if seed is not None else 0,
        )

    @property
    def name(self) -> str:
        return "vllm"

    def generate(self, prompt: str) -> GenerationResult:
        params = SamplingParams(
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
            seed=self._seed,
        )
        start = time.perf_counter()
        outputs = self._llm.generate([prompt], params)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        output = outputs[0]
        completion = output.outputs[0]

        ttft_ms: float | None = None
        metrics = output.metrics
        if (
            metrics is not None
            and getattr(metrics, "first_token_time", None) is not None
            and getattr(metrics, "arrival_time", None) is not None
        ):
            ttft_ms = (metrics.first_token_time - metrics.arrival_time) * 1000.0

        return GenerationResult(
            text=completion.text,
            input_tokens=len(output.prompt_token_ids),
            output_tokens=len(completion.token_ids),
            latency_ms=elapsed_ms,
            ttft_ms=ttft_ms,
        )

    def compute_perplexity(self, texts: list[str]) -> float | None:
        """Corpus-level perplexity via prompt logprobs (teacher-forced, one forward pass per text).

        Texts that tokenize to fewer than 2 tokens are skipped — there is no
        next-token target to score at position 0.
        """
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        outputs = self._llm.generate(texts, params)
        total_nll = 0.0
        total_tokens = 0
        for output in outputs:
            if not output.prompt_logprobs or len(output.prompt_token_ids) < 2:
                continue
            # prompt_logprobs[0] is always None (no context for first token).
            # From index 1 onward, the dict maps token_id → Logprob for that position.
            for token_id, position_logprobs in zip(
                output.prompt_token_ids[1:], output.prompt_logprobs[1:], strict=True
            ):
                if position_logprobs is None or token_id not in position_logprobs:
                    continue
                total_nll += -position_logprobs[token_id].logprob
                total_tokens += 1
        if total_tokens == 0:
            return None
        return perplexity_from_nll(total_nll, total_tokens)
