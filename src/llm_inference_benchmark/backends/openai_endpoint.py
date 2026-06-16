"""OpenAI-compatible HTTP endpoint backend (/v1/chat/completions).

Works with any server that speaks the OpenAI chat-completions protocol —
llama.cpp server, Ollama, LM Studio, vLLM, and others.

No additional runtime dependency is required; the standard library handles
the HTTP request.

IMPORTANT: Reported latency includes network round-trip and any server-side
queuing overhead.  It is not directly comparable to in-process backend latency
from the transformers or llama-cpp backends.

API key handling: if api_key_env is set, the key is read from that environment
variable at call time.  The key is never logged, printed, or stored in config.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from urllib.request import urlopen

from llm_inference_benchmark.backends.base import Backend, GenerationResult


class OpenAIEndpointBackend(Backend):
    """Backend that calls a /v1/chat/completions HTTP endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        api_key_env: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._api_key_env = api_key_env

    @property
    def name(self) -> str:
        return "openai"

    def generate(self, prompt: str) -> GenerationResult:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if key:
                req.add_header("Authorization", f"Bearer {key}")

        try:
            start = time.perf_counter()
            with urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"OpenAI-compatible endpoint request failed: {exc}") from exc

        data: dict = json.loads(raw)
        text: str = data["choices"][0]["message"]["content"]

        usage = data.get("usage")
        if usage is not None:
            input_tokens: int = usage["prompt_tokens"]
            output_tokens: int = usage["completion_tokens"]
        else:
            input_tokens = len(prompt.split())
            output_tokens = len(text.split())

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )
