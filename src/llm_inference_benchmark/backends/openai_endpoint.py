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
        stream: bool = False,
        seed: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._api_key_env = api_key_env
        self._stream = stream
        self._seed = seed

    @property
    def name(self) -> str:
        return "openai"

    def _make_request(self, payload: dict) -> urllib.request.Request:
        url = f"{self._base_url}/chat/completions"
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
        return req

    def generate(self, prompt: str) -> GenerationResult:
        if self._stream:
            return self._generate_streaming(prompt)
        return self._generate_blocking(prompt)

    def _generate_blocking(self, prompt: str) -> GenerationResult:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._seed is not None:
            payload["seed"] = self._seed  # advisory: server may ignore
        req = self._make_request(payload)

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

    def _generate_streaming(self, prompt: str) -> GenerationResult:
        """Send a streaming request and record time-to-first-token (TTFT).

        Parses the server-sent events (SSE) stream from /v1/chat/completions.
        ttft_ms is the wall-clock time from request start to the first non-empty
        content delta received from the server.
        """
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": True,
        }
        if self._seed is not None:
            payload["seed"] = self._seed  # advisory: server may ignore
        req = self._make_request(payload)

        try:
            start = time.perf_counter()
            with urlopen(req, timeout=self._timeout_s) as resp:
                ttft_ms: float | None = None
                content_parts: list[str] = []
                usage: dict | None = None
                token_times: list[float] = []

                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].lstrip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk: dict = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        delta_content: str = choices[0].get("delta", {}).get("content") or ""
                        if delta_content:
                            t = time.perf_counter()
                            token_times.append(t)
                            if ttft_ms is None:
                                ttft_ms = (t - start) * 1000.0
                        content_parts.append(delta_content)
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage = chunk_usage

            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"OpenAI-compatible endpoint request failed: {exc}") from exc

        text = "".join(content_parts)

        if usage is not None:
            input_tokens: int = usage["prompt_tokens"]
            output_tokens: int = usage["completion_tokens"]
        else:
            input_tokens = len(prompt.split())
            output_tokens = len(text.split())

        itl_values = (
            [(token_times[i] - token_times[i - 1]) * 1000.0 for i in range(1, len(token_times))]
            if len(token_times) >= 2
            else None
        )

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            itl_values=itl_values,
        )
