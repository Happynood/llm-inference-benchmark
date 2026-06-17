from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float | None = None


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def compute_perplexity(self, texts: list[str]) -> float | None:
        """Return corpus-level perplexity of texts under this backend's model.

        Default: None (backend has no token-level log-probability access).
        Backends that expose logits (e.g. transformers) override this.
        """
        return None

    def compute_judge_score(self, prompts: list[str], texts: list[str]) -> float | None:
        """Return a logprob-based self-judge score for (prompt, completion) pairs.

        Default: None (backend has no token-level logit access).
        Backends that expose logits (e.g. transformers) override this.
        """
        return None
