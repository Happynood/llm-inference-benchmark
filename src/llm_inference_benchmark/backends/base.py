from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


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
