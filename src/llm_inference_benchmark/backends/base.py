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
