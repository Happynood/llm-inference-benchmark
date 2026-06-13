from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class MockBackendConfig(BaseModel):
    latency_ms: float = 10.0
    tokens_per_response: int = 50


class BenchmarkConfig(BaseModel):
    backend: Literal["mock"] = "mock"
    model: str = "mock-gpt2"
    requests: int = Field(default=20, ge=1)
    concurrency: int = Field(default=1, ge=1)
    prompts_file: str = "data/prompts/smoke.txt"
    warmup_requests: int = Field(default=2, ge=0)
    mock: MockBackendConfig = Field(default_factory=MockBackendConfig)


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a YAML benchmark config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig.model_validate(data or {})
