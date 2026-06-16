from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class MockBackendConfig(BaseModel):
    latency_ms: float = 10.0
    tokens_per_response: int = 50


class HFBackendConfig(BaseModel):
    max_new_tokens: int = Field(default=50, ge=1)
    device: str = "cpu"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    do_sample: bool = False


class LlamaCppBackendConfig(BaseModel):
    n_ctx: int = Field(default=2048, ge=1)
    n_gpu_layers: int = 0
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    n_threads: int | None = None
    verbose: bool = False


class BenchmarkConfig(BaseModel):
    backend: Literal["mock", "transformers", "llama-cpp"] = "mock"
    model: str = "mock-gpt2"
    requests: int = Field(default=20, ge=1)
    concurrency: int = Field(default=1, ge=1)
    prompts_file: str = "data/prompts/smoke.txt"
    quality_file: str | None = None
    workload_profile: str | None = None
    warmup_requests: int = Field(default=2, ge=0)
    repeats: int = Field(default=1, ge=1)
    measure_perplexity: bool = False
    mock: MockBackendConfig = Field(default_factory=MockBackendConfig)
    hf: HFBackendConfig = Field(default_factory=HFBackendConfig)
    llama_cpp: LlamaCppBackendConfig = Field(default_factory=LlamaCppBackendConfig)

    @model_validator(mode="after")
    def _validate_workload_profile(self) -> BenchmarkConfig:
        if self.workload_profile is not None:
            from llm_inference_benchmark.profiles import get_profile

            get_profile(self.workload_profile)
        return self

    def resolve_prompts_file(self) -> str:
        """Return the effective prompts file path.

        When workload_profile is set the profile's path takes precedence.
        Otherwise prompts_file is used directly (backward-compatible default).
        """
        if self.workload_profile is not None:
            from llm_inference_benchmark.profiles import get_profile

            return get_profile(self.workload_profile).prompts_file
        return self.prompts_file


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a YAML benchmark config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig.model_validate(data or {})
