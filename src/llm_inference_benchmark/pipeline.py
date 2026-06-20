"""Pipeline config: matrix run declarations plus optional post-processing steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llm_inference_benchmark.matrix import MatrixConfig


class PipelineSteps(BaseModel):
    """Post-processing steps that run after all matrix cells complete."""

    compare_sort: str = "p95"
    compare_filter: list[str] = Field(default_factory=list)
    compare_limit: int | None = None
    pareto: bool = False
    recommend: dict[str, Any] | None = None


class PipelineConfig(MatrixConfig):
    """Matrix config extended with an optional pipeline post-processing block."""

    pipeline: PipelineSteps = Field(default_factory=PipelineSteps)


def load_pipeline(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return PipelineConfig.model_validate(data or {})
