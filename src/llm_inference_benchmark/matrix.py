"""Matrix config for running multiple benchmark experiments sequentially."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Run names become file stems: require alphanumeric start, allow .-_ only,
# no path separators or parent-traversal sequences.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MatrixRunConfig(BaseModel):
    """One run entry in a benchmark matrix."""

    name: str
    config: str
    workload_profile: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                f"Run name {v!r} is invalid. Names must start with a letter or digit and "
                "contain only letters, digits, dots, underscores, and hyphens "
                "(no path separators or special characters)."
            )
        return v

    @model_validator(mode="after")
    def _validate_profile(self) -> MatrixRunConfig:
        if self.workload_profile is not None:
            from llm_inference_benchmark.profiles import get_profile

            get_profile(self.workload_profile)
        return self


class MatrixConfig(BaseModel):
    """Full matrix config: a list of runs sharing a common results directory."""

    results_dir: str = "results"
    runs: list[MatrixRunConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_names(self) -> MatrixConfig:
        seen: set[str] = set()
        for run in self.runs:
            if run.name in seen:
                raise ValueError(f"Duplicate run name {run.name!r} in matrix config")
            seen.add(run.name)
        return self


def load_matrix(path: str | Path) -> MatrixConfig:
    """Load and validate a YAML matrix config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return MatrixConfig.model_validate(data or {})
