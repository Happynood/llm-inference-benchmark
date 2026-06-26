"""Matrix config for running multiple benchmark experiments sequentially."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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
    overrides: dict[str, Any] = Field(default_factory=dict)
    dataset: str | None = None

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

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, v: str | None) -> str | None:
        if v is None:
            return v
        from llm_inference_benchmark.datasets import REGISTRY

        if v not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(f"Unknown dataset {v!r}. Known datasets: {known}")
        return v

    @model_validator(mode="after")
    def _validate_profile(self) -> MatrixRunConfig:
        if self.workload_profile is not None:
            from llm_inference_benchmark.profiles import get_profile

            get_profile(self.workload_profile)
        return self


class MatrixConfig(BaseModel):
    """Full matrix config: a list of runs sharing a common results directory.

    Two mutually exclusive formats are supported:

    **Explicit runs** (existing format, backward-compatible)::

        results_dir: results
        runs:
          - name: run-a
            config: configs/a.yaml
          - name: run-b
            config: configs/b.yaml

    **Sweep** (v0.17, generates cartesian product)::

        base_config: configs/example.yaml
        results_dir: results
        sweep:
          mock.latency_ms: [5, 10, 20]
          mock.tokens_per_response: [25, 50]

    ``sweep`` and ``runs`` cannot be combined in the same config.
    """

    results_dir: str = "results"
    base_config: str | None = None
    sweep: dict[str, list[Any]] | None = None
    runs: list[MatrixRunConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _expand_sweep(cls, data: Any) -> Any:
        """Expand sweep grid into explicit run entries before field validation."""
        if not isinstance(data, dict):
            return data

        sweep: dict[str, list[Any]] | None = data.get("sweep")
        if not sweep:
            return data

        base_config: str | None = data.get("base_config")
        if not base_config:
            raise ValueError("'base_config' is required when 'sweep' is set")

        if data.get("runs"):
            raise ValueError(
                "Cannot combine 'runs' and 'sweep' in the same matrix config. "
                "Use one format or the other."
            )

        from llm_inference_benchmark.sweep import expand_sweep

        expanded = expand_sweep(base_config, sweep)
        data = {
            **data,
            "runs": [
                {"name": name, "config": config, "overrides": overrides}
                for name, config, overrides in expanded
            ],
        }
        return data

    @model_validator(mode="after")
    def _validate_runs(self) -> MatrixConfig:
        if not self.runs:
            raise ValueError(
                "Matrix config must define either 'runs' (non-empty list) "
                "or 'sweep' + 'base_config'"
            )
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
