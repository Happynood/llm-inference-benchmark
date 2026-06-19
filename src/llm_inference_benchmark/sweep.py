"""Parameter sweep expansion: cartesian product of override grids → run list."""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_inference_benchmark.config import BenchmarkConfig

_SAFE_RE = re.compile(r"[^A-Za-z0-9._]")


def _to_name_part(value: Any) -> str:
    """Convert a sweep value to a fragment safe for use in a run name."""
    s = _SAFE_RE.sub("-", str(value)).strip("-")
    return s or "x"


def validate_override_path(path: str) -> None:
    """Raise ValueError if *path* is not a valid BenchmarkConfig override path.

    Accepts up to two dot-separated components, e.g. ``llama_cpp.n_gpu_layers``.
    Raises ``ValueError`` with a descriptive message on any invalid path.
    """
    from llm_inference_benchmark.config import (
        BenchmarkConfig,
        HFBackendConfig,
        LlamaCppBackendConfig,
        MockBackendConfig,
        OnnxBackendConfig,
        OpenAIEndpointConfig,
        VLLMBackendConfig,
    )

    _sub_models: dict[str, type] = {
        "llama_cpp": LlamaCppBackendConfig,
        "hf": HFBackendConfig,
        "mock": MockBackendConfig,
        "openai": OpenAIEndpointConfig,
        "onnx": OnnxBackendConfig,
        "vllm": VLLMBackendConfig,
    }

    parts = path.split(".")
    if len(parts) > 2:
        raise ValueError(
            f"Override path {path!r} is too deep (max 2 levels, e.g. 'llama_cpp.n_gpu_layers')"
        )

    top = parts[0]
    if top not in BenchmarkConfig.model_fields:
        valid = ", ".join(sorted(BenchmarkConfig.model_fields))
        raise ValueError(f"Unknown override path {path!r}. Valid top-level fields: {valid}")

    if len(parts) == 2:
        key = parts[1]
        sub_cls = _sub_models.get(top)
        if sub_cls is None:
            raise ValueError(f"Section {top!r} does not support nested overrides in path {path!r}")
        if key not in sub_cls.model_fields:
            valid = ", ".join(sorted(sub_cls.model_fields))
            raise ValueError(f"Unknown field {key!r} in section {top!r}. Valid fields: {valid}")


def apply_overrides(cfg: BenchmarkConfig, overrides: dict[str, Any]) -> BenchmarkConfig:
    """Apply dot-path overrides to a BenchmarkConfig, returning a new config instance."""
    for path, value in overrides.items():
        parts = path.split(".", 1)
        if len(parts) == 1:
            cfg = cfg.model_copy(update={parts[0]: value})
        else:
            sub = getattr(cfg, parts[0])
            cfg = cfg.model_copy(update={parts[0]: sub.model_copy(update={parts[1]: value})})
    return cfg


def expand_sweep(
    base_config: str,
    sweep: dict[str, list[Any]],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Expand a sweep grid into ``(name, base_config, overrides)`` triples.

    Parameters
    ----------
    base_config:
        Path to the base YAML config file shared by all sweep runs.
    sweep:
        Mapping from dot-path override keys to lists of values.
        Every list must be non-empty.

    Returns
    -------
    list of ``(name, base_config, overrides)`` where *name* is a deterministic
    run identifier and *overrides* maps each key to one value from the grid.
    """
    if not sweep:
        raise ValueError("'sweep' must have at least one parameter with non-empty values")

    for path, values in sweep.items():
        validate_override_path(path)
        if not values:
            raise ValueError(
                f"Sweep parameter {path!r} has an empty value list; provide at least one value"
            )

    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]
    result: list[tuple[str, str, dict[str, Any]]] = []
    seen_names: set[str] = set()

    for combo in itertools.product(*value_lists):
        overrides: dict[str, Any] = dict(zip(keys, combo, strict=True))

        name_parts: list[str] = ["sweep"]
        for key, val in zip(keys, combo, strict=True):
            leaf = key.rsplit(".", 1)[-1]
            name_parts.append(_to_name_part(leaf))
            name_parts.append(_to_name_part(val))
        name = "-".join(name_parts)

        if name in seen_names:
            raise ValueError(
                f"Sweep expansion produced duplicate run name {name!r}. "
                "Use explicit 'runs:' when parameter values have ambiguous string representations."
            )
        seen_names.add(name)
        result.append((name, base_config, overrides))

    return result
