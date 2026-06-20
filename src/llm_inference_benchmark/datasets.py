from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Registry of supported dataset names → HuggingFace coordinates and extraction logic
REGISTRY: dict[str, dict[str, Any]] = {
    "lmsys-chat": {
        "hf_repo": "lmsys/lmsys-chat-1m",
        "split": "train",
        "max_samples": 500,
        "extractor": "lmsys_chat",
        "description": "lmsys/lmsys-chat-1m — real multi-turn chat (first user turn)",
    },
    "hermes-fn": {
        "hf_repo": "NousResearch/hermes-function-calling-v1",
        "split": "train",
        "max_samples": 200,
        "extractor": "hermes_fn",
        "description": "NousResearch/hermes-function-calling-v1 — function-calling prompts",
    },
}


def cache_dir() -> Path:
    """Return (and create) the cache directory for llm-bench datasets."""
    path = Path.home() / ".cache" / "llm-bench" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_lmsys_chat(row: dict[str, Any]) -> str | None:
    """Return the first user message from a lmsys-chat-1m row, or None to skip."""
    conversation = row.get("conversation")
    if not conversation:
        return None
    for turn in conversation:
        if turn.get("role") == "user":
            content = str(turn.get("content", "")).strip()
            return content or None
    return None


def _extract_hermes_fn(row: dict[str, Any]) -> str | None:
    """Return the first human message from a hermes-function-calling row, or None to skip."""
    conversations = row.get("conversations")
    if not conversations:
        return None
    for msg in conversations:
        if msg.get("from") == "human":
            content = str(msg.get("value", "")).strip()
            return content or None
    return None


_EXTRACTORS = {
    "lmsys_chat": _extract_lmsys_chat,
    "hermes_fn": _extract_hermes_fn,
}


def pull(name: str, hf_token: str | None = None, max_samples: int | None = None) -> Path:
    """Download up to *max_samples* prompts for *name* and write them to the JSONL cache.

    Requires the ``datasets`` package (``pip install datasets``).
    Returns the path to the written JSONL file.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Known: {', '.join(REGISTRY)}")

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to pull datasets. "
            "Install it with:  pip install datasets"
        ) from exc

    spec = REGISTRY[name]
    extractor = _EXTRACTORS[spec["extractor"]]
    limit = max_samples if max_samples is not None else spec["max_samples"]
    out_path = cache_dir() / f"{name}.jsonl"

    print(f"Downloading {spec['hf_repo']} (up to {limit} samples, streaming)…")
    ds = load_dataset(
        spec["hf_repo"],
        split=spec["split"],
        streaming=True,
        token=hf_token,
    )

    samples: list[dict[str, str]] = []
    for row in ds:
        if len(samples) >= limit:
            break
        prompt = extractor(row)  # type: ignore[arg-type]
        if prompt and len(prompt) >= 10:
            samples.append({"prompt": prompt})

    lines = "\n".join(json.dumps(s) for s in samples)
    out_path.write_text(lines + "\n" if lines else "\n")
    print(f"Cached {len(samples)} samples → {out_path}")
    return out_path


def list_cached() -> list[tuple[str, int]]:
    """Return ``(name, sample_count)`` pairs for every cached JSONL dataset."""
    result: list[tuple[str, int]] = []
    for path in sorted(cache_dir().glob("*.jsonl")):
        count = sum(1 for ln in path.read_text().splitlines() if ln.strip())
        result.append((path.stem, count))
    return result


def load_prompts(name: str, n: int | None = None, seed: int | None = None) -> list[str]:
    """Load prompts from a cached dataset JSONL file.

    If *n* is given and smaller than the total cached count, sample *n* prompts
    without replacement using *seed* for reproducibility.
    Raises ``FileNotFoundError`` when the dataset has not been pulled yet.
    """
    path = cache_dir() / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {name!r} is not cached. Run:  llm-bench datasets pull {name}"
        )

    prompts: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            prompts.append(json.loads(line)["prompt"])

    if not prompts:
        raise ValueError(f"Cached dataset {name!r} contains no prompts")

    if n is None or n >= len(prompts):
        return prompts

    rng = random.Random(seed)
    return rng.sample(prompts, n)
