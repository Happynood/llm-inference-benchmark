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
    "long-context-4k": {
        "hf_repo": "deepmind/pg19",
        "split": "train",
        "max_samples": 100,
        "extractor": "pg19_4k",
        "description": "deepmind/pg19 — ~4k-token passages for prefill latency profiling",
    },
    "long-context-16k": {
        "hf_repo": "deepmind/pg19",
        "split": "train",
        "max_samples": 50,
        "extractor": "pg19_16k",
        "description": "deepmind/pg19 — ~16k-token passages for prefill latency profiling",
    },
    "long-context-64k": {
        "hf_repo": "deepmind/pg19",
        "split": "train",
        "max_samples": 10,
        "extractor": "pg19_64k",
        "description": "deepmind/pg19 — ~64k-token passages for prefill latency profiling",
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


# Approximate characters per token for English prose (used for pg19 slicing).
_CHARS_PER_TOKEN = 4


def _make_pg19_extractor(target_tokens: int):
    """Return an extractor that slices a ~*target_tokens*-token passage from a PG19 book.

    Books shorter than half the target are skipped (return None).
    The slice starts after the first ~5 % of text to avoid title/TOC pages.
    """
    target_chars = target_tokens * _CHARS_PER_TOKEN
    min_chars = target_chars // 2

    def _extract(row: dict[str, Any]) -> str | None:
        text = str(row.get("text", "")).strip()
        if len(text) < min_chars:
            return None
        # Skip front matter: first ~5 %, clamped to [500, 10 000] chars.
        start = min(max(500, len(text) // 20), 10_000)
        chunk = text[start : start + target_chars]
        if len(chunk) < min_chars:
            return None
        # Trim to the last word boundary to avoid cutting mid-word.
        last_space = chunk.rfind(" ")
        if last_space > int(len(chunk) * 0.9):
            chunk = chunk[:last_space]
        return f"Summarize the following passage in 2-3 sentences:\n\n{chunk}"

    return _extract


_EXTRACTORS = {
    "lmsys_chat": _extract_lmsys_chat,
    "hermes_fn": _extract_hermes_fn,
    "pg19_4k": _make_pg19_extractor(4_096),
    "pg19_16k": _make_pg19_extractor(16_384),
    "pg19_64k": _make_pg19_extractor(65_536),
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
