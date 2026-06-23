from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub import model_info as hf_model_info

_DEFAULT_MODELS_DIR = Path.home() / "models"
_DEFAULT_MAX_SIZE_GB = 10.0
_BYTES_PER_GB = 1024**3


@dataclass
class PullResult:
    path: Path
    backend: Literal["gguf", "transformers"]
    skipped: bool


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _find_gguf_sibling(repo_id: str, quant: str, token: str | None) -> tuple[str, int, str | None]:
    """Return (filename, size_bytes, sha256_or_None) for the best GGUF match in *repo_id*."""
    info = hf_model_info(repo_id, token=token, files_metadata=True)
    quant_upper = quant.upper()
    matches: list[tuple[str, int, str | None]] = []
    for sibling in info.siblings or []:
        name: str = sibling.rfilename
        if not name.lower().endswith(".gguf"):
            continue
        if quant_upper not in name.upper():
            continue
        size: int = sibling.size or 0
        sha256: str | None = sibling.lfs.sha256 if sibling.lfs is not None else None
        matches.append((name, size, sha256))

    if not matches:
        raise ValueError(
            f"No GGUF file matching {quant!r} found in {repo_id!r}. "
            "Check the repo on HuggingFace Hub for available quantization filenames."
        )
    # Prefer shortest filename — the unsplit single-file variant over shards
    matches.sort(key=lambda t: len(t[0]))
    return matches[0]


def pull_gguf(
    repo_id: str,
    quant: str,
    *,
    dest_dir: Path | None = None,
    token: str | None = None,
    max_size_gb: float = _DEFAULT_MAX_SIZE_GB,
) -> PullResult:
    """Download a GGUF file from HuggingFace Hub to *dest_dir* (default ~/models/).

    Raises ValueError if the remote file exceeds *max_size_gb* or if the
    post-download SHA-256 does not match the Hub's LFS metadata.
    Returns a PullResult with skipped=True when the file is already present
    and its hash matches.
    """
    dest = (dest_dir or _DEFAULT_MODELS_DIR).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    filename, size_bytes, expected_sha256 = _find_gguf_sibling(repo_id, quant, token)

    if size_bytes:
        size_gb = size_bytes / _BYTES_PER_GB
        if size_gb > max_size_gb:
            raise ValueError(
                f"{filename} is {size_gb:.1f} GB — exceeds --max-size-gb {max_size_gb}. "
                "Raise the limit or choose a smaller quantization."
            )

    local_path = dest / Path(filename).name

    if local_path.exists() and expected_sha256:
        if _sha256_file(local_path) == expected_sha256:
            return PullResult(path=local_path, backend="gguf", skipped=True)

    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest),
            token=token,
        )
    )

    if expected_sha256:
        actual = _sha256_file(downloaded)
        if actual != expected_sha256:
            downloaded.unlink(missing_ok=True)
            raise ValueError(
                f"Hash mismatch for {downloaded.name}: "
                f"expected {expected_sha256[:16]}…, got {actual[:16]}…. "
                "The partial file has been deleted."
            )

    return PullResult(path=downloaded, backend="gguf", skipped=False)


def pull_transformers(
    repo_id: str,
    *,
    token: str | None = None,
    max_size_gb: float = _DEFAULT_MAX_SIZE_GB,
) -> PullResult:
    """Download a full Transformers model snapshot from HuggingFace Hub.

    Uses the default HF cache directory (~/.cache/huggingface/hub).
    Raises ValueError if the total repository size exceeds *max_size_gb*.
    """
    info = hf_model_info(repo_id, token=token, files_metadata=True)
    total_bytes = sum(s.size or 0 for s in (info.siblings or []))
    if total_bytes:
        total_gb = total_bytes / _BYTES_PER_GB
        if total_gb > max_size_gb:
            raise ValueError(
                f"{repo_id} is {total_gb:.1f} GB — exceeds --max-size-gb {max_size_gb}. "
                "Raise the limit or choose a smaller model."
            )

    cache_path = snapshot_download(repo_id=repo_id, token=token)
    return PullResult(path=Path(cache_path), backend="transformers", skipped=False)
