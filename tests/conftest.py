from __future__ import annotations

import os
from pathlib import Path

import pytest

# HF Hub (via httpx) rejects the socks:// scheme in ALL_PROXY during commit-hash
# validation even when files are fully cached. Clear it for the test session;
# http_proxy / https_proxy remain set and still serve as the HTTP forward proxy.
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)


@pytest.fixture
def tmp_prompts(tmp_path: Path) -> Path:
    p = tmp_path / "prompts.txt"
    p.write_text(
        "What is the capital of France?\n"
        "Explain gradient descent.\n"
        "Write a haiku about neural networks.\n"
    )
    return p


@pytest.fixture
def tmp_config(tmp_path: Path, tmp_prompts: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: mock\n"
        f"model: test-model\n"
        f"requests: 5\n"
        f"warmup_requests: 1\n"
        f"prompts_file: {tmp_prompts}\n"
        f"mock:\n"
        f"  latency_ms: 0\n"
        f"  tokens_per_response: 10\n"
    )
    return cfg
