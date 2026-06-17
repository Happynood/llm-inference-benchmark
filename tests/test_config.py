from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_inference_benchmark.config import BenchmarkConfig, load_config


def test_load_config(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.backend == "mock"
    assert cfg.model == "test-model"
    assert cfg.requests == 5
    assert cfg.warmup_requests == 1
    assert cfg.mock.latency_ms == 0
    assert cfg.mock.tokens_per_response == 10


def test_config_defaults() -> None:
    cfg = BenchmarkConfig()
    assert cfg.backend == "mock"
    assert cfg.model == "mock-gpt2"
    assert cfg.requests == 20
    assert cfg.warmup_requests == 2
    assert cfg.mock.latency_ms == 10.0
    assert cfg.measure_perplexity is False
    assert cfg.measure_judge is False


def test_config_rejects_zero_requests() -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(requests=0)


def test_config_rejects_negative_concurrency() -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(concurrency=0)


def test_config_accepts_concurrency_1() -> None:
    cfg = BenchmarkConfig(concurrency=1)
    assert cfg.concurrency == 1


def test_config_accepts_concurrency_gt_1() -> None:
    cfg = BenchmarkConfig(concurrency=4)
    assert cfg.concurrency == 4


def test_config_seed_defaults_to_none() -> None:
    cfg = BenchmarkConfig()
    assert cfg.seed is None


def test_config_seed_accepts_int() -> None:
    cfg = BenchmarkConfig(seed=42)
    assert cfg.seed == 42


def test_config_seed_zero_accepted() -> None:
    cfg = BenchmarkConfig(seed=0)
    assert cfg.seed == 0


def test_config_seed_parsed_from_yaml(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg_file = tmp_path / "seed.yaml"
    cfg_file.write_text(
        f"backend: mock\nmodel: x\nrequests: 1\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\nseed: 7\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.seed == 7
