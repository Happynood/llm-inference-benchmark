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


def test_config_rejects_zero_requests() -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(requests=0)


def test_config_rejects_negative_concurrency() -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(concurrency=0)
