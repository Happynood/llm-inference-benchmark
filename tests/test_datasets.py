"""Tests for the datasets module and CLI sub-commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.datasets import (
    REGISTRY,
    _extract_hermes_fn,
    _extract_lmsys_chat,
    cache_dir,
    list_cached,
    load_prompts,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, prompts: list[str]) -> None:
    path.write_text("\n".join(json.dumps({"prompt": p}) for p in prompts) + "\n")


# ── Unit tests: extractors ─────────────────────────────────────────────────────


def test_extract_lmsys_chat_returns_first_user_turn() -> None:
    row = {
        "conversation": [
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is Python?"},
        ]
    }
    assert _extract_lmsys_chat(row) == "What is Python?"


def test_extract_lmsys_chat_missing_user_returns_none() -> None:
    row = {"conversation": [{"role": "assistant", "content": "Hi"}]}
    assert _extract_lmsys_chat(row) is None


def test_extract_lmsys_chat_empty_conversation_returns_none() -> None:
    assert _extract_lmsys_chat({"conversation": []}) is None
    assert _extract_lmsys_chat({}) is None


def test_extract_hermes_fn_returns_first_human_message() -> None:
    row = {
        "conversations": [
            {"from": "system", "value": "You are an AI."},
            {"from": "human", "value": "Call the weather API for London."},
        ]
    }
    assert _extract_hermes_fn(row) == "Call the weather API for London."


def test_extract_hermes_fn_missing_human_returns_none() -> None:
    row = {"conversations": [{"from": "gpt", "value": "Sure."}]}
    assert _extract_hermes_fn(row) is None


def test_extract_hermes_fn_empty_returns_none() -> None:
    assert _extract_hermes_fn({}) is None
    assert _extract_hermes_fn({"conversations": []}) is None


# ── Unit tests: cache_dir ─────────────────────────────────────────────────────


def test_cache_dir_creates_directory(tmp_path: Path) -> None:
    expected = tmp_path / ".cache" / "llm-bench" / "datasets"
    with patch("llm_inference_benchmark.datasets.Path.home", return_value=tmp_path):
        result = cache_dir()
    assert result == expected
    assert result.is_dir()


# ── Unit tests: list_cached ───────────────────────────────────────────────────


def test_list_cached_empty(tmp_path: Path) -> None:
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        assert list_cached() == []


def test_list_cached_returns_name_and_count(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "lmsys-chat.jsonl", ["Hello world", "How are you?", "Explain async."])
    _write_jsonl(tmp_path / "hermes-fn.jsonl", ["Call the API."])
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        rows = list_cached()
    assert ("hermes-fn", 1) in rows
    assert ("lmsys-chat", 3) in rows


# ── Unit tests: load_prompts ──────────────────────────────────────────────────


def test_load_prompts_raises_when_not_cached(tmp_path: Path) -> None:
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        with pytest.raises(FileNotFoundError, match="not cached"):
            load_prompts("lmsys-chat")


def test_load_prompts_returns_all_prompts(tmp_path: Path) -> None:
    samples = ["Alpha prompt", "Beta prompt", "Gamma prompt"]
    _write_jsonl(tmp_path / "lmsys-chat.jsonl", samples)
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        result = load_prompts("lmsys-chat")
    assert result == samples


def test_load_prompts_samples_n_prompts(tmp_path: Path) -> None:
    samples = [f"Prompt number {i}" for i in range(20)]
    _write_jsonl(tmp_path / "lmsys-chat.jsonl", samples)
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        result = load_prompts("lmsys-chat", n=5, seed=42)
    assert len(result) == 5
    assert all(p in samples for p in result)


def test_load_prompts_sampling_is_reproducible(tmp_path: Path) -> None:
    samples = [f"Prompt {i}" for i in range(30)]
    _write_jsonl(tmp_path / "lmsys-chat.jsonl", samples)
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        r1 = load_prompts("lmsys-chat", n=10, seed=99)
        r2 = load_prompts("lmsys-chat", n=10, seed=99)
    assert r1 == r2


def test_load_prompts_raises_on_empty_file(tmp_path: Path) -> None:
    (tmp_path / "lmsys-chat.jsonl").write_text("\n")
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        with pytest.raises(ValueError, match="no prompts"):
            load_prompts("lmsys-chat")


# ── Unit tests: pull (mocked) ─────────────────────────────────────────────────


def test_pull_unknown_dataset_raises(tmp_path: Path) -> None:
    from llm_inference_benchmark.datasets import pull

    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            pull("nonexistent")


def test_pull_missing_datasets_package_raises(tmp_path: Path) -> None:
    from llm_inference_benchmark.datasets import pull

    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                pull("lmsys-chat")


def test_pull_writes_jsonl(tmp_path: Path) -> None:
    from llm_inference_benchmark.datasets import pull

    fake_rows = [
        {"conversation": [{"role": "user", "content": f"Question number {i}"}]} for i in range(10)
    ]
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(return_value=iter(fake_rows))
    mock_load = MagicMock(return_value=mock_ds)

    with (
        patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path),
        patch("llm_inference_benchmark.datasets.load_dataset", mock_load, create=True),
        patch.dict(
            "sys.modules",
            {"datasets": MagicMock(load_dataset=mock_load)},
        ),
    ):
        out = pull("lmsys-chat", max_samples=5)

    assert out == tmp_path / "lmsys-chat.jsonl"
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5
    for line in lines:
        obj = json.loads(line)
        assert "prompt" in obj


# ── CLI tests: datasets pull ──────────────────────────────────────────────────


def test_cli_datasets_pull_unknown_name() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["datasets", "pull", "bogus-name"])
    assert result.exit_code != 0
    assert "Unknown dataset" in result.output


def test_cli_datasets_list_empty(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        result = runner.invoke(main, ["datasets", "list"])
    assert result.exit_code == 0
    assert "No datasets cached" in result.output


def test_cli_datasets_list_shows_cached(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "lmsys-chat.jsonl", ["Hello there", "What is AI?"])
    runner = CliRunner()
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        result = runner.invoke(main, ["datasets", "list"])
    assert result.exit_code == 0
    assert "lmsys-chat" in result.output
    assert "2" in result.output


# ── CLI tests: --dataset flag ─────────────────────────────────────────────────


def test_cli_run_with_dataset_flag(tmp_path: Path) -> None:
    """--dataset loads prompts from cache instead of the prompts_file."""
    samples = ["Tell me about quantum computing.", "Explain transformers in NLP."]
    _write_jsonl(tmp_path / "lmsys-chat.jsonl", samples)

    config_yaml = tmp_path / "cfg.yaml"
    config_yaml.write_text("backend: mock\nmodel: test-model\nrequests: 2\nwarmup_requests: 0\n")

    runner = CliRunner()
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        result = runner.invoke(
            main, ["--config", str(config_yaml), "--dataset", "lmsys-chat", "--requests", "2"]
        )
    assert result.exit_code == 0, result.output
    assert "Backend: mock" in result.output


def test_cli_run_dataset_not_cached_gives_usage_error(tmp_path: Path) -> None:
    config_yaml = tmp_path / "cfg.yaml"
    config_yaml.write_text("backend: mock\nmodel: test-model\nrequests: 2\nwarmup_requests: 0\n")

    runner = CliRunner()
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        result = runner.invoke(main, ["--config", str(config_yaml), "--dataset", "lmsys-chat"])
    assert result.exit_code != 0
    assert "not cached" in result.output.lower() or "Error" in result.output


# ── Registry sanity check ─────────────────────────────────────────────────────


def test_registry_has_expected_entries() -> None:
    assert "lmsys-chat" in REGISTRY
    assert "hermes-fn" in REGISTRY
    for spec in REGISTRY.values():
        assert "hf_repo" in spec
        assert "extractor" in spec
        assert "max_samples" in spec
