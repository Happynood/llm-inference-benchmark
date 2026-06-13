"""Tests for workload profiles — loading, resolution, backward compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.config import BenchmarkConfig, load_config
from llm_inference_benchmark.profiles import PROFILE_NAMES, WorkloadProfile, get_profile

# ---------------------------------------------------------------------------
# WorkloadProfile dataclass
# ---------------------------------------------------------------------------


def test_all_four_profiles_are_registered() -> None:
    assert {"short_chat", "summarization", "code_completion", "long_context_smoke"} == PROFILE_NAMES


def test_get_profile_short_chat() -> None:
    p = get_profile("short_chat")
    assert isinstance(p, WorkloadProfile)
    assert p.name == "short_chat"
    assert p.prompts_file == "data/prompts/short_chat.txt"
    assert p.input_length == "short"
    assert p.output_length == "short"


def test_get_profile_summarization() -> None:
    p = get_profile("summarization")
    assert p.prompts_file == "data/prompts/summarization.txt"
    assert p.input_length == "medium"


def test_get_profile_code_completion() -> None:
    p = get_profile("code_completion")
    assert p.prompts_file == "data/prompts/code_completion.txt"
    assert p.output_length == "medium"


def test_get_profile_long_context_smoke() -> None:
    p = get_profile("long_context_smoke")
    assert p.prompts_file == "data/prompts/long_context_smoke.txt"
    assert p.input_length == "long"


def test_get_profile_unknown_raises_with_helpful_message() -> None:
    with pytest.raises(ValueError, match="Unknown workload profile 'bogus'"):
        get_profile("bogus")


def test_get_profile_error_lists_valid_profiles() -> None:
    with pytest.raises(ValueError, match="short_chat"):
        get_profile("bogus")


def test_all_profiles_have_non_empty_description() -> None:
    for name in PROFILE_NAMES:
        p = get_profile(name)
        assert p.description.strip(), f"Profile {name!r} has an empty description"


def test_workload_profile_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    p = get_profile("short_chat")
    with pytest.raises(FrozenInstanceError):
        p.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchmarkConfig with workload_profile
# ---------------------------------------------------------------------------


def test_config_with_valid_workload_profile() -> None:
    cfg = BenchmarkConfig(workload_profile="short_chat")
    assert cfg.workload_profile == "short_chat"


def test_config_with_invalid_profile_fails_validation() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Unknown workload profile"):
        BenchmarkConfig(workload_profile="not_a_profile")


def test_resolve_prompts_file_uses_profile_path() -> None:
    cfg = BenchmarkConfig(workload_profile="short_chat")
    assert cfg.resolve_prompts_file() == "data/prompts/short_chat.txt"


def test_resolve_prompts_file_uses_explicit_prompts_file_when_no_profile() -> None:
    cfg = BenchmarkConfig(prompts_file="custom/prompts.txt")
    assert cfg.resolve_prompts_file() == "custom/prompts.txt"


def test_resolve_prompts_file_returns_default_when_neither_set() -> None:
    cfg = BenchmarkConfig()
    assert cfg.resolve_prompts_file() == "data/prompts/smoke.txt"


# ---------------------------------------------------------------------------
# load_config with workload_profile key in YAML
# ---------------------------------------------------------------------------


def test_load_config_yaml_with_workload_profile(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "backend: mock\n"
        "model: test-model\n"
        "requests: 5\n"
        "workload_profile: summarization\n"
        "mock:\n"
        "  latency_ms: 0\n"
        "  tokens_per_response: 10\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.workload_profile == "summarization"
    assert cfg.resolve_prompts_file() == "data/prompts/summarization.txt"


def test_load_config_yaml_with_invalid_profile_fails(tmp_path: Path) -> None:
    from pydantic import ValidationError

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("workload_profile: does_not_exist\n")
    with pytest.raises(ValidationError, match="Unknown workload profile"):
        load_config(cfg_file)


# ---------------------------------------------------------------------------
# Backward compatibility — existing configs without workload_profile still work
# ---------------------------------------------------------------------------


def test_existing_config_without_profile_loads(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    assert cfg.workload_profile is None
    assert cfg.resolve_prompts_file() != ""


def test_existing_config_resolve_uses_prompts_file(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    # tmp_config sets prompts_file to an absolute tmp path; resolve must return it
    assert cfg.resolve_prompts_file() == cfg.prompts_file


# ---------------------------------------------------------------------------
# CLI integration with workload_profile config
# ---------------------------------------------------------------------------


def test_cli_with_workload_profile_config(tmp_path: Path) -> None:
    prompts = tmp_path / "short_chat.txt"
    prompts.write_text("What is AI?\nExplain overfitting.\n")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "backend: mock\n"
        "model: test-model\n"
        "requests: 2\n"
        "warmup_requests: 0\n"
        "workload_profile: short_chat\n"
        "mock:\n"
        "  latency_ms: 0\n"
        "  tokens_per_response: 5\n"
    )
    # Override the profile's prompts path via monkeypatching resolve_prompts_file
    from unittest.mock import patch

    with patch(
        "llm_inference_benchmark.config.BenchmarkConfig.resolve_prompts_file",
        return_value=str(prompts),
    ):
        result = CliRunner().invoke(main, ["--config", str(cfg_file)])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Manifest uses resolved prompts path
# ---------------------------------------------------------------------------


def test_manifest_prompts_sha256_uses_resolved_path(tmp_path: Path) -> None:
    import hashlib
    from unittest.mock import patch

    from llm_inference_benchmark.config import BenchmarkConfig, load_config
    from llm_inference_benchmark.manifest import collect_manifest

    prompts = tmp_path / "p.txt"
    prompts.write_text("Hello\n")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"backend: mock\n"
        f"model: test-model\n"
        f"requests: 2\n"
        f"warmup_requests: 0\n"
        f"prompts_file: {prompts}\n"
        f"mock:\n"
        f"  latency_ms: 0\n"
        f"  tokens_per_response: 5\n"
    )
    cfg = load_config(cfg_file)
    expected_sha = hashlib.sha256(prompts.read_bytes()).hexdigest()

    # Patch the class method so collect_manifest calls our overridden resolve
    with patch.object(BenchmarkConfig, "resolve_prompts_file", return_value=str(prompts)):
        m = collect_manifest(cfg_file, cfg)

    assert m.prompts_sha256 == expected_sha
