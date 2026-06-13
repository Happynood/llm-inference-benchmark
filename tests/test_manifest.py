"""Tests for manifest.py — no GPU or model downloads needed."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_inference_benchmark.cli import main
from llm_inference_benchmark.config import load_config
from llm_inference_benchmark.manifest import (
    RunManifest,
    collect_manifest,
    write_manifest,
)


@pytest.fixture
def manifest(tmp_config: Path) -> RunManifest:
    cfg = load_config(tmp_config)
    return collect_manifest(tmp_config, cfg)


# ---------------------------------------------------------------------------
# collect_manifest — field types and values
# ---------------------------------------------------------------------------


def test_collect_manifest_returns_run_manifest(manifest: RunManifest) -> None:
    assert isinstance(manifest, RunManifest)


def test_timestamp_is_iso_format(manifest: RunManifest) -> None:
    # datetime.fromisoformat accepts the ISO 8601 strings we produce
    from datetime import datetime

    parsed = datetime.fromisoformat(manifest.timestamp)
    assert parsed.tzinfo is not None  # must be timezone-aware


def test_backend_matches_config(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    m = collect_manifest(tmp_config, cfg)
    assert m.backend == cfg.backend


def test_model_matches_config(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    m = collect_manifest(tmp_config, cfg)
    assert m.model == cfg.model


def test_config_sha256_is_64_hex_chars(manifest: RunManifest) -> None:
    assert len(manifest.config_sha256) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", manifest.config_sha256)


def test_prompts_sha256_is_64_hex_chars(manifest: RunManifest) -> None:
    assert len(manifest.prompts_sha256) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", manifest.prompts_sha256)


def test_config_and_prompts_hashes_differ(tmp_config: Path) -> None:
    cfg = load_config(tmp_config)
    m = collect_manifest(tmp_config, cfg)
    assert m.config_sha256 != m.prompts_sha256


def test_python_version_is_nonempty(manifest: RunManifest) -> None:
    assert manifest.python_version


def test_platform_info_is_nonempty(manifest: RunManifest) -> None:
    assert manifest.platform_info


def test_cpu_model_is_nonempty(manifest: RunManifest) -> None:
    assert manifest.cpu_model


def test_cpu_count_is_positive_or_none(manifest: RunManifest) -> None:
    if manifest.cpu_count is not None:
        assert manifest.cpu_count > 0


def test_package_version_is_nonempty(manifest: RunManifest) -> None:
    assert manifest.package_version


def test_git_commit_is_none_or_40_hex(manifest: RunManifest) -> None:
    if manifest.git_commit is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", manifest.git_commit)


def test_git_dirty_is_bool_or_none(manifest: RunManifest) -> None:
    assert manifest.git_dirty is None or isinstance(manifest.git_dirty, bool)


def test_psutil_version_is_nonempty(manifest: RunManifest) -> None:
    assert manifest.psutil_version  # psutil is a required dep


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------


def test_write_manifest_creates_file(manifest: RunManifest, tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)
    assert out.exists()


def test_write_manifest_is_valid_json(manifest: RunManifest, tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)
    data = json.loads(out.read_text())
    assert isinstance(data, dict)


def test_write_manifest_contains_required_keys(manifest: RunManifest, tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)
    data = json.loads(out.read_text())
    required = {
        "timestamp",
        "backend",
        "model",
        "git_commit",
        "git_dirty",
        "config_sha256",
        "prompts_sha256",
        "python_version",
        "platform_info",
        "cpu_model",
        "cpu_count",
        "package_version",
        "psutil_version",
    }
    assert required.issubset(data.keys())


def test_write_manifest_ends_with_newline(manifest: RunManifest, tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)
    assert out.read_text().endswith("\n")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_manifest_option_creates_file(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--manifest", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_manifest_is_valid_json(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    CliRunner().invoke(main, ["--config", str(tmp_config), "--manifest", str(out)])
    data = json.loads(out.read_text())
    assert "backend" in data
    assert "config_sha256" in data
    assert "git_commit" in data


def test_cli_manifest_output_echoed(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--manifest", str(out)])
    assert "Manifest written to" in result.output


def test_cli_without_manifest_still_works(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config)])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.output
