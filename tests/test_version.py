"""Tests for package version consistency — no GPU or model downloads needed."""

from __future__ import annotations

from importlib.metadata import version

from llm_inference_benchmark import __version__


def test_version_matches_installed_package_metadata() -> None:
    assert __version__ == version("llm-inference-benchmark")


def test_version_is_not_stale_scaffold_placeholder() -> None:
    assert __version__ != "0.1.0"
