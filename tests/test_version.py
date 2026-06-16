"""Tests for package version consistency — no GPU or model downloads needed."""

from __future__ import annotations

from importlib.metadata import version

from click.testing import CliRunner

from llm_inference_benchmark import __version__
from llm_inference_benchmark.cli import main


def test_version_matches_installed_package_metadata() -> None:
    assert __version__ == version("llm-inference-benchmark")


def test_version_is_not_stale_scaffold_placeholder() -> None:
    assert __version__ != "0.1.0"


def test_cli_version_flag_exits_zero() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0


def test_cli_version_flag_output_matches_package_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert __version__ in result.output
