from __future__ import annotations

import json

from click.testing import CliRunner

from llm_inference_benchmark.cli import main


def test_env_exits_zero() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert result.exit_code == 0


def test_env_table_shows_python() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert "python" in result.output


def test_env_table_shows_platform() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert "platform" in result.output


def test_env_table_shows_cpu() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert "cpu" in result.output


def test_env_table_shows_package() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert "llm-inference-benchmark" in result.output


def test_env_table_is_not_json() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert not result.output.strip().startswith("{")


def test_env_format_json_exits_zero() -> None:
    result = CliRunner().invoke(main, ["env", "--format", "json"])
    assert result.exit_code == 0


def test_env_format_json_valid() -> None:
    result = CliRunner().invoke(main, ["env", "--format", "json"])
    data = json.loads(result.output)
    assert isinstance(data, dict)


def test_env_format_json_required_keys() -> None:
    result = CliRunner().invoke(main, ["env", "--format", "json"])
    data = json.loads(result.output)
    for key in ("python_version", "platform_info", "cpu_model", "cpu_count", "package_version"):
        assert key in data, f"missing key: {key}"


def test_env_format_json_optional_keys_present() -> None:
    result = CliRunner().invoke(main, ["env", "--format", "json"])
    data = json.loads(result.output)
    optional_keys = (
        "torch_version",
        "transformers_version",
        "optimum_version",
        "vllm_version",
        "psutil_version",
        "gpu",
    )
    for key in optional_keys:
        assert key in data, f"missing key: {key}"


def test_env_format_json_cpu_count_int_or_null() -> None:
    result = CliRunner().invoke(main, ["env", "--format", "json"])
    data = json.loads(result.output)
    assert data["cpu_count"] is None or isinstance(data["cpu_count"], int)


def test_env_invalid_format_fails() -> None:
    result = CliRunner().invoke(main, ["env", "--format", "xml"])
    assert result.exit_code != 0
