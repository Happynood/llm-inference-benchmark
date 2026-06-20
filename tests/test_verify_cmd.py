from __future__ import annotations

import json
import sys
from unittest.mock import patch

from click.testing import CliRunner

from llm_inference_benchmark.cli import main


def test_verify_exits_zero() -> None:
    result = CliRunner().invoke(main, ["verify"])
    assert result.exit_code == 0, result.output


def test_verify_shows_mock_ok() -> None:
    result = CliRunner().invoke(main, ["verify"])
    assert "mock" in result.output
    assert "OK" in result.output


def test_verify_shows_all_backends() -> None:
    result = CliRunner().invoke(main, ["verify"])
    for backend in ("mock", "transformers", "llama-cpp", "openai", "onnx", "vllm"):
        assert backend in result.output, f"missing backend {backend!r} in output:\n{result.output}"


def test_verify_shows_latency_column() -> None:
    result = CliRunner().invoke(main, ["verify"])
    assert "ms" in result.output


def test_verify_format_json_exits_zero() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "json"])
    assert result.exit_code == 0, result.output


def test_verify_format_json_is_list() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_verify_format_json_entry_count() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    assert len(data) == 6


def test_verify_format_json_required_keys() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    for row in data:
        for key in ("backend", "status", "latency_ms", "reason"):
            assert key in row, f"missing key {key!r} in row {row}"


def test_verify_mock_ok_in_json() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    mock_row = next((r for r in data if r["backend"] == "mock"), None)
    assert mock_row is not None
    assert mock_row["status"] == "OK"


def test_verify_mock_has_latency_in_json() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    mock_row = next(r for r in data if r["backend"] == "mock")
    assert mock_row["latency_ms"] is not None
    assert isinstance(mock_row["latency_ms"], float)


def test_verify_all_statuses_valid() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    valid = {"OK", "SKIP", "FAIL"}
    for row in data:
        assert row["status"] in valid, f"unexpected status {row['status']!r}"


def test_verify_openai_always_ok() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    openai_row = next((r for r in data if r["backend"] == "openai"), None)
    assert openai_row is not None
    assert openai_row["status"] == "OK"


def test_verify_missing_dep_reported_as_skip() -> None:
    """A backend whose import fails should be SKIP, not FAIL."""
    fake_modules = dict(sys.modules)
    fake_modules["llama_cpp"] = None  # type: ignore[assignment]
    with patch.dict("sys.modules", fake_modules):
        result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    llama_row = next(r for r in data if r["backend"] == "llama-cpp")
    assert llama_row["status"] == "SKIP"
    assert "llama_cpp" in llama_row["reason"]


def test_verify_skip_reason_names_missing_package() -> None:
    fake_modules = dict(sys.modules)
    fake_modules["llama_cpp"] = None  # type: ignore[assignment]
    with patch.dict("sys.modules", fake_modules):
        result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    llama_row = next(r for r in data if r["backend"] == "llama-cpp")
    assert "missing" in llama_row["reason"]


def test_verify_invalid_format_exits_nonzero() -> None:
    result = CliRunner().invoke(main, ["verify", "--format", "xml"])
    assert result.exit_code != 0


def test_verify_mock_fail_exits_nonzero() -> None:
    with patch(
        "llm_inference_benchmark.backends.mock.MockBackend.generate",
        side_effect=RuntimeError("injected failure"),
    ):
        result = CliRunner().invoke(main, ["verify"])
    assert result.exit_code != 0


def test_verify_mock_fail_shows_fail_in_json() -> None:
    with patch(
        "llm_inference_benchmark.backends.mock.MockBackend.generate",
        side_effect=RuntimeError("injected failure"),
    ):
        result = CliRunner().invoke(main, ["verify", "--format", "json"])
    data = json.loads(result.output)
    mock_row = next(r for r in data if r["backend"] == "mock")
    assert mock_row["status"] == "FAIL"
    assert result.exit_code != 0
