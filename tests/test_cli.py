from pathlib import Path

from click.testing import CliRunner

from llm_inference_benchmark.cli import main


def test_cli_smoke(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config)])
    assert result.exit_code == 0, result.output
    assert "Benchmark Results" in result.output
    assert "p50_latency_ms" in result.output


def test_cli_csv_output(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    content = out.read_text()
    assert "p50_latency_ms" in content
    assert "mock" in content


def test_cli_csv_has_one_data_row(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--output", str(out)])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in out.read_text().splitlines() if ln]
    assert len(lines) == 2  # header + 1 data row


def test_cli_missing_config_fails() -> None:
    result = CliRunner().invoke(main, ["--config", "nonexistent.yaml"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# --format json on the main benchmark command
# ---------------------------------------------------------------------------


def test_cli_format_json_emits_valid_json(tmp_config: Path) -> None:
    import json

    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["backend"] == "mock"
    assert "p50_latency_ms" in data
    assert "timestamp" in data


def test_cli_format_json_null_for_none_fields(tmp_config: Path) -> None:
    import json

    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    # perplexity is None for mock backend — must be null in JSON, not ""
    assert data["perplexity"] is None


def test_cli_format_json_no_table_text(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--format", "json"])
    assert result.exit_code == 0, result.output
    assert "Benchmark Results" not in result.output
    assert "Backend:" not in result.output


def test_cli_format_json_output_writes_json_file(tmp_config: Path, tmp_path: Path) -> None:
    import json

    out = tmp_path / "results.json"
    result = CliRunner().invoke(
        main, ["--config", str(tmp_config), "--format", "json", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["backend"] == "mock"
    assert "p50_latency_ms" in data


def test_cli_format_json_output_no_stdout_json(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    result = CliRunner().invoke(
        main, ["--config", str(tmp_config), "--format", "json", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    # When writing to file, stdout should not contain the JSON payload
    assert result.output.strip() == "" or not _is_json(result.output)


def _is_json(text: str) -> bool:
    import json

    try:
        json.loads(text)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# validate-config subcommand
# ---------------------------------------------------------------------------


def test_validate_config_exits_zero(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", str(tmp_config)])
    assert result.exit_code == 0, result.output


def test_validate_config_shows_ok(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", str(tmp_config)])
    assert "OK" in result.output


def test_validate_config_shows_config_path(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", str(tmp_config)])
    assert str(tmp_config) in result.output


def test_validate_config_shows_backend_and_model(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", str(tmp_config)])
    assert "mock" in result.output
    assert "test-model" in result.output


def test_validate_config_shows_requests(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", str(tmp_config)])
    assert "requests" in result.output
    assert "5" in result.output


def test_validate_config_shows_mock_backend_fields(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", str(tmp_config)])
    assert "mock.latency_ms" in result.output
    assert "mock.tokens_per_response" in result.output


def test_validate_config_optional_fields_hidden_when_not_set(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", str(tmp_config)])
    assert "workload_profile" not in result.output
    assert "quality_file" not in result.output
    assert "measure_perplexity" not in result.output
    assert "measure_judge" not in result.output
    assert "seed" not in result.output


def test_validate_config_shows_seed_when_set(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "seeded.yaml"
    cfg.write_text(
        f"backend: mock\nmodel: x\nrequests: 1\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\nseed: 42\n"
        f"mock:\n  latency_ms: 0\n  tokens_per_response: 5\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "seed" in result.output
    assert "42" in result.output


def test_validate_config_missing_config_fails() -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_validate_config_invalid_backend_fails(tmp_path: Path, tmp_prompts: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"backend: unknown_backend\nmodel: x\nprompts_file: {tmp_prompts}\n")
    result = CliRunner().invoke(main, ["validate-config", "--config", str(bad)])
    assert result.exit_code != 0


def test_validate_config_workload_profile_shown(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: mock\nmodel: x\nrequests: 2\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\nworkload_profile: short_chat\n"
        f"mock:\n  latency_ms: 0\n  tokens_per_response: 5\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg)])
    assert "workload_profile" in result.output
    assert "short_chat" in result.output


def test_validate_config_quality_file_shown(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: mock\nmodel: x\nrequests: 2\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\nquality_file: /some/quality.jsonl\n"
        f"mock:\n  latency_ms: 0\n  tokens_per_response: 5\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg)])
    assert "quality_file" in result.output
    assert "/some/quality.jsonl" in result.output


def test_validate_config_measure_flags_shown(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: mock\nmodel: x\nrequests: 2\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\nmeasure_perplexity: true\nmeasure_judge: true\n"
        f"mock:\n  latency_ms: 0\n  tokens_per_response: 5\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg)])
    assert "measure_perplexity" in result.output
    assert "measure_judge" in result.output


def test_validate_config_openai_backend(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: openai\nmodel: my-model\nrequests: 2\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"openai:\n  base_url: http://localhost:1234/v1\n  max_tokens: 64\n  timeout_s: 30\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "openai.base_url" in result.output
    assert "http://localhost:1234/v1" in result.output
    assert "OK" in result.output


def test_validate_config_openai_api_key_env_shown(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: openai\nmodel: my-model\nrequests: 2\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"openai:\n  api_key_env: MY_API_KEY\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg)])
    assert "api_key_env" in result.output
    assert "MY_API_KEY" in result.output


def test_validate_config_llama_cpp_backend(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: llama-cpp\nmodel: /path/to/model.gguf\nrequests: 2\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"llama_cpp:\n  n_ctx: 512\n  n_gpu_layers: 0\n  max_tokens: 64\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "llama_cpp.n_ctx" in result.output
    assert "OK" in result.output


def test_validate_config_shows_concurrency(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["validate-config", "--config", str(tmp_config)])
    assert result.exit_code == 0, result.output
    assert "concurrency" in result.output


def test_validate_config_concurrency_gt1_passes(tmp_path: Path, tmp_prompts: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: mock\nmodel: x\nrequests: 2\nwarmup_requests: 0\n"
        f"concurrency: 2\nprompts_file: {tmp_prompts}\n"
        f"mock:\n  latency_ms: 0\n  tokens_per_response: 5\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "concurrency" in result.output


# ---------------------------------------------------------------------------
# validate-config --format json
# ---------------------------------------------------------------------------


def test_validate_config_json_exits_zero(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(tmp_config), "--format", "json"]
    )
    assert result.exit_code == 0, result.output


def test_validate_config_json_emits_valid_json(tmp_config: Path) -> None:
    import json

    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(tmp_config), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["backend"] == "mock"
    assert data["model"] == "test-model"
    assert data["valid"] is True


def test_validate_config_json_contains_core_fields(tmp_config: Path) -> None:
    import json

    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(tmp_config), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    for field in ("requests", "concurrency", "warmup_requests", "repeats", "prompts_file"):
        assert field in data, f"missing field: {field}"


def test_validate_config_json_optional_fields_null_when_unset(tmp_config: Path) -> None:
    import json

    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(tmp_config), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["workload_profile"] is None
    assert data["quality_file"] is None
    assert data["seed"] is None


def test_validate_config_json_mock_backend_config(tmp_config: Path) -> None:
    import json

    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(tmp_config), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "backend_config" in data
    assert "latency_ms" in data["backend_config"]
    assert "tokens_per_response" in data["backend_config"]


def test_validate_config_json_no_table_text(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(tmp_config), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    assert "OK" not in result.output
    assert "Config:" not in result.output


def test_validate_config_json_seed_present_when_set(tmp_path: Path, tmp_prompts: Path) -> None:
    import json

    cfg = tmp_path / "seeded.yaml"
    cfg.write_text(
        f"backend: mock\nmodel: x\nrequests: 1\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\nseed: 7\n"
        f"mock:\n  latency_ms: 0\n  tokens_per_response: 5\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["seed"] == 7


def test_validate_config_json_openai_backend(tmp_path: Path, tmp_prompts: Path) -> None:
    import json

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: openai\nmodel: my-model\nrequests: 2\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"openai:\n  base_url: http://localhost:1234/v1\n  max_tokens: 64\n  timeout_s: 30\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["backend"] == "openai"
    assert data["backend_config"]["base_url"] == "http://localhost:1234/v1"
    assert data["valid"] is True


def test_validate_config_json_llama_cpp_backend(tmp_path: Path, tmp_prompts: Path) -> None:
    import json

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"backend: llama-cpp\nmodel: /path/to/model.gguf\nrequests: 2\nwarmup_requests: 0\n"
        f"prompts_file: {tmp_prompts}\n"
        f"llama_cpp:\n  n_ctx: 512\n  n_gpu_layers: 0\n  max_tokens: 64\n"
    )
    result = CliRunner().invoke(main, ["validate-config", "--config", str(cfg), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["backend"] == "llama-cpp"
    assert data["backend_config"]["n_ctx"] == 512
    assert data["valid"] is True


# ---------------------------------------------------------------------------
# CLI run-time overrides (--requests / --warmup-requests / --concurrency)
# ---------------------------------------------------------------------------


def test_requests_override(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--requests", "3"])
    assert result.exit_code == 0, result.output
    assert "request_count: 3" in result.output


def test_warmup_requests_override_zero(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--warmup-requests", "0"])
    assert result.exit_code == 0, result.output
    assert "Benchmark Results" in result.output


def test_concurrency_override(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--concurrency", "2"])
    assert result.exit_code == 0, result.output
    assert "Benchmark Results" in result.output


def test_requests_override_below_minimum_fails(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--requests", "0"])
    assert result.exit_code != 0


def test_concurrency_override_below_minimum_fails(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--concurrency", "0"])
    assert result.exit_code != 0


def test_all_overrides_combined(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(tmp_config),
            "--requests",
            "2",
            "--warmup-requests",
            "0",
            "--concurrency",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "request_count: 2" in result.output


# ---------------------------------------------------------------------------
# --seed override
# ---------------------------------------------------------------------------


def test_seed_override_exits_zero(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--seed", "42"])
    assert result.exit_code == 0, result.output


def test_seed_override_shown_in_header(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--seed", "42"])
    assert result.exit_code == 0, result.output
    assert "Seed: 42" in result.output


def test_seed_not_shown_when_absent(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config)])
    assert result.exit_code == 0, result.output
    assert "Seed:" not in result.output


def test_seed_override_zero(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--seed", "0"])
    assert result.exit_code == 0, result.output
    assert "Seed: 0" in result.output


def test_seed_override_combined_with_requests(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main, ["--config", str(tmp_config), "--seed", "7", "--requests", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "Seed: 7" in result.output
    assert "request_count: 2" in result.output


# ---------------------------------------------------------------------------
# --set KEY=VALUE overrides
# ---------------------------------------------------------------------------


def test_set_override_top_level_field(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--set", "requests=3"])
    assert result.exit_code == 0, result.output
    assert "request_count: 3" in result.output


def test_set_override_nested_field(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--set", "mock.latency_ms=0"])
    assert result.exit_code == 0, result.output
    assert "p50_latency_ms: 0" in result.output


def test_set_override_multiple(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["--config", str(tmp_config), "--set", "requests=2", "--set", "mock.latency_ms=0"],
    )
    assert result.exit_code == 0, result.output
    assert "request_count: 2" in result.output


def test_set_override_named_wins_over_set(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["--config", str(tmp_config), "--set", "requests=10", "--requests", "2"],
    )
    assert result.exit_code == 0, result.output
    assert "request_count: 2" in result.output


def test_set_override_missing_equals_fails(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--set", "noequals"])
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_set_override_unknown_path_fails(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main, ["--config", str(tmp_config), "--set", "doesnotexist.field=1"]
    )
    assert result.exit_code != 0
    assert "Unknown override path" in result.output


def test_set_override_wrong_type_fails(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main, ["--config", str(tmp_config), "--set", "mock.latency_ms=notanumber"]
    )
    assert result.exit_code != 0
    assert "Invalid --set value" in result.output


def test_set_override_bool_value(tmp_config: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--set", "mock.latency_ms=0"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# validate-config --set
# ---------------------------------------------------------------------------


def test_validate_config_set_override_reflects_value(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["validate-config", "--config", str(tmp_config), "--set", "mock.latency_ms=999"],
    )
    assert result.exit_code == 0, result.output
    assert "999" in result.output
    assert "OK" in result.output


def test_validate_config_set_override_nested_field(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["validate-config", "--config", str(tmp_config), "--set", "mock.tokens_per_response=77"],
    )
    assert result.exit_code == 0, result.output
    assert "77" in result.output


def test_validate_config_set_unknown_path_fails(tmp_config: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["validate-config", "--config", str(tmp_config), "--set", "bad.path=1"],
    )
    assert result.exit_code != 0
    assert "Unknown override path" in result.output


# env subcommand


def test_env_exits_zero() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert result.exit_code == 0, result.output


def test_env_includes_python_and_platform() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert result.exit_code == 0, result.output
    assert "python" in result.output
    assert "platform" in result.output
    assert "cpu" in result.output
    assert "package" in result.output


def test_env_includes_psutil() -> None:
    result = CliRunner().invoke(main, ["env"])
    assert result.exit_code == 0, result.output
    assert "psutil" in result.output


# help-text smoke tests — verify key flags are discoverable


def test_help_includes_set_flag() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--set" in result.output


def test_help_includes_seed_flag() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--seed" in result.output


def test_compare_help_includes_limit_flag() -> None:
    result = CliRunner().invoke(main, ["compare", "--help"])
    assert result.exit_code == 0, result.output
    assert "--limit" in result.output


def test_compare_help_includes_format_flag() -> None:
    result = CliRunner().invoke(main, ["compare", "--help"])
    assert result.exit_code == 0, result.output
    assert "--format" in result.output


def test_compare_help_includes_ttft_sort_option() -> None:
    result = CliRunner().invoke(main, ["compare", "--help"])
    assert result.exit_code == 0, result.output
    assert "ttft" in result.output


# ---------------------------------------------------------------------------
# Auto-mkdir: --output and --manifest create missing parent directories
# ---------------------------------------------------------------------------


def test_output_csv_auto_creates_parent_dir(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "new_subdir" / "deep" / "bench.csv"
    assert not out.parent.exists()
    result = CliRunner().invoke(main, ["--config", str(tmp_config), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_output_json_auto_creates_parent_dir(tmp_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "new_subdir" / "results.json"
    assert not out.parent.exists()
    result = CliRunner().invoke(
        main, ["--config", str(tmp_config), "--format", "json", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


# ---------------------------------------------------------------------------
# --base-url / --api-key flags
# ---------------------------------------------------------------------------


def test_base_url_requires_no_config() -> None:
    """--base-url alone (with mock server) should succeed without --config."""
    import json as _json
    from unittest.mock import MagicMock, patch

    fake_body = _json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": "Paris"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch(
        "llm_inference_benchmark.backends.openai_endpoint.urlopen",
        return_value=mock_resp,
    ):
        result = CliRunner().invoke(
            main,
            [
                "--base-url",
                "http://localhost:11434/v1",
                "--set",
                "model=llama3:3b",
                "--requests",
                "2",
                "--set",
                "warmup_requests=0",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Benchmark Results" in result.output


def test_base_url_overrides_config_openai_url(tmp_config: Path) -> None:
    """--base-url with --config switches backend to openai and overrides base_url."""
    import json as _json
    from unittest.mock import MagicMock, patch

    fake_body = _json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    captured_urls: list[str] = []

    def _mock_urlopen(req, **_kw):  # type: ignore[override]
        captured_urls.append(req.full_url)
        return mock_resp

    with patch(
        "llm_inference_benchmark.backends.openai_endpoint.urlopen",
        side_effect=_mock_urlopen,
    ):
        result = CliRunner().invoke(
            main,
            [
                "--config",
                str(tmp_config),
                "--base-url",
                "http://custom-host:1234/v1",
                "--requests",
                "1",
                "--set",
                "warmup_requests=0",
            ],
        )
    assert result.exit_code == 0, result.output
    assert any("custom-host:1234" in u for u in captured_urls)


def test_api_key_sent_as_bearer_header(tmp_config: Path) -> None:
    """--api-key value is sent as Authorization: Bearer header."""
    import json as _json
    from unittest.mock import MagicMock, patch

    fake_body = _json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    captured_headers: list[dict] = []

    def _mock_urlopen(req, **_kw):  # type: ignore[override]
        captured_headers.append(dict(req.headers))
        return mock_resp

    with patch(
        "llm_inference_benchmark.backends.openai_endpoint.urlopen",
        side_effect=_mock_urlopen,
    ):
        result = CliRunner().invoke(
            main,
            [
                "--base-url",
                "http://localhost:11434/v1",
                "--api-key",
                "sk-testkey",
                "--set",
                "model=llama3:3b",
                "--requests",
                "1",
                "--set",
                "warmup_requests=0",
            ],
        )
    assert result.exit_code == 0, result.output
    assert any(h.get("Authorization") == "Bearer sk-testkey" for h in captured_headers)


def test_no_config_no_base_url_fails() -> None:
    """Running without --config and without --base-url must fail with a clear error."""
    result = CliRunner().invoke(main, ["--requests", "1"])
    assert result.exit_code != 0
    assert "--config" in result.output or "--base-url" in result.output or "Error" in result.output
