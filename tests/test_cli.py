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
    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(cfg), "--format", "json"]
    )
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
    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(cfg), "--format", "json"]
    )
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
    result = CliRunner().invoke(
        main, ["validate-config", "--config", str(cfg), "--format", "json"]
    )
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
