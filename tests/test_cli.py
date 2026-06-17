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
