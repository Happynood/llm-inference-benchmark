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
