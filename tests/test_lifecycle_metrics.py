"""Tests for lifecycle metrics: model_load_ms and warmup_p50_latency_ms."""

from __future__ import annotations

import csv
import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_inference_benchmark.backends.mock import MockBackend
from llm_inference_benchmark.compare import load_csv
from llm_inference_benchmark.config import BenchmarkConfig
from llm_inference_benchmark.metrics import MetricsReport, RequestMetrics, compute_metrics
from llm_inference_benchmark.runner import load_prompts, run_benchmark

# ---------------------------------------------------------------------------
# MetricsReport field presence and defaults
# ---------------------------------------------------------------------------


def _results(n: int = 3) -> list[RequestMetrics]:
    return [RequestMetrics(latency_ms=10.0, input_tokens=5, output_tokens=5) for _ in range(n)]


def test_model_load_ms_defaults_to_none() -> None:
    report = compute_metrics(_results(), backend="mock", model="t")
    assert report.model_load_ms is None


def test_warmup_p50_defaults_to_none() -> None:
    report = compute_metrics(_results(), backend="mock", model="t")
    assert report.warmup_p50_latency_ms is None


def test_model_load_ms_passed_through() -> None:
    report = compute_metrics(_results(), backend="mock", model="t", model_load_ms=123.4)
    assert report.model_load_ms == pytest.approx(123.4)


def test_warmup_p50_passed_through() -> None:
    report = compute_metrics(_results(), backend="mock", model="t", warmup_p50_latency_ms=7.5)
    assert report.warmup_p50_latency_ms == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# run_benchmark: warmup latency capture
# ---------------------------------------------------------------------------


def test_run_benchmark_warmup_p50_none_when_zero_warmup(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text("Hello\nWorld\n")
    backend = MockBackend(model="test", latency_ms=5)
    cfg = BenchmarkConfig(requests=3, warmup_requests=0, prompts_file=str(p))
    report = run_benchmark(backend, cfg, load_prompts(p))
    assert report.warmup_p50_latency_ms is None


def test_run_benchmark_warmup_p50_present_when_warmup_gt_zero(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text("Hello\nWorld\n")
    backend = MockBackend(model="test", latency_ms=5)
    cfg = BenchmarkConfig(requests=3, warmup_requests=2, prompts_file=str(p))
    report = run_benchmark(backend, cfg, load_prompts(p))
    assert report.warmup_p50_latency_ms is not None
    assert report.warmup_p50_latency_ms >= 0.0


def test_run_benchmark_model_load_ms_none_by_default(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text("Hello\n")
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(requests=2, warmup_requests=0, prompts_file=str(p))
    report = run_benchmark(backend, cfg, load_prompts(p))
    assert report.model_load_ms is None


def test_run_benchmark_model_load_ms_passed_through(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text("Hello\n")
    backend = MockBackend(model="test", latency_ms=0)
    cfg = BenchmarkConfig(requests=2, warmup_requests=0, prompts_file=str(p))
    report = run_benchmark(backend, cfg, load_prompts(p), model_load_ms=42.0)
    assert report.model_load_ms == pytest.approx(42.0)


def test_run_benchmark_warmup_p50_single_request(tmp_path: Path) -> None:
    """With warmup_requests=1 the p50 equals that single request's latency."""
    p = tmp_path / "prompts.txt"
    p.write_text("Hello\n")
    backend = MockBackend(model="test", latency_ms=10)
    cfg = BenchmarkConfig(requests=1, warmup_requests=1, prompts_file=str(p))
    report = run_benchmark(backend, cfg, load_prompts(p))
    assert report.warmup_p50_latency_ms is not None
    # MockBackend sleeps for latency_ms; expect at least 10 ms
    assert report.warmup_p50_latency_ms >= 10.0


# ---------------------------------------------------------------------------
# CSV round-trip: new fields written and blank when None
# ---------------------------------------------------------------------------


def test_csv_roundtrip_lifecycle_fields_written(tmp_path: Path) -> None:
    """lifecycle fields appear in CSV; blank when None, numeric when set."""
    report = MetricsReport(
        request_count=1,
        p50_latency_ms=5.0,
        p95_latency_ms=5.0,
        tokens_per_second=100.0,
        total_tokens=10,
        backend="mock",
        model="m",
        peak_cpu_memory_mb=50.0,
        peak_cuda_memory_mb=None,
        peak_vram_memory_mb=None,
        model_load_ms=12.3,
        warmup_p50_latency_ms=None,
        timestamp=datetime.now(UTC).isoformat(),
    )
    csv_path = tmp_path / "out.csv"
    row = {k: ("" if v is None else v) for k, v in dataclasses.asdict(report).items()}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["model_load_ms"] == "12.3"
    assert rows[0]["warmup_p50_latency_ms"] == ""  # None → blank


# ---------------------------------------------------------------------------
# Backward compatibility: older CSVs without lifecycle columns still load
# ---------------------------------------------------------------------------

_LEGACY_FIXTURE = Path(__file__).parent / "fixtures" / "mock_run.csv"


def test_load_csv_backward_compat_no_lifecycle_columns() -> None:
    """CSVs without model_load_ms / warmup_p50_latency_ms load without error."""
    assert _LEGACY_FIXTURE.exists(), f"fixture missing: {_LEGACY_FIXTURE}"
    with open(_LEGACY_FIXTURE) as f:
        headers = next(csv.reader(f))
    assert "model_load_ms" not in headers
    assert "warmup_p50_latency_ms" not in headers
    # Must not raise
    row = load_csv(_LEGACY_FIXTURE)
    assert row.backend == "mock"
