"""Tests for the FastAPI Web UI backend (server.py)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import llm_inference_benchmark.server as server_mod
from llm_inference_benchmark.server import (
    _buffers,
    _get_db,
    _now_iso,
    _pareto_mask,
    _parse_metrics_from_output,
    _pull_errors,
    _render_pareto_html,
    _render_runs_table_rows,
    _row_to_result,
    _set_db_path,
    app,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path) -> Generator[Path, None, None]:
    """Redirect DB to a temp file and reset after each test."""
    db_file = tmp_path / "test_runs.db"
    _set_db_path(db_file)
    _buffers.clear()
    _pull_errors.clear()
    yield db_file
    _set_db_path(Path.home() / ".llm-bench" / "results.db")
    _buffers.clear()
    _pull_errors.clear()


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────


async def test_health(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_capabilities_returns_llama_cpp_gpu_flag(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "llama_cpp_gpu" in data
    assert isinstance(data["llama_cpp_gpu"], bool)


# ── Models ────────────────────────────────────────────────────────────────────


async def test_models_empty(client: httpx.AsyncClient) -> None:
    with patch.object(server_mod, "_discover_models", return_value=[]):
        resp = await client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == {"models": []}


async def test_models_returns_list(client: httpx.AsyncClient) -> None:
    fake_models = [
        {"type": "gguf", "name": "llama-q4.gguf", "path": "/models/llama-q4.gguf"},
        {"type": "hf", "name": "mistralai/Mistral-7B", "path": "/cache/models--mistralai--Mistral"},
    ]
    with patch.object(server_mod, "_discover_models", return_value=fake_models):
        resp = await client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json()["models"] == fake_models


def test_discover_models_no_dirs(tmp_path: Path) -> None:
    """Returns [] when neither ~/models nor HF cache dirs exist."""
    result = _fake_discover_with_root(tmp_path)
    assert result == []


def test_discover_models_gguf(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "llama-q4.gguf").write_text("fake")
    (models_dir / "mistral-q8.gguf").write_text("fake")

    result = _fake_discover_with_root(tmp_path)
    assert len(result) == 2
    assert all(m["type"] == "gguf" for m in result)
    names = {m["name"] for m in result}
    assert {"llama-q4.gguf", "mistral-q8.gguf"} == names


def test_discover_models_hf_cache(tmp_path: Path) -> None:
    hf_dir = tmp_path / ".cache" / "huggingface" / "hub"
    hf_dir.mkdir(parents=True)
    (hf_dir / "models--mistralai--Mistral-7B-v0.1").mkdir()
    (hf_dir / "models--meta-llama--Llama-2-7b-hf").mkdir()
    (hf_dir / "datasets--some--dataset").mkdir()  # should be ignored

    result = _fake_discover_with_root(tmp_path)
    assert len(result) == 2
    names = {m["name"] for m in result}
    assert "meta-llama/Llama-2-7b-hf" in names
    assert "mistralai/Mistral-7B-v0.1" in names
    assert all(m["type"] == "hf" for m in result)


def _fake_discover_with_root(root: Path) -> list[dict[str, str]]:
    """Run _discover_models logic against a custom root (test helper)."""
    results: list[dict[str, str]] = []
    gguf_root = root / "models"
    if gguf_root.is_dir():
        for p in sorted(gguf_root.glob("**/*.gguf")):
            results.append({"type": "gguf", "name": p.name, "path": str(p)})
    hf_hub = root / ".cache" / "huggingface" / "hub"
    if hf_hub.is_dir():
        for d in sorted(hf_hub.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                slug = d.name[len("models--") :]
                model_name = slug.replace("--", "/", 1)
                results.append({"type": "hf", "name": model_name, "path": str(d)})
    return results


# ── Runs — listing and querying ───────────────────────────────────────────────


async def test_list_runs_empty(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


async def test_get_run_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/runs/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


async def test_list_runs_with_data(client: httpx.AsyncClient, isolated_db: Path) -> None:
    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, output, created_at) VALUES (?,?,?,?,?)",
        ("run-abc", "done", '{"backend":"mock"}', "ok output", _now_iso()),
    )
    db.commit()

    resp = await client.get("/api/runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-abc"
    assert runs[0]["status"] == "done"
    assert runs[0]["output"] == "ok output"


async def test_get_run_found(client: httpx.AsyncClient, isolated_db: Path) -> None:
    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?,?,?,?)",
        ("run-xyz", "running", '{"backend":"mock"}', _now_iso()),
    )
    db.commit()

    resp = await client.get("/api/runs/run-xyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run-xyz"
    assert body["status"] == "running"
    assert body["config"] == {"backend": "mock"}


# ── Submit run ────────────────────────────────────────────────────────────────


async def test_submit_run_returns_run_id(client: httpx.AsyncClient) -> None:
    def _fake_do_run(run_id: str, config: dict[str, Any], dataset: str | None = None) -> None:
        db = _get_db()
        db.execute(
            "UPDATE runs SET status='done', output='bench output', finished_at=? WHERE id=?",
            (_now_iso(), run_id),
        )
        db.commit()

    with patch.object(server_mod, "_do_run", _fake_do_run):
        resp = await client.post("/api/runs", json={"config": {"backend": "mock"}})

    assert resp.status_code == 202
    assert "run_id" in resp.json()


async def test_submit_run_status_becomes_done(client: httpx.AsyncClient) -> None:
    def _fake_do_run(run_id: str, config: dict[str, Any], dataset: str | None = None) -> None:
        db = _get_db()
        db.execute(
            "UPDATE runs SET status='done', output='bench output', finished_at=? WHERE id=?",
            (_now_iso(), run_id),
        )
        db.commit()

    with patch.object(server_mod, "_do_run", _fake_do_run):
        resp = await client.post("/api/runs", json={"config": {"backend": "mock"}})

    run_id = resp.json()["run_id"]
    poll = await client.get(f"/api/runs/{run_id}")
    assert poll.json()["status"] == "done"


async def test_submit_run_appears_in_list(client: httpx.AsyncClient) -> None:
    def _noop(run_id: str, config: dict[str, Any], dataset: str | None = None) -> None:
        pass

    with patch.object(server_mod, "_do_run", _noop):
        resp = await client.post("/api/runs", json={"config": {"backend": "mock", "model": "test"}})

    run_id = resp.json()["run_id"]
    list_resp = await client.get("/api/runs")
    run_ids = [r["run_id"] for r in list_resp.json()["runs"]]
    assert run_id in run_ids


async def test_submit_run_error_status(client: httpx.AsyncClient) -> None:
    """Background task can mark run as error; API reflects that status."""

    def _error_do_run(run_id: str, config: dict[str, Any], dataset: str | None = None) -> None:
        db = _get_db()
        db.execute(
            "UPDATE runs SET status='error', output='crashed', finished_at=? WHERE id=?",
            (_now_iso(), run_id),
        )
        db.commit()

    with patch.object(server_mod, "_do_run", _error_do_run):
        resp = await client.post("/api/runs", json={"config": {"backend": "bad"}})

    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    poll = await client.get(f"/api/runs/{run_id}")
    assert poll.json()["status"] == "error"


# ── SSE streaming ─────────────────────────────────────────────────────────────


async def test_stream_run_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/runs/missing-id/stream")
    assert resp.status_code == 404


async def test_stream_completed_run_from_db(client: httpx.AsyncClient, isolated_db: Path) -> None:
    db = _get_db()
    db.execute(
        "INSERT INTO runs"
        " (id, status, config, output, created_at, finished_at)"
        " VALUES (?,?,?,?,?,?)",
        ("run-done", "done", '{"backend":"mock"}', "line1\nline2\nline3", _now_iso(), _now_iso()),
    )
    db.commit()

    resp = await client.get("/api/runs/run-done/stream")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "data: line1" in text
    assert "data: line2" in text
    assert "data: line3" in text
    assert "data: [done:done]" in text


async def test_stream_run_from_active_buffer(client: httpx.AsyncClient, isolated_db: Path) -> None:
    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?,?,?,?)",
        ("run-live", "done", '{"backend":"mock"}', _now_iso()),
    )
    db.commit()
    _buffers["run-live"] = ["alpha", "beta", "gamma"]

    resp = await client.get("/api/runs/run-live/stream")
    assert resp.status_code == 200
    text = resp.text
    assert "data: alpha" in text
    assert "data: beta" in text
    assert "data: gamma" in text
    assert "data: [done:done]" in text


# ── Internal helpers ──────────────────────────────────────────────────────────


def test_now_iso_format() -> None:
    ts = _now_iso()
    assert "T" in ts
    assert ts.endswith("+00:00")


def test_row_to_result(isolated_db: Path) -> None:
    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, output, created_at, finished_at)"
        " VALUES (?,?,?,?,?,?)",
        ("r1", "done", '{"backend":"mock"}', "out", "2026-01-01T00:00:00+00:00", None),
    )
    db.commit()
    row = db.execute("SELECT * FROM runs WHERE id='r1'").fetchone()
    result = _row_to_result(row)
    assert result.run_id == "r1"
    assert result.status == "done"
    assert result.config == {"backend": "mock"}
    assert result.output == "out"


# ── do_run unit tests (mocked subprocess) ────────────────────────────────────


def test_do_run_success(isolated_db: Path) -> None:
    """_do_run marks run as 'done' when subprocess exits 0."""
    from llm_inference_benchmark.server import _do_run

    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?,?,?,?)",
        ("run-sub", "pending", '{"backend":"mock"}', _now_iso()),
    )
    db.commit()

    mock_proc = _MockProc(rc=0, lines=["bench line 1", "bench line 2"])
    with patch("llm_inference_benchmark.server.subprocess.Popen", return_value=mock_proc):
        _do_run("run-sub", {"backend": "mock"})

    row = db.execute("SELECT * FROM runs WHERE id='run-sub'").fetchone()
    assert row["status"] == "done"
    assert "bench line 1" in row["output"]
    assert _buffers.get("run-sub") is not None


def test_do_run_failure(isolated_db: Path) -> None:
    """_do_run marks run as 'error' when subprocess exits non-zero."""
    from llm_inference_benchmark.server import _do_run

    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?,?,?,?)",
        ("run-fail", "pending", '{"backend":"mock"}', _now_iso()),
    )
    db.commit()

    mock_proc = _MockProc(rc=1, lines=["error: something went wrong"])
    with patch("llm_inference_benchmark.server.subprocess.Popen", return_value=mock_proc):
        _do_run("run-fail", {"backend": "mock"})

    row = db.execute("SELECT * FROM runs WHERE id='run-fail'").fetchone()
    assert row["status"] == "error"


class _MockProc:
    """Minimal subprocess.Popen mock for _do_run tests."""

    def __init__(self, rc: int, lines: list[str]) -> None:
        self.returncode = rc
        self.stdout: Any = iter(line + "\n" for line in lines)

    def wait(self) -> int:
        return self.returncode


# ── _parse_metrics_from_output ────────────────────────────────────────────────


def test_parse_metrics_none_output() -> None:
    m = _parse_metrics_from_output(None)
    assert m["tokens_per_second"] is None
    assert m["p50_ttft_ms"] is None
    assert m["p95_latency_ms"] is None


def test_parse_metrics_from_typical_output() -> None:
    output = (
        "Backend: mock  Model: gpt2  Requests: 10\n"
        "=== Benchmark Results ===\n"
        "  request_count: 10\n"
        "  tokens_per_second: 123.45\n"
        "  p95_latency_ms: 210.00\n"
        "  p50_ttft_ms: 55.20\n"
    )
    m = _parse_metrics_from_output(output)
    assert m["tokens_per_second"] == pytest.approx(123.45)
    assert m["p95_latency_ms"] == pytest.approx(210.0)
    assert m["p50_ttft_ms"] == pytest.approx(55.20)


def test_parse_metrics_na_values() -> None:
    output = "  tokens_per_second: 80.00\n  p50_ttft_ms: N/A\n"
    m = _parse_metrics_from_output(output)
    assert m["tokens_per_second"] == pytest.approx(80.0)
    assert m["p50_ttft_ms"] is None


# ── _pareto_mask ──────────────────────────────────────────────────────────────


def test_pareto_mask_single_point() -> None:
    assert _pareto_mask([(100.0, 50.0)]) == [True]


def test_pareto_mask_dominated() -> None:
    # point 0 is dominated by point 1 (lower latency, higher throughput)
    mask = _pareto_mask([(200.0, 40.0), (100.0, 60.0)])
    assert mask == [False, True]


def test_pareto_mask_two_on_front() -> None:
    # Neither dominates the other (tradeoff)
    mask = _pareto_mask([(100.0, 40.0), (200.0, 80.0)])
    assert mask == [True, True]


# ── Dashboard and fragment endpoints ─────────────────────────────────────────


async def test_dashboard_returns_html(client: httpx.AsyncClient) -> None:
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "llm-bench" in resp.text
    assert "run-list" in resp.text


async def test_runs_table_fragment_empty(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/ui/runs-table")
    assert resp.status_code == 200
    assert "No runs yet" in resp.text


async def test_runs_table_fragment_with_data(client: httpx.AsyncClient, isolated_db: Path) -> None:
    db = _get_db()
    output = (
        "=== Benchmark Results ===\n"
        "  tokens_per_second: 95.10\n"
        "  p95_latency_ms: 180.00\n"
        "  p50_ttft_ms: N/A\n"
    )
    db.execute(
        "INSERT INTO runs (id, status, config, output, created_at) VALUES (?,?,?,?,?)",
        ("abc12345", "done", '{"backend":"mock","model":"gpt2"}', output, _now_iso()),
    )
    db.commit()

    resp = await client.get("/api/ui/runs-table")
    assert resp.status_code == 200
    assert "abc123" in resp.text
    assert "95.1" in resp.text
    assert "mock" in resp.text
    assert "badge-done" in resp.text


async def test_pareto_page_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/runs/nonexistent-id/pareto.html")
    assert resp.status_code == 404


async def test_pareto_page_no_metrics(client: httpx.AsyncClient, isolated_db: Path) -> None:
    _get_db().execute(
        "INSERT INTO runs (id, status, config, output, created_at) VALUES (?,?,?,?,?)",
        ("run-p1", "done", '{"backend":"mock","model":"m"}', None, _now_iso()),
    )
    _get_db().commit()
    resp = await client.get("/runs/run-p1/pareto.html")
    assert resp.status_code == 200
    assert "No completed runs" in resp.text


async def test_pareto_page_with_metrics(client: httpx.AsyncClient, isolated_db: Path) -> None:
    output = "  tokens_per_second: 70.0\n  p95_latency_ms: 150.0\n"
    _get_db().execute(
        "INSERT INTO runs (id, status, config, output, created_at) VALUES (?,?,?,?,?)",
        ("run-p2", "done", '{"backend":"mock","model":"m"}', output, _now_iso()),
    )
    _get_db().commit()
    resp = await client.get("/runs/run-p2/pareto.html")
    assert resp.status_code == 200
    assert "Plotly.newPlot" in resp.text
    assert "run-p2"[:8] in resp.text


# ── _render_runs_table_rows and _render_pareto_html unit tests ────────────────


def test_render_runs_table_rows_empty() -> None:
    html_out = _render_runs_table_rows([])
    assert "No runs yet" in html_out


def test_render_pareto_html_no_metrics(isolated_db: Path) -> None:
    from llm_inference_benchmark.server import RunResult

    result = RunResult(
        run_id="abc",
        status="done",
        config={},
        output=None,
        created_at=_now_iso(),
        finished_at=None,
    )
    out = _render_pareto_html("abc", [result])
    assert "No completed runs" in out


def test_render_pareto_html_with_metrics(isolated_db: Path) -> None:
    from llm_inference_benchmark.server import RunResult

    output = "  tokens_per_second: 60.0\n  p95_latency_ms: 200.0\n"
    result = RunResult(
        run_id="abc",
        status="done",
        config={"backend": "mock", "model": "m"},
        output=output,
        created_at=_now_iso(),
        finished_at=None,
    )
    out = _render_pareto_html("abc", [result])
    assert "Plotly.newPlot" in out
    assert "60.0" in out


# ── Datasets API ──────────────────────────────────────────────────────────────


async def test_list_datasets_returns_registry(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert "datasets" in data
    names = [d["name"] for d in data["datasets"]]
    assert "lmsys-chat" in names
    assert "hermes-fn" in names
    for entry in data["datasets"]:
        assert "name" in entry
        assert "description" in entry
        assert "cached" in entry
        assert "samples" in entry


async def test_list_datasets_cached_false_when_not_pulled(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        resp = await client.get("/api/datasets")
    assert resp.status_code == 200
    for entry in resp.json()["datasets"]:
        assert entry["cached"] is False
        assert entry["samples"] == 0


async def test_pull_dataset_unknown_name(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/datasets/pull", json={"name": "does-not-exist"})
    assert resp.status_code == 422


async def test_pull_dataset_starts_background_task(client: httpx.AsyncClient) -> None:
    with patch("llm_inference_benchmark.server._do_pull_dataset") as mock_pull:
        resp = await client.post("/api/datasets/pull", json={"name": "lmsys-chat"})
    assert resp.status_code == 202
    assert resp.json() == {"status": "started"}
    mock_pull.assert_called_once_with("lmsys-chat")


async def test_datasets_table_fragment_returns_html(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/ui/datasets-table")
    assert resp.status_code == 200
    assert "lmsys-chat" in resp.text
    assert "hermes-fn" in resp.text


def test_render_datasets_table_empty() -> None:
    from llm_inference_benchmark.server import _render_datasets_table

    out = _render_datasets_table([])
    assert "No datasets registered" in out


def test_render_datasets_table_cached_entry() -> None:
    from llm_inference_benchmark.server import _render_datasets_table

    statuses = [
        {
            "name": "wildchat",
            "description": "Public chat",
            "cached": True,
            "samples": 42,
            "error": None,
        }
    ]
    out = _render_datasets_table(statuses)
    assert "wildchat" in out
    assert "42" in out
    assert "✓" in out


def test_render_datasets_table_uncached_entry() -> None:
    from llm_inference_benchmark.server import _render_datasets_table

    statuses = [
        {
            "name": "hermes-fn",
            "description": "Function calls",
            "cached": False,
            "samples": 0,
            "error": None,
        }
    ]
    out = _render_datasets_table(statuses)
    assert "✗" in out
    assert "—" in out


def test_render_datasets_table_shows_error() -> None:
    from llm_inference_benchmark.server import _render_datasets_table

    statuses = [
        {
            "name": "lmsys-chat",
            "description": "Chat logs",
            "cached": False,
            "samples": 0,
            "error": "Connection timed out",
        }
    ]
    out = _render_datasets_table(statuses)
    assert "ds-pull-error" in out
    assert "Pull failed:" in out
    assert "Connection timed out" in out


def test_render_datasets_table_no_error_when_none() -> None:
    from llm_inference_benchmark.server import _render_datasets_table

    statuses = [
        {
            "name": "lmsys-chat",
            "description": "Chat logs",
            "cached": False,
            "samples": 0,
            "error": None,
        }
    ]
    out = _render_datasets_table(statuses)
    assert "ds-pull-error" not in out
    assert "Pull failed" not in out


def test_do_pull_dataset_records_error_on_failure() -> None:
    from llm_inference_benchmark.server import _do_pull_dataset

    with patch("llm_inference_benchmark.server._datasets_mod") as mock_ds:
        mock_ds.pull.side_effect = RuntimeError("disk full")
        _do_pull_dataset("lmsys-chat")

    assert _pull_errors.get("lmsys-chat") == "disk full"


def test_do_pull_dataset_clears_error_on_success() -> None:
    from llm_inference_benchmark.server import _do_pull_dataset

    _pull_errors["lmsys-chat"] = "previous error"
    with patch("llm_inference_benchmark.server._datasets_mod") as mock_ds:
        mock_ds.pull.return_value = None
        _do_pull_dataset("lmsys-chat")

    assert "lmsys-chat" not in _pull_errors


async def test_list_datasets_includes_error_field(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _pull_errors["lmsys-chat"] = "test error"
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        resp = await client.get("/api/datasets")
    assert resp.status_code == 200
    entry = next(d for d in resp.json()["datasets"] if d["name"] == "lmsys-chat")
    assert entry["error"] == "test error"


async def test_datasets_table_fragment_shows_error(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _pull_errors["lmsys-chat"] = "Network unreachable"
    with patch("llm_inference_benchmark.datasets.cache_dir", return_value=tmp_path):
        resp = await client.get("/api/ui/datasets-table")
    assert resp.status_code == 200
    assert "Pull failed:" in resp.text
    assert "Network unreachable" in resp.text


# ── Dataset selector in run form ──────────────────────────────────────────────


async def test_submit_run_with_dataset_passes_to_do_run(client: httpx.AsyncClient) -> None:
    """POST /api/runs with dataset= calls _do_run with that dataset name."""
    captured: dict[str, Any] = {}

    def _capture(run_id: str, config: dict[str, Any], dataset: str | None = None) -> None:
        captured["dataset"] = dataset

    with patch.object(server_mod, "_do_run", _capture):
        resp = await client.post(
            "/api/runs",
            json={"config": {"backend": "mock"}, "dataset": "wildchat"},
        )

    assert resp.status_code == 202
    assert captured["dataset"] == "wildchat"


async def test_submit_run_without_dataset_defaults_none(client: httpx.AsyncClient) -> None:
    """POST /api/runs without dataset= passes dataset=None to _do_run."""
    captured: dict[str, Any] = {}

    def _capture(run_id: str, config: dict[str, Any], dataset: str | None = None) -> None:
        captured["dataset"] = dataset

    with patch.object(server_mod, "_do_run", _capture):
        resp = await client.post("/api/runs", json={"config": {"backend": "mock"}})

    assert resp.status_code == 202
    assert captured["dataset"] is None


def test_do_run_passes_dataset_flag_to_subprocess(isolated_db: Path) -> None:
    """_do_run appends --dataset <name> to the subprocess command when dataset is set."""
    from llm_inference_benchmark.server import _do_run

    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?,?,?,?)",
        ("run-ds", "pending", '{"backend":"mock"}', _now_iso()),
    )
    db.commit()

    captured_cmd: list[list[str]] = []

    class _MockProcDs:
        returncode = 0
        stdout: Any = iter([])

        def wait(self) -> int:
            return 0

    def _fake_popen(cmd: list[str], **_: Any) -> _MockProcDs:
        captured_cmd.append(cmd)
        return _MockProcDs()

    with patch("llm_inference_benchmark.server.subprocess.Popen", _fake_popen):
        _do_run("run-ds", {"backend": "mock"}, dataset="wildchat")

    assert len(captured_cmd) == 1
    cmd = captured_cmd[0]
    assert "--dataset" in cmd
    assert cmd[cmd.index("--dataset") + 1] == "wildchat"


def test_do_run_no_dataset_flag_when_none(isolated_db: Path) -> None:
    """_do_run does not add --dataset when dataset is None."""
    from llm_inference_benchmark.server import _do_run

    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?,?,?,?)",
        ("run-nods", "pending", '{"backend":"mock"}', _now_iso()),
    )
    db.commit()

    captured_cmd: list[list[str]] = []

    class _MockProcNone:
        returncode = 0
        stdout: Any = iter([])

        def wait(self) -> int:
            return 0

    def _fake_popen(cmd: list[str], **_: Any) -> _MockProcNone:
        captured_cmd.append(cmd)
        return _MockProcNone()

    with patch("llm_inference_benchmark.server.subprocess.Popen", _fake_popen):
        _do_run("run-nods", {"backend": "mock"}, dataset=None)

    assert "--dataset" not in captured_cmd[0]


async def test_dashboard_contains_dataset_selector(client: httpx.AsyncClient) -> None:
    """The dashboard HTML includes the dataset selector element for the run form."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "f-dataset" in resp.text
    assert "Default prompts" in resp.text


async def test_dashboard_contains_model_select(client: httpx.AsyncClient) -> None:
    """The dashboard HTML includes the model select element for the run form."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "f-model" in resp.text
    # loadModels lives in the external app.js file
    js_resp = await client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "loadModels" in js_resp.text


async def test_dashboard_loadmodels_uses_type_tag(client: httpx.AsyncClient) -> None:
    """The dashboard JS maps model types to human-readable backend tags."""
    js_resp = await client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "typeTag" in js_resp.text
    assert "llama.cpp" in js_resp.text
    assert "transformers" in js_resp.text


async def test_models_api_includes_type_and_name(
    client: httpx.AsyncClient,
) -> None:
    """/api/models returns type and name fields used by the dropdown renderer."""
    fake_models = [
        {"type": "gguf", "name": "llama-q4.gguf", "path": "/models/llama-q4.gguf"},
        {
            "type": "hf",
            "name": "meta-llama/Llama-3-8B",
            "path": "/cache/models--meta-llama--Llama-3-8B",
        },
    ]
    with patch.object(server_mod, "_discover_models", return_value=fake_models):
        resp = await client.get("/api/models")
    data = resp.json()
    assert data["models"][0]["type"] == "gguf"
    assert data["models"][0]["name"] == "llama-q4.gguf"
    assert data["models"][1]["type"] == "hf"
    assert data["models"][1]["name"] == "meta-llama/Llama-3-8B"


# ── Delete run ────────────────────────────────────────────────────────────────


async def test_delete_run_done(client: httpx.AsyncClient) -> None:
    """DELETE /api/runs/{id} removes a completed run and returns 204."""
    db = _get_db()
    run_id = "aaaaaaaa-0000-0000-0000-000000000001"
    db.execute(
        "INSERT INTO runs (id, status, config, output, created_at, finished_at) "
        "VALUES (?, 'done', '{}', 'ok', ?, ?)",
        (run_id, _now_iso(), _now_iso()),
    )
    db.commit()

    resp = await client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 204

    row = db.execute("SELECT id FROM runs WHERE id=?", (run_id,)).fetchone()
    assert row is None


async def test_delete_run_error_status(client: httpx.AsyncClient) -> None:
    """DELETE /api/runs/{id} also removes runs in 'error' status."""
    db = _get_db()
    run_id = "aaaaaaaa-0000-0000-0000-000000000002"
    db.execute(
        "INSERT INTO runs (id, status, config, output, created_at, finished_at) "
        "VALUES (?, 'error', '{}', 'boom', ?, ?)",
        (run_id, _now_iso(), _now_iso()),
    )
    db.commit()

    resp = await client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 204


async def test_delete_run_not_found(client: httpx.AsyncClient) -> None:
    """DELETE /api/runs/{id} returns 404 for an unknown run ID."""
    resp = await client.delete("/api/runs/does-not-exist")
    assert resp.status_code == 404


async def test_delete_run_pending_returns_409(client: httpx.AsyncClient) -> None:
    """DELETE /api/runs/{id} returns 409 when the run is still pending."""
    db = _get_db()
    run_id = "aaaaaaaa-0000-0000-0000-000000000003"
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?, 'pending', '{}', ?)",
        (run_id, _now_iso()),
    )
    db.commit()

    resp = await client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 409


async def test_delete_run_running_returns_409(client: httpx.AsyncClient) -> None:
    """DELETE /api/runs/{id} returns 409 when the run is actively running."""
    db = _get_db()
    run_id = "aaaaaaaa-0000-0000-0000-000000000004"
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?, 'running', '{}', ?)",
        (run_id, _now_iso()),
    )
    db.commit()

    resp = await client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 409


async def test_delete_run_removes_buffer(client: httpx.AsyncClient) -> None:
    """DELETE /api/runs/{id} also clears the in-memory streaming buffer."""
    db = _get_db()
    run_id = "aaaaaaaa-0000-0000-0000-000000000005"
    db.execute(
        "INSERT INTO runs (id, status, config, output, created_at, finished_at) "
        "VALUES (?, 'done', '{}', 'output', ?, ?)",
        (run_id, _now_iso(), _now_iso()),
    )
    db.commit()
    _buffers[run_id] = ["line1", "line2"]

    resp = await client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 204
    assert run_id not in _buffers
