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
    yield db_file
    _set_db_path(Path.home() / ".llm-bench" / "results.db")
    _buffers.clear()


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
    def _fake_do_run(run_id: str, config: dict[str, Any]) -> None:
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
    def _fake_do_run(run_id: str, config: dict[str, Any]) -> None:
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
    def _noop(run_id: str, config: dict[str, Any]) -> None:
        pass

    with patch.object(server_mod, "_do_run", _noop):
        resp = await client.post("/api/runs", json={"config": {"backend": "mock", "model": "test"}})

    run_id = resp.json()["run_id"]
    list_resp = await client.get("/api/runs")
    run_ids = [r["run_id"] for r in list_resp.json()["runs"]]
    assert run_id in run_ids


async def test_submit_run_error_status(client: httpx.AsyncClient) -> None:
    """Background task can mark run as error; API reflects that status."""

    def _error_do_run(run_id: str, config: dict[str, Any]) -> None:
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
