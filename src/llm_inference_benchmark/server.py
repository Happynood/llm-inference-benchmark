"""FastAPI HTTP backend for llm-bench Web UI.

Exposes six endpoints:
  GET  /api/health
  GET  /api/models
  GET  /api/runs
  POST /api/runs
  GET  /api/runs/{run_id}
  GET  /api/runs/{run_id}/stream   (Server-Sent Events)
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Database ─────────────────────────────────────────────────────────────────

_DB_PATH: Path = Path.home() / ".llm-bench" / "results.db"
_thread_local = threading.local()

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS runs (
        id          TEXT PRIMARY KEY,
        status      TEXT NOT NULL,
        config      TEXT NOT NULL,
        output      TEXT,
        created_at  TEXT NOT NULL,
        finished_at TEXT
    )
"""


def _set_db_path(path: Path) -> None:
    """Override DB path; closes any open thread-local connection (used in tests)."""
    global _DB_PATH
    _DB_PATH = path
    conn: sqlite3.Connection | None = getattr(_thread_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _thread_local.conn = None


def _get_db() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_thread_local, "conn", None)
    if conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_SCHEMA)
        conn.commit()
        _thread_local.conn = conn
    return conn


# ── Streaming buffers (keyed by run_id, alive while the run is active) ───────

_buffers: dict[str, list[str]] = {}


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="llm-bench Web API", version="1.0.0")


# ── Pydantic models ───────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    config: dict[str, Any]


class RunSubmitted(BaseModel):
    run_id: str


class RunResult(BaseModel):
    run_id: str
    status: str
    config: dict[str, Any] | None = None
    output: str | None = None
    created_at: str
    finished_at: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_result(row: sqlite3.Row) -> RunResult:
    return RunResult(
        run_id=row["id"],
        status=row["status"],
        config=json.loads(row["config"]),
        output=row["output"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
    )


def _discover_models() -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    gguf_root = Path.home() / "models"
    if gguf_root.is_dir():
        for p in sorted(gguf_root.glob("**/*.gguf")):
            results.append({"type": "gguf", "name": p.name, "path": str(p)})

    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.is_dir():
        for d in sorted(hf_hub.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                slug = d.name[len("models--") :]
                model_name = slug.replace("--", "/", 1)
                results.append({"type": "hf", "name": model_name, "path": str(d)})

    return results


# ── Background benchmark runner ───────────────────────────────────────────────


def _do_run(run_id: str, config: dict[str, Any]) -> None:
    """Execute llm-bench with *config*; updates DB status and streaming buffer."""
    import os

    import yaml

    buf: list[str] = []
    _buffers[run_id] = buf
    db = _get_db()

    db.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
    db.commit()

    rc = 1
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(config, tmp)
            cfg_path = tmp.name

        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "llm_inference_benchmark.cli", "--config", cfg_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                buf.append(raw.rstrip("\n"))
            proc.wait()
            rc = proc.returncode
        finally:
            os.unlink(cfg_path)

        status = "done" if rc == 0 else "error"
        db.execute(
            "UPDATE runs SET status=?, output=?, finished_at=? WHERE id=?",
            (status, "\n".join(buf), _now_iso(), run_id),
        )
        db.commit()
    except Exception as exc:
        buf.append(f"ERROR: {exc}")
        db.execute(
            "UPDATE runs SET status='error', output=?, finished_at=? WHERE id=?",
            ("\n".join(buf), _now_iso(), run_id),
        )
        db.commit()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def list_models() -> dict[str, list[dict[str, str]]]:
    return {"models": _discover_models()}


@app.get("/api/runs")
async def list_runs() -> dict[str, list[RunResult]]:
    rows = _get_db().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    return {"runs": [_row_to_result(r) for r in rows]}


@app.post("/api/runs", status_code=202)
async def submit_run(body: RunRequest, background_tasks: BackgroundTasks) -> RunSubmitted:
    run_id = str(uuid.uuid4())
    db = _get_db()
    db.execute(
        "INSERT INTO runs (id, status, config, created_at) VALUES (?, 'pending', ?, ?)",
        (run_id, json.dumps(body.config), _now_iso()),
    )
    db.commit()
    background_tasks.add_task(_do_run, run_id, body.config)
    return RunSubmitted(run_id=run_id)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> RunResult:
    row = _get_db().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return _row_to_result(row)


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    row = _get_db().execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    async def _generate() -> AsyncGenerator[str, None]:
        sent = 0
        while True:
            buf = _buffers.get(run_id)
            if buf is not None:
                new_lines = buf[sent:]
                for line in new_lines:
                    yield f"data: {line}\n\n"
                sent += len(new_lines)

            current = (
                _get_db()
                .execute("SELECT status, output FROM runs WHERE id=?", (run_id,))
                .fetchone()
            )
            if current is None:
                yield "data: [error: run disappeared]\n\n"
                return

            status: str = current["status"]
            if status in ("done", "error"):
                # If the run completed before we had a buffer (e.g. server restart),
                # stream from stored output instead.
                if buf is None:
                    stored: str = current["output"] or ""
                    for line in stored.split("\n"):
                        yield f"data: {line}\n\n"
                yield f"data: [done:{status}]\n\n"
                return

            await asyncio.sleep(0.2)

    return StreamingResponse(_generate(), media_type="text/event-stream")
