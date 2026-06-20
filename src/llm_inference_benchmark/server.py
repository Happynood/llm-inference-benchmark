"""FastAPI HTTP backend for llm-bench Web UI.

Exposes the following endpoints:
  GET  /                              dashboard (HTML)
  GET  /api/health
  GET  /api/models
  GET  /api/runs
  POST /api/runs
  GET  /api/runs/{run_id}
  GET  /api/runs/{run_id}/stream      Server-Sent Events
  GET  /api/ui/runs-table             HTMX HTML fragment
  GET  /runs/{run_id}/pareto.html     interactive Plotly scatter page
"""

from __future__ import annotations

import asyncio
import html
import json
import re
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
from fastapi.responses import HTMLResponse, StreamingResponse
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


# ── Metrics parsing ───────────────────────────────────────────────────────────


def _parse_metrics_from_output(output: str | None) -> dict[str, float | None]:
    """Extract key numeric metrics from benchmark stdout text."""
    if not output:
        return {"tokens_per_second": None, "p50_ttft_ms": None, "p95_latency_ms": None}
    result: dict[str, float | None] = {}
    for key in ("tokens_per_second", "p50_ttft_ms", "p95_latency_ms"):
        m = re.search(rf"^\s+{re.escape(key)}: ([0-9]+(?:\.[0-9]+)?)", output, re.MULTILINE)
        result[key] = float(m.group(1)) if m else None
    return result


def _pareto_mask(points: list[tuple[float, float]]) -> list[bool]:
    """Return True for each non-dominated point (lower latency AND higher throughput is better)."""
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lj, tj = points[j]
            li, ti = points[i]
            if lj <= li and tj >= ti and (lj < li or tj > ti):
                dominated[i] = True
                break
    return [not d for d in dominated]


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>llm-bench</title>
  <script src="https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
  <style>
    *{box-sizing:border-box}
    body{font-family:system-ui,-apple-system,sans-serif;
         max-width:1400px;margin:0 auto;padding:1.5rem;color:#111}
    h1{margin:0 0 .25rem}
    h2{margin:1.5rem 0 .5rem;font-size:.9rem;text-transform:uppercase;
       letter-spacing:.06em;color:#64748b}
    table{border-collapse:collapse;width:100%}
    th{text-align:left;padding:.45rem .75rem;border-bottom:2px solid #e2e8f0;
       font-size:.8rem;color:#64748b;white-space:nowrap}
    td{padding:.45rem .75rem;border-bottom:1px solid #f1f5f9;
       font-size:.875rem;vertical-align:middle}
    tr:hover td{background:#f8fafc}
    .badge{display:inline-block;padding:.1rem .45rem;border-radius:.25rem;
           font-size:.75rem;font-weight:600}
    .badge-running{background:#fef3c7;color:#92400e}
    .badge-done{background:#d1fae5;color:#065f46}
    .badge-error{background:#fee2e2;color:#991b1b}
    .badge-pending{background:#f1f5f9;color:#475569}
    pre{background:#0f172a;color:#e2e8f0;padding:1rem;border-radius:.5rem;
        max-height:320px;overflow-y:auto;font-size:.82rem;line-height:1.55;margin:0}
    .btn{cursor:pointer;padding:.3rem .8rem;border:none;border-radius:.3rem;
         background:#3b82f6;color:#fff;font-size:.82rem;text-decoration:none}
    .btn:hover{background:#2563eb}
    .btn-sm{padding:.18rem .5rem}
    .btn-outline{background:transparent;color:#3b82f6;border:1px solid #3b82f6}
    .btn-outline:hover{background:#eff6ff}
    .mono{font-family:ui-monospace,monospace;font-size:.82rem}
    section+section{border-top:1px solid #e2e8f0;padding-top:1rem}
    #log-section,#chart-section{display:none}
  </style>
</head>
<body>
<h1>llm-bench</h1>
<p style="color:#64748b;margin:.25rem 0 1.5rem">Benchmark Runs Dashboard</p>

<section>
  <h2>Runs</h2>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th><input type="checkbox" id="select-all" title="Select all"></th>
      <th>Run</th><th>Model</th><th>Backend</th>
      <th>Tok/s</th><th>TTFT&nbsp;p50</th><th>Status</th>
      <th>Created</th><th>Actions</th>
    </tr></thead>
    <tbody id="runs-tbody"
           hx-get="/api/ui/runs-table"
           hx-trigger="load, every 5s"
           hx-swap="innerHTML">
      <tr><td colspan="9" style="color:#94a3b8;padding:1.5rem .75rem">Loading…</td></tr>
    </tbody>
  </table>
  </div>
  <div style="margin-top:.75rem">
    <button class="btn" onclick="compareSelected()">Compare Selected</button>
  </div>
</section>

<section id="log-section">
  <h2>Live Log — <span id="log-run-id" class="mono"></span></h2>
  <pre id="log-output"></pre>
</section>

<section id="chart-section">
  <h2>Throughput Comparison</h2>
  <div id="comparison-chart"></div>
</section>

<script>
document.getElementById('select-all').addEventListener('change', function() {
  document.querySelectorAll('.run-cb').forEach(function(cb){ cb.checked = this.checked; }, this);
});

var _sse = null;
function streamLog(runId) {
  if (_sse) { _sse.close(); }
  document.getElementById('log-section').style.display = 'block';
  document.getElementById('log-run-id').textContent = runId.slice(0, 8);
  var pre = document.getElementById('log-output');
  pre.textContent = '';
  _sse = new EventSource('/api/runs/' + runId + '/stream');
  _sse.onmessage = function(e) {
    if (e.data.startsWith('[done:')) { _sse.close(); return; }
    pre.textContent += e.data + '\\n';
    pre.scrollTop = pre.scrollHeight;
  };
}

function compareSelected() {
  var cbs = Array.from(document.querySelectorAll('.run-cb:checked'));
  if (cbs.length < 2) { alert('Select at least 2 runs to compare.'); return; }
  var labels = cbs.map(function(cb){ return cb.dataset.label; });
  var toks   = cbs.map(function(cb){ return parseFloat(cb.dataset.toks) || 0; });
  var ttfts  = cbs.map(function(cb){
    return cb.dataset.ttft ? parseFloat(cb.dataset.ttft) : null; });
  document.getElementById('chart-section').style.display = 'block';
  var traces = [{type:'bar', name:'Tok/s', x:labels, y:toks}];
  if (ttfts.some(function(v){ return v !== null; })) {
    traces.push({type:'bar', name:'TTFT p50 (ms)', x:labels, y:ttfts});
  }
  Plotly.newPlot('comparison-chart', traces,
    {barmode:'group', title:'Throughput & Latency Comparison',
     yaxis:{title:'Value'}, legend:{orientation:'h'}});
}
</script>
</body>
</html>"""


def _render_runs_table_rows(results: list[RunResult]) -> str:
    if not results:
        return (
            '<tr><td colspan="9" style="color:#94a3b8;padding:1.5rem .75rem">'
            "No runs yet. Submit a benchmark via POST /api/runs.</td></tr>"
        )
    parts: list[str] = []
    for run in results:
        m = _parse_metrics_from_output(run.output)
        toks = m.get("tokens_per_second")
        ttft = m.get("p50_ttft_ms")
        cfg = run.config or {}
        model = html.escape(str(cfg.get("model", "—")))
        backend = html.escape(str(cfg.get("backend", "—")))
        toks_str = f"{toks:.1f}" if toks is not None else "N/A"
        ttft_str = f"{ttft:.0f} ms" if ttft is not None else "N/A"
        toks_data = str(toks) if toks is not None else ""
        ttft_data = str(ttft) if ttft is not None else ""
        created = run.created_at[:16].replace("T", " ")
        short_id = run.run_id[:8]
        label = html.escape(f"{backend}/{model[:20]}")
        rid = run.run_id
        parts.append(
            f"<tr>\n"
            f'  <td><input type="checkbox" class="run-cb" value="{rid}"'
            f' data-label="{label}" data-toks="{toks_data}" data-ttft="{ttft_data}"></td>\n'
            f'  <td class="mono" title="{html.escape(rid)}">{short_id}</td>\n'
            f"  <td>{model}</td>\n"
            f"  <td>{backend}</td>\n"
            f"  <td>{toks_str}</td>\n"
            f"  <td>{ttft_str}</td>\n"
            f'  <td><span class="badge badge-{run.status}">{run.status}</span></td>\n'
            f"  <td>{created}</td>\n"
            f"  <td>\n"
            f'    <button class="btn btn-sm btn-outline"'
            f" onclick=\"streamLog('{rid}')\">Log</button>\n"
            f'    <a href="/runs/{rid}/pareto.html"'
            f' class="btn btn-sm btn-outline" style="margin-left:.25rem">Pareto</a>\n'
            f"  </td>\n"
            f"</tr>"
        )
    return "\n".join(parts)


def _render_pareto_html(run_id: str, results: list[RunResult]) -> str:
    pts: list[dict[str, Any]] = []
    for r in results:
        m = _parse_metrics_from_output(r.output)
        lat = m.get("p95_latency_ms")
        tok = m.get("tokens_per_second")
        if lat is None or tok is None:
            continue
        cfg = r.config or {}
        pts.append(
            {
                "run_id": r.run_id,
                "latency": lat,
                "toks": tok,
                "backend": cfg.get("backend", "?"),
                "model": cfg.get("model", "?"),
                "highlight": r.run_id == run_id,
            }
        )

    if not pts:
        return (
            f"<!DOCTYPE html><html lang='en'><body>"
            f"<h2>Pareto — {html.escape(run_id[:8])}</h2>"
            f"<p>No completed runs with parseable metrics yet.</p>"
            f"<p><a href='/'>&#8592; Dashboard</a></p>"
            f"</body></html>"
        )

    coords = [(p["latency"], p["toks"]) for p in pts]
    mask = _pareto_mask(coords) if len(pts) > 1 else [True]

    normal = [p for p, is_p in zip(pts, mask, strict=True) if not p["highlight"] and not is_p]
    pareto = [p for p, is_p in zip(pts, mask, strict=True) if is_p and not p["highlight"]]
    hi = [p for p in pts if p["highlight"]]

    traces: list[dict[str, Any]] = []
    if normal:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": "Other runs",
                "x": [p["latency"] for p in normal],
                "y": [p["toks"] for p in normal],
                "text": [
                    f"{p['backend']}/{str(p['model'])[:20]}<br>{p['run_id'][:8]}" for p in normal
                ],
                "marker": {"color": "#94a3b8", "size": 10},
                "hovertemplate": "%{text}<br>p95: %{x:.0f} ms<br>tok/s: %{y:.1f}<extra></extra>",
            }
        )
    if pareto:
        pf = sorted(pareto, key=lambda p: p["latency"])
        traces.append(
            {
                "type": "scatter",
                "mode": "markers+lines",
                "name": "Pareto front",
                "x": [p["latency"] for p in pf],
                "y": [p["toks"] for p in pf],
                "text": [f"{p['backend']}/{str(p['model'])[:20]}<br>{p['run_id'][:8]}" for p in pf],
                "marker": {"color": "#3b82f6", "size": 12},
                "line": {"dash": "dot", "color": "#3b82f6"},
                "hovertemplate": "%{text}<br>p95: %{x:.0f} ms<br>tok/s: %{y:.1f}<extra></extra>",
            }
        )
    if hi:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": "This run",
                "x": [p["latency"] for p in hi],
                "y": [p["toks"] for p in hi],
                "text": [f"{p['backend']}/{str(p['model'])[:20]}<br>{p['run_id'][:8]}" for p in hi],
                "marker": {"color": "#f59e0b", "size": 16, "symbol": "star"},
                "hovertemplate": "%{text}<br>p95: %{x:.0f} ms<br>tok/s: %{y:.1f}<extra></extra>",
            }
        )

    layout: dict[str, Any] = {
        "title": f"Pareto — Run {run_id[:8]}",
        "xaxis": {"title": "p95 Latency (ms)"},
        "yaxis": {"title": "Throughput (tok/s)"},
        "hovermode": "closest",
        "legend": {"orientation": "h"},
    }
    traces_json = json.dumps(traces)
    layout_json = json.dumps(layout)
    short = html.escape(run_id[:8])

    return (
        f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        f"  <meta charset='UTF-8'>\n"
        f"  <title>llm-bench Pareto — {short}</title>\n"
        f"  <script src='https://cdn.plot.ly/plotly-2.32.0.min.js' charset='utf-8'></script>\n"
        f"  <style>"
        f"body{{font-family:system-ui,sans-serif;margin:1.5rem;max-width:1000px}}"
        f"</style>\n"
        f"</head>\n<body>\n"
        f"  <h2>Pareto Chart — <span style='font-family:monospace'>{short}</span></h2>\n"
        f"  <p><a href='/'>&#8592; Dashboard</a></p>\n"
        f"  <div id='chart'></div>\n"
        f"  <script>Plotly.newPlot('chart', {traces_json}, {layout_json});</script>\n"
        f"</body>\n</html>"
    )


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


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return _DASHBOARD_HTML


@app.get("/api/ui/runs-table", response_class=HTMLResponse)
async def runs_table_fragment() -> str:
    rows = _get_db().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    results = [_row_to_result(r) for r in rows]
    return _render_runs_table_rows(results)


@app.get("/runs/{run_id}/pareto.html", response_class=HTMLResponse)
async def pareto_page(run_id: str) -> str:
    row = _get_db().execute("SELECT id FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    all_rows = _get_db().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    results = [_row_to_result(r) for r in all_rows]
    return _render_pareto_html(run_id, results)
