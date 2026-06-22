"""FastAPI HTTP backend for llm-bench Web UI.

Endpoints:
  GET  /                              dashboard (HTML)
  GET  /static/app.css               CSS
  GET  /static/app.js                JS
  GET  /api/health
  GET  /api/models
  GET  /api/datasets
  POST /api/datasets/pull
  GET  /api/runs
  POST /api/runs
  GET    /api/runs/{run_id}
  DELETE /api/runs/{run_id}
  GET    /api/runs/{run_id}/results.csv  downloadable CSV of parsed metrics
  GET    /api/runs/{run_id}/stream      Server-Sent Events
  GET  /api/ui/run-list              HTMX HTML fragment — sidebar card list
  GET  /api/ui/run-detail/{run_id}   HTMX HTML fragment — run detail panel
  GET  /api/ui/runs-table            HTMX HTML fragment — legacy table rows
  GET  /api/ui/datasets-table        HTMX HTML fragment
  GET  /runs/{run_id}/pareto.html    interactive Plotly scatter page
  GET  /runs/pareto                  multi-run Pareto page
"""

from __future__ import annotations

import asyncio
import csv
import html
import io
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

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from llm_inference_benchmark import datasets as _datasets_mod

# ── Paths ─────────────────────────────────────────────────────────────────────

_UI_DIR = Path(__file__).parent / "ui"
_templates = Jinja2Templates(directory=str(_UI_DIR / "templates"))

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


# ── Streaming buffers ─────────────────────────────────────────────────────────

_buffers: dict[str, list[str]] = {}

# ── Dataset pull error store ──────────────────────────────────────────────────

_pull_errors: dict[str, str] = {}


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="llm-bench Web API", version="1.0.0")


# ── Pydantic models ───────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    config: dict[str, Any]
    dataset: str | None = None


class RunSubmitted(BaseModel):
    run_id: str


class DatasetPullRequest(BaseModel):
    name: str


class RunResult(BaseModel):
    run_id: str
    status: str
    config: dict[str, Any] | None = None
    output: str | None = None
    created_at: str
    finished_at: str | None = None


# ── Core helpers ──────────────────────────────────────────────────────────────


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
    """Return GGUF files and HF-cached models found on the local machine."""
    results: list[dict[str, str]] = []

    gguf_root = Path.home() / "models"
    if gguf_root.is_dir():
        for p in sorted(gguf_root.glob("**/*.gguf")):
            results.append({"type": "gguf", "name": p.name, "path": str(p), "value": str(p)})

    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.is_dir():
        for d in sorted(hf_hub.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                snapshots = d / "snapshots"
                if not snapshots.is_dir() or not any(snapshots.iterdir()):
                    continue  # partial/empty download — skip
                slug = d.name[len("models--") :]
                model_name = slug.replace("--", "/", 1)
                # Use the model name (not the hub dir path) so from_pretrained works.
                results.append(
                    {"type": "hf", "name": model_name, "path": str(d), "value": model_name}
                )

    return results


# ── Dataset helpers ───────────────────────────────────────────────────────────


def _dataset_statuses() -> list[dict[str, Any]]:
    cached = dict(_datasets_mod.list_cached())
    result: list[dict[str, Any]] = []
    for name, spec in _datasets_mod.REGISTRY.items():
        result.append(
            {
                "name": name,
                "description": spec.get("description", ""),
                "cached": name in cached,
                "samples": cached.get(name, 0),
                "error": None if name in cached else _pull_errors.get(name),
            }
        )
    return result


def _do_pull_dataset(name: str) -> None:
    import os

    hf_token = os.environ.get("HF_TOKEN") or None
    try:
        _datasets_mod.pull(name, hf_token=hf_token)
        _pull_errors.pop(name, None)
    except Exception as exc:
        _pull_errors[name] = str(exc) or type(exc).__name__


# ── Background benchmark runner ───────────────────────────────────────────────


def _do_run(run_id: str, config: dict[str, Any], dataset: str | None = None) -> None:
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
            cmd = [str(Path(sys.executable).parent / "llm-bench"), "--config", cfg_path]
            if dataset:
                cmd.extend(["--dataset", dataset])
            proc = subprocess.Popen(
                cmd,
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

_NUMERIC_METRIC_KEYS = [
    "tokens_per_second",
    "decode_tokens_per_second",
    "p50_latency_ms",
    "p95_latency_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "mean_input_tokens",
    "mean_output_tokens",
    "peak_cpu_memory_mb",
    "peak_cuda_memory_mb",
    "peak_vram_memory_mb",
    "model_load_ms",
    "warmup_p50_latency_ms",
    "energy_joules",
    "tokens_per_joule",
    "sanity_pass_rate",
    "task_quality_pass_rate",
    "total_tokens",
    "request_count",
    "empty_output_count",
    "repeated_output_count",
    "thermal_throttle_pct",
]


def _parse_metrics_from_output(output: str | None) -> dict[str, float | None]:
    """Extract numeric metrics from benchmark stdout text."""
    if not output:
        return {k: None for k in _NUMERIC_METRIC_KEYS}
    result: dict[str, float | None] = {}
    for key in _NUMERIC_METRIC_KEYS:
        m = re.search(rf"^\s+{re.escape(key)}: ([0-9]+(?:\.[0-9]+)?)", output, re.MULTILINE)
        result[key] = float(m.group(1)) if m else None
    return result


_HW_KEYS = ["hw_cpu", "hw_gpu", "hw_os", "hw_cpu_cores", "hw_ram_gb", "hw_vram_gb"]


def _parse_hw_from_output(output: str) -> dict[str, str]:
    """Extract hardware info lines from benchmark stdout."""
    result: dict[str, str] = {}
    for key in _HW_KEYS:
        m = re.search(rf"^\s+{re.escape(key)}: (.+)$", output, re.MULTILINE)
        if m:
            result[key] = m.group(1).strip()
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


# ── HTML rendering helpers ────────────────────────────────────────────────────

_DISPLAY_METRICS: list[tuple[str, str, Any]] = [
    ("tokens_per_second", "Tok/s", lambda v: f"{v:.1f}"),
    ("p50_latency_ms", "p50 Latency", lambda v: f"{v:.0f} ms"),
    ("p95_latency_ms", "p95 Latency", lambda v: f"{v:.0f} ms"),
    ("p50_ttft_ms", "TTFT p50", lambda v: f"{v:.0f} ms"),
    ("peak_cuda_memory_mb", "CUDA Mem", lambda v: f"{v:.0f} MB"),
    ("peak_vram_memory_mb", "VRAM", lambda v: f"{v:.0f} MB"),
    ("model_load_ms", "Load Time", lambda v: f"{v:.0f} ms"),
    ("energy_joules", "Energy", lambda v: f"{v:.1f} J"),
    ("tokens_per_joule", "Efficiency", lambda v: f"{v:.2f} tok/J"),
    ("sanity_pass_rate", "Sanity", lambda v: f"{v * 100:.0f}%"),
]


def _render_metric_cards(metrics: dict[str, float | None]) -> str:
    cards = []
    for key, label, fmt in _DISPLAY_METRICS:
        val = metrics.get(key)
        if val is None:
            continue
        val_str = fmt(val)
        cards.append(
            f'<div class="metric-card">'
            f'<div class="metric-value">{html.escape(val_str)}</div>'
            f'<div class="metric-label">{html.escape(label)}</div>'
            f"</div>"
        )
    if not cards:
        return ""
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def _render_hw_info(hw: dict[str, str]) -> str:
    if not hw:
        return ""
    parts = []
    if "hw_cpu" in hw:
        parts.append(f"CPU: {html.escape(hw['hw_cpu'])}")
    if "hw_cpu_cores" in hw:
        parts.append(f"{html.escape(hw['hw_cpu_cores'])} cores")
    if "hw_ram_gb" in hw:
        parts.append(f"RAM: {html.escape(hw['hw_ram_gb'])} GB")
    if "hw_gpu" in hw:
        parts.append(f"GPU: {html.escape(hw['hw_gpu'])}")
    if "hw_vram_gb" in hw:
        parts.append(f"VRAM: {html.escape(hw['hw_vram_gb'])} GB")
    if not parts:
        return ""
    return f'<div class="hw-info"><span class="hw-label">Hardware:</span> {" · ".join(parts)}</div>'


def _render_run_detail(run: RunResult) -> str:
    """Return the HTMX HTML fragment for the run detail panel."""
    cfg = run.config or {}
    model = str(cfg.get("model", "—"))
    backend = str(cfg.get("backend", "—"))
    short_id = run.run_id[:8]
    created = run.created_at[:16].replace("T", " ")

    metrics = _parse_metrics_from_output(run.output)
    hw = _parse_hw_from_output(run.output or "")
    metric_html = _render_metric_cards(metrics)
    hw_html = _render_hw_info(hw)

    log_content = ""
    if run.status in ("done", "error") and run.output:
        log_content = html.escape(run.output)

    log_header = "Output" if run.status in ("done", "error") else "Live Log"
    rid = html.escape(run.run_id)

    model_display = html.escape(model)
    if len(model) > 80:
        model_display = html.escape("…" + model[-77:])

    csv_btn = (
        f'    <a href="/api/runs/{rid}/results.csv"'
        f' class="btn btn-sm btn-outline" download>Download CSV</a>\n'
        if run.status == "done"
        else ""
    )

    return (
        f'<div id="detail-inner" data-run-id="{rid}" data-status="{html.escape(run.status)}">\n'
        f'  <div class="detail-header">\n'
        f"    <div>\n"
        f'      <div class="detail-title">\n'
        f'        <span class="detail-id mono">{html.escape(short_id)}</span>\n'
        f'        <span class="detail-backend">{html.escape(backend)}</span>\n'
        f'        <span class="badge badge-{html.escape(run.status)}">'
        f"{html.escape(run.status)}</span>\n"
        f"      </div>\n"
        f'      <div class="detail-meta">\n'
        f'        <span class="detail-model">{model_display}</span> · {html.escape(created)}\n'
        f"      </div>\n"
        f"    </div>\n"
        f"  </div>\n"
        f"\n"
        f"  {metric_html}\n"
        f"  {hw_html}\n"
        f"\n"
        f'  <div class="detail-actions">\n'
        f'    <button class="btn btn-sm btn-outline btn-danger"'
        f" onclick=\"deleteRun('{rid}')\">Delete</button>\n"
        f'    <a href="/runs/{rid}/pareto.html"'
        f' class="btn btn-sm btn-outline" target="_blank">Pareto Chart</a>\n'
        f"{csv_btn}"
        f"  </div>\n"
        f"\n"
        f'  <div class="log-section">\n'
        f'    <div class="log-header">{html.escape(log_header)}</div>\n'
        f'    <pre class="log-output" id="log-output">{log_content}</pre>\n'
        f"  </div>\n"
        f"</div>\n"
    )


def _render_run_list_cards(results: list[RunResult]) -> str:
    """Return sidebar run card HTML (HTMX fragment)."""
    if not results:
        return (
            '<div class="run-empty">No runs yet.'
            " Click <strong>+ New Run</strong> to start a benchmark.</div>"
        )
    parts: list[str] = []
    for run in results:
        cfg = run.config or {}
        model = str(cfg.get("model", "—"))
        backend = str(cfg.get("backend", "—"))
        model_short = model.split("/")[-1] if "/" in model else model
        model_short = ("…" + model_short[-37:]) if len(model_short) > 40 else model_short

        metrics = _parse_metrics_from_output(run.output)
        toks = metrics.get("tokens_per_second")
        toks_str = f"{toks:.1f} tok/s" if toks is not None else ""

        created = run.created_at[:16].replace("T", " ")
        rid = run.run_id
        short_id = rid[:8]

        meta_parts = [html.escape(backend)]
        if toks_str:
            meta_parts.append(html.escape(toks_str))
        meta_parts.append(html.escape(created))

        parts.append(
            f'<div class="run-card" data-run-id="{html.escape(rid)}"'
            f" onclick=\"selectRun('{html.escape(rid)}')\">\n"
            f'  <div class="run-card-header">\n'
            f'    <span class="run-card-id mono">{html.escape(short_id)}</span>\n'
            f'    <span class="badge badge-{html.escape(run.status)}">'
            f"{html.escape(run.status)}</span>\n"
            f"  </div>\n"
            f'  <div class="run-card-model">{html.escape(model_short)}</div>\n'
            f'  <div class="run-card-meta">{" · ".join(meta_parts)}</div>\n'
            f"</div>"
        )
    return "\n".join(parts)


def _render_runs_table_rows(results: list[RunResult]) -> str:
    """Legacy table row renderer (kept for backward compatibility and tests)."""
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
        ttft_str = f"{ttft:.0f} ms" if ttft is not None else "N/A"
        toks_data = str(toks) if toks is not None else ""
        ttft_data = str(ttft) if ttft is not None else ""
        created = run.created_at[:16].replace("T", " ")
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
            f' class="btn btn-sm btn-outline">Pareto</a>\n'
            f'    <button class="btn btn-sm btn-outline btn-danger"'
            f" onclick=\"deleteRun('{rid}')\">Delete</button>\n"
            f"  </td>\n"
            f"</tr>"
        )
    return "\n".join(parts)


def _render_datasets_table(statuses: list[dict[str, Any]]) -> str:
    if not statuses:
        return (
            '<tr><td colspan="4" style="color:#94a3b8;padding:1.5rem .75rem">'
            "No datasets registered.</td></tr>"
        )
    parts: list[str] = []
    for ds in statuses:
        name = html.escape(ds["name"])
        desc = html.escape(ds["description"])
        error = ds["error"]
        cached_icon = "✓" if ds["cached"] else "✗"
        cached_color = "#065f46" if ds["cached"] else "#991b1b"
        samples = str(ds["samples"]) if ds["cached"] else "—"
        error_html = (
            f'<br><span class="ds-pull-error" style="color:#b91c1c;font-size:.78rem">'
            f"Pull failed: {html.escape(error.replace(chr(10), ' '))}</span>"
            if error is not None
            else ""
        )
        parts.append(
            f"<tr>\n"
            f'  <td class="mono">{name}</td>\n'
            f'  <td style="color:#64748b;font-size:.82rem">{desc}{error_html}</td>\n'
            f'  <td style="color:{cached_color};font-weight:600">{cached_icon}</td>\n'
            f"  <td>{samples}</td>\n"
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
        "  <style>body{font-family:system-ui,sans-serif;"
        "margin:1.5rem;max-width:1000px}</style>\n"
        f"</head>\n<body>\n"
        f"  <h2>Pareto Chart — <span style='font-family:monospace'>{short}</span></h2>\n"
        f"  <p><a href='/'>&#8592; Dashboard</a></p>\n"
        f"  <div id='chart'></div>\n"
        f"  <script>Plotly.newPlot('chart', {traces_json}, {layout_json});</script>\n"
        f"</body>\n</html>"
    )


# ── Static file routes ────────────────────────────────────────────────────────


@app.get("/static/app.css")
async def serve_css() -> Response:
    return Response((_UI_DIR / "static" / "app.css").read_text(), media_type="text/css")


@app.get("/static/app.js")
async def serve_js() -> Response:
    return Response(
        (_UI_DIR / "static" / "app.js").read_text(), media_type="application/javascript"
    )


# ── API endpoints ─────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/capabilities")
async def capabilities() -> dict[str, bool]:
    """Return runtime capability flags for optional backends."""
    llama_gpu = False
    try:
        from llama_cpp import llama_supports_gpu_offload  # type: ignore[import-untyped]

        llama_gpu = bool(llama_supports_gpu_offload())
    except Exception:
        pass
    return {"llama_cpp_gpu": llama_gpu}


@app.get("/api/models")
async def list_models() -> dict[str, list[dict[str, str]]]:
    return {"models": _discover_models()}


@app.get("/api/datasets")
async def list_datasets() -> dict[str, list[dict[str, Any]]]:
    return {"datasets": _dataset_statuses()}


@app.post("/api/datasets/pull", status_code=202)
async def pull_dataset(
    body: DatasetPullRequest, background_tasks: BackgroundTasks
) -> dict[str, str]:
    if body.name not in _datasets_mod.REGISTRY:
        raise HTTPException(status_code=422, detail=f"Unknown dataset {body.name!r}")
    background_tasks.add_task(_do_pull_dataset, body.name)
    return {"status": "started"}


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
    background_tasks.add_task(_do_run, run_id, body.config, dataset=body.dataset)
    return RunSubmitted(run_id=run_id)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> RunResult:
    row = _get_db().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return _row_to_result(row)


@app.delete("/api/runs/{run_id}", status_code=204)
async def delete_run(run_id: str) -> None:
    db = _get_db()
    row = db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    if row["status"] in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id!r} is still {row['status']}; wait for it to finish",
        )
    db.execute("DELETE FROM runs WHERE id=?", (run_id,))
    db.commit()
    _buffers.pop(run_id, None)


@app.get("/api/runs/{run_id}/results.csv")
async def download_run_csv(run_id: str) -> Response:
    db = _get_db()
    row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    run = _row_to_result(row)
    if run.status in ("pending", "running"):
        raise HTTPException(status_code=409, detail="Run is not finished")
    metrics = _parse_metrics_from_output(run.output)
    cfg = run.config or {}
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["run_id", "backend", "model", "status", "created_at", "finished_at"]
        + _NUMERIC_METRIC_KEYS
    )
    writer.writerow(
        [
            run.run_id,
            cfg.get("backend", ""),
            cfg.get("model", ""),
            run.status,
            run.created_at,
            run.finished_at or "",
        ]
        + [("" if metrics.get(k) is None else metrics[k]) for k in _NUMERIC_METRIC_KEYS]
    )
    short = run_id[:8]
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="run_{short}.csv"'},
    )


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
                if buf is None:
                    stored: str = current["output"] or ""
                    for line in stored.split("\n"):
                        yield f"data: {line}\n\n"
                yield f"data: [done:{status}]\n\n"
                return

            await asyncio.sleep(0.2)

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── UI / HTMX fragment endpoints ──────────────────────────────────────────────


@app.get("/api/ui/run-list", response_class=HTMLResponse)
async def run_list_fragment() -> str:
    rows = _get_db().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    results = [_row_to_result(r) for r in rows]
    return _render_run_list_cards(results)


@app.get("/api/ui/run-detail/{run_id}", response_class=HTMLResponse)
async def run_detail_fragment(run_id: str) -> str:
    row = _get_db().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        return f"<p>Run {html.escape(run_id[:8])} not found.</p>"
    return _render_run_detail(_row_to_result(row))


@app.get("/api/ui/runs-table", response_class=HTMLResponse)
async def runs_table_fragment() -> str:
    rows = _get_db().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    results = [_row_to_result(r) for r in rows]
    return _render_runs_table_rows(results)


@app.get("/api/ui/datasets-table", response_class=HTMLResponse)
async def datasets_table_fragment() -> str:
    return _render_datasets_table(_dataset_statuses())


# ── Dashboard ─────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> Any:
    return _templates.TemplateResponse(request=request, name="dashboard.html")


# ── Pareto pages ──────────────────────────────────────────────────────────────


@app.get("/runs/{run_id}/pareto.html", response_class=HTMLResponse)
async def pareto_page(run_id: str) -> str:
    row = _get_db().execute("SELECT id FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    all_rows = _get_db().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    results = [_row_to_result(r) for r in all_rows]
    return _render_pareto_html(run_id, results)


@app.get("/runs/pareto", response_class=HTMLResponse)
async def pareto_multi_page(ids: str = Query(..., description="Comma-separated run IDs")) -> str:
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if len(id_list) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 run IDs via ?ids=")
    placeholders = ",".join("?" * len(id_list))
    rows = _get_db().execute(f"SELECT * FROM runs WHERE id IN ({placeholders})", id_list).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No matching runs found")
    results = [_row_to_result(r) for r in rows]
    return _render_pareto_html(id_list[0], results)
