"""FastAPI HTTP backend for llm-bench Web UI.

Endpoints:
  GET  /                              dashboard (HTML)
  GET  /static/app.css               CSS
  GET  /static/app.js                JS
  GET  /api/health
  GET  /api/models
  GET  /api/datasets
  POST /api/datasets/pull
  GET    /api/runs
  POST   /api/runs
  GET    /api/runs/{run_id}
  PATCH  /api/runs/{run_id}              update label ({"label": "…"})
  DELETE /api/runs/{run_id}
  GET    /api/runs/{run_id}/results.csv  downloadable CSV of parsed metrics
  GET    /api/runs/{run_id}/stream      Server-Sent Events
  GET  /api/ui/run-list              HTMX HTML fragment — sidebar card list
  GET  /api/ui/run-detail/{run_id}   HTMX HTML fragment — run detail panel
  GET  /api/ui/runs-table            HTMX HTML fragment — legacy table rows
  GET  /api/ui/datasets-table        HTMX HTML fragment
  GET  /api/ui/compare-table         HTMX HTML fragment — side-by-side metric comparison
  GET  /api/ui/compare-chart         HTMX HTML fragment — normalised bar chart comparison
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

_MIGRATE_LABEL = "ALTER TABLE runs ADD COLUMN label TEXT"


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
        try:
            conn.execute(_MIGRATE_LABEL)
        except sqlite3.OperationalError:
            pass  # column already exists
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
    label: str | None = None


class LabelUpdate(BaseModel):
    label: str


# ── Core helpers ──────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _format_duration(created_at: str, finished_at: str | None) -> str | None:
    if not finished_at:
        return None
    try:
        start = datetime.fromisoformat(created_at)
        end = datetime.fromisoformat(finished_at)
        secs = max(0, int((end - start).total_seconds()))
        if secs < 60:
            return f"{secs}s"
        return f"{secs // 60}m {secs % 60}s"
    except Exception:
        return None


def _row_to_result(row: sqlite3.Row) -> RunResult:
    cols = row.keys()
    return RunResult(
        run_id=row["id"],
        status=row["status"],
        config=json.loads(row["config"]),
        output=row["output"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
        label=row["label"] if "label" in cols else None,
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

# (key, short label, formatter, higher_is_better) — used for both table and chart
_COMPARE_METRICS: list[tuple[str, str, Any, bool]] = [
    ("tokens_per_second", "Throughput", lambda v: f"{v:.1f} tok/s", True),
    ("p50_latency_ms", "p50 Latency", lambda v: f"{v:.0f} ms", False),
    ("p95_latency_ms", "p95 Latency", lambda v: f"{v:.0f} ms", False),
    ("p50_ttft_ms", "TTFT p50", lambda v: f"{v:.0f} ms", False),
    ("model_load_ms", "Load Time", lambda v: f"{v:.0f} ms", False),
    ("peak_vram_memory_mb", "VRAM", lambda v: f"{v:.0f} MB", False),
    ("peak_cuda_memory_mb", "CUDA Mem", lambda v: f"{v:.0f} MB", False),
    ("energy_joules", "Energy", lambda v: f"{v:.1f} J", False),
    ("tokens_per_joule", "Efficiency", lambda v: f"{v:.2f} tok/J", True),
    ("sanity_pass_rate", "Sanity", lambda v: f"{v * 100:.0f}%", True),
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


def _render_compare_table(runs: list[RunResult]) -> str:
    """Return HTML fragment for the side-by-side metric comparison table."""
    if len(runs) < 2:
        return "<p class='muted'>Select at least 2 runs to compare.</p>"

    all_metrics = [_parse_metrics_from_output(r.output) for r in runs]

    def _col_header(r: RunResult) -> str:
        cfg = r.config or {}
        model = str(cfg.get("model", "—")).split("/")[-1]
        backend = str(cfg.get("backend", "—"))
        label = r.label or ""
        short_id = r.run_id[:8]
        lbl_html = f"<br><em>{html.escape(label)}</em>" if label else ""
        return (
            f"<th class='cmp-col'>"
            f"<span class='cmp-model'>{html.escape(model)}</span>"
            f"<br><span class='cmp-backend'>{html.escape(backend)}</span>"
            f"{lbl_html}"
            f"<br><span class='mono cmp-id'>{html.escape(short_id)}</span>"
            f"</th>"
        )

    header_cells = "".join(_col_header(r) for r in runs)
    header_row = f"<tr><th class='cmp-metric-hd'>Metric</th>{header_cells}</tr>"

    ref_metrics = all_metrics[0]
    rows_html: list[str] = []

    for key, label, fmt, higher_is_better in _COMPARE_METRICS:
        if all(m.get(key) is None for m in all_metrics):
            continue
        ref_val = ref_metrics.get(key)
        cells: list[str] = [f"<td class='cmp-metric-cell'>{html.escape(label)}</td>"]
        for i, m in enumerate(all_metrics):
            val = m.get(key)
            if val is None:
                cells.append("<td class='cmp-val muted'>—</td>")
                continue
            val_str = fmt(val)
            if i == 0 or ref_val is None or ref_val == 0:
                cells.append(f"<td class='cmp-val'>{html.escape(val_str)}</td>")
            else:
                delta = (val - ref_val) / abs(ref_val) * 100
                good = (delta > 0) == higher_is_better
                delta_cls = "delta-good" if good else "delta-bad"
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{delta:.1f}%"
                cells.append(
                    f"<td class='cmp-val'>{html.escape(val_str)}"
                    f" <span class='{delta_cls}'>{html.escape(delta_str)}</span></td>"
                )
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    if not rows_html:
        table_body = "<tr><td colspan='99' class='muted'>No metrics available yet.</td></tr>"
    else:
        table_body = "\n".join(rows_html)

    run_ids = " vs ".join(r.run_id[:8] for r in runs)
    return (
        f"<div id='detail-inner' class='compare-table-wrap'>\n"
        f"<div class='detail-header'><div>"
        f"<div class='detail-title'>Metric Comparison</div>"
        f"<div class='detail-meta muted'>{html.escape(run_ids)}</div>"
        f"</div></div>\n"
        f"<div class='compare-table-scroll'>"
        f"<table class='cmp-table data-table'>\n"
        f"<thead>{header_row}</thead>\n"
        f"<tbody>{table_body}</tbody>\n"
        f"</table></div>\n"
        f"</div>\n"
    )


_CHART_METRICS: list[tuple[str, str, str, bool]] = [
    ("tokens_per_second", "Throughput", "{:.1f} tok/s", True),
    ("p50_latency_ms", "p50 Lat", "{:.0f} ms", False),
    ("p95_latency_ms", "p95 Lat", "{:.0f} ms", False),
    ("p50_ttft_ms", "TTFT", "{:.0f} ms", False),
    ("peak_vram_memory_mb", "VRAM", "{:.0f} MB", False),
    ("energy_joules", "Energy", "{:.1f} J", False),
]

_CHART_COLORS = [
    "#6366f1",
    "#f59e0b",
    "#10b981",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
    "#0ea5e9",
    "#84cc16",
]


def _render_compare_chart(runs: list[RunResult]) -> str:
    """Return HTML fragment with an embedded Plotly grouped bar chart."""
    if len(runs) < 2:
        return "<p class='muted'>Select at least 2 runs to compare.</p>"

    all_metrics = [_parse_metrics_from_output(r.output) for r in runs]

    active: list[tuple[str, str, str, bool, list[float | None]]] = []
    for key, label, fmt, hib in _CHART_METRICS:
        vals: list[float | None] = [m.get(key) for m in all_metrics]
        if any(v is not None for v in vals):
            active.append((key, label, fmt, hib, vals))

    if not active:
        run_ids = " vs ".join(r.run_id[:8] for r in runs)
        return (
            f"<div id='detail-inner' class='compare-table-wrap'>"
            f"<div class='detail-header'><div><div class='detail-title'>Metric Chart</div>"
            f"<div class='detail-meta muted'>{html.escape(run_ids)}</div></div></div>"
            f"<p class='muted' style='padding:1.5rem'>No metrics available for chart.</p>"
            f"</div>"
        )

    traces: list[dict[str, Any]] = []
    for i, run in enumerate(runs):
        cfg = run.config or {}
        model = str(cfg.get("model", "—")).split("/")[-1]
        backend = str(cfg.get("backend", "—"))
        run_label = run.label or f"{model}/{backend}"
        short_id = run.run_id[:8]
        color = _CHART_COLORS[i % len(_CHART_COLORS)]

        x_labels: list[str] = []
        y_vals: list[float | None] = []
        hover: list[str] = []

        for key, label, fmt, hib, all_vals in active:
            val = all_metrics[i].get(key)
            valid = [v for v in all_vals if v is not None and v > 0]
            x_labels.append(label)
            if val is None or not valid:
                y_vals.append(None)
                hover.append("—")
                continue
            norm = (val / max(valid) * 100) if hib else (min(valid) / val * 100)
            y_vals.append(round(norm, 1))
            hover.append(fmt.format(val))

        traces.append(
            {
                "type": "bar",
                "name": f"{run_label} [{short_id}]",
                "x": x_labels,
                "y": y_vals,
                "text": hover,
                "hovertemplate": "%{text}<extra>%{fullData.name}</extra>",
                "marker": {"color": color},
            }
        )

    layout: dict[str, Any] = {
        "barmode": "group",
        "yaxis": {
            "title": "Normalised score — higher is better",
            "range": [0, 115],
            "ticksuffix": "%",
        },
        "legend": {"orientation": "h", "y": -0.25},
        "margin": {"t": 20, "b": 80, "l": 55, "r": 20},
        "plot_bgcolor": "#f8f9fa",
        "paper_bgcolor": "#ffffff",
    }

    traces_json = html.escape(json.dumps(traces))
    layout_json = html.escape(json.dumps(layout))
    run_ids = " vs ".join(r.run_id[:8] for r in runs)

    return (
        f"<div id='detail-inner' class='compare-table-wrap'>\n"
        f"<div class='detail-header'><div>"
        f"<div class='detail-title'>Metric Chart</div>"
        f"<div class='detail-meta muted'>{html.escape(run_ids)}</div>"
        f"</div></div>\n"
        f"<div id='cmp-chart-div' style='width:100%;height:420px'"
        f" data-traces='{traces_json}'"
        f" data-layout='{layout_json}'></div>\n"
        f"</div>\n"
    )


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

    lbl = run.label or ""
    detail_label_text = html.escape(lbl) if lbl else "Add label…"
    detail_label_cls = "run-label-text" if lbl else "run-label-text run-label-placeholder"
    duration = _format_duration(run.created_at, run.finished_at)
    duration_str = f" · {html.escape(duration)}" if duration else ""

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
        f'        <span class="detail-model">{model_display}</span>'
        f" · {html.escape(created)}{duration_str}\n"
        f"      </div>\n"
        f'      <div class="detail-label" id="detail-lbl-{rid}"'
        f' data-label="{html.escape(lbl)}">\n'
        f'        <span class="{detail_label_cls}"'
        f" onclick=\"startEditLabel(event,'{rid}')\">"
        f"{detail_label_text}</span>\n"
        f"      </div>\n"
        f"    </div>\n"
        f"  </div>\n"
        f"\n"
        f"  {metric_html}\n"
        f"  {hw_html}\n"
        f"\n"
        f'  <div class="detail-actions">\n'
        f'    <button class="btn btn-sm btn-outline"'
        f" onclick=\"cloneRun('{rid}')\">Clone</button>\n"
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

        duration = _format_duration(run.created_at, run.finished_at)

        meta_parts = [html.escape(backend)]
        if toks_str:
            meta_parts.append(html.escape(toks_str))
        if duration:
            meta_parts.append(html.escape(duration))
        meta_parts.append(html.escape(created))

        label = run.label or ""
        label_text = html.escape(label) if label else "Add label…"
        label_cls = "run-label-text" if label else "run-label-text run-label-placeholder"

        parts.append(
            f'<div class="run-card" data-run-id="{html.escape(rid)}"'
            f" onclick=\"selectRun('{html.escape(rid)}')\">\n"
            f'  <div class="run-card-header">\n'
            f'    <span class="run-card-id mono">{html.escape(short_id)}</span>\n'
            f'    <span class="badge badge-{html.escape(run.status)}">'
            f"{html.escape(run.status)}</span>\n"
            f'    <input type="checkbox" class="compare-cb"'
            f' value="{html.escape(rid)}"'
            f" onclick=\"toggleCompare(event,'{html.escape(rid)}')\">\n"
            f"  </div>\n"
            f'  <div class="run-card-model">{html.escape(model_short)}</div>\n'
            f'  <div class="run-card-label" id="lbl-{html.escape(rid)}"'
            f' data-label="{html.escape(label)}">\n'
            f'    <span class="{label_cls}"'
            f" onclick=\"startEditLabel(event,'{html.escape(rid)}')\">"
            f"{label_text}</span>\n"
            f"  </div>\n"
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


_PARETO_CHART_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>llm-bench Pareto — PARETO_SHORT</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
  <style>
    body{font-family:system-ui,sans-serif;margin:1.5rem;max-width:1000px}
    .controls{display:flex;gap:1rem;align-items:center;margin-bottom:1rem;flex-wrap:wrap}
    label{font-size:.875rem;color:#475569}
    select{padding:.25rem .5rem;border:1px solid #cbd5e1;border-radius:.375rem;font-size:.875rem}
    .btn{padding:.375rem .75rem;background:#3b82f6;color:#fff;border:none;
         border-radius:.375rem;cursor:pointer;font-size:.875rem}
    .btn:hover{background:#2563eb}
  </style>
</head>
<body>
  <h2>Pareto Chart — <span style="font-family:monospace">PARETO_SHORT</span></h2>
  <p><a href="/">← Dashboard</a></p>
  <div class="controls">
    <label>X axis: <select id="x-axis"></select></label>
    <label>Y axis: <select id="y-axis"></select></label>
    <button class="btn" onclick="downloadPNG()">Download PNG</button>
  </div>
  <div id="chart"></div>
  <script>
const PTS = PARETO_PTS_JSON;
const AXES = [
  {key:'p95_latency_ms',           label:'p95 Latency (ms)',          minimize:true},
  {key:'p50_latency_ms',           label:'p50 Latency (ms)',          minimize:true},
  {key:'tokens_per_second',        label:'Throughput (tok/s)',        minimize:false},
  {key:'decode_tokens_per_second', label:'Decode Throughput (tok/s)', minimize:false},
  {key:'p50_ttft_ms',              label:'TTFT p50 (ms)',             minimize:true},
  {key:'p95_ttft_ms',              label:'TTFT p95 (ms)',             minimize:true},
  {key:'peak_vram_memory_mb',      label:'VRAM (MB)',                 minimize:true},
  {key:'peak_cuda_memory_mb',      label:'CUDA Mem (MB)',             minimize:true},
  {key:'peak_cpu_memory_mb',       label:'CPU Mem (MB)',              minimize:true},
  {key:'model_load_ms',            label:'Load Time (ms)',            minimize:true},
  {key:'tokens_per_joule',         label:'Efficiency (tok/J)',        minimize:false},
  {key:'sanity_pass_rate',         label:'Sanity Pass Rate',         minimize:false},
];

function paretoMask(pts, xKey, yKey, xMin, yMin) {
  const dominated = pts.map(() => false);
  for (let i = 0; i < pts.length; i++) {
    for (let j = 0; j < pts.length; j++) {
      if (i === j) continue;
      const xi = pts[i][xKey], yi = pts[i][yKey];
      const xj = pts[j][xKey], yj = pts[j][yKey];
      if (xi == null || yi == null || xj == null || yj == null) continue;
      const xBetter = xMin ? xj <= xi : xj >= xi;
      const yBetter = yMin ? yj <= yi : yj >= yi;
      const xStrict = xMin ? xj < xi : xj > xi;
      const yStrict = yMin ? yj < yi : yj > yi;
      if (xBetter && yBetter && (xStrict || yStrict)) { dominated[i] = true; break; }
    }
  }
  return dominated.map(d => !d);
}

function redraw() {
  const xKey  = document.getElementById('x-axis').value;
  const yKey  = document.getElementById('y-axis').value;
  const xMeta = AXES.find(a => a.key === xKey) || {label: xKey, minimize: true};
  const yMeta = AXES.find(a => a.key === yKey) || {label: yKey, minimize: false};

  const valid = PTS.filter(p => p[xKey] != null && p[yKey] != null);
  if (!valid.length) {
    Plotly.react('chart', [], {title: 'No data for selected axes'});
    return;
  }

  const mask   = valid.length > 1
    ? paretoMask(valid, xKey, yKey, xMeta.minimize, yMeta.minimize)
    : [true];
  const normal = valid.filter((p, i) => !p.highlight && !mask[i]);
  const pareto = valid.filter((p, i) =>  mask[i] && !p.highlight);
  const hi     = valid.filter(p => p.highlight);
  const ht     = '%{text}<br>' + xMeta.label + ': %{x:.3g}<br>'
    + yMeta.label + ': %{y:.3g}<extra></extra>';

  const traces = [];
  if (normal.length) traces.push({
    type:'scatter', mode:'markers', name:'Other runs',
    x: normal.map(p => p[xKey]), y: normal.map(p => p[yKey]),
    text: normal.map(p => p.label),
    marker: {color:'#94a3b8', size:10},
    hovertemplate: ht,
  });
  if (pareto.length) {
    const pf = [...pareto].sort((a, b) => a[xKey] - b[xKey]);
    traces.push({
      type:'scatter', mode:'markers+lines', name:'Pareto front',
      x: pf.map(p => p[xKey]), y: pf.map(p => p[yKey]),
      text: pf.map(p => p.label),
      marker: {color:'#3b82f6', size:12},
      line: {dash:'dot', color:'#3b82f6'},
      hovertemplate: ht,
    });
  }
  if (hi.length) traces.push({
    type:'scatter', mode:'markers', name:'This run',
    x: hi.map(p => p[xKey]), y: hi.map(p => p[yKey]),
    text: hi.map(p => p.label),
    marker: {color:'#f59e0b', size:16, symbol:'star'},
    hovertemplate: ht,
  });

  Plotly.react('chart', traces, {
    title: 'Pareto — Run PARETO_SHORT',
    xaxis: {title: xMeta.label},
    yaxis: {title: yMeta.label},
    hovermode: 'closest',
    legend: {orientation: 'h'},
  });
}

function downloadPNG() {
  Plotly.downloadImage('chart', {
    format: 'png', filename: 'pareto-PARETO_SHORT', width: 1200, height: 700,
  });
}

const available = AXES.filter(a => PTS.some(p => p[a.key] != null));
const xSel = document.getElementById('x-axis');
const ySel = document.getElementById('y-axis');
available.forEach(a => {
  xSel.appendChild(new Option(a.label, a.key));
  ySel.appendChild(new Option(a.label, a.key));
});
const defX = available.find(a => a.key === 'p95_latency_ms') || available[0];
const defY = available.find(a => a.key === 'tokens_per_second') || available[1] || available[0];
if (defX) xSel.value = defX.key;
if (defY) ySel.value = defY.key;
xSel.addEventListener('change', redraw);
ySel.addEventListener('change', redraw);
redraw();
  </script>
</body>
</html>"""


def _render_pareto_html(run_id: str, results: list[RunResult]) -> str:
    pts: list[dict[str, Any]] = []
    for r in results:
        m = _parse_metrics_from_output(r.output)
        if m.get("p95_latency_ms") is None or m.get("tokens_per_second") is None:
            continue
        cfg = r.config or {}
        pt: dict[str, Any] = {
            "run_id": r.run_id,
            "label": (
                f"{cfg.get('backend', '?')}/{str(cfg.get('model', '?'))[:20]}<br>{r.run_id[:8]}"
            ),
            "highlight": r.run_id == run_id,
        }
        for key in _NUMERIC_METRIC_KEYS:
            pt[key] = m.get(key)
        pts.append(pt)

    short = html.escape(run_id[:8])

    if not pts:
        return (
            f"<!DOCTYPE html><html lang='en'><body>"
            f"<h2>Pareto — {short}</h2>"
            f"<p>No completed runs with parseable metrics yet.</p>"
            f"<p><a href='/'>&#8592; Dashboard</a></p>"
            f"</body></html>"
        )

    return _PARETO_CHART_TMPL.replace("PARETO_PTS_JSON", json.dumps(pts)).replace(
        "PARETO_SHORT", short
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
    # Preload bundled nvidia-* CUDA libs so llama_cpp's native extension can find them
    # on driver-only systems (prebuilt CUDA wheel install).  No-op when not installed.
    try:
        from llm_inference_benchmark.backends.llama_cpp import _preload_nvidia_cuda_libs

        _preload_nvidia_cuda_libs()
    except Exception:
        pass
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


@app.patch("/api/runs/{run_id}", status_code=204)
async def update_run_label(run_id: str, body: LabelUpdate) -> None:
    label: str | None = body.label.strip()[:80] or None
    db = _get_db()
    row = db.execute("SELECT id FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    db.execute("UPDATE runs SET label=? WHERE id=?", (label, run_id))
    db.commit()


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
        ["run_id", "label", "backend", "model", "status", "created_at", "finished_at"]
        + _NUMERIC_METRIC_KEYS
    )
    writer.writerow(
        [
            run.run_id,
            run.label or "",
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
async def run_list_fragment(
    q: str | None = None,
    status: str | None = None,
    sort: str | None = None,
) -> str:
    sql_order = "created_at ASC" if sort == "oldest" else "created_at DESC"
    rows = _get_db().execute(f"SELECT * FROM runs ORDER BY {sql_order}").fetchall()
    results = [_row_to_result(r) for r in rows]
    if status:
        results = [r for r in results if r.status == status]
    if q:
        q_lower = q.lower()
        results = [
            r
            for r in results
            if q_lower in (r.config or {}).get("model", "").lower()
            or q_lower in (r.config or {}).get("backend", "").lower()
            or r.run_id.startswith(q_lower)
            or q_lower in (r.label or "").lower()
        ]
    if sort == "model":
        results.sort(key=lambda r: (r.config or {}).get("model", "").lower())
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


@app.get("/api/ui/compare-table", response_class=HTMLResponse)
async def compare_table_fragment(
    ids: str = Query(..., description="Comma-separated run IDs"),
) -> str:
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if len(id_list) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 run IDs via ?ids=")
    placeholders = ",".join("?" * len(id_list))
    rows = (
        _get_db()
        .execute(
            f"SELECT * FROM runs WHERE id IN ({placeholders})",
            id_list,
        )
        .fetchall()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No matching runs found")
    results = [_row_to_result(r) for r in rows]
    id_order = {rid: i for i, rid in enumerate(id_list)}
    results.sort(key=lambda r: id_order.get(r.run_id, 999))
    return _render_compare_table(results)


@app.get("/api/ui/compare-chart", response_class=HTMLResponse)
async def compare_chart_fragment(
    ids: str = Query(..., description="Comma-separated run IDs"),
) -> str:
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if len(id_list) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 run IDs via ?ids=")
    placeholders = ",".join("?" * len(id_list))
    rows = (
        _get_db()
        .execute(
            f"SELECT * FROM runs WHERE id IN ({placeholders})",
            id_list,
        )
        .fetchall()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No matching runs found")
    results = [_row_to_result(r) for r in rows]
    id_order = {rid: i for i, rid in enumerate(id_list)}
    results.sort(key=lambda r: id_order.get(r.run_id, 999))
    return _render_compare_chart(results)


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
