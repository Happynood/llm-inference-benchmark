"""FastAPI HTTP backend for llm-bench Web UI.

Exposes the following endpoints:
  GET  /                              dashboard (HTML)
  GET  /api/health
  GET  /api/models
  GET  /api/datasets
  POST /api/datasets/pull
  GET  /api/runs
  POST /api/runs
  GET  /api/runs/{run_id}
  GET  /api/runs/{run_id}/stream      Server-Sent Events
  GET  /api/ui/datasets-table         HTMX HTML fragment
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

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from llm_inference_benchmark import datasets as _datasets_mod

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

# ── Dataset pull error store (keyed by dataset name) ─────────────────────────

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
    .modal-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);
      z-index:100;align-items:center;justify-content:center}
    .modal-backdrop.open{display:flex}
    .modal{background:#fff;border-radius:.5rem;padding:1.5rem;width:100%;max-width:480px;
      box-shadow:0 8px 32px rgba(0,0,0,.18)}
    .modal h3{margin:0 0 1rem;font-size:1rem}
    .form-row{margin-bottom:.75rem}
    .form-row label{display:block;font-size:.8rem;color:#475569;margin-bottom:.25rem;
      font-weight:500}
    .form-row input,.form-row select,.form-row textarea{
      width:100%;padding:.35rem .6rem;border:1px solid #cbd5e1;border-radius:.3rem;
      font-size:.875rem;font-family:inherit}
    .form-row textarea{resize:vertical;min-height:60px}
    .modal-actions{display:flex;justify-content:flex-end;gap:.5rem;margin-top:1rem}
    #f-gpu-row{display:none}
    .ds-select{padding:.3rem .6rem;border:1px solid #cbd5e1;border-radius:.3rem;font-size:.875rem}
  </style>
</head>
<body>
<h1>llm-bench</h1>
<p style="color:#64748b;margin:.25rem 0 1.5rem">Benchmark Runs Dashboard</p>

<!-- New Run modal -->
<div class="modal-backdrop" id="run-modal" role="dialog" aria-modal="true"
     aria-labelledby="modal-title">
  <div class="modal">
    <h3 id="modal-title">New Benchmark Run</h3>
    <div class="form-row">
      <label for="f-model">Model</label>
      <select id="f-model"><option value="">Loading…</option></select>
    </div>
    <div class="form-row">
      <label for="f-dataset">Dataset (prompt source)</label>
      <select id="f-dataset"><option value="">Default prompts</option></select>
    </div>
    <div class="form-row">
      <label for="f-backend">Backend</label>
      <select id="f-backend" onchange="toggleGpuRow()">
        <option value="mock">mock</option>
        <option value="llama-cpp">llama-cpp</option>
        <option value="transformers">transformers</option>
        <option value="openai">openai</option>
        <option value="vllm">vllm</option>
        <option value="onnx">onnx</option>
      </select>
    </div>
    <div class="form-row">
      <label for="f-requests">Requests</label>
      <input type="number" id="f-requests" value="10" min="1">
    </div>
    <div class="form-row">
      <label for="f-concurrency">Concurrency</label>
      <input type="number" id="f-concurrency" value="1" min="1">
    </div>
    <div class="form-row">
      <label for="f-warmup">Warmup requests</label>
      <input type="number" id="f-warmup" value="2" min="0">
    </div>
    <div class="form-row" id="f-gpu-row">
      <label for="f-gpu-layers">GPU layers (llama-cpp)</label>
      <input type="number" id="f-gpu-layers" value="28" min="0">
    </div>
    <div class="form-row">
      <label for="f-extra">Extra YAML (optional)</label>
      <textarea id="f-extra" placeholder="key: value"></textarea>
    </div>
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="closeModal()">Cancel</button>
      <button class="btn" id="submit-btn" onclick="submitRun()">Run Benchmark</button>
    </div>
  </div>
</div>

<section>
  <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem">
    <h2 style="margin:0">Runs</h2>
    <button class="btn" id="new-run-btn" onclick="openModal()">+ New Run</button>
  </div>
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
    <button class="btn btn-outline" onclick="compareSelected()">Compare Selected</button>
    <button id="pareto-btn" class="btn btn-outline" onclick="paretoSelected()"
            style="display:none;margin-left:.5rem">Pareto</button>
  </div>
</section>

<section>
  <h2>Datasets</h2>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>Name</th><th>Description</th><th>Cached</th><th>Samples</th>
    </tr></thead>
    <tbody id="datasets-tbody"
           hx-get="/api/ui/datasets-table"
           hx-trigger="load, every 10s"
           hx-swap="innerHTML">
      <tr><td colspan="4" style="color:#94a3b8;padding:1.5rem .75rem">Loading…</td></tr>
    </tbody>
  </table>
  </div>
  <div style="display:flex;align-items:center;gap:.5rem;margin-top:.75rem">
    <select id="dataset-select" class="ds-select"></select>
    <button class="btn" onclick="pullDataset()">Pull</button>
    <span id="pull-status" style="font-size:.8rem;color:#64748b"></span>
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
function _updateToolbar() {
  var checked = Array.from(document.querySelectorAll('.run-cb:checked'));
  var pb = document.getElementById('pareto-btn');
  if (pb) pb.style.display = checked.length >= 2 ? '' : 'none';
}

document.getElementById('select-all').addEventListener('change', function() {
  document.querySelectorAll('.run-cb').forEach(function(cb){ cb.checked = this.checked; }, this);
  _updateToolbar();
});
document.body.addEventListener('change', function(evt) {
  if (evt.target && evt.target.classList.contains('run-cb')) _updateToolbar();
});

var _checkedRuns = new Set();
document.body.addEventListener('htmx:beforeSwap', function(evt) {
  if (evt.detail.target && evt.detail.target.id === 'runs-tbody') {
    _checkedRuns = new Set(
      Array.from(document.querySelectorAll('.run-cb:checked')).map(function(cb){ return cb.value; })
    );
  }
});
document.body.addEventListener('htmx:afterSettle', function(evt) {
  if (evt.detail.target && evt.detail.target.id === 'runs-tbody') {
    var all = document.querySelectorAll('.run-cb');
    all.forEach(function(cb){ cb.checked = _checkedRuns.has(cb.value); });
    var checked = Array.from(all).filter(function(cb){ return cb.checked; });
    var sa = document.getElementById('select-all');
    if (sa) {
      sa.checked = all.length > 0 && checked.length === all.length;
      sa.indeterminate = checked.length > 0 && checked.length < all.length;
    }
    _updateToolbar();
  }
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

function openModal() {
  document.getElementById('run-modal').classList.add('open');
  loadModels();
  loadModalDatasets();
}

function closeModal() {
  document.getElementById('run-modal').classList.remove('open');
}

function toggleGpuRow() {
  var backend = document.getElementById('f-backend').value;
  document.getElementById('f-gpu-row').style.display =
    backend === 'llama-cpp' ? 'block' : 'none';
}

function loadModels() {
  fetch('/api/models').then(function(r){ return r.json(); }).then(function(data) {
    var sel = document.getElementById('f-model');
    var models = data.models || [];
    if (!models.length) {
      sel.innerHTML = '<option value="">&lt;no models found&gt;</option>';
      return;
    }
    var typeTag = {gguf: 'llama.cpp', hf: 'transformers'};
    sel.innerHTML = models.map(function(m) {
      var val = m.path || m.id || '';
      var name = m.name || val;
      var short = name.indexOf('/') !== -1 ? name.substring(name.indexOf('/') + 1) : name;
      var tag = typeTag[m.type] || m.type || '';
      var label = tag ? short + ' (' + tag + ')' : short;
      var ev = val.replace(/&/g,'&amp;').replace(/"/g,'&quot;');
      var el = label.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      return '<option value="' + ev + '" title="' + ev + '">' + el + '</option>';
    }).join('');
  }).catch(function(){
    document.getElementById('f-model').innerHTML =
      '<option value="">(failed to load)</option>';
  });
}

function loadModalDatasets() {
  fetch('/api/datasets').then(function(r){ return r.json(); }).then(function(data) {
    var sel = document.getElementById('f-dataset');
    var cached = (data.datasets || []).filter(function(d){ return d.cached; });
    var opts = '<option value="">Default prompts</option>';
    cached.forEach(function(d){
      var label = d.name + ' (' + d.samples + ' samples)';
      opts += '<option value="' + d.name.replace(/"/g,'&quot;') + '">' + label + '</option>';
    });
    sel.innerHTML = opts;
  }).catch(function(){});
}

function submitRun() {
  var model    = document.getElementById('f-model').value;
  var dataset  = document.getElementById('f-dataset').value || null;
  var backend  = document.getElementById('f-backend').value;
  var requests = parseInt(document.getElementById('f-requests').value) || 10;
  var conc     = parseInt(document.getElementById('f-concurrency').value) || 1;
  var warmup   = parseInt(document.getElementById('f-warmup').value) || 0;
  var config   = {model: model, backend: backend, requests: requests,
                  concurrency: conc, warmup_requests: warmup};
  if (backend === 'llama-cpp') {
    var gpuLayers = parseInt(document.getElementById('f-gpu-layers').value);
    if (!isNaN(gpuLayers)) { config['llama_cpp'] = {n_gpu_layers: gpuLayers}; }
  }
  var extraYaml = (document.getElementById('f-extra').value || '').trim();
  if (extraYaml) {
    try {
      var parsed = jsyaml ? jsyaml.load(extraYaml) : null;
      if (parsed && typeof parsed === 'object') {
        Object.assign(config, parsed);
      }
    } catch(e) { /* ignore parse errors */ }
  }
  var body = {config: config};
  if (dataset) { body['dataset'] = dataset; }
  var btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.textContent = 'Submitting…';
  fetch('/api/runs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  }).then(function(r){
    if (!r.ok) { throw new Error('HTTP ' + r.status); }
    return r.json();
  }).then(function(data) {
    closeModal();
    btn.disabled = false;
    btn.textContent = 'Run Benchmark';
    htmx.trigger('#runs-tbody', 'load');
    if (data.run_id) {
      setTimeout(function(){ streamLog(data.run_id); }, 500);
    }
  }).catch(function(err) {
    alert('Failed to start run: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Run Benchmark';
  });
}

function loadDatasetNames() {
  fetch('/api/datasets').then(function(r){ return r.json(); }).then(function(data) {
    var sel = document.getElementById('dataset-select');
    sel.innerHTML = (data.datasets || []).map(function(d){
      return '<option value="' + d.name.replace(/"/g,'&quot;') + '">' + d.name + '</option>';
    }).join('');
  }).catch(function(){});
}
loadDatasetNames();

function pullDataset() {
  var name = document.getElementById('dataset-select').value;
  if (!name) return;
  var statusEl = document.getElementById('pull-status');
  statusEl.textContent = 'Starting pull…';
  fetch('/api/datasets/pull', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: name})
  }).then(function(r){
    if (!r.ok) { throw new Error('HTTP ' + r.status); }
    return r.json();
  }).then(function(){
    statusEl.textContent = 'Pull started — table will refresh automatically.';
    htmx.trigger('#datasets-tbody', 'load');
  }).catch(function(err){
    statusEl.textContent = 'Error: ' + err.message;
  });
}

function compareSelected() {
  var cbs = Array.from(document.querySelectorAll('.run-cb:checked'));
  if (cbs.length < 2) { alert('Select at least 2 runs to compare.'); return; }
  var labels = cbs.map(function(cb){ return cb.dataset.label; });
  var toks   = cbs.map(function(cb){ return parseFloat(cb.dataset.toks) || 0; });
  var ttfts  = cbs.map(function(cb){
    return cb.dataset.ttft ? parseFloat(cb.dataset.ttft) : null; });
  document.getElementById('chart-section').style.display = 'block';
  var traces = [
    {type:'bar', name:'Tokens / s', x:labels, y:toks, marker:{color:'#3b82f6'}}
  ];
  if (ttfts.some(function(v){ return v !== null; })) {
    traces.push({
      type:'bar', name:'TTFT p50 (ms)', x:labels,
      y:ttfts.map(function(v){ return v === null ? 0 : v; }),
      marker:{color:'#f59e0b'}, yaxis:'y2'
    });
  }
  Plotly.newPlot('comparison-chart', traces, {
    barmode:'group',
    title:{text:'Throughput & Latency', font:{size:14}},
    yaxis:{title:'Tokens / s', titlefont:{color:'#3b82f6'}, tickfont:{color:'#3b82f6'}},
    yaxis2:{title:'TTFT p50 (ms)', titlefont:{color:'#f59e0b'}, tickfont:{color:'#f59e0b'},
            overlaying:'y', side:'right'},
    legend:{orientation:'h', y:-0.2},
    margin:{t:40, b:80}
  });
}

function paretoSelected() {
  var cbs = Array.from(document.querySelectorAll('.run-cb:checked'));
  if (cbs.length < 2) { alert('Select at least 2 runs for Pareto analysis.'); return; }
  var ids = cbs.map(function(cb){ return cb.value; }).join(',');
  window.open('/runs/pareto?ids=' + encodeURIComponent(ids), '_blank');
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
            f' class="btn btn-sm btn-outline">Pareto</a>\n'
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


@app.get("/api/ui/datasets-table", response_class=HTMLResponse)
async def datasets_table_fragment() -> str:
    return _render_datasets_table(_dataset_statuses())


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
    # Use the first selected run as the "highlight" anchor
    return _render_pareto_html(id_list[0], results)
