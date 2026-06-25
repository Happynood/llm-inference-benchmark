/* llm-bench dashboard UI */
'use strict';

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function setNestedValue(obj, path, value) {
  const parts = path.split('.');
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!cur[parts[i]] || typeof cur[parts[i]] !== 'object') cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}

/* ── Backend field definitions ───────────────────────────────────────────── */

const BACKEND_FIELDS = {
  'mock': [
    {id: 'f-mock-latency',  key: 'mock.latency_ms',           label: 'Latency (ms)',       type: 'number', default: 10},
    {id: 'f-mock-tokens',   key: 'mock.tokens_per_response',  label: 'Tokens / response',  type: 'number', default: 50},
  ],
  'llama-cpp': [
    {id: 'f-llama-gpu',    key: 'llama_cpp.n_gpu_layers', label: 'GPU Layers',    type: 'number', default: -1,   hint: '-1 offloads all layers to GPU; 0 = CPU only'},
    {id: 'f-llama-ctx',    key: 'llama_cpp.n_ctx',        label: 'Context Size',  type: 'number', default: 4096},
    {id: 'f-llama-tokens', key: 'llama_cpp.max_tokens',   label: 'Max Tokens',    type: 'number', default: 50},
  ],
  'transformers': [
    {id: 'f-hf-device', key: 'hf.device',          label: 'Device',         type: 'select', options: ['cpu', 'cuda', 'mps'], default: 'cpu', hint: 'Use "cuda" for NVIDIA GPU acceleration'},
    {id: 'f-hf-dtype',  key: 'hf.torch_dtype',     label: 'Precision',      type: 'select', options: ['float32', 'float16', 'bfloat16'], default: 'float32'},
    {id: 'f-hf-tokens', key: 'hf.max_new_tokens',  label: 'Max New Tokens', type: 'number', default: 50},
  ],
  'openai': [
    {id: 'f-oai-url',    key: 'openai.base_url',   label: 'Base URL',     type: 'text',   default: 'http://localhost:8080/v1', hint: 'Set OPENAI_API_KEY env var for authentication'},
    {id: 'f-oai-tokens', key: 'openai.max_tokens', label: 'Max Tokens',   type: 'number', default: 50},
    {id: 'f-oai-stream', key: 'openai.stream',     label: 'Streaming',    type: 'select', options: ['false', 'true'], default: 'false'},
  ],
  'vllm': [
    {id: 'f-vllm-tp',     key: 'vllm.tensor_parallel_size',   label: 'Tensor Parallel',  type: 'number', default: 1},
    {id: 'f-vllm-gpu',    key: 'vllm.gpu_memory_utilization', label: 'GPU Memory Util',  type: 'number', default: 0.9, step: 0.05, min: 0.1, max: 1.0},
    {id: 'f-vllm-tokens', key: 'vllm.max_new_tokens',         label: 'Max New Tokens',   type: 'number', default: 50},
  ],
  'onnx': [
    {id: 'f-onnx-device', key: 'onnx.device',          label: 'Device',         type: 'select', options: ['cpu', 'cuda'], default: 'cpu'},
    {id: 'f-onnx-tokens', key: 'onnx.max_new_tokens',  label: 'Max New Tokens', type: 'number', default: 50},
  ],
};

/* Maps model type to human-readable backend tag shown in dropdowns */
const typeTag = {gguf: 'llama.cpp', hf: 'transformers'};

/* ── State ───────────────────────────────────────────────────────────────── */

let selectedRunId = null;
let sseConn = null;
const compareSet = new Set();

/* ── Tab switching ───────────────────────────────────────────────────────── */

function showTab(name) {
  ['runs', 'datasets'].forEach(function(t) {
    const panel = document.getElementById('tab-' + t);
    const btn   = document.getElementById('tab-btn-' + t);
    if (panel) panel.hidden = (t !== name);
    if (btn)   btn.classList.toggle('active', t === name);
  });
  if (name === 'datasets') {
    loadDatasetNames();
  }
}

/* ── Run list ────────────────────────────────────────────────────────────── */

function selectRun(runId) {
  selectedRunId = runId;
  document.querySelectorAll('.run-card').forEach(function(el) {
    el.classList.toggle('selected', el.dataset.runId === runId);
  });
  htmx.ajax('GET', '/api/ui/run-detail/' + runId, {target: '#run-detail', swap: 'innerHTML'});
}

/* Preserve selected card after HTMX run-list refresh */
document.addEventListener('htmx:afterSettle', function(evt) {
  const target = evt.detail && evt.detail.target;
  if (!target) return;

  if (target.id === 'run-list') {
    if (selectedRunId) {
      const card = document.querySelector('[data-run-id="' + selectedRunId + '"]');
      if (card) card.classList.add('selected');
    }
    compareSet.forEach(function(rid) {
      const cb = document.querySelector('.compare-cb[value="' + rid + '"]');
      if (cb) cb.checked = true;
    });
    return;
  }

  if (target.id === 'run-detail') {
    const inner = document.getElementById('detail-inner');
    if (inner) {
      const status = inner.dataset.status;
      const runId  = inner.dataset.runId;
      if (status === 'running' || status === 'pending') {
        startSSE(runId);
      }
    }

    const chartDiv = document.getElementById('cmp-chart-div');
    if (chartDiv && typeof Plotly !== 'undefined') {
      try {
        var traces = JSON.parse(chartDiv.dataset.traces || '[]');
        var layout = JSON.parse(chartDiv.dataset.layout || '{}');
        Plotly.newPlot(chartDiv, traces, layout, {responsive: true});
      } catch (_e) { /* malformed data — ignore */ }
    }

    const trendDiv = document.getElementById('cmp-trend-div');
    if (trendDiv && typeof Plotly !== 'undefined') {
      try {
        var tTraces = JSON.parse(trendDiv.dataset.traces || '[]');
        var tLayout = JSON.parse(trendDiv.dataset.layout || '{}');
        Plotly.newPlot(trendDiv, tTraces, tLayout, {responsive: true});
      } catch (_e) { /* malformed data — ignore */ }
    }
  }
});

/* ── SSE live log ────────────────────────────────────────────────────────── */

function startSSE(runId) {
  if (sseConn) { sseConn.close(); sseConn = null; }
  const logEl = document.getElementById('log-output');
  if (!logEl) return;
  logEl.textContent = '';

  sseConn = new EventSource('/api/runs/' + runId + '/stream');
  sseConn.onmessage = function(e) {
    if (e.data.startsWith('[done:')) {
      sseConn.close(); sseConn = null;
      /* Reload detail panel to show final metrics */
      if (selectedRunId === runId) {
        setTimeout(function() {
          htmx.ajax('GET', '/api/ui/run-detail/' + runId, {target: '#run-detail', swap: 'innerHTML'});
        }, 400);
      }
      return;
    }
    if (logEl) {
      logEl.textContent += e.data + '\n';
      logEl.scrollTop = logEl.scrollHeight;
    }
  };
  sseConn.onerror = function() { if (sseConn) { sseConn.close(); sseConn = null; } };
}

/* ── Delete run ──────────────────────────────────────────────────────────── */

function deleteRun(runId) {
  if (!confirm('Delete run ' + runId.slice(0, 8) + '? This cannot be undone.')) return;
  fetch('/api/runs/' + runId, {method: 'DELETE'}).then(function(r) {
    if (r.status === 409) {
      return r.json().then(function(d) { alert(d.detail || 'Run is in progress.'); });
    }
    if (!r.ok) { alert('Delete failed (HTTP ' + r.status + ').'); return; }
    if (selectedRunId === runId) {
      selectedRunId = null;
      document.getElementById('run-detail').innerHTML =
        '<div class="empty-state"><p class="empty-sub">Run deleted.</p></div>';
    }
    compareSet.delete(runId);
    updateCompareBar();
    htmx.trigger('#run-list', 'load');
  }).catch(function(err) { alert('Delete failed: ' + err.message); });
}

/* ── Clone run ───────────────────────────────────────────────────────────── */

function cloneRun(runId) {
  fetch('/api/runs/' + runId).then(function(r) {
    if (!r.ok) { alert('Could not load run config.'); return; }
    return r.json();
  }).then(function(run) {
    if (!run) return;
    _pendingCloneConfig = run.config || {};
    openNewRunModal();
  }).catch(function(err) { alert('Clone failed: ' + err.message); });
}

function applyCloneConfig(cfg) {
  /* Set model */
  var modelSel = document.getElementById('f-model');
  if (modelSel && cfg.model) {
    for (var i = 0; i < modelSel.options.length; i++) {
      if (modelSel.options[i].value === cfg.model) {
        modelSel.selectedIndex = i;
        break;
      }
    }
    var hintEl = document.getElementById('model-hint');
    if (hintEl) {
      hintEl.textContent = cfg.model.length > 60 ? '…' + cfg.model.slice(-57) : cfg.model;
    }
  }

  /* Set backend and render its fields */
  var backendSel = document.getElementById('f-backend');
  if (backendSel && cfg.backend) {
    backendSel.value = cfg.backend;
  }
  onBackendChange();

  /* Set run settings */
  if (cfg.requests !== undefined) {
    var el = document.getElementById('f-requests');
    if (el) el.value = cfg.requests;
  }
  if (cfg.concurrency !== undefined) {
    var el2 = document.getElementById('f-concurrency');
    if (el2) el2.value = cfg.concurrency;
  }
  if (cfg.warmup_requests !== undefined) {
    var el3 = document.getElementById('f-warmup');
    if (el3) el3.value = cfg.warmup_requests;
  }

  /* Set backend-specific fields */
  var backend = (backendSel && backendSel.value) || cfg.backend || '';
  var fields = BACKEND_FIELDS[backend] || [];
  fields.forEach(function(f) {
    var fieldEl = document.getElementById(f.id);
    if (!fieldEl) return;
    var parts = f.key.split('.');
    var val = cfg;
    for (var j = 0; j < parts.length; j++) {
      val = val && val[parts[j]];
    }
    if (val !== undefined && val !== null) {
      fieldEl.value = String(val);
    }
  });
}

/* ── Multi-run comparison ────────────────────────────────────────────────── */

function toggleCompare(evt, runId) {
  evt.stopPropagation();
  if (compareSet.has(runId)) {
    compareSet.delete(runId);
  } else {
    compareSet.add(runId);
  }
  updateCompareBar();
}

function updateCompareBar() {
  const bar   = document.getElementById('compare-bar');
  const count = document.getElementById('compare-count');
  if (!bar) return;
  if (compareSet.size >= 2) {
    bar.hidden = false;
    count.textContent = compareSet.size + ' runs selected';
  } else {
    bar.hidden = true;
  }
}

function openComparison() {
  if (compareSet.size < 2) return;
  window.open('/runs/pareto?ids=' + Array.from(compareSet).join(','), '_blank');
}

function openCompareExport() {
  if (compareSet.size < 1) return;
  window.location.href = '/api/runs/export.csv?ids=' + Array.from(compareSet).join(',');
}

function openCompareTable() {
  if (compareSet.size < 2) return;
  var ids = Array.from(compareSet).join(',');
  htmx.ajax('GET', '/api/ui/compare-table?ids=' + ids, {
    target: '#run-detail',
    swap: 'innerHTML'
  });
}

function openCompareChart() {
  if (compareSet.size < 2) return;
  var ids = Array.from(compareSet).join(',');
  function doLoad() {
    htmx.ajax('GET', '/api/ui/compare-chart?ids=' + ids, {
      target: '#run-detail',
      swap: 'innerHTML'
    });
  }
  if (typeof Plotly !== 'undefined') {
    doLoad();
  } else {
    var s = document.createElement('script');
    s.src = 'https://cdn.plot.ly/plotly-2.32.0.min.js';
    s.onload = doLoad;
    document.head.appendChild(s);
  }
}

function openCompareTrend() {
  if (compareSet.size < 2) return;
  var ids = Array.from(compareSet).join(',');
  function doLoad() {
    htmx.ajax('GET', '/api/ui/compare-trend?ids=' + ids, {
      target: '#run-detail',
      swap: 'innerHTML'
    });
  }
  if (typeof Plotly !== 'undefined') {
    doLoad();
  } else {
    var s = document.createElement('script');
    s.src = 'https://cdn.plot.ly/plotly-2.32.0.min.js';
    s.onload = doLoad;
    document.head.appendChild(s);
  }
}

function clearComparison() {
  compareSet.clear();
  document.querySelectorAll('.compare-cb').forEach(function(cb) { cb.checked = false; });
  updateCompareBar();
}

/* ── New Run Modal ───────────────────────────────────────────────────────── */

function openNewRunModal() {
  document.getElementById('modal-overlay').hidden = false;
  loadModels();
  loadModalDatasets();
  onBackendChange();
}

function closeNewRunModal() {
  document.getElementById('modal-overlay').hidden = true;
}

function overlayClick(evt) {
  if (evt.target === document.getElementById('modal-overlay')) closeNewRunModal();
}

/* ── Model loading ───────────────────────────────────────────────────────── */

var _pendingCloneConfig = null;

function loadModels() {
  fetch('/api/models').then(function(r) { return r.json(); }).then(function(data) {
    const sel = document.getElementById('f-model');
    const models = data.models || [];
    if (!models.length) {
      sel.innerHTML = '<option value="">(no models found in ~/models or HF cache)</option>';
      return;
    }
    sel.innerHTML = models.map(function(m) {
      const val  = m.value || (m.type === 'gguf' ? m.path : m.name);
      const tag  = typeTag[m.type] || m.type || '';
      const name = m.name || val;
      const disp = name.split('/').pop() || name;
      const label = tag ? (disp + ' (' + tag + ')') : disp;
      return '<option value="' + esc(val) + '" data-type="' + esc(m.type) + '" title="' + esc(val) + '">' + esc(label) + '</option>';
    }).join('');
    if (_pendingCloneConfig) {
      applyCloneConfig(_pendingCloneConfig);
      _pendingCloneConfig = null;
    } else {
      onModelChange();
    }
  }).catch(function() {
    document.getElementById('f-model').innerHTML = '<option value="">(failed to load)</option>';
  });
}

function onModelChange() {
  const sel = document.getElementById('f-model');
  const opt = sel && sel.options[sel.selectedIndex];
  if (!opt) return;

  const type = opt.dataset.type;
  const backendSel = document.getElementById('f-backend');
  if (type === 'gguf' && backendSel.value !== 'llama-cpp') {
    backendSel.value = 'llama-cpp';
    onBackendChange();
  } else if (type === 'hf' && backendSel.value !== 'transformers') {
    backendSel.value = 'transformers';
    onBackendChange();
  }

  const hintEl = document.getElementById('model-hint');
  if (hintEl) {
    const path = opt.value;
    hintEl.textContent = path.length > 60 ? '…' + path.slice(-57) : path;
  }
}

/* ── Backend-specific fields ─────────────────────────────────────────────── */

function onBackendChange() {
  const backend = document.getElementById('f-backend').value;
  renderBackendFields(backend);
  if (backend === 'llama-cpp') {
    fetch('/api/capabilities').then(function(r) { return r.json(); }).then(function(caps) {
      if (!caps.llama_cpp_gpu) {
        appendLlamaCppGpuWarning();
      }
    }).catch(function() {});
  }
}

function appendLlamaCppGpuWarning() {
  const container = document.getElementById('backend-fields');
  if (!container || container.querySelector('.gpu-warning')) return;
  const warn = document.createElement('div');
  warn.className = 'gpu-warning';
  warn.innerHTML =
    '<strong>GPU offload unavailable.</strong> ' +
    'The installed llama-cpp-python was built without CUDA support — ' +
    'the model will run on CPU regardless of the GPU Layers setting. ' +
    'To enable GPU acceleration, reinstall with CUDA: ' +
    '<code>pip install llama-cpp-python ' +
    '--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121</code>' +
    ' or build from source: ' +
    '<code>CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp</code>';
  container.appendChild(warn);
}

function renderBackendFields(backend) {
  const fields = BACKEND_FIELDS[backend] || [];
  const container = document.getElementById('backend-fields');
  if (!fields.length) { container.innerHTML = ''; return; }

  const inner = fields.map(function(f) {
    let input;
    if (f.type === 'select') {
      const opts = f.options.map(function(o) {
        return '<option value="' + esc(o) + '"' + (o === String(f.default) ? ' selected' : '') + '>' + esc(o) + '</option>';
      }).join('');
      input = '<select id="' + esc(f.id) + '" class="form-input">' + opts + '</select>';
    } else {
      const extras = [
        f.step !== undefined ? 'step="' + f.step + '"' : '',
        f.min  !== undefined ? 'min="'  + f.min  + '"' : '',
        f.max  !== undefined ? 'max="'  + f.max  + '"' : '',
      ].filter(Boolean).join(' ');
      input = '<input type="' + esc(f.type) + '" id="' + esc(f.id) + '" class="form-input" value="' + esc(f.default) + '"' + (extras ? ' ' + extras : '') + '>';
    }
    const hint = f.hint ? '<div class="form-hint">' + esc(f.hint) + '</div>' : '';
    return '<div class="form-group"><label class="form-label" for="' + esc(f.id) + '">' + esc(f.label) + '</label>' + input + hint + '</div>';
  }).join('');

  container.innerHTML = '<fieldset class="fieldset"><legend>' + esc(backend) + ' Settings</legend>' + inner + '</fieldset>';
}

/* ── Build config object from form ──────────────────────────────────────── */

function buildConfig() {
  const backend = document.getElementById('f-backend').value;
  const model   = document.getElementById('f-model').value;

  const config = {
    backend:         backend,
    model:           model,
    requests:        parseInt(document.getElementById('f-requests').value)    || 10,
    concurrency:     parseInt(document.getElementById('f-concurrency').value) || 1,
    warmup_requests: parseInt(document.getElementById('f-warmup').value)      || 0,
  };

  const fields = BACKEND_FIELDS[backend] || [];
  fields.forEach(function(f) {
    const el = document.getElementById(f.id);
    if (!el) return;
    let val = el.value;
    if (f.type === 'number') {
      val = parseFloat(val);
      if (isNaN(val)) return;
    }
    if (f.type === 'select' && (val === 'true' || val === 'false')) {
      val = (val === 'true');
    }
    setNestedValue(config, f.key, val);
  });

  return config;
}

/* ── Submit run ──────────────────────────────────────────────────────────── */

function submitRun() {
  const dataset = document.getElementById('f-dataset').value || null;
  const config  = buildConfig();

  if (!config.model) { alert('Please select a model.'); return; }

  const btn = document.getElementById('submit-btn');
  btn.disabled = true; btn.textContent = 'Submitting…';

  fetch('/api/runs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({config: config, dataset: dataset}),
  }).then(function(r) {
    if (!r.ok) { return r.json().then(function(d) { throw new Error(d.detail || ('HTTP ' + r.status)); }); }
    return r.json();
  }).then(function(data) {
    closeNewRunModal();
    btn.disabled = false; btn.textContent = 'Run Benchmark';
    htmx.trigger('#run-list', 'load');
    if (data.run_id) {
      setTimeout(function() { selectRun(data.run_id); }, 300);
    }
  }).catch(function(err) {
    alert('Failed to start run: ' + err.message);
    btn.disabled = false; btn.textContent = 'Run Benchmark';
  });
}

/* ── Dataset helpers ─────────────────────────────────────────────────────── */

function loadDatasetNames() {
  fetch('/api/datasets').then(function(r) { return r.json(); }).then(function(data) {
    const sel = document.getElementById('dataset-select');
    if (!sel) return;
    sel.innerHTML = '<option value="">Select dataset…</option>' +
      (data.datasets || []).map(function(d) {
        return '<option value="' + esc(d.name) + '">' + esc(d.name) + '</option>';
      }).join('');
  }).catch(function() {});
}

function loadModalDatasets() {
  fetch('/api/datasets').then(function(r) { return r.json(); }).then(function(data) {
    const sel = document.getElementById('f-dataset');
    if (!sel) return;
    const cached = (data.datasets || []).filter(function(d) { return d.cached; });
    sel.innerHTML = '<option value="">Default prompts</option>' +
      cached.map(function(d) {
        return '<option value="' + esc(d.name) + '">' + esc(d.name) + ' (' + d.samples + ' samples)</option>';
      }).join('');
  }).catch(function() {});
}

function pullSelectedDataset() {
  const name = document.getElementById('dataset-select').value;
  if (!name) { alert('Select a dataset first.'); return; }
  const statusEl = document.getElementById('pull-status');
  statusEl.textContent = 'Starting pull…';
  fetch('/api/datasets/pull', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: name}),
  }).then(function(r) {
    if (!r.ok) { return r.json().then(function(d) { throw new Error(d.detail || ('HTTP ' + r.status)); }); }
    return r.json();
  }).then(function() {
    statusEl.textContent = 'Pulling — table refreshes automatically.';
    htmx.trigger('#datasets-tbody', 'load');
  }).catch(function(err) { statusEl.textContent = 'Error: ' + err.message; });
}

/* ── Run label inline editing ────────────────────────────────────────────── */

function startEditLabel(evt, runId) {
  evt.stopPropagation();
  var cardContainer = document.getElementById('lbl-' + runId);
  var detailContainer = document.getElementById('detail-lbl-' + runId);
  var container = cardContainer || detailContainer;
  if (!container) return;
  var currentLabel = container.dataset.label || '';
  container.innerHTML =
    '<input class="run-label-input form-input form-input-sm"' +
    ' value="' + esc(currentLabel) + '"' +
    ' maxlength="80"' +
    ' onclick="event.stopPropagation()"' +
    ' onblur="saveRunLabel(event,\'' + runId + '\')"' +
    ' onkeydown="labelKeydown(event,\'' + runId + '\')">';
  container.querySelector('input').focus();
  container.querySelector('input').select();
}

function labelKeydown(evt, runId) {
  if (evt.key === 'Enter') { evt.preventDefault(); evt.target.blur(); }
  if (evt.key === 'Escape') { cancelEditLabel(evt, runId); }
}

function cancelEditLabel(evt, runId) {
  var cardContainer = document.getElementById('lbl-' + runId);
  var detailContainer = document.getElementById('detail-lbl-' + runId);
  [cardContainer, detailContainer].forEach(function(c) {
    if (c) renderLabelSpan(c, runId, c.dataset.label || '');
  });
}

function saveRunLabel(evt, runId) {
  var input = evt.target;
  var label = input.value.trim().slice(0, 80);
  fetch('/api/runs/' + runId, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({label: label}),
  }).then(function(r) {
    if (!r.ok) { alert('Failed to save label.'); cancelEditLabel(evt, runId); return; }
    var cardContainer = document.getElementById('lbl-' + runId);
    var detailContainer = document.getElementById('detail-lbl-' + runId);
    [cardContainer, detailContainer].forEach(function(c) {
      if (c) renderLabelSpan(c, runId, label);
    });
  }).catch(function() { alert('Failed to save label.'); cancelEditLabel(evt, runId); });
}

function renderLabelSpan(container, runId, label) {
  var text = label || 'Add label…';
  var cls = label ? 'run-label-text' : 'run-label-text run-label-placeholder';
  container.dataset.label = label;
  container.innerHTML =
    '<span class="' + cls + '"' +
    ' onclick="startEditLabel(event,\'' + runId + '\')">' +
    esc(text) + '</span>';
}

/* ── Init ────────────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', function() {
  /* Nothing needed at startup; runs and datasets load via HTMX hx-trigger=load */
});
