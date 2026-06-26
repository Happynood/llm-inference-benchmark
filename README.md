# llm-inference-benchmark

[![CI](https://github.com/Happynood/llm-inference-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Happynood/llm-inference-benchmark/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Happynood/llm-inference-benchmark?color=blue)](https://github.com/Happynood/llm-inference-benchmark/releases)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-informational)](https://github.com/Happynood/llm-inference-benchmark/pkgs/container/llm-inference-benchmark)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet)](https://docs.astral.sh/uv/)

A reproducible harness for LLM inference experiments. Compare backends, quantizations, and
runtime parameters under identical workloads — then get a constraint-based recommendation
for the best configuration. Includes a built-in browser dashboard.

---

## Quick Start

```bash
pip install llm-inference-benchmark
llm-bench serve
# Open http://localhost:8080
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Happynood/llm-inference-benchmark
cd llm-inference-benchmark
uv sync && uv run llm-bench serve
```

Open **http://localhost:8080** and submit your first benchmark from the **+ New Run** form.

---

## GPU Quick Start (NVIDIA CUDA)

```bash
uv sync --extra all-backends          # installs all backends, CPU llama-cpp included
make install-llama-cpp-prebuilt       # replaces CPU wheel with pre-built CUDA wheel
uv run llm-bench serve                # or: make webui-gpu
```

> After any `uv sync`, re-run `make install-llama-cpp-prebuilt` to restore GPU support.
> See **[docs/quickstart.md](docs/quickstart.md)** for full options and the `make setup-gpu` shortcut.

---

## Backends

| Backend | Install | GPU | Notes |
|---------|---------|-----|-------|
| `mock` | — | — | Deterministic CI backend, no model |
| `transformers` | `--extra transformers` | CUDA | HuggingFace `AutoModelForCausalLM`; set `hf.device: cuda` |
| `llama-cpp` | `--extra llama-cpp` | CUDA (pre-built wheel) | GGUF quantized inference, no nvcc needed |
| `openai` | — | server-side | Any `/v1/chat/completions`-compatible server |
| `onnx` | `--extra onnx` | CUDA (`onnxruntime-gpu`) | ONNX Runtime via Optimum; INT8/FP16 export |
| `vllm` | `--extra vllm` | CUDA, Linux only | High-throughput vLLM engine |

```bash
uv sync --extra all-backends   # install everything at once
```

---

## Web UI

`llm-bench serve` (or `make webui`) starts a local dashboard at **http://localhost:8080**.

- **Leaderboard** — sidebar tab showing the single best run for each key metric
  (Throughput, p50/p95 Latency, TTFT, VRAM, Energy) at a glance.
- **Recommend** — sidebar tab with a constraint form (max VRAM, max p95 latency, max TTFT,
  min sanity/quality rate); click **Find Best Run** to get an instant recommendation from
  all stored runs, with runners-up and excluded-run counts.
- **Compare bar** — select two or more runs to reveal Table / Chart / Trend / Pareto / CSV
  buttons above the run list.
- **Pareto chart** — interactive Plotly scatter at `/runs/{id}/pareto.html`; X/Y axis
  dropdowns let you explore any metric pair; **Download PNG** exports at 1200×700 px.
- **Live log streaming** — stdout streams into the detail panel via Server-Sent Events while
  a benchmark runs; metrics appear automatically on completion.
- **Clone run** — pre-fills the New Run modal from an existing run's config for fast iteration.
- **Run labels** — inline-editable name on every run card; included in keyword search and CSV export.
- **Search and filter** — keyword + status dropdown above the run list; auto-refresh respects the active filter.
- **Download CSV** — exports parsed metrics from any completed run.

```bash
llm-bench serve --host 0.0.0.0 --port 9000   # bind to all interfaces
make webui                                     # localhost:8080 shortcut
make webui-gpu                                 # install CUDA wheel then serve
```

---

## CLI

```
llm-bench [OPTIONS] [--config YAML] [--output CSV]       # single run
llm-bench compare   FILE [FILE...] [--sort p95|toks]     # Markdown comparison table
llm-bench diff      BASELINE CURRENT [--fail-on-regression PCT]  # per-metric % change
llm-bench pareto    FILE [FILE...]                        # Pareto classification
llm-bench recommend FILE [FILE...] [CONSTRAINTS]          # best config under constraints
llm-bench matrix    --config MATRIX_YAML [--dry-run]     # multi-run sweep / parameter grid
llm-bench sweep     --config YAML --concurrency-range 1,2,4,8   # throughput-vs-latency ramp
llm-bench pull      REPO_ID [--quant Q4_K_M]             # download GGUF / HF model
llm-bench datasets  pull <name>                           # cache real-world prompt dataset
llm-bench serve     [--host HOST] [--port PORT]           # Web UI server
llm-bench env [--format json]                             # print hardware / backend info
llm-bench validate-config --config YAML                   # validate config without running
llm-bench profiles                                        # list built-in workload profiles
```

```bash
# Override config values on the command line without editing YAML
uv run llm-bench --config configs/example.yaml --requests 50 --concurrency 4 --seed 42

# Test any OpenAI-compatible server (Ollama, LM Studio, vLLM) without a config file
uv run llm-bench --base-url http://localhost:11434/v1 --set model=llama3.2:3b

# Open-loop load mode: dispatch at a fixed arrival rate (reveals queueing latency)
uv run llm-bench --config configs/example.yaml --arrival-rate 5 --requests 50

# CI regression gate: fail if any metric degrades more than 5 %
llm-bench diff results/before.csv results/after.csv --fail-on-regression 5
```

Full reference: **[docs/cli.md](docs/cli.md)**

### Constraint-based recommendation

```bash
llm-bench recommend results/*.csv \
  --max-vram-mb 4096 \
  --max-p95-ms 1000 \
  --max-ttft-ms 200 \
  --min-sanity 1.0
```

```
Recommendation
──────────────────────────────────────────
  Backend  : llama-cpp
  Model    : Llama-3.2-3B-Instruct-Q4_K_M.gguf
  p95      : 915.86 ms
  tok/s    : 55.5
  TTFT p50 : 142.0 ms
  VRAM     : 2361.0 MB
  Sanity   : 100.0%

Why: lowest p95 among 1 candidate(s) passing all constraints; Pareto-optimal.
```

---

## Benchmark Results

### Llama 3.2 3B Instruct — Q4\_K\_M vs Q8\_0 · RTX 3050 (4 GB)

> llama.cpp backend, all 28 layers on GPU, 10 prompts per run.

| Quantization | p50 (ms) | p95 (ms) | tok/s | VRAM (MiB) | Sanity |
|---|---|---|---|---|---|
| **Q4\_K\_M** | **904** | **915** | **55.3** | 2361 (58%) | 100% |
| Q8\_0 | 1185 | 1187 | 42.2 | 3697 (90%) | 100% |

Q4\_K\_M is **1.31× faster** and uses **1.57× less VRAM**.

### n\_gpu\_layers Sweep — Llama 3.2 3B · RTX 3050

| Layers on GPU | p95 (ms) | tok/s | VRAM (MiB) |
|---|---|---|---|
| 0 / 28 (CPU only) | 3093 | 17.5 | 655 |
| 20 / 28 (partial) | 1420 | 36.6 | 1829 |
| **28 / 28 (full)** | **984** | **51.4** | **2361** |

Full offload is **3.1× faster** than CPU-only.

Harness validation (mock backend, no model download):

| Metric | Value | What it validates |
|--------|-------|-------------------|
| p50 latency | ~5 ms | configured `latency_ms` is measured |
| p95 latency | ~5 ms | p95 ≈ p50 for deterministic mock |
| tokens/sec | ~10,000 | tok/s formula is correct |

> Never compare mock numbers to real backend results — they validate pipeline plumbing only.

Full metric definitions: **[docs/metrics.md](docs/metrics.md)** · Curated reports: **[docs/results/](docs/results/)**

---

## Features

- **YAML config** — backend, model, requests, warmup, prompts file or workload profile
- **Workload profiles** — `short_chat`, `summarization`, `code_completion`, `long_context_smoke`
- **Run matrix** — cartesian-product sweep from one YAML; one CSV + manifest per combination
- **Metrics** — p50/p95 latency, TTFT p50/p95 (streaming), tok/s, ITL σ, decode throughput
- **Memory** — CPU RSS, PyTorch CUDA peak, driver-level VRAM via `nvidia-smi`
- **Lifecycle** — model load time, warmup latency
- **Variance** — `repeats: N` reports median ± std dev across N trial loops
- **Sanity checks** — empty, min/mean chars, repeated output, `sanity_pass_rate`
- **Task quality** — `quality_file:` YAML rubric (`contains_all`, `regex`, `forbidden`, …)
- **Self-perplexity** and **LLM-as-judge** score (`transformers` backend only)
- **Pareto analysis** — classifies configs as optimal or dominated across all metrics
- **Constraint-based recommender** — `--max-vram-mb`, `--max-p95-ms`, `--max-ttft-ms`,
  `--min-sanity`, `--min-quality`, `--max-perplexity`, `--min-judge`, `--max-load-ms`
- **JSON run manifest** — git commit, config/prompts SHA256, Python/OS/CPU/GPU fingerprint
- **Energy efficiency** — `energy_joules` and `tokens_per_joule` via `nvidia-smi` / Intel RAPL
- **Real-world datasets** — pull and cache HuggingFace prompt sets for contamination-resistant runs
- **Model downloads** — `llm-bench pull` downloads GGUF or HF snapshots with hash verification
- **Docker images** — CPU, GPU, Web UI, and Web UI + GPU variants on GHCR
- **HTML reports** — `llm-bench report` generates a shareable, self-contained HTML page from one or more CSVs

---

## Shareable Reports

Generate a single HTML file from any benchmark CSV — no server needed:

```bash
llm-bench report result.csv                              # → report.html
llm-bench report run1.csv run2.csv --output compare.html
llm-bench report *.csv --title "Llama-3.2 Quant Sweep"
```

The report includes an interactive Plotly chart (throughput vs p95 latency), a full metrics
table with Pareto-optimal runs highlighted, and optional hardware details.  Open it in any
browser or attach it to a GitHub issue.

---

## Configuration

Every benchmark is driven by a YAML file:

```yaml
# configs/llama-cpp-gpu.yaml
backend: llama-cpp
model: ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
requests: 20
warmup_requests: 2
prompts_file: data/prompts/smoke.txt   # or use workload_profile:
repeats: 3                              # median ± std dev across 3 trial loops

llama_cpp:
  n_ctx: 2048
  n_gpu_layers: 99     # offload all layers; 0 = CPU only
  max_tokens: 50
  temperature: 0.0     # greedy / deterministic
```

Run it:

```bash
uv run llm-bench --config configs/llama-cpp-gpu.yaml --output results/run.csv
```

Override any field without editing the YAML:

```bash
uv run llm-bench --config configs/llama-cpp-gpu.yaml \
  --set llama_cpp.n_gpu_layers=20 --requests 50 --seed 42
```

Use `validate-config` to catch typos before a long run:

```bash
uv run llm-bench validate-config --config configs/llama-cpp-gpu.yaml
```

---

## Real-World Datasets

```bash
llm-bench datasets pull wildchat          # 500 samples — real-world chat (public)
llm-bench datasets pull lmsys-chat        # 500 samples — multi-turn chat (gated, HF_TOKEN required)
llm-bench datasets pull long-context-4k   # 100 passages @ ~4k tokens (prefill benchmark)
llm-bench datasets pull gsm8k             # 200 grade-school math problems
llm-bench datasets pull humaneval         # 164 Python coding problems

# Use a cached dataset instead of the config prompts_file
llm-bench --config configs/example.yaml --dataset wildchat --requests 50
```

Samples are cached in `~/.cache/llm-bench/datasets/` as JSONL. No network access needed after the initial pull.

---

## Model Downloads

```bash
llm-bench pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --quant Q4_K_M
llm-bench pull HuggingFaceTB/SmolLM2-360M-Instruct --backend transformers
HF_TOKEN=hf_... llm-bench pull meta-llama/Llama-3.2-3B-Instruct-GGUF --quant Q4_K_M
```

Verifies SHA-256 against HuggingFace LFS metadata. Re-running skips already-cached files.

---

## Docker

Four image variants are published to GHCR on every release:

| Tag | Use case |
|-----|----------|
| `cpu-latest` | llama.cpp CPU benchmark |
| `gpu-latest` | llama.cpp + CUDA benchmark |
| `webui-latest` | Web dashboard (CPU) |
| `webui-gpu-latest` | Web dashboard + CUDA |

```bash
docker compose up webui          # CPU dashboard at http://localhost:8080
docker compose up webui-gpu      # GPU dashboard (requires NVIDIA Container Toolkit)
```

GGUF models at `~/models/` and the HuggingFace cache at `~/.cache/huggingface` are mounted
automatically. Results persist in a named Docker volume (`llm-bench-data`).

---

## Limitations

- Mock backend numbers validate pipeline plumbing only — never compare them to real backends.
- `peak_cuda_memory_mb = 0.0` for llama.cpp runs — use `peak_vram_memory_mb` instead.
- `measure_perplexity` and `measure_judge` are `transformers` backend only.
- OpenAI endpoint latency includes network round-trip and server-side queueing.
- Tested on Linux x86-64 (Ubuntu). macOS should work; Windows untested.

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for development setup, test instructions, and PR guidelines.

## Security

See **[SECURITY.md](SECURITY.md)** for the vulnerability disclosure policy.

## License

MIT — see [LICENSE](LICENSE).
