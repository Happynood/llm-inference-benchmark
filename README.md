# llm-inference-benchmark

[![CI](https://github.com/Happynood/llm-inference-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Happynood/llm-inference-benchmark/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Happynood/llm-inference-benchmark?color=blue)](https://github.com/Happynood/llm-inference-benchmark/releases)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-informational)](https://github.com/Happynood/llm-inference-benchmark/pkgs/container/llm-inference-benchmark)
[![uv](https://img.shields.io/badge/uv-managed-blueviolet)](https://docs.astral.sh/uv/)

A reproducible harness for LLM inference optimization experiments. Compare backends, quantization
modes, runtime parameters, latency, throughput, TTFT, and quality under identical workloads —
then get a constraint-based recommendation for the best configuration.

---

## Why

Choosing between `transformers`, `llama.cpp`, `onnxruntime`, `vllm`, precision modes, and runtime
parameters requires apples-to-apples experiments under identical prompts and hardware.
Ad-hoc timing scripts produce inconsistent results that cannot be compared or reproduced.

This harness wraps the benchmark loop in a typed, config-driven CLI so every run is reproducible,
comparable, and feeds directly into Pareto analysis and constraint-based recommendation.

```
model × backend × precision × parameters × workload  →  Pareto-optimal configuration
```

---

## Goals

1. Make benchmark runs reproducible from repository files alone.
2. Compare multiple inference backends under the same prompts, config schema, and metric definitions.
3. Track quantization and precision metadata so runs are compared as configurations, not just backend names.
4. Report latency (p50/p95/TTFT), throughput, peak memory, and lifecycle timings transparently.
5. Add lightweight quality checks so speed and memory are not optimized in isolation.
6. Produce comparison tables, Pareto analysis, and constraint-based recommendations.
7. Keep the implementation small, typed, tested, CI-friendly, and easy to extend.
8. Separate harness validation results from production performance claims.

---

## Quick Start

```bash
# Install (no GPU required)
git clone https://github.com/Happynood/llm-inference-benchmark
cd llm-inference-benchmark
uv sync

# Run mock backend — validates the harness pipeline, no model download
uv run llm-bench --config configs/example.yaml --output results/mock.csv

# Compare results
uv run llm-bench compare results/mock.csv
```

For full installation options (transformers, llama.cpp, GPU), see **[docs/quickstart.md](docs/quickstart.md)**.

---

## Backends

| Backend | Install extra | GPU support | Notes |
|---|---|---|---|
| `mock` | — | — | Deterministic CI backend, no model |
| `transformers` | `--extra transformers` | CUDA | HuggingFace `AutoModelForCausalLM` |
| `llama-cpp` | `--extra llama-cpp` | CUDA (pre-built wheel) | GGUF quantized inference |
| `openai` | — | server-side | Any `/v1/chat/completions`-compatible server |
| `onnx` | `--extra onnx` | CUDA (via ORT provider) | ONNX Runtime via Optimum; supports INT8/FP16 export |
| `vllm` | `--extra vllm` | CUDA | High-throughput serving via vLLM engine (requires `vllm>=0.4`) |

---

## CLI

```
llm-bench [OPTIONS] [--config YAML] [--output CSV]     # single run
llm-bench compare   FILE [FILE...]                     # Markdown comparison table
llm-bench pareto    FILE [FILE...]                     # Pareto classification
llm-bench recommend FILE [FILE...] [CONSTRAINTS]       # best config under constraints
llm-bench matrix    --config MATRIX_YAML               # multi-run sweep
llm-bench profiles                                     # list built-in workload profiles
llm-bench validate-config --config YAML                # validate config without running
llm-bench env [--format json]                          # print environment/hardware info
llm-bench --version
```

Full reference: **[docs/cli.md](docs/cli.md)**

### Runtime overrides

```bash
# Override config values without editing YAML
uv run llm-bench --config configs/example.yaml --requests 50 --concurrency 4 --warmup-requests 2 --seed 42
```

### Constraint-based recommendation

```bash
llm-bench recommend results/q4km.csv results/q8.csv \
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

Excluded (1)
──────────────────────────────────────────
  llama-cpp  Llama-3.2-3B-Instruct-Q8_0.gguf  →  p95 latency too high (1227 ms > 1000 ms)
```

---

## Architecture

```
configs/*.yaml
      │
      ▼
load_config() → BenchmarkConfig (Pydantic v2)
      │
      ▼
_build_backend() → Backend (ABC)
                       ├── MockBackend            ← zero-dep, CI-safe
                       ├── HFBackend              ← transformers + PyTorch
                       ├── LlamaCppBackend        ← GGUF, n_gpu_layers
                       ├── OpenAIEndpointBackend  ← HTTP /v1/chat/completions
                       ├── OnnxBackend            ← optimum + onnxruntime
                       └── VllmBackend            ← vLLM high-throughput engine
      │
      ▼
run_repeated(backend, config, prompts)
      ├── warmup loop   (excluded from metrics)
      └── benchmark loop → [RequestMetrics]
                                  │
                                  ▼
                           compute_metrics()
                                  │
                                  ▼
                           MetricsReport → CSV / stdout
                                  │
                            ┌─────┴─────┐
                            ▼           ▼
                         pareto     recommend
                      (Pareto table) (best config)

llm-bench env          → EnvInfo (Python, CPU, GPU, installed backends)
```

---

## Benchmark Results

Full metric definitions: **[docs/metrics.md](docs/metrics.md)** · Curated hardware reports: **[docs/results/](docs/results/)**

### Llama 3.2 3B Instruct — Q4\_K\_M vs Q8\_0 · RTX 3050 (4 GB)

> llama.cpp backend, all 28 layers on GPU, 10 prompts per run.

| Quantization | p50 (ms) | p95 (ms) | tok/s | VRAM (MiB) | Sanity |
|---|---|---|---|---|---|
| **Q4\_K\_M** | **904** | **915** | **55.3** | 2361 (58%) | 100% |
| Q8\_0 | 1185 | 1187 | 42.2 | 3697 (90%) | 100% |

Q4\_K\_M is **1.31× faster** and uses **1.57× less VRAM** — memory bandwidth bound, not compute bound.

### n\_gpu\_layers Sweep — Llama 3.2 3B · RTX 3050

> VRAM scales ~60 MiB/layer after a 655 MiB CUDA-init baseline.

| Layers on GPU | p95 (ms) | tok/s | VRAM (MiB) |
|---|---|---|---|
| 0 / 28 (CPU only) | 3093 | 17.5 | 655 |
| 20 / 28 (partial) | 1420 | 36.6 | 1829 |
| **28 / 28 (full)** | **984** | **51.4** | **2361** |

Full offload is **3.1× faster** than CPU-only. Partial offload (20/28) captures 71% of the speedup at 78% of the VRAM cost.

### Harness Validation — mock backend

> The mock backend does `time.sleep(latency_ms / 1000)`. These numbers validate measurement
> plumbing, not inference speed.

| Metric | Value |
|---|---|
| p50 latency | ~5 ms |
| p95 latency | ~5 ms |
| tokens/sec | ~10,000 (simulated) |

---

## Features

- **YAML config** — backend, model, requests, warmup, prompts file or workload profile
- **CLI overrides** — `--requests`, `--warmup-requests`, `--concurrency`, `--seed` at run time
- **Environment introspection** — `llm-bench env` reports Python, CPU, GPU, driver, and installed backend versions; `--format json` for CI/scripting
- **Workload profiles** — `short_chat`, `summarization`, `code_completion`, `long_context_smoke`
- **Run matrix** — cartesian-product sweep from one YAML; one CSV + manifest per combination
- **Metrics** — p50/p95 latency, TTFT p50/p95 (streaming), tok/s, total tokens, decode throughput
- **Memory** — CPU RSS, PyTorch CUDA peak, driver-level VRAM via `nvidia-smi`
- **Lifecycle** — model load time, warmup latency
- **Variance** — `repeats: N` reports median ± std dev across N trial loops
- **Sanity checks** — empty, min/mean chars, repeated output, `sanity_pass_rate`
- **Task quality** — `quality_file:` YAML rubric (`contains_all`, `regex`, `forbidden`, …)
- **Self-perplexity** — teacher-forcing PPL on generated completions (`transformers` only)
- **LLM-as-judge** — P(Yes) from fixed yes/no relevance question (`transformers` only)
- **Pareto analysis** — classifies configs as optimal or dominated across all metrics
- **Constraint-based recommender** — `--max-vram-mb`, `--max-p95-ms`, `--max-ttft-ms`, `--min-sanity`, `--min-quality`, `--max-perplexity`, `--min-judge`, `--max-load-ms`
- **JSON run manifest** — git commit, config/prompts SHA256, Python/OS/CPU/GPU fingerprint
- **Docker image** — mock + transformers CPU published to ghcr.io on each release

---

## Docker

```bash
docker pull ghcr.io/happynood/llm-inference-benchmark:latest

# Mock backend — no model download
docker run --rm \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/results:/app/results" \
  ghcr.io/happynood/llm-inference-benchmark:latest \
  --config /app/configs/example.yaml --output /app/results/bench.csv

# Transformers CPU — reuse HuggingFace cache
docker run --rm \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/results:/app/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/happynood/llm-inference-benchmark:latest \
  --config /app/configs/transformers-cpu.yaml --output /app/results/bench-hf.csv
```

### Running with Docker + GPU (llama.cpp / CUDA)

`Dockerfile.cuda` builds `llama-cpp-python` with `GGML_CUDA=ON` so all GPU layers run on the
device. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host.

```bash
# Build once (no GPU needed for the build step)
docker build -f Dockerfile.cuda -t llm-bench-cuda .

# Verify — prints CLI help
docker run --rm llm-bench-cuda --help

# Run a llama.cpp benchmark with full GPU offload
# Place your GGUF model under ~/models/ on the host
docker run --rm --gpus all \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/results:/app/results" \
  -v "$HOME/models:/models:ro" \
  llm-bench-cuda \
  --config /app/configs/example.yaml \
  --backend llama-cpp \
  --model /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --n-gpu-layers 28 \
  --output /app/results/bench-cuda.csv
```

Or use the compose file:

```bash
docker compose run --rm llm-bench-cuda \
  --config /app/configs/example.yaml --output /app/results/bench-cuda.csv
```

> **Tip:** set `n_gpu_layers: 28` (or `-1` for all layers) in your YAML config instead of
> passing `--n-gpu-layers` each time.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Runtime | Python 3.11+, [uv](https://docs.astral.sh/uv/) |
| Config validation | Pydantic v2, PyYAML |
| CLI | Click |
| Transformers backend | HuggingFace `transformers` + PyTorch |
| OpenAI endpoint backend | stdlib `urllib` |
| Tests | pytest, pytest-cov |
| Lint / format | Ruff |
| Type checking | Pyright |
| CI | GitHub Actions |
| Container | Docker, GitHub Container Registry |

---

## Limitations

- Mock backend numbers validate harness plumbing only — never compare them to real backends.
- `sshleifer/tiny-gpt2` results validate the transformers backend; GPU is slower than CPU at 117 K params (kernel-launch overhead dominates). Production models reverse this.
- `peak_cuda_memory_mb = 0.0` for llama.cpp runs — use `peak_vram_memory_mb` from `nvidia-smi` instead.
- `measure_perplexity` and `measure_judge` are `transformers` backend only.
- OpenAI endpoint latency includes network round-trip and server-side queueing.
- Concurrent execution (`concurrency > 1`) uses `asyncio.to_thread`; CPU-bound backends may not scale linearly.
- Tested on Linux x86-64 (Ubuntu). macOS should work; Windows untested.

---

## Roadmap

**Stable (v1.0.0+)**
- [x] Mock, transformers, llama.cpp, OpenAI endpoint, ONNX Runtime backends
- [x] p50/p95 latency, TTFT, tok/s, VRAM, lifecycle, variance metrics
- [x] Sanity checks, task quality, perplexity, LLM-as-judge
- [x] Pareto analysis, constraint-based recommender
- [x] Run matrix, parameter sweeps, workload profiles
- [x] JSON manifest, CSV output, Markdown comparison table
- [x] Docker image (ghcr.io), GitHub Actions CI

**Planned**
- [x] `vllm` backend — high-throughput GPU serving
- [x] Real parameter sweep evidence: RTX 3050, n\_gpu\_layers sweep on Llama 3.2 3B

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for development setup, test instructions, and PR guidelines.

## Security

See **[SECURITY.md](SECURITY.md)** for the vulnerability disclosure policy.

## License

MIT — see [LICENSE](LICENSE).
