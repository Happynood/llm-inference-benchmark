# LLM Inference Benchmark

A reproducible harness for LLM inference optimization experiments: compare backends,
quantization modes, runtime parameters, latency, throughput, memory, and eventually quality under
identical workloads.

Project direction is fixed in [docs/project-charter.md](docs/project-charter.md): start with a
small benchmark harness, then grow toward configuration comparison, Pareto analysis, and
constraint-based recommendations.

## Problem

Choosing between `transformers`, `llama.cpp`, `onnxruntime`, `vllm`, precision modes, and runtime
parameters requires apples-to-apples experiments under identical prompts and hardware.
Ad-hoc timing scripts per experiment produce inconsistent results that can't be compared or reproduced later.
This project wraps the benchmark loop in a typed, config-driven CLI so every run is reproducible,
comparable, and suitable for later recommendation logic.

## Demo

**CI/harness validation (mock backend, no model downloads):**
```bash
uv run llm-bench --config configs/example.yaml --output benchmark.csv
```
> The mock backend sleeps for a configured `latency_ms` — no model is loaded.
> These numbers validate that the measurement pipeline is wired correctly, not that any model is fast.
```
Backend: mock  Model: mock-gpt2  Requests: 20
p50_latency_ms: 5.01  p95_latency_ms: 5.09  tokens_per_second: 9971.18  (simulated)
```

**Real inference (transformers backend, CPU):**
```bash
make install-hf   # uv sync --extra transformers
uv run llm-bench --config configs/transformers-cpu.yaml --output benchmark-hf.csv
```
```
Backend: transformers  Model: sshleifer/tiny-gpt2  Requests: 10

=== Benchmark Results ===
  request_count: 10
  p50_latency_ms: 40.95
  p95_latency_ms: 44.67
  tokens_per_second: 1211.23
  total_tokens: 598
  backend: transformers
  model: sshleifer/tiny-gpt2
```

**Run manifest (environment fingerprint):**
```bash
uv run llm-bench --config configs/example.yaml --output results.csv --manifest results/manifest.json
```
```json
{
  "timestamp": "2026-06-14T10:00:00+00:00",
  "backend": "mock",
  "model": "mock-gpt2",
  "git_commit": "a43a16f...",
  "git_dirty": false,
  "config_sha256": "e3b0c442...",
  "prompts_sha256": "d4a3c1f...",
  "python_version": "3.12.13",
  "platform_info": "Linux-7.0.0-x86_64",
  "cpu_model": "Intel(R) Core(TM) i5-11400H @ 2.70GHz",
  "cpu_count": 12,
  "package_version": "0.1.0",
  "torch_version": "2.12.0",
  "transformers_version": "5.12.0",
  "psutil_version": "6.1.1",
  "gpu": {
    "name": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "driver_version": "535.183.01",
    "cuda_version": "12.2",
    "vram_total_mb": 4096,
    "torch_cuda_available": true,
    "torch_cuda_device_name": "NVIDIA GeForce RTX 3050 Laptop GPU"
  }
}
```

**Workload profiles:**
```bash
uv run llm-bench --config configs/profile-short-chat.yaml --output results/short-chat.csv
uv run llm-bench --config configs/profile-summarization.yaml --output results/summarization.csv
```

**llama.cpp backend (GGUF quantized inference, local model):**
```bash
make install-llama-cpp           # uv sync --extra llama-cpp
# Edit configs/llama-cpp-cpu.yaml to set model: /path/to/model.gguf
uv run llm-bench --config configs/llama-cpp-cpu.yaml --output results/llama-cpp.csv
# GPU (CUDA build required):
make install-llama-cpp-cuda     # CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp
uv run llm-bench --config configs/llama-cpp-gpu.yaml --output results/llama-cpp-gpu.csv
```

**Run matrix (multiple configs in one command):**
```bash
uv run llm-bench matrix --config configs/matrix-example.yaml
# Preview without running:
uv run llm-bench matrix --config configs/matrix-example.yaml --dry-run
# Compare all results:
uv run llm-bench compare results/*.csv --sort p95
```

**Comparison table (across saved CSVs):**
```bash
llm-bench compare mock.csv transformers.csv --sort p95
```
```
| Backend      | Model               | N  | p50 (ms) | p95 (ms) | tok/s  | CPU mem (MB) | CUDA mem (MB) | VRAM mem (MB) | Sanity % |
|--------------|---------------------|----|----------|----------|--------|--------------|---------------|---------------|----------|
| mock         | mock-gpt2           | 20 | 5.01     | 5.09     | 9971.2 | 45.2         | N/A           | N/A           | 100.0%   |
| transformers | sshleifer/tiny-gpt2 | 10 | 40.95    | 44.67    | 1211.2 | 721.4        | 0.0           | N/A           | 100.0%   |
```

> `CUDA mem` is PyTorch allocator memory (zero for CPU runs, absent when torch unavailable).
> `VRAM mem` is driver-level VRAM from `nvidia-smi` — captures llama.cpp GPU usage that
> `peak_cuda_memory_mb` misses. `N/A` when `nvidia-smi` is not available.

> **Note**: Each row is one unrepeated benchmark run. p95 at N=10 requests is the single
> worst-latency observation, not a stable statistical estimate.

> See [docs/metrics.md](docs/metrics.md) for benchmark results and hardware context.

## Features

- **YAML-driven config** — backend, model, request count, warmup, prompts file or workload profile
- **Workload profiles** — named prompt sets (`short_chat`, `summarization`, `code_completion`, `long_context_smoke`) for reproducible cross-experiment comparisons
- **Run matrix** — define multiple experiment runs in one YAML; `llm-bench matrix` executes all sequentially with one CSV + manifest per run
- **p50/p95 latency, tokens/sec, total tokens** per run
- **Peak memory reporting** — CPU RSS via `psutil`; PyTorch allocator CUDA peak via `torch.cuda`; driver-level VRAM via `nvidia-smi` (`peak_vram_memory_mb`) for non-PyTorch GPU backends such as llama.cpp
- **Output sanity checks** — `empty_output_count`, `min/mean_output_chars`, `repeated_output_count`, `sanity_pass_rate` computed per run; shown as `Sanity %` in `llm-bench compare`
- **CSV output** + **Markdown comparison table** across multiple runs (`llm-bench compare`)
- **JSON run manifest** — git commit, config/prompts SHA256, Python/OS/CPU, dep versions, optional GPU fingerprint (`--manifest`)
- **Optimization-oriented roadmap** — run manifests, workload profiles, quality checks, Pareto
  analysis, and constraint-based recommendations
- **Pluggable backends** — add a new backend by subclassing one abstract class
- **Mock backend** — deterministic, zero-dependency, CI-friendly
- **Transformers backend** — real CPU/GPU inference via `AutoModelForCausalLM` (optional extra)
- **llama.cpp backend** — GGUF quantized inference via `llama-cpp-python` (optional extra); GPU via CUDA build
- **Type-checked and tested** — Pyright (basic) + Ruff + pytest
- **GitHub Actions CI** — lint + type-check + mock tests on every push

## Tech Stack

| Layer | Tool |
|-------|------|
| Runtime | Python 3.11+, [uv](https://docs.astral.sh/uv/) |
| Config | Pydantic v2, PyYAML |
| CLI | Click |
| Transformers backend | HuggingFace `transformers` + PyTorch (optional) |
| Tests | pytest, pytest-cov |
| Lint/format | Ruff |
| Type checking | Pyright |
| CI | GitHub Actions |

## Architecture

```
configs/*.yaml
        │
        ▼
  load_config() → BenchmarkConfig (Pydantic v2)
        │
        ▼
  _build_backend() → Backend (ABC)
                         ├── MockBackend     ← zero-dep, CI-safe    ✓ v0.1
                         ├── HFBackend       ← transformers extra   ✓ v0.2
                         ├── LlamaCppBackend ← GGUF quantization    ✓ v0.10
                         └── ONNXBackend     ← ONNX export          roadmap
        │
        ▼
  run_benchmark(backend, config, prompts)
        ├── warmup loop   (excluded from metrics)
        └── benchmark loop → [RequestMetrics]
                                    │
                                    ▼
                             compute_metrics()
                                    │
                                    ▼
                             MetricsReport → CSV / stdout
```

## Results

Metric definitions and computation notes: [docs/metrics.md](docs/metrics.md).  
Real-run curated reports: [docs/results/](docs/results/).

### CI / Harness Validation — mock backend

> These numbers validate that the harness measures correctly. They are not inference benchmarks.
> The mock backend does `time.sleep(latency_ms / 1000)` — no model is loaded.

| Metric | Value | What it validates |
|--------|-------|-------------------|
| p50 latency | ~5 ms | configured latency is measured |
| p95 latency | ~5 ms | p95 ≈ p50 for deterministic mock |
| tokens/sec | ~10,000 (simulated) | tokens/sec formula is correct |

### Real Hardware — `sshleifer/tiny-gpt2`, i5-11400H + RTX 3050 Laptop

> `sshleifer/tiny-gpt2` is a 2-layer toy model (~117 K params, ~4 MB). It is used to
> validate that the harness measures **real** inference (actual model weights, tokenizer,
> CPU/CUDA kernels). Numbers are not representative of production-size models.
> Full report: [docs/results/gpu-rtx3050-tiny-gpt2.md](docs/results/gpu-rtx3050-tiny-gpt2.md)

| Metric | CPU (float32) | GPU — RTX 3050 (float16) |
|--------|--------------|--------------------------|
| p50 latency | 40.95 ms | 59.95 ms |
| p95 latency | 44.67 ms | 61.86 ms |
| tokens/sec | 1211.23 | 829.60 |
| Peak CPU memory | 721 MB | 1383 MB |
| Peak CUDA memory | 0 MB | **8.82 MB** |

> GPU is slower than CPU here — expected for a 2-layer toy model where kernel-launch overhead
> exceeds compute savings. Production models (Llama 3 8B Q4) reverse this decisively.

### Real Hardware — Llama 3.2 3B Instruct Q4\_K\_M, i5-11400H + RTX 3050 (llama.cpp)

> 3B parameter production LLM, Q4\_K\_M quantization, 1.9 GB GGUF, all 28 layers on CUDA0.
> Full report: [docs/results/llama-cpp-rtx3050-llama32-3b.md](docs/results/llama-cpp-rtx3050-llama32-3b.md)

| Metric | CPU (`n_gpu_layers=0`) | GPU — RTX 3050 (`n_gpu_layers=99`) |
|--------|----------------------|-------------------------------------|
| p50 latency | 2750.56 ms | 931.18 ms |
| p95 latency | 2939.79 ms | 939.69 ms |
| tokens/sec | 18.01 | **53.71** |
| Peak VRAM | — | 2361 MiB |

> GPU is **2.95× faster** — all 28 model layers offloaded to CUDA0. 53.7 tok/s is
> real-time capable for interactive inference.

### Real Hardware — n\_gpu\_layers Sweep (0 / 20 / 99), RTX 3050 (llama.cpp)

> Three-point sweep via `llm-bench matrix`. Quantifies the latency and VRAM trade-off
> from CPU-only to full GPU offload.
> Full report: [docs/results/llama-cpp-rtx3050-vram-sweep.md](docs/results/llama-cpp-rtx3050-vram-sweep.md)

| `n_gpu_layers` | Layers on GPU | p50 (ms) | p95 (ms) | tok/s | Peak VRAM (MiB) |
|----------------|---------------|----------|----------|-------|-----------------|
| 0 | 0 / 28 (CPU only) | 2836.98 | 3092.79 | 17.52 | 655 |
| 20 | 20 / 28 (partial) | 1357.39 | 1420.41 | 36.59 | 1829 |
| 99 | 28 / 28 (full) | **970.65** | **984.31** | **51.44** | **2361** |

> VRAM scales **~60 MiB/layer** after a 655 MiB CUDA-init baseline (present even at
> `n_gpu_layers=0`). Partial offload (20/28 layers) delivers 71% of the full-offload
> speedup at 78% of the VRAM cost. Full offload uses 2361/4096 MiB (57.6%).
> `peak_cuda_memory_mb = 0.0` for all llama.cpp runs — use `peak_vram_memory_mb` instead.

### Real Hardware — Q4\_K\_M vs Q8\_0 Quantization Comparison, RTX 3050 (llama.cpp)

> Same model, same prompts, same GPU offload — different quantization.
> Full report: [docs/results/llama-cpp-rtx3050-quant-compare.md](docs/results/llama-cpp-rtx3050-quant-compare.md)

| Quantization | n\_gpu\_layers | p50 (ms) | p95 (ms) | tok/s | Peak VRAM (MiB) | Sanity % |
|---|---|---|---|---|---|---|
| Q4\_K\_M | 99 (28/28) | **904.33** | **915.22** | **55.28** | 2361 (57.6%) | 100% |
| Q8\_0 | 99 (28/28) | 1185.23 | 1186.75 | 42.21 | 3697 (90.2%) | 100% |

> Q4\_K\_M is **1.31× faster** and uses **1.57× less VRAM** than Q8\_0. The speedup comes from
> memory bandwidth: fewer bits per weight = fewer bytes read from VRAM per forward pass.
> Q8\_0 fits in 4 GB but leaves only 399 MiB headroom — practical for benchmarking,
> tight for interactive use with longer contexts. Sanity % = fraction of non-empty completions
> (100% = all outputs contained text; repeated outputs due to deterministic cycling are expected).

## How to Run

**Prerequisites**: Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/happynood/llm-inference-benchmark
cd llm-inference-benchmark
```

**Mock backend (no downloads):**
```bash
make install    # uv sync
make run        # llm-bench --config configs/example.yaml
```

**Transformers backend (downloads ~4 MB model on first run):**
```bash
make install-hf  # uv sync --extra transformers
make run-hf      # llm-bench --config configs/transformers-cpu.yaml
```

**GPU benchmark — transformers backend (requires CUDA):**
```bash
make install-hf
make run-gpu   # llm-bench --config configs/transformers-gpu.yaml ...
```

**llama.cpp backend — CPU:**
```bash
make install-llama-cpp         # uv sync --extra llama-cpp
# set model: /path/to/model.gguf in configs/llama-cpp-cpu.yaml
make run-llama-cpp-cpu
```

**llama.cpp backend — GPU (CUDA build):**
```bash
make install-llama-cpp-cuda   # CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp
# set model: /path/to/model.gguf in configs/llama-cpp-gpu.yaml, tune n_gpu_layers
make run-llama-cpp-gpu
```

**Run matrix (all four profiles, one command):**
```bash
make run-matrix  # llm-bench matrix --config configs/matrix-example.yaml
```

**Quantization comparison — Q4\_K\_M vs Q8\_0 (llama.cpp, CUDA):**
```bash
# 1. Download both quantizations (Llama 3.2 3B Instruct example):
#    huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
#      Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models/
#    huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
#      Llama-3.2-3B-Instruct-Q8_0.gguf   --local-dir ~/models/

# 2. Set model: paths in configs/llama-cpp-q4km-best.yaml and configs/llama-cpp-q8-best.yaml

# 3. Run the comparison matrix (CUDA 12/13 workaround for pre-built cu124 wheel):
CUDA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name "*.so*" \
  | xargs -I{} dirname {} | sort -u | tr '\n' ':')
LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" \
  uv run llm-bench matrix --config configs/llama-cpp-quant-compare.yaml

# 4. Compare results:
uv run llm-bench compare results/quant-q4km.csv results/quant-q8.csv
```

**Save a run manifest (environment fingerprint):**
```bash
llm-bench --config configs/example.yaml --output results.csv --manifest manifest.json
```

**Compare multiple runs into a Markdown table:**
```bash
llm-bench compare results_a.csv results_b.csv --sort p95
llm-bench compare results_a.csv results_b.csv --sort backend --output table.md
```

**Run all checks:**
```bash
make test       # pytest -v (mock tests only, CI-safe)
make test-hf    # pytest -m integration (requires install-hf)
make lint       # ruff check .
make format     # ruff format .
make typecheck  # pyright
```

## Limitations

**Mock backend**
- Produces no real inference — latency is `time.sleep(latency_ms / 1000)`. Numbers validate
  the harness pipeline only; never compare mock results to real backend results.

**Real backends (current)**
- `sshleifer/tiny-gpt2` results are harness validation for the `transformers` backend, not
  production benchmarks. Llama 3 8B latency is 100–1000× higher.
- `peak_cpu_memory_mb` is total process RSS including PyTorch runtime (~700 MB base overhead).
  For tiny models the runtime dominates; for production models the weights dominate.
- GPU is slower than CPU for tiny-gpt2 — kernel-launch overhead exceeds compute savings at
  117 K params. Production models reverse this.
- Concurrency > 1 is config-exposed but sequential execution only.
- Tested on Linux x86-64 (Ubuntu-family). macOS should work; Windows untested.
- `prompts_file` and `config` paths resolve relative to the working directory. Run from the
  project root.

**llama.cpp backend**
- Requires a local GGUF model file — no automatic download. Obtain models from Hugging Face Hub
  (e.g. `huggingface_hub.hf_hub_download`).
- GPU support: pre-built cu124 wheels are available at `https://abetlen.github.io/llama-cpp-python/whl/cu124`.
  On systems without nvcc (no CUDA toolkit), install `nvidia-cublas-cu12` + `nvidia-cuda-runtime-cu12`
  alongside the wheel and set `LD_LIBRARY_PATH` to cover all `nvidia/*/lib/` dirs in the venv
  (see [docs/results/llama-cpp-rtx3050-llama32-3b.md](docs/results/llama-cpp-rtx3050-llama32-3b.md)).
- `n_gpu_layers` must be tuned to your VRAM budget. For Llama 3.2 3B Q4\_K\_M, `n_gpu_layers: 99`
  offloads all 28 layers (2361 MiB VRAM on RTX 3050). For larger models, increase gradually.
- `peak_cuda_memory_mb` will be `0.0` even on GPU runs — llama-cpp uses its own VRAM allocator,
  not PyTorch's. Use `peak_vram_memory_mb` instead: it captures driver-level VRAM via `nvidia-smi`
  automatically during the benchmark loop (blank when `nvidia-smi` is not on `PATH`).

## Roadmap

**Harness foundation (complete)**
- [x] Mock backend — CI/harness validation, zero deps (v0.1)
- [x] `transformers` backend — real CPU inference (v0.2)
- [x] Peak memory reporting — CPU RSS + CUDA peak (v0.3)
- [x] Markdown comparison table (`llm-bench compare`) (v0.4)
- [x] Run manifest and environment fingerprint (`--manifest`) (v0.5)
- [x] Optional NVIDIA GPU fingerprint in manifest (v0.6)
- [x] Workload profiles (`short_chat`, `summarization`, `code_completion`, `long_context_smoke`) (v0.7)
- [x] GPU baseline — RTX 3050 Laptop, tiny-gpt2 (v0.7)
- [x] Run matrix — multi-experiment YAML (`llm-bench matrix`) (v0.8)
- [x] CI/real-evidence separation — docs/results/ registry (v0.9)

**Next: production-size models on 4 GB VRAM (active)**
- [x] `llama-cpp-python` backend — GGUF quantization, `n_gpu_layers` for partial GPU offload (v0.10)
- [x] First real run: Llama 3.2 3B Instruct Q4\_K\_M on RTX 3050 — curated results report (v0.11)
- [x] n\_gpu\_layers sweep (0 / 20 / 99) via `llm-bench matrix` — VRAM scaling and latency documented
- [x] Quantization comparison: Q4\_K\_M vs Q8\_0 — 1.28× speed difference, 1.57× VRAM difference documented

**Optimization analysis (planned)**
- [ ] Lightweight output quality checks (perplexity or judge scoring) so speed isn't reported without correctness
- [ ] Parameter sweeps: batch size, concurrency, `max_new_tokens`, context length
- [ ] Pareto table: latency vs memory vs quality across configurations
- [ ] Constraint-based recommender: "best config under 4 GB VRAM, p95 < 2 s"

**Additional backends (later)**
- [ ] `onnxruntime` (ONNX export + quantization)
- [ ] `vllm` (high-throughput GPU serving)
- [ ] OpenAI-compatible endpoint (local or remote)
- [ ] Pareto analysis and constraint-based recommendation report
