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

**Mock backend (no deps):**
```bash
uv run llm-bench --config configs/example.yaml --output benchmark.csv
```
```
Backend: mock  Model: mock-gpt2  Requests: 20

=== Benchmark Results ===
  request_count: 20
  p50_latency_ms: 5.01
  p95_latency_ms: 5.09
  tokens_per_second: 9971.18
  total_tokens: 1100
  backend: mock
  model: mock-gpt2
```

**Transformers backend (CPU, real inference):**
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
  "psutil_version": "6.1.1"
}
```

**Comparison table (across saved CSVs):**
```bash
llm-bench compare mock.csv transformers.csv --sort p95
```
```
| Backend      | Model               | N  | p50 (ms) | p95 (ms) | tok/s  | CPU mem (MB) | CUDA mem (MB) |
|--------------|---------------------|----|----------|----------|--------|--------------|---------------|
| mock         | mock-gpt2           | 20 | 5.01     | 5.09     | 9971.2 | 45.2         | N/A           |
| transformers | sshleifer/tiny-gpt2 | 10 | 40.95    | 44.67    | 1211.2 | 721.4        | 0.0           |
```

> **Note**: Each row is one unrepeated benchmark run. p95 at N=10 requests is the single
> worst-latency observation, not a stable statistical estimate.

> See [docs/metrics.md](docs/metrics.md) for benchmark results and hardware context.

## Features

- **YAML-driven config** — backend, model, request count, warmup, prompts file
- **p50/p95 latency, tokens/sec, total tokens** per run
- **Peak memory reporting** — CPU RSS via `psutil`, CUDA peak via `torch.cuda` when available
- **CSV output** + **Markdown comparison table** across multiple runs (`llm-bench compare`)
- **JSON run manifest** — git commit, config/prompts SHA256, Python/OS/CPU, dep versions (`--manifest`)
- **Optimization-oriented roadmap** — run manifests, workload profiles, quality checks, Pareto
  analysis, and constraint-based recommendations
- **Pluggable backends** — add a new backend by subclassing one abstract class
- **Mock backend** — deterministic, zero-dependency, CI-friendly
- **Transformers backend** — real CPU inference via `AutoModelForCausalLM` (optional extra)
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
                         ├── LlamaCppBackend ← GGUF quantization    roadmap
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

Full metric definitions and hardware context: [docs/metrics.md](docs/metrics.md).

### Mock backend (harness validation)

| Metric | Value |
|--------|-------|
| p50 latency | ~5 ms |
| p95 latency | ~5 ms |
| Output tokens/sec | ~10,000 (simulated) |
| Backend | mock |

### Transformers backend — `sshleifer/tiny-gpt2`, CPU

> **Note**: `sshleifer/tiny-gpt2` is a 2-layer toy model (~4 MB, ~117 K params) used to
> validate the harness produces real inference latency. It is not representative of
> production-size models (GPT-2 medium 345 M, Llama 3 8B, etc.).
> See [docs/metrics.md](docs/metrics.md) for the full run log and hardware details.

| Metric | Value |
|--------|-------|
| p50 latency | 40.95 ms |
| p95 latency | 44.67 ms |
| Output tokens/sec | 1211.23 |
| Peak CPU memory | 721 MB (total process RSS) |
| Peak CUDA memory | 0 MB (CUDA present, inference on CPU) |
| Backend | transformers |
| Model | sshleifer/tiny-gpt2 |
| Hardware | Intel i5-11400H, CPU only |
| max_new_tokens | 50 |
| device | cpu |

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

- Transformers backend tested only with GPT-2 architecture models on CPU.
  Real-world latency for production models (Llama 3, Mistral 7B) is 100–1000× higher.
- Concurrency > 1 is config-exposed but not yet implemented (sequential execution only).
- `peak_cpu_memory_mb` is total process RSS (interpreter + PyTorch runtime + model weights +
  activations). For tiny models, PyTorch runtime overhead dominates (~700 MB for tiny-gpt2).
- `sshleifer/tiny-gpt2` results are harness validation, not production benchmarks.
- Tested on Linux x86-64; macOS should work, Windows untested.
- `prompts_file` resolves relative to the working directory — run from the project root.

## Roadmap

- [x] Mock backend (v0.1)
- [x] `transformers` backend — CPU inference (v0.2)
- [ ] `llama-cpp-python` backend (GGUF quantization)
- [ ] `onnxruntime` backend (ONNX export + quantization)
- [ ] `vllm` backend (high-throughput GPU serving)
- [x] Peak memory reporting — CPU RSS (`psutil`) + CUDA peak (`torch.cuda`) (v0.3)
- [x] Markdown comparison table across runs (`llm-bench compare`) (v0.4)
- [ ] Async concurrent request execution
- [ ] Benchmark comparison table across backends in README
- [x] Run manifest and environment fingerprint per benchmark (`--manifest`) (v0.5)
- [ ] Workload profiles for short chat, summarization, code completion, and longer-context smoke tests
- [ ] Lightweight output sanity / quality-retention checks
- [ ] Pareto analysis and constraint-based recommendation report
