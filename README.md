# LLM Inference Benchmark

A reproducible harness for comparing LLM inference backends — latency, throughput, and memory — across backends and quantization modes.

## Problem

Choosing between `transformers`, `llama.cpp`, `onnxruntime`, and `vllm` for production requires apples-to-apples benchmarks under identical prompts and hardware.
Ad-hoc timing scripts per experiment produce inconsistent results that can't be compared or reproduced later.
This project wraps the benchmark loop in a typed, config-driven CLI so every run is reproducible and comparable.

## Demo

```bash
uv run llm-bench --config configs/example.yaml --output benchmark.csv
```

```
Backend: mock  Model: mock-gpt2  Requests: 20
Results written to benchmark.csv

=== Benchmark Results ===
  request_count: 20
  p50_latency_ms: 5.01
  p95_latency_ms: 5.09
  tokens_per_second: 9971.18
  total_tokens: 1100
  backend: mock
  model: mock-gpt2
  timestamp: 2026-06-13T10:30:00+00:00
```

> **v0.1 uses a deterministic mock backend.** No model weights or GPU required.
> Real backend results (`transformers`, `llama-cpp`, `onnxruntime`) are in the roadmap.

## Features

- **YAML-driven config** — backend, model, request count, warmup, prompts file
- **p50/p95 latency, tokens/sec, total tokens** per run
- **CSV output** for downstream comparison tables
- **Pluggable backends** — add a backend by implementing one abstract class
- **Mock backend** — deterministic, zero-dependency, CI-friendly
- **Type-checked and tested** — Pyright (basic) + Ruff + pytest, 80%+ coverage
- **GitHub Actions CI** — runs on every push and PR

## Tech Stack

| Layer | Tool |
|-------|------|
| Runtime | Python 3.11+, [uv](https://docs.astral.sh/uv/) |
| Config | Pydantic v2, PyYAML |
| CLI | Click |
| Tests | pytest, pytest-cov |
| Lint/format | Ruff |
| Type checking | Pyright |
| CI | GitHub Actions |

## Architecture

```
configs/example.yaml
        │
        ▼
  load_config() → BenchmarkConfig (Pydantic v2)
        │
        ▼
  _build_backend() → Backend (ABC)
                         ├── MockBackend         ← v0.1 ✓
                         ├── HFBackend           ← roadmap
                         ├── LlamaCppBackend     ← roadmap
                         └── ONNXBackend         ← roadmap
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

> v0.1 results use a mock backend only. Numbers validate the harness, not real inference.

| Metric | Mock backend |
|--------|-------------|
| p50 latency | ~5 ms |
| p95 latency | ~5 ms |
| Output tokens/sec | ~10,000 (simulated) |
| Backend | mock |

Real backend tables will be added as backends ship. See [docs/metrics.md](docs/metrics.md).

## How to Run

**Prerequisites**: Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/yourusername/llm-inference-benchmark
cd llm-inference-benchmark
uv sync
uv run llm-bench --config configs/example.yaml --output benchmark.csv
```

Run all checks:

```bash
make install    # uv sync
make test       # pytest -v
make lint       # ruff check .
make format     # ruff format .
make typecheck  # pyright
```

## Limitations

- **v0.1 is mock-only** — no real model inference yet
- Concurrency is config-exposed but not yet implemented (sequential execution)
- No peak memory measurement yet
- Tested on Linux (Ubuntu 22.04); macOS should work, Windows untested
- `prompts_file` path resolves relative to the working directory — run from the project root

## Roadmap

- [ ] `transformers` backend (HuggingFace hub, CPU/GPU, auto device selection)
- [ ] `llama-cpp-python` backend (GGUF quantization via llama.cpp)
- [ ] `onnxruntime` backend (ONNX export + quantization)
- [ ] `vllm` backend (high-throughput GPU serving)
- [ ] Async concurrent request execution
- [ ] Peak memory profiling (`tracemalloc` / `psutil`)
- [ ] Multi-run comparison table auto-generated in README
- [ ] Gradio demo for interactive backend comparison
