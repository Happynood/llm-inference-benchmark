# Project Context

## Purpose

Reproducible harness for benchmarking LLM inference backends (latency, throughput, memory).
Production-grade engineering reference: CLI, config, tests, CI, and honest metrics.

## Current State (v0.1)

- Mock backend only — architecture, CLI, tests, and CSV output are complete
- No real model inference yet
- All quality checks pass: pytest, ruff, pyright, CI

## Architecture

```
configs/*.yaml
    │
    ▼
load_config()  →  BenchmarkConfig (Pydantic v2)
    │
    ▼
_build_backend()  →  Backend (ABC)
                         └─ MockBackend (v0.1)
    │
    ▼
run_benchmark(backend, config, prompts)
    │
    ├─ warmup loop  (excluded from metrics)
    └─ benchmark loop  →  [RequestMetrics]
                               │
                               ▼
                         compute_metrics()  →  MetricsReport
                               │
                               ▼
                         CSV / stdout
```

## File Layout

```
src/llm_inference_benchmark/
├── __init__.py
├── cli.py          ← Click entry point (llm-bench)
├── config.py       ← Pydantic config models + YAML loader
├── metrics.py      ← RequestMetrics, MetricsReport, compute_metrics
├── runner.py       ← load_prompts, run_benchmark
└── backends/
    ├── base.py     ← Backend ABC + GenerationResult
    └── mock.py     ← Deterministic mock backend
```

## Next Steps

1. Add `transformers` backend (HuggingFace, CPU/GPU)
2. Add `llama-cpp-python` backend (GGUF quantization)
3. Add concurrent execution (`concurrency > 1`)
4. Add peak memory measurement
5. Publish first real benchmark table in README + docs/metrics.md
