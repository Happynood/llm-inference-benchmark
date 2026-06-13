# Task Log

## 2026-06-13 — v0.1: Mock backend vertical slice

**Goal**: Portfolio-ready scaffold with architecture, CLI, tests, CI — no model weights required.

**Delivered**:
- `pyproject.toml` with uv, Pydantic v2, Click, Ruff, Pyright, pytest
- `BenchmarkConfig` (Pydantic) + YAML loader
- `Backend` ABC + `MockBackend` (deterministic, latency configurable)
- `run_benchmark` → `MetricsReport` with p50/p95, tokens/sec, total tokens
- `llm-bench` CLI (`--config`, `--output` CSV)
- 16 tests across config, metrics, runner, CLI
- GitHub Actions CI (lint + format + typecheck + pytest)
- README with honest mock-backend disclaimer and roadmap

**Quality checks passed**:
- `uv run pytest -v` — all tests pass
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — no errors

**Next iteration**:
- `transformers` backend (CPU inference, real latency numbers)
