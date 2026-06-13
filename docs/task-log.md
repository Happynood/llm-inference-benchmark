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

## 2026-06-13 — v0.2: HuggingFace Transformers backend

**Goal**: First real inference backend using `AutoModelForCausalLM`; keep mock for CI.

**Delivered**:
- `HFBackend` (optional extra, excluded from pyright CI check)
- `HFBackendConfig` in Pydantic config (`max_new_tokens`, `device`, `torch_dtype`, `do_sample`)
- Lazy import in `_build_backend()` — ImportError only on actual use, not on `mock` config
- `configs/transformers-cpu.yaml` with `sshleifer/tiny-gpt2`
- 6 integration tests (`pytest -m integration`), auto-skipped without extras
- `conftest.py` + `hf.py` both clear `ALL_PROXY`/`all_proxy` (bare `socks://` rejected by httpx 0.28)
- `Makefile`: `install-hf`, `test-hf`, `run-hf`
- README and `docs/metrics.md` updated with real run numbers

**Quality checks passed**:
- `uv run pytest -v` — 26/26 tests pass (20 mock + 6 HF integration)
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — no errors

**Real benchmark** (Intel i5-11400H, CPU, `sshleifer/tiny-gpt2`):
- p50: 40.70 ms, p95: 46.19 ms, tokens/sec: 1192.87

**Next iteration**:
- `llama-cpp-python` backend (GGUF quantization, real production-size model)
