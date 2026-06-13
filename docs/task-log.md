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

## 2026-06-13 — v0.3: Peak memory metrics

**Goal**: Report peak CPU RSS and CUDA memory for every benchmark run.

**Delivered**:
- `memory.py`: `MemorySampler` (RSS polling via background thread + `threading.Lock`),
  `reset_cuda_peak()`, `cuda_peak_mb()` (lazy import, None when CUDA absent)
- `MetricsReport`: `peak_cpu_memory_mb: float`, `peak_cuda_memory_mb: float | None`
- `compute_metrics()`: backward-compatible default args; p50 now uses `_percentile_sorted`
  consistently with p95 (was `statistics.median`, numerically identical but inconsistent)
- `runner.py`: warmup excluded from measurement window; `reset_cuda_peak()` called inside
  `MemorySampler` context so CPU and CUDA windows are co-incident
- `cli.py`: `None → ""` in CSV; `N/A` in stdout for absent CUDA
- 6 new tests in `test_memory.py`; 3 new tests in `test_metrics.py`; 2 new tests in `test_runner.py`
- `psutil>=5.9` added as required dependency
- `docs/metrics.md`: Memory Measurement section with caveats and CUDA interpretation table
- README: stale "No peak memory measurement yet" removed; roadmap updated; real run numbers added

**Quality checks passed**:
- `uv run pytest -v` — 37/37 tests pass
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — no errors

**Real benchmark** (Intel i5-11400H, CPU, `sshleifer/tiny-gpt2`):
- peak_cpu_memory_mb: 721.37 (dominated by PyTorch runtime, not model weights)
- peak_cuda_memory_mb: 0.00 (CUDA toolkit present, inference ran on CPU device)

**Next iteration**:
- `llama-cpp-python` backend (GGUF quantization)

## 2026-06-13 — v0.4: Markdown comparison table

**Goal**: `llm-bench compare *.csv` turns saved benchmark CSVs into a GFM table.

**Delivered**:
- `compare.py`: `RunRow` dataclass, `load_csv` (validates required columns, handles empty
  CUDA field), `sort_rows` (by p95/backend/model), `render_table` (padded GFM),
  `build_comparison_table` (load + sort + render)
- `cli.py`: converted from `@click.command` to `@click.group(invoke_without_command=True)`;
  added `compare` subcommand with `--sort` (p95/backend/model) and `--output` options;
  `llm-bench --config ...` remains fully backward-compatible
- `tests/fixtures/mock_run.csv`, `tests/fixtures/transformers_run.csv` — committed fixtures
- `tests/test_compare.py` — 24 new tests covering load, sort, render, CLI subcommand,
  backward-compat, and error paths (incl. whitespace CUDA and empty-paths guards)
- README: comparison table demo updated with N column + single-run caveat;
  transformers demo numbers corrected to match fixture values
- `docs/metrics.md`: ~1193 → ~1211 tok/s in interpretation

**Quality checks passed**:
- `uv run pytest -v` — 61/61 tests pass
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — 0 errors

**Next iteration**:
- `llama-cpp-python` backend (GGUF quantization)

## 2026-06-14 — v0.5: Run manifest and environment fingerprint

**Goal**: Save a JSON manifest per benchmark run for full reproducibility.

**Delivered**:
- `manifest.py`: `RunManifest` frozen dataclass; `collect_manifest(config_path, cfg)`
  gathers git commit + dirty flag, SHA256 of config + prompts files, Python version,
  platform, CPU model/count, and dep versions (package, torch, transformers, psutil);
  `write_manifest(manifest, path)` writes pretty-printed JSON; all subprocess/import
  calls are guarded — manifest collection never crashes a benchmark run
- `cli.py`: `--manifest` option added to `main`; manifest is written after `run_benchmark`
  completes; CSV output and all existing options remain backward-compatible
- `tests/test_manifest.py` — 23 new tests covering field types, SHA256 format, ISO
  timestamp, git state, write output, JSON structure, CLI integration, and backward compat
- `docs/metrics.md`: "Reproducing a Run" section with field reference table
- README: manifest demo block + `--manifest` in How-to section; roadmap item checked off

**Quality checks passed**:
- `uv run pytest -v` — 84/84 tests pass
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — 0 errors

**Next iteration**:
- `llama-cpp-python` backend (GGUF quantization)

## 2026-06-14 — v0.6: Optional NVIDIA GPU fingerprint in run manifest

**Goal**: Extend `RunManifest` with an optional `gpu` section from `nvidia-smi` and `torch.cuda`.

**Delivered**:
- `manifest.py`: `GpuInfo` frozen dataclass (name, driver_version, cuda_version,
  vram_total_mb, torch_cuda_available, torch_cuda_device_name); `_collect_gpu_info()`
  tries `nvidia-smi --query-gpu` then `torch.cuda`; returns `None` when both are
  unavailable; all subprocess/import calls guarded
- `RunManifest`: new `gpu: GpuInfo | None` field; `dataclasses.asdict` serializes
  it as a nested dict or `null`
- `tests/test_gpu_fingerprint.py` — 14 tests; `subprocess.run` and `sys.modules["torch"]`
  mocked so all tests pass on CPU-only CI machines
- README and `docs/metrics.md` updated with GPU fields and optional-field caveats

**Quality checks passed**:
- `uv run pytest -v` — 98/98 tests pass
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — 0 errors

**Next iteration**:
- `llama-cpp-python` backend (GGUF quantization)
