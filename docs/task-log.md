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

## 2026-06-14 — v0.10: llama.cpp GGUF backend

**Goal**: Add `llama-cpp` backend for quantized inference on production-size models;
keep CI safe with mock-based unit tests (no GGUF download, no GPU required).

**Delivered**:
- `backends/llama_cpp.py`: `LlamaCppBackend` implementing `Backend` ABC via `llama-cpp-python`;
  lazy import guarded by `_AVAILABLE`; `n_gpu_layers` for partial GPU offload on 4 GB VRAM;
  `n_threads` omitted from kwargs when `None` (llama.cpp auto-detect); `echo=False` so
  completion text excludes the prompt; `time.perf_counter()` latency measurement consistent
  with HFBackend
- `config.py`: `LlamaCppBackendConfig` with `n_ctx`, `n_gpu_layers`, `max_tokens`,
  `temperature`, `n_threads`, `verbose`; `backend` Literal extended to include `"llama-cpp"`;
  `BenchmarkConfig.llama_cpp` sub-config field (default-factory, backward-compatible)
- `cli.py`: `_build_backend()` case for `"llama-cpp"` with lazy import (same pattern as HF)
- `pyproject.toml`: `[llama-cpp]` optional extra (`llama-cpp-python>=0.2`); added
  `backends/llama_cpp.py` to pyright ignore list (optional dep pattern)
- `configs/llama-cpp-cpu.yaml`: CPU config with placeholder model path and install instructions
- `configs/llama-cpp-gpu.yaml`: partial GPU offload config for RTX 3050 4 GB with VRAM
  budget notes and CUDA build instructions
- `Makefile`: `install-llama-cpp`, `install-llama-cpp-cuda`, `run-llama-cpp-cpu`,
  `run-llama-cpp-gpu` targets
- `tests/test_llama_cpp_backend.py`: 23 mock-based unit tests + 1 skipped integration
  test; covers ImportError path, constructor kwargs, `generate()` output, config validation,
  YAML loading, CLI dispatch, full `run_benchmark` integration — zero GGUF downloads
- `docs/metrics.md`: llama.cpp backend section with config schema, install steps, VRAM budget
  table for RTX 3050 4 GB, CUDA memory caveat
- `README.md`: demo block, feature bullet, How-to-Run instructions, Limitations section,
  Roadmap item checked off, architecture diagram updated

**Quality checks passed**:
- `uv run pytest -v` — 179/179 pass (23 new llama-cpp tests, 1 skipped integration)
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — 0 errors

**Next iteration**:
- Real run: Llama 3 8B Q4_K_M on RTX 3050 with llama-cpp GPU build; curated report in
  `docs/results/`; quantization comparison Q4_K_M vs Q8_0

## 2026-06-14 — v0.9: Security hardening + CI/real-evidence separation

**Goal**: Fix path-traversal risk in run names; make portfolio clearly distinguish mock
CI validation from real hardware benchmark evidence.

**Security fixes**:
- `matrix.py`: added `@field_validator("name")` enforcing `^[A-Za-z0-9][A-Za-z0-9._-]*$` —
  rejects path separators, leading dots, spaces, control chars, and empty strings
- `cli.py`: added `resolve().is_relative_to(resolved_dir)` containment check as defense-in-depth
  after computing `csv_path`/`manifest_path`
- `test_matrix.py`: 8 new parametrized tests for bad name rejection, 6 for valid name acceptance

**Issue #17 — CI/real-evidence separation**:
- `docs/results/README.md`: real-run registry with criteria for what qualifies, mock exclusion
  rationale, run table, and planned-runs table
- `docs/results/gpu-rtx3050-tiny-gpt2.md`: curated report for RTX 3050 + tiny-gpt2 runs
  (CPU vs GPU), with hardware table, software table, config YAML, results table,
  interpretation, and reproduction instructions
- `docs/metrics.md`: added explicit CI/Harness Validation vs Real Hardware Evidence sections
  with rule ("never compare mock to real in the same table"); updated CPU/GPU headings to
  match new structure; linked to curated reports
- `README.md`: mock demo labeled as CI/harness validation; Results section split into
  CI-validation table (with "what it validates" column) and Real Hardware table (linked to
  curated report); roadmap restructured into Harness Foundation (done), Next (active), and
  later phases; Limitations section split by backend

**Quality checks passed**:
- `uv run pytest -v` — 156/156 tests pass (14 new security tests)
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — 0 errors

**Next iteration**:
- `llama-cpp-python` backend (GGUF quantization, Llama 3 8B Q4_K_M on RTX 3050)

## 2026-06-14 — v0.8: Run matrix for multi-experiment configs

**Goal**: Run multiple benchmark configurations sequentially from one YAML file.

**Delivered**:
- `matrix.py`: `MatrixRunConfig(BaseModel)` with `name`, `config`, optional `workload_profile`
  (validated against the profile registry at parse time); `MatrixConfig(BaseModel)` with
  `results_dir` (default `"results"`) and `runs: list[MatrixRunConfig]` with unique-name
  validator; `load_matrix()` loader
- `cli.py`: `matrix` subcommand — `--config` (matrix YAML, required) and `--dry-run` (list
  without executing); runs each entry sequentially using existing `_build_backend` + `run_benchmark`
  flow; writes `{results_dir}/{name}.csv` and `{results_dir}/{name}.manifest.json` per run;
  prints `compare` hint on completion; existing `--config` / `compare` paths unchanged
- `configs/matrix-example.yaml` — four-profile mock matrix ready to run with `make run-matrix`
- `Makefile`: `run-matrix` target
- `.gitignore`: `results/` added
- `tests/test_matrix.py` — 24 tests: config parsing, validation (empty runs, duplicate names,
  bad profile), `load_matrix`, dry-run (no files created, lists all runs, shows profile), full
  execution (CSV + manifest per run, nested results dir creation, compare hint), backward compat
  (single-run CLI and compare subcommand still work)
- `docs/metrics.md`: Run Matrix section with YAML schema and output structure
- README: matrix demo block, feature bullet, `run-matrix` make target, roadmap item

**Quality checks passed**:
- `uv run pytest -v` — 142/142 tests pass
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — 0 errors

**Next iteration**:
- `llama-cpp-python` backend (GGUF quantization)

## 2026-06-14 — v0.7: Workload profiles + GPU benchmark results

**Goal**: Reproducible named prompt sets for cross-experiment comparisons; first GPU run.

**Delivered**:
- `profiles.py`: `WorkloadProfile` frozen dataclass; `_PROFILES` registry for `short_chat`,
  `summarization`, `code_completion`, `long_context_smoke`; `get_profile()` with helpful error
- `data/prompts/short_chat.txt`, `summarization.txt`, `code_completion.txt`, `long_context_smoke.txt`
  — committed prompt fixtures for each profile
- `config.py`: `workload_profile: str | None = None` field; `@model_validator` validates profile
  name at parse time; `resolve_prompts_file()` returns profile path when set, else `prompts_file`
  — existing `prompts_file` configs unchanged
- `cli.py`, `manifest.py`: updated to call `cfg.resolve_prompts_file()` instead of `cfg.prompts_file`
- `configs/profile-short-chat.yaml`, `configs/profile-summarization.yaml` — example configs
- `configs/transformers-gpu.yaml` — GPU config (`device: cuda`, `torch_dtype: float16`)
- `Makefile`: `run-gpu` target
- `tests/test_profiles.py` — 20 tests: profile registry, config validation, YAML loading,
  `resolve_prompts_file` logic, backward compat, CLI integration, manifest SHA256
- `docs/metrics.md`: GPU results section + workload profiles table
- README: GPU results side-by-side with CPU; profiles quick-start; roadmap updated

**GPU run** (Intel i5-11400H + NVIDIA RTX 3050 Laptop 4 GB, `sshleifer/tiny-gpt2`, float16):
- p50: 59.95 ms, p95: 61.86 ms, tok/s: 829.60, peak_cuda_memory_mb: 8.82
- GPU is slower than CPU for this 2-layer toy model — kernel-launch overhead dominates.

**Quality checks passed**:
- `uv run pytest -v` — 118/118 tests pass
- `uv run ruff check .` — no issues
- `uv run ruff format --check .` — no issues
- `uv run pyright` — 0 errors

**Next iteration**:
- `llama-cpp-python` backend (GGUF quantization, real production-size model)

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
