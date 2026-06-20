# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versions before
1.0.0 may include breaking changes between minor releases.

## [Unreleased]

### Added
- `llm-bench pipeline --config <file>`: run a full benchmark study from a single YAML config.
  Executes all matrix cells in sequence, then writes compare, Pareto, and recommendation
  outputs to `results_dir/`. Supports `--dry-run` to preview the plan without executing,
  `--continue-on-error` to keep running after a cell failure, and `--format table|json`
  for terminal progress. The `pipeline:` block is optional — a plain matrix YAML is valid
  pipeline input. Example:
  ```
  llm-bench pipeline --config configs/pipeline-example.yaml
  llm-bench pipeline --config configs/pipeline-example.yaml --dry-run
  llm-bench pipeline --config configs/pipeline-example.yaml --continue-on-error
  ```
  Output files written to `results_dir/`: `compare.md`, `compare.json`; `pareto.md`,
  `pareto.json` (when `pipeline.pareto: true`); `recommend.md`, `recommend.json` (when
  `pipeline.recommend:` block is present). Exits 1 when any cell failed or when a
  `recommend` block finds no winner. Calls library functions directly — no subprocess
  calls to other `llm-bench` subcommands. A `configs/pipeline-example.yaml` ships with
  the repo and uses the mock backend (no model download required).

- `llm-bench compare --filter FIELD=PATTERN`: filter comparison rows before sorting and
  `--limit`. Pattern is a case-insensitive substring match. Supported fields: `backend`,
  `model`. The flag is repeatable; multiple filters are ANDed:
  ```
  llm-bench compare results/*.csv --filter backend=llama_cpp
  llm-bench compare results/*.csv --filter model=Llama-3.2 --sort toks
  llm-bench compare results/*.csv --filter backend=llama_cpp --filter model=Q4_K_M --limit 3
  llm-bench compare results/*.csv --filter backend=llama_cpp --format json
  ```
  An unsupported field name produces a clear `UsageError` listing valid fields. When no rows
  survive filtering, an empty table or empty JSON array is returned (not an error). Composes
  naturally with `--sort`, `--limit`, and `--format`.


- `llm-bench compare --limit N`: cap the output table or JSON array to the top N rows after
  sorting. Composes naturally with `--sort` to get a focused view of the best runs:
  ```
  llm-bench compare results/*.csv --sort toks --limit 5
  llm-bench compare results/*.csv --sort p95 --limit 3 --format json
  ```
  Values below 1 are rejected with a usage error. When N exceeds the number of available
  rows all rows are returned unchanged.

- `--set KEY=VALUE` flag on the main `llm-bench` run command and on `llm-bench validate-config`:
  overrides any config field via dot-path without editing the YAML. Values are parsed as YAML
  scalars (so `200` becomes `int`, `3.14` becomes `float`, `true` becomes `bool`, `cuda`
  stays `str`). The flag is repeatable:
  ```
  llm-bench --config my.yaml --set llama_cpp.max_tokens=200 --set llama_cpp.n_gpu_layers=20
  llm-bench validate-config --config my.yaml --set hf.max_new_tokens=256
  ```
  Unknown paths and type mismatches produce a clear `UsageError` with the list of valid
  fields. Named flags (`--requests`, `--seed`, etc.) take precedence over `--set` when both
  target the same field. Reuses `apply_overrides` from the existing `sweep`/`matrix`
  subsystem.

## [1.3.0] — 2026-06-19

### Added
- `llm-bench matrix --format json`: machine-readable JSON output for the matrix command.
  `--dry-run --format json` emits a single JSON object listing all planned runs with their
  index, name, config, overrides, workload profile, and expected output/manifest paths —
  no files are created. `--format json` (actual run) routes all progress messages to stderr
  so stdout remains a single parseable JSON object; always collects outcomes from all runs
  (continues past failures); emits a summary with `total`/`succeeded`/`failed` counts and
  per-run `status`, `output`, and `error` fields; exits 1 when any run failed. Makes
  `matrix` consistent with every other subcommand that supports `--format json`.

### Changed
- `llama-cpp` backend now pre-loads CUDA shared libraries bundled by `nvidia-*` pip packages
  at Python import time using `ctypes.CDLL(..., mode=RTLD_GLOBAL)`. This makes GPU runs
  work on driver-only systems (no CUDA toolkit / nvcc) without requiring `LD_LIBRARY_PATH`
  to be set manually. The preload is a silent no-op on CPU-only machines and on systems
  without `nvidia-*` packages installed.

## [1.2.0] — 2026-06-18

### Added
- `llm-bench env`: prints the current Python, package, and hardware environment without
  requiring a benchmark config or a full run. Reports Python version, OS/platform, CPU model
  and core count, `llm-inference-benchmark` version, and optional package versions
  (`torch`, `transformers`, `optimum`, `vllm`, `psutil`) when installed. GPU section shows
  device name, driver version, CUDA version, and total VRAM from `nvidia-smi`; reports
  "not detected" on CPU-only machines. Supports `--format json` for machine-readable output
  and CI integration.
- `llm-bench --seed N` CLI flag: overrides the `seed:` field in the YAML config without
  editing the file. Follows the same pattern as `--requests`, `--warmup-requests`, and
  `--concurrency`. The seed value appears in the human-readable run header when set
  (`Seed: N`). Has no effect when the backend uses greedy decoding (`temperature: 0.0`).
- `llm-bench recommend --format json`: machine-readable JSON output from the recommend command.
  Returns an object with `winner` (all RunRow fields; `null` when no run satisfies constraints),
  `is_pareto_optimal` (bool), `candidates_count` (int), and `excluded` (array of objects with
  `backend`, `model`, and `reason`). Exit code behaviour (0 = winner found, 1 = no winner) is
  unchanged. Compatible with `--output`; useful for CI pipelines and scripting without
  screen-scraping the human-readable text output. The default format remains `table`.
- `seed` field in `BenchmarkConfig`: optional random seed for stochastic sampling.
  Applied per-request via `torch.manual_seed` in the `transformers` backend and at model
  load via `Llama(seed=...)` in the `llama-cpp` backend. Forwarded as a best-effort hint
  in the `openai` backend (server support varies). Recorded in `RunManifest` to complete
  the reproducibility fingerprint. Has no effect on greedy decoding (`temperature: 0.0`).
- `llm-bench compare --format json`: machine-readable JSON output from the compare command.
  Returns a JSON array of objects — one per CSV — with all fields present (`null` for absent
  optional metrics). Compatible with `--sort` and `--output`; useful for CI dashboards and
  scripting without screen-scraping Markdown. The default format remains `table` (Markdown).
- `vllm` backend (`backend: vllm`): high-throughput GPU inference via the vLLM engine
  (`vllm>=0.4`). Install with `uv sync --extra vllm`. Supports `max_new_tokens`,
  `temperature`, `tensor_parallel_size`, `gpu_memory_utilization`, and `dtype` under
  `vllm:` in the config.

### Changed
- `llm-bench compare` now suppresses optional columns (Out tok/s, In/Out tok, Load, TTFT,
  CUDA/VRAM mem, Sanity, Task Q, PPL, Judge) when every row in the table shows N/A for that
  column. Mandatory columns (Backend, Model, N, p50, p95, tok/s, CPU mem) are always shown.
  Mixed tables (some rows with data, some without) still show the column with N/A for the
  rows that lack data.
- `llm-bench diff --format json`: machine-readable JSON output from the diff command.
  Returns an object with `baseline`/`current` run metadata and a `metrics` list — one
  entry per tracked metric — each with `label`, `baseline`, `current`, `change_pct`
  (null when not computable), and `direction` (`"improvement"`, `"regression"`,
  `"neutral"`, or `"n/a"`). Optional metrics absent from both runs are omitted,
  consistent with Markdown table output. Compatible with `--output` and
  `--fail-on-regression`; useful for CI dashboards and scripting. The default format
  remains `table` (Markdown).
- `llm-bench diff baseline.csv current.csv`: per-metric percentage-change comparison between
  two benchmark runs. Shows p50/p95 latency, tok/s, TTFT, VRAM, load time, and quality
  metrics side-by-side with ✓ (improvement) / ✗ (regression) annotations. Optional metrics
  are suppressed when absent from both runs. Accepts `--output` to write Markdown to a file.
- `llm-bench diff --fail-on-regression PCT`: exit code 1 when any tracked metric degrades
  beyond the given percentage threshold. Use `0` to fail on any regression; use a positive
  value (e.g. `5`) to tolerate noise. The diff table is always printed before the exit code
  check, so CI logs retain full regression detail. Enables `llm-bench diff` as a CI gate
  without any external tooling.

## [1.1.0] — 2026-06-17

### Added
- `--max-ttft-ms` constraint for `llm-bench recommend`: exclude runs where TTFT p50 exceeds the
  threshold. Runs without a TTFT reading are excluded when the constraint is active (requires
  `stream: true` backend run). Consistent with the `--max-load-ms` pattern added in v0.23.
- TTFT (time-to-first-token) measurement for the `llama-cpp` backend: set `stream: true`
  under `llama_cpp:` in the config to enable streaming mode. `p50_ttft_ms` and `p95_ttft_ms`
  are populated in CSV output, matching the existing `transformers` and `openai` backend
  behaviour. Input tokens are counted via the model's tokenizer; output tokens are counted
  as non-empty streaming chunks (each chunk = one decoded token). Blocking mode (`stream:
  false`, the default) is unchanged.
- TTFT (time-to-first-token) measurement for the `transformers` backend: `p50_ttft_ms` and
  `p95_ttft_ms` are now populated for HuggingFace runs alongside the existing OpenAI-endpoint
  metrics. Implemented via a `LogitsProcessor` hook — no threading, token counts unchanged.
- Docker image (mock + transformers CPU) published to GitHub Container Registry on each
  release tag; `Dockerfile` uses a two-layer uv install for optimal cache reuse.
- `.github/workflows/release.yml` — creates a GitHub Release and pushes versioned
  Docker tags (`v1.1`, `v1.1.0`, `latest`) to `ghcr.io` on every `v*.*.*` tag push.
- `llm-bench --requests N`, `--warmup-requests N`, `--concurrency N` flags let callers
  override the corresponding YAML config values from the command line without editing the
  file. Useful for quick one-off experiments, especially with the concurrent-execution
  backend (`--concurrency 4`). CLI values take precedence over YAML; all existing
  behaviour when the flags are absent is unchanged.

## [1.0.0] — 2026-06-17

### Added
- `llm-bench --version` prints the installed package version and exits 0.

### Changed
- Version bumped to 1.0.0, signalling a stable public API. The CLI commands
  (`llm-bench`, `compare`, `pareto`, `recommend`, `matrix`), YAML config schema,
  CSV output format, and manifest format are now considered stable and will not
  change in an incompatible way without a major version bump.

## [0.23.0] — 2026-06-17

### Added
- `compare` table now shows a `Load (ms)` column (`model_load_ms` from v0.18 runs; `N/A`
  for pre-v0.18 CSVs). When a run used `repeats > 1`, the `p95 (ms)` and `tok/s` cells
  show the median with its standard deviation as `value ± std`.
- `pareto` table now shows a `Load (ms)` column and includes `model_load_ms` as a
  Pareto dimension (lower is better) when both compared runs have a value.
- `recommend` displays `Load` in the winning-configuration summary and accepts a new
  `--max-load-ms` constraint: runs where load time exceeds the threshold (or where load
  time is unknown) are excluded with a clear reason.

## [0.22.0] — 2026-06-17

### Added
- OpenAI-compatible endpoint backend (`backend: openai`): benchmarks any server that
  exposes the `/v1/chat/completions` HTTP API — llama.cpp server, Ollama, LM Studio,
  vLLM, and others. Uses `urllib` (no extra dependency). Token counts from the `usage`
  field when present; word-count fallback otherwise. API key read from a named
  environment variable (`api_key_env`); never written to config, logs, or output.
  Example config at `configs/openai-endpoint.yaml`.

**Limitation:** reported latency includes network round-trip and server-side queueing
overhead and is not directly comparable to in-process backend latency.

## [0.21.0] — 2026-06-16

### Added
- LLM-as-judge quality score: `measure_judge: true` reports the mean P(yes) from a fixed
  self-judge yes/no question asked about each of a backend's own generated completions.
  `transformers` backend only; `None` elsewhere. `Judge` column in `compare`/`pareto` tables;
  `--min-judge` constraint in `llm-bench recommend`.

## [0.20.0] — 2026-06-16

### Added
- Self-perplexity quality metric: `measure_perplexity: true` reports corpus-level perplexity
  of a backend's own generated completions via teacher forcing. `transformers` backend only;
  `None` elsewhere. `PPL` column in `compare`/`pareto` tables; `--max-perplexity` constraint
  in `llm-bench recommend`.

## [0.19.0] — 2026-06-16

### Added
- Repeated-trial variance reporting: `repeats: N` in a config runs the benchmark loop N
  times; reported p95 latency and tokens/sec become the median across repeats.
  `p95_latency_ms_std` and `tokens_per_second_std` CSV columns carry the sample standard
  deviation. Single-run CSVs (`repeats: 1`, the default) are unchanged.

## [0.18.0] — 2026-06-16

### Added
- Lifecycle metrics: model load time (`model_load_ms`) and warmup latency
  (`warmup_p50_latency_ms`), wired into CSV output and the CLI.

## [0.17.0] — 2026-06-15

### Added
- Parameter sweep matrix: `base_config:` + `sweep:` in a matrix YAML expands a cartesian
  product of overrides (including dot-path nested fields such as `llama_cpp.n_gpu_layers`
  or `hf.max_new_tokens`) into deterministic per-combination run names, with a `--dry-run`
  preview.

## [0.16.0] — 2026-06-15

### Added
- Task-quality evaluation: optional `quality_file:` in a config points to a YAML rubric
  spec (`contains_all`, `contains_any`, `forbidden`, `regex`, `min_chars` per prompt).
  `task_quality_pass_rate` / `task_quality_checked_count` in CSV output; `Task Q %` column
  in `compare`/`pareto`; `--min-quality` constraint in `llm-bench recommend`.

## [0.15.0] — 2026-06-14

### Added
- Constraint-based recommender: `llm-bench recommend` filters benchmark CSVs by
  `--max-vram-mb`, `--max-p95-ms`, and `--min-sanity`, returning the Pareto-optimal winner
  or exiting with code 1 and a clear exclusion table.

## [0.14.0] — 2026-06-14

### Added
- Pareto analysis: `llm-bench pareto` classifies configurations across saved CSVs as
  optimal or dominated on p95 latency, throughput, VRAM, and sanity, narrowing the
  comparison rather than crashing when optional metrics are missing.

## [0.13.0] — 2026-06-14

### Added
- Output sanity checks: `empty_output_count`, `min_output_chars`, `mean_output_chars`,
  `repeated_output_count`, and `sanity_pass_rate` computed per run; surfaced as `Sanity %`
  in `compare` tables.

## [0.12.0] — 2026-06-14

### Added
- Quantization comparison evidence: Q4_K_M vs Q8_0 for Llama 3.2 3B Instruct on an RTX 3050
  (llama.cpp) — curated report documenting a 1.31× speed and 1.57× VRAM difference.
- `n_gpu_layers` sweep (0 / 20 / 99) via `llm-bench matrix`, quantifying VRAM and latency
  scaling from CPU-only to full GPU offload.

## [0.11.0] — 2026-06-14

### Added
- First real production-size model benchmark: Llama 3.2 3B Instruct Q4_K_M on an RTX 3050
  Laptop GPU via the llama.cpp backend — curated report in `docs/results/`.

## [0.10.0] — 2026-06-14

### Added
- `llama-cpp` backend: GGUF quantized inference via `llama-cpp-python`, with
  `n_gpu_layers` for partial GPU offload on constrained VRAM budgets (optional extra).

## [0.9.0] — 2026-06-14

### Fixed
- Path-traversal hardening in run/matrix names (`matrix.py` validator, CLI containment check).

### Changed
- Documentation restructured to explicitly separate CI/harness validation (mock backend)
  from real hardware benchmark evidence (`docs/results/` registry).

## [0.8.0] — 2026-06-14

### Added
- Run matrix: `llm-bench matrix` executes multiple experiment configs from one YAML file
  sequentially, writing one CSV and manifest per run.

## [0.7.0] — 2026-06-14

### Added
- Workload profiles: named, reproducible prompt sets (`short_chat`, `summarization`,
  `code_completion`, `long_context_smoke`).
- First GPU benchmark baseline (RTX 3050 Laptop, `sshleifer/tiny-gpt2`).

## [0.6.0] — 2026-06-14

### Added
- Optional NVIDIA GPU fingerprint (`nvidia-smi` + `torch.cuda`) in the run manifest.

## [0.5.0] — 2026-06-14

### Added
- JSON run manifest (`--manifest`): git commit/dirty flag, config and prompts SHA256,
  Python/OS/CPU info, and dependency versions, for full run reproducibility.

## [0.4.0] — 2026-06-14

### Added
- `llm-bench compare`: Markdown comparison table across multiple saved benchmark CSVs.

## [0.3.0] — 2026-06-13

### Added
- Peak memory reporting: CPU RSS via `psutil`, PyTorch CUDA allocator peak via `torch.cuda`.

## [0.2.0] — 2026-06-13

### Added
- `transformers` backend: real CPU/GPU inference via `AutoModelForCausalLM` (optional extra).

## [0.1.0] — 2026-06-13

### Added
- Initial vertical slice: YAML-driven `BenchmarkConfig`, pluggable `Backend` ABC, zero-dependency
  `MockBackend`, `llm-bench` CLI, p50/p95 latency and tokens/sec metrics, GitHub Actions CI.
