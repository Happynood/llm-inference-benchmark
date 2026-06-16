# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versions before
1.0.0 may include breaking changes between minor releases.

## [Unreleased]

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
