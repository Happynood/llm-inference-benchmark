# CLI Reference

Complete reference for the `llm-bench` command-line tool.

## Global options

```
llm-bench [OPTIONS] COMMAND [ARGS]...
```

| Option | Type | Description |
|--------|------|-------------|
| `--version` | flag | Print version and exit |
| `--help` | flag | Show help and exit |

---

## Default mode — run a benchmark

Run without a subcommand to execute a single benchmark.

```
llm-bench [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config PATH` | path (required) | — | YAML benchmark config file |
| `--output PATH` | path | — | Write results CSV to this path (omit for stdout summary only) |
| `--manifest PATH` | path | — | Write JSON environment fingerprint to this path |
| `--requests N` | int ≥ 1 | — | Override `requests` from the config |
| `--warmup-requests N` | int ≥ 0 | — | Override `warmup_requests` from the config |
| `--concurrency N` | int ≥ 1 | — | Override `concurrency` from the config |

**Examples**

```bash
# Run and print summary to stdout
llm-bench --config configs/example.yaml

# Save CSV result
llm-bench --config configs/example.yaml --output results/run-a.csv

# Save CSV + environment manifest
llm-bench --config configs/example.yaml --output results/run-a.csv --manifest results/run-a.manifest.json

# Quick one-off: override request count and concurrency without editing YAML
llm-bench --config configs/example.yaml --requests 50 --concurrency 4

# Disable warmup for a fast sanity check
llm-bench --config configs/example.yaml --warmup-requests 0 --requests 5
```

The YAML config file controls which backend is used, the model, request count, and all
backend-specific parameters. CLI flags (`--requests`, `--warmup-requests`, `--concurrency`)
take precedence over the YAML values when provided. See [metrics.md](metrics.md) for CSV
column definitions.

---

## compare

Generate a Markdown comparison table from two or more benchmark CSVs.

```
llm-bench compare [OPTIONS] CSV_FILES...
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sort` | choice | `p95` | Sort column. Choices: `backend`, `model`, `p95`, `toks`, `load` |
| `--output PATH` | path | — | Write Markdown to file instead of stdout |

Sort values:

| Value | Order |
|-------|-------|
| `p95` | p95 latency ascending (fastest first) |
| `toks` | tokens/sec descending (highest throughput first) |
| `load` | model load time ascending (fastest load first; N/A rows last) |
| `backend` | alphabetical by backend, then model |
| `model` | alphabetical by model, then backend |

**Examples**

```bash
llm-bench compare results/mock.csv results/transformers.csv
llm-bench compare results/*.csv --sort toks
llm-bench compare results/a.csv results/b.csv --sort backend --output table.md
```

**Table columns**: Backend, Model, N, p50 (ms), p95 (ms) ±std, tok/s ±std, Load (ms),
CPU mem (MB), CUDA mem (MB), VRAM mem (MB), Sanity %, Task Q %, PPL, Judge.
Optional columns show `N/A` when the data was not collected.

---

## pareto

Identify Pareto-optimal configurations from two or more benchmark CSVs.

```
llm-bench pareto [OPTIONS] CSV_FILES...
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output PATH` | path | — | Write Markdown to file instead of stdout |

A configuration is **optimal** when no other configuration is at least as good on every
metric and strictly better on at least one. Metrics considered: p95 latency (lower is
better), tok/s (higher is better), VRAM (lower is better), sanity pass rate (higher is
better), task quality (higher is better), perplexity (lower is better), judge score
(higher is better). Missing optional metrics are excluded from comparisons rather than
penalising either run.

**Examples**

```bash
llm-bench pareto results/q4km.csv results/q8.csv
llm-bench pareto results/*.csv --output pareto.md
```

---

## recommend

Return the best Pareto-optimal configuration that satisfies all given constraints.

```
llm-bench recommend [OPTIONS] CSV_FILES...
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-vram-mb FLOAT` | float | — | Maximum peak VRAM in MB |
| `--max-p95-ms FLOAT` | float | — | Maximum p95 latency in ms |
| `--min-sanity FLOAT` | float | — | Minimum sanity pass rate `[0, 1]` |
| `--min-quality FLOAT` | float | — | Minimum task quality pass rate `[0, 1]` |
| `--max-perplexity FLOAT` | float | — | Maximum perplexity |
| `--min-judge FLOAT` | float | — | Minimum judge score `[0, 1]` |
| `--max-load-ms FLOAT` | float | — | Maximum model load time in ms |
| `--output PATH` | path | — | Write recommendation to file instead of stdout |

Exits with code **0** when a winner is found, **1** when no run satisfies all constraints
(all excluded runs are listed with their rejection reason).

**Examples**

```bash
# Recommend under VRAM and latency budget
llm-bench recommend results/*.csv --max-vram-mb 4096 --max-p95-ms 1000

# Add quality floors
llm-bench recommend results/*.csv --max-vram-mb 4096 --min-sanity 1.0 --min-quality 0.9

# Add load time constraint (requires runs from v0.18+)
llm-bench recommend results/*.csv --max-load-ms 5000
```

---

## profiles

List all built-in workload profiles with descriptions.

```
llm-bench profiles
```

No options. Prints each profile name, input/output length class, and description.

Profiles can be referenced by name in a config YAML (`workload_profile: short_chat`) or
in a matrix run entry.

| Profile name | Input length | Output length |
|---|---|---|
| `short_chat` | short | short |
| `summarization` | medium | short |
| `code_completion` | short | medium |
| `long_context_smoke` | long | short |

**Example**

```bash
llm-bench profiles
```

---

## matrix

Execute all benchmark runs defined in a matrix config file.

```
llm-bench matrix [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config PATH` | path (required) | — | YAML matrix config file |
| `--dry-run` | flag | off | List runs without executing them |
| `--continue-on-error` | flag | off | Continue remaining runs after a failure; exit 1 if any failed |

Each run writes one CSV and one manifest JSON into the `results_dir` defined in the matrix
config. Run names must be alphanumeric + `-_.` (no slashes or spaces).

**Matrix config format**

```yaml
results_dir: results            # optional, default: "results"
runs:
  - name: mock-chat
    config: configs/example.yaml
    workload_profile: short_chat   # optional: override config's profile
    overrides:                     # optional: dot-path overrides of config fields
      mock.latency_ms: 5
      mock.tokens_per_response: 50
  - name: mock-code
    config: configs/example.yaml
    workload_profile: code_completion
```

**Parameter sweep shorthand** (`base_config` + `sweep`):

```yaml
base_config: configs/example.yaml
results_dir: results
sweep:
  mock.latency_ms: [5, 10, 20]
  mock.tokens_per_response: [25, 50, 100]
```

Generates the cartesian product of all sweep axes as individual runs.

**Examples**

```bash
# Preview all runs
llm-bench matrix --config configs/matrix-example.yaml --dry-run

# Execute all runs
llm-bench matrix --config configs/matrix-example.yaml

# Continue past failures, report summary at the end
llm-bench matrix --config configs/matrix-example.yaml --continue-on-error

# Compare all results afterwards
llm-bench compare results/*.csv --sort p95
```

---

## validate-config

Validate a benchmark config file and print a summary of resolved settings.

```
llm-bench validate-config [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config PATH` | path (required) | — | YAML benchmark config file to validate |

Reads the YAML, runs full Pydantic validation, resolves the effective prompts file, and
prints a structured summary of all settings including backend-specific fields. Optional
fields (`workload_profile`, `quality_file`, `measure_perplexity`, `measure_judge`) are
shown only when set.

Exits **0** on success, **1** with an error message on validation failure.

Useful for catching typos and invalid values before committing to a long benchmark run.

**Example**

```bash
llm-bench validate-config --config configs/example.yaml
```

```
Config: configs/example.yaml
  backend          : mock
  model            : mock-gpt2
  requests         : 20
  warmup_requests  : 2
  repeats          : 1
  prompts_file     : data/prompts/smoke.txt
  mock.latency_ms  : 5.0
  mock.tokens_per_response: 50
OK
```

---

## Config YAML reference

All fields are optional unless marked required.

### Common fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string | `mock` | Backend to use: `mock`, `transformers`, `llama-cpp`, `openai` |
| `model` | string | `mock-gpt2` | Model identifier (path or HuggingFace repo ID) |
| `requests` | int ≥ 1 | `20` | Number of benchmark requests (excluding warmup) |
| `concurrency` | int ≥ 1 | `1` | Maximum number of requests in-flight at once |
| `warmup_requests` | int ≥ 0 | `2` | Warmup requests (excluded from metrics) |
| `repeats` | int ≥ 1 | `1` | Repeat the full benchmark loop N times; p95/tok/s become the median |
| `prompts_file` | path | `data/prompts/smoke.txt` | Path to newline-delimited prompts file |
| `workload_profile` | string | — | Named profile: `short_chat`, `summarization`, `code_completion`, `long_context_smoke` |
| `quality_file` | path | — | YAML rubric spec for per-prompt quality evaluation |
| `measure_perplexity` | bool | `false` | Compute self-perplexity on generated completions (`transformers` only) |
| `measure_judge` | bool | `false` | Score yes/no self-judge relevance from logprobs (`transformers` only) |

### Backend-specific fields

**`mock` backend**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mock.latency_ms` | float | `10.0` | Simulated per-request latency in ms |
| `mock.tokens_per_response` | int | `50` | Simulated token count per response |

**`transformers` backend**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hf.max_new_tokens` | int ≥ 1 | `50` | Maximum tokens to generate per request |
| `hf.device` | string | `cpu` | PyTorch device string (`cpu`, `cuda`, `cuda:0`, …) |
| `hf.torch_dtype` | string | `float32` | Model dtype: `float32`, `float16`, `bfloat16` |
| `hf.do_sample` | bool | `false` | Enable sampling (deterministic greedy decoding when false) |

**`llama-cpp` backend**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llama_cpp.n_ctx` | int ≥ 1 | `2048` | Context window size in tokens |
| `llama_cpp.n_gpu_layers` | int | `0` | Number of model layers to offload to GPU (`99` = all) |
| `llama_cpp.max_tokens` | int ≥ 1 | `50` | Maximum tokens to generate per request |
| `llama_cpp.temperature` | float ≥ 0 | `0.0` | Sampling temperature (0 = greedy) |
| `llama_cpp.n_threads` | int | — | CPU threads to use (default: llama.cpp auto-detect) |
| `llama_cpp.verbose` | bool | `false` | Enable llama.cpp verbose logging |

**`openai` backend**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `openai.base_url` | string | `http://localhost:8080/v1` | Base URL of the OpenAI-compatible server |
| `openai.max_tokens` | int ≥ 1 | `50` | Maximum tokens to request per completion |
| `openai.temperature` | float ≥ 0 | `0.0` | Sampling temperature |
| `openai.timeout_s` | float > 0 | `60.0` | Per-request HTTP timeout in seconds |
| `openai.api_key_env` | string | — | Environment variable name holding the API key |
