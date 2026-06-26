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
| `--config PATH` | path | — | YAML benchmark config file. Required unless `--base-url` is given |
| `--format` | choice | `table` | Output format: `table` = human-readable text, `json` = machine-readable JSON object |
| `--output PATH` | path | — | Write results to this path (`--format table` → CSV, `--format json` → JSON file) |
| `--manifest PATH` | path | — | Write JSON environment fingerprint to this path |
| `--requests N` | int ≥ 1 | — | Override `requests` from the config |
| `--warmup-requests N` | int ≥ 0 | — | Override `warmup_requests` from the config |
| `--concurrency N` | int ≥ 1 | — | Override `concurrency` from the config |
| `--arrival-rate RPS` | float > 0 | — | Open-loop mode: dispatch requests at a constant rate (req/s), regardless of response time. Overrides `--concurrency`. Reveals queuing latency under sustained load |
| `--seed N` | int | — | Override `seed` from the config for reproducible prompt sampling |
| `--dataset NAME` | string | — | Use a cached real-world dataset as the prompt source instead of `prompts_file`. Pull first with `llm-bench datasets pull <name>` |
| `--base-url URL` | string | — | Base URL of an OpenAI-compatible endpoint (e.g. `http://localhost:11434/v1`). Sets backend to `openai` and makes `--config` optional; use `--set model=<name>` to specify the model |
| `--api-key KEY` | string | — | API key for `--base-url`. Omit for local servers that do not require authentication |
| `--set KEY=VALUE` | str (repeatable) | — | Override any config field via dot-path, e.g. `--set llama_cpp.max_tokens=200`. Values are parsed as YAML scalars (int, float, bool, str). Can be repeated. |

**Examples**

```bash
# Run and print summary to stdout
llm-bench --config configs/example.yaml

# Save CSV result
llm-bench --config configs/example.yaml --output results/run-a.csv

# Save CSV + environment manifest
llm-bench --config configs/example.yaml --output results/run-a.csv --manifest results/run-a.manifest.json

# Emit machine-readable JSON to stdout
llm-bench --config configs/example.yaml --format json

# Write JSON result to file (for scripting or CI pipelines)
llm-bench --config configs/example.yaml --format json --output results/run-a.json

# Quick one-off: override request count and concurrency without editing YAML
llm-bench --config configs/example.yaml --requests 50 --concurrency 4

# Disable warmup for a fast sanity check
llm-bench --config configs/example.yaml --warmup-requests 0 --requests 5

# Override backend-specific fields without editing YAML (--set accepts dot-paths)
llm-bench --config configs/example.yaml --set mock.latency_ms=5

# Multiple --set overrides in one command
llm-bench --config configs/llama-cpp-cpu.yaml --set llama_cpp.max_tokens=200 --set llama_cpp.n_threads=4

# Reproducible run with a fixed seed for prompt sampling
llm-bench --config configs/example.yaml --seed 42 --requests 20

# Open-loop mode: dispatch 5 requests/second regardless of response time
llm-bench --config configs/example.yaml --arrival-rate 5 --requests 50

# Test any OpenAI-compatible server without a config file
llm-bench --base-url http://localhost:11434/v1 --set model=llama3.2:3b

# Use a real-world dataset instead of the default prompts file (pull it first)
llm-bench --config configs/example.yaml --dataset lmsys-chat
```

The YAML config file controls which backend is used, the model, request count, and all
backend-specific parameters. CLI flags (`--requests`, `--warmup-requests`, `--concurrency`)
take precedence over the YAML values when provided. See [metrics.md](metrics.md) for CSV
column definitions.

**`--format json` output**

When `--format json` is used, the output is a single JSON object containing all
`MetricsReport` fields. Optional metrics that were not measured appear as `null` (not as
empty strings). Numeric fields are numbers; the `timestamp` field is an ISO-8601 string.

```json
{
  "backend": "mock",
  "model": "mock-model",
  "p50_latency_ms": 10.5,
  "p95_latency_ms": 12.1,
  "tokens_per_second": 950.0,
  "peak_cuda_memory_mb": null,
  "perplexity": null,
  "timestamp": "2026-01-01T00:00:00+00:00",
  ...
}
```

No progress or results-table text is printed to stdout in JSON mode. When `--output` is
given, the JSON is written to the file and nothing is printed to stdout.

---

## compare

Generate a Markdown comparison table from two or more benchmark CSVs.

```
llm-bench compare [OPTIONS] CSV_FILES...
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sort` | choice | `p95` | Sort column. Choices: `backend`, `model`, `p95`, `toks`, `load`, `ttft` |
| `--limit N` | int ≥ 1 | — | Show only the top N rows after sorting (omit to show all rows) |
| `--format` | choice | `table` | Output format: `table` = Markdown table, `json` = JSON array of row objects |
| `--output PATH` | path | — | Write output to file instead of stdout |

Sort values:

| Value | Order |
|-------|-------|
| `p95` | p95 latency ascending (fastest first) |
| `toks` | tokens/sec descending (highest throughput first) |
| `load` | model load time ascending (fastest load first; N/A rows last) |
| `ttft` | TTFT p50 ascending (lowest time-to-first-token first; N/A rows last) |
| `backend` | alphabetical by backend, then model |
| `model` | alphabetical by model, then backend |

**Examples**

```bash
llm-bench compare results/mock.csv results/transformers.csv
llm-bench compare results/*.csv --sort toks
llm-bench compare results/a.csv results/b.csv --sort backend --output table.md

# Focus on the 5 fastest runs by throughput
llm-bench compare results/*.csv --sort toks --limit 5

# Machine-readable output for CI scripting
llm-bench compare results/*.csv --sort p95 --format json

# Top 3 by latency, emitted as JSON
llm-bench compare results/*.csv --sort p95 --limit 3 --format json
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
| `--max-ttft-ms FLOAT` | float | — | Maximum TTFT p50 in ms (requires `stream: true` run) |
| `--max-load-ms FLOAT` | float | — | Maximum model load time in ms |
| `--format` | choice | `table` | Output format: `table` = human-readable text, `json` = machine-readable JSON |
| `--output PATH` | path | — | Write recommendation to file instead of stdout |

Exits with code **0** when a winner is found, **1** when no run satisfies all constraints
(all excluded runs are listed with their rejection reason).

**`--format json` output**

When `--format json` is used, the output is a JSON object with four keys:

| Key | Type | Description |
|-----|------|-------------|
| `winner` | object \| null | The recommended run (all RunRow fields; `null` when no run satisfies constraints) |
| `is_pareto_optimal` | bool | Whether the winner is Pareto-optimal among passing candidates |
| `candidates_count` | int | Number of runs that passed all constraints |
| `excluded` | array | Each entry has `backend`, `model`, and `reason` keys |

The exit code is identical to the default format: 0 when a winner exists, 1 otherwise.

**Examples**

```bash
# Recommend under VRAM and latency budget
llm-bench recommend results/*.csv --max-vram-mb 4096 --max-p95-ms 1000

# Add quality floors
llm-bench recommend results/*.csv --max-vram-mb 4096 --min-sanity 1.0 --min-quality 0.9

# Add a TTFT budget (requires stream: true runs)
llm-bench recommend results/*.csv --max-vram-mb 4096 --max-ttft-ms 100

# Add load time constraint (requires runs from v0.18+)
llm-bench recommend results/*.csv --max-load-ms 5000

# Machine-readable output for CI or scripting
llm-bench recommend results/*.csv --max-vram-mb 4096 --format json
llm-bench recommend results/*.csv --max-p95-ms 500 --format json --output rec.json
```

---

## diff

Compare two benchmark CSVs and show per-metric percentage change.

```
llm-bench diff [OPTIONS] BASELINE_CSV CURRENT_CSV
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output PATH` | path | — | Write diff table to file instead of stdout |
| `--fail-on-regression PCT` | float ≥ 0 | — | Exit 1 if any metric regresses by more than PCT% |

Each metric row shows the baseline value, the current value, and the percentage change.
Changes are annotated with **✓** (improvement) or **✗** (regression) based on direction
(lower is better for latency, TTFT, VRAM, CPU mem, perplexity, and load time; higher is
better for tok/s, sanity, task quality, and judge score). Optional metrics (TTFT, VRAM,
load time, quality, perplexity, judge) are omitted when absent from both runs, or shown
with `N/A` when only one side has data. Changes smaller than 0.05 % are displayed without
an annotation.

**`--fail-on-regression`** makes `llm-bench diff` usable as a CI gate. Pass `0` to fail
on any regression, or a positive number to tolerate small changes:

```bash
# fail if any metric regresses at all
llm-bench diff baseline.csv current.csv --fail-on-regression 0

# tolerate up to 5% degradation (noise-tolerant CI)
llm-bench diff baseline.csv current.csv --fail-on-regression 5
```

The diff table is always printed before the exit code check, so the regression detail is
visible in CI logs regardless of outcome.

**Example**

```bash
llm-bench diff results/before.csv results/after.csv
llm-bench diff results/before.csv results/after.csv --output diff.md
llm-bench diff results/before.csv results/after.csv --fail-on-regression 5
```

```
## Benchmark Diff

Baseline : llama-cpp | llama3 | N=20  (before.csv)
Current  : llama-cpp | llama3 | N=20  (after.csv)

| Metric        | Baseline |  Current |    Change |
|:--------------|----------|---------:|----------:|
| p50 (ms)      |  1169.00 |  1050.00 | -10.2% ✓ |
| p95 (ms)      |  1206.00 |  1100.00 |  -8.8% ✓ |
| tok/s         |     54.4 |     59.1 |  +8.6% ✓ |
| TTFT p50 (ms) |    40.80 |    38.20 |  -6.4% ✓ |
| VRAM (MB)     |   2361.0 |   2361.0 |   0.0%   |
| CPU mem (MB)  |    800.0 |    790.0 |  -1.2% ✓ |

✓ = improvement  ✗ = regression  (lower is better for latency/VRAM/PPL; higher for tok/s, sanity, quality, judge)
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

## env

Print current Python, package, and hardware environment information.

```
llm-bench env
```

No options. Outputs a one-line-per-field summary of the runtime environment useful for
reproducing or reporting benchmark results.

**Fields printed**

| Field | Description |
|-------|-------------|
| `python` | Python version and build info |
| `platform` | OS, kernel, and CPU architecture |
| `cpu` | CPU model and core count |
| `package` | `llm-inference-benchmark` version |
| `torch` | PyTorch version (if installed) |
| `transformers` | Transformers version (if installed) |
| `psutil` | psutil version |
| `gpu` | `detected` if a CUDA GPU is visible, otherwise `none` |

**Example**

```bash
llm-bench env
```

```
python      : 3.11.9 (main, Apr 19 2024, 16:48:33) [GCC 11.2.0]
platform    : Linux-6.8.0-generic-x86_64-with-glibc2.35
cpu         : AMD Ryzen 9 7950X (32 cores)
package     : llm-inference-benchmark 1.3.0
torch       : 2.3.0
transformers: 4.40.0
psutil      : 5.9.8
gpu         : detected
```

Use `llm-bench env` to capture the environment fingerprint before a benchmark session, or
include its output in bug reports and result comparisons.

---

## verify

Check which backends are installed and run a smoke test on the mock backend.

```
llm-bench verify [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | `table` | Output format: `table` = human-readable, `json` = machine-readable JSON array |

Runs a minimal inference call on the `mock` backend (always available) and checks whether
optional backend dependencies (`transformers`, `llama-cpp-python`, `optimum`, `vllm`) are
installed. Use this to confirm your environment is working before running expensive benchmarks.

Exits **0** when all installed backends pass, **1** if any backend reports a failure.

**Examples**

```bash
llm-bench verify
llm-bench verify --format json
```

```
BACKEND        STATUS     LATENCY  NOTES
----------------------------------------------------
mock           PASS        10.0 ms  smoke test OK
transformers   PASS        98.2 ms  OK
llama-cpp      NOT_INSTALLED  N/A   llama_cpp package not found
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
    dataset: lmsys-chat            # optional: use a cached real-world dataset as prompt source
    overrides:                     # optional: dot-path overrides of config fields
      mock.latency_ms: 5
      mock.tokens_per_response: 50
  - name: mock-code
    config: configs/example.yaml
    workload_profile: code_completion
```

`dataset` must be one of the names in `llm-bench datasets list`.  Pull the dataset before running
the matrix: `llm-bench datasets pull <name>`.  When `dataset` is set the run ignores
`prompts_file` from the config and loads prompts from `~/.cache/llm-bench/datasets/<name>.jsonl`
instead.

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

## pipeline

Run a full benchmark study: all matrix cells followed by compare, Pareto, and recommend.

```
llm-bench pipeline [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config PATH` | path (required) | — | YAML pipeline config file |
| `--dry-run` | flag | off | Print the full run plan and exit without executing |
| `--continue-on-error` | flag | off | Continue remaining cells after a failure; run post-processing on successful CSVs; exits 1 when any cell failed |
| `--format` | choice | `table` | Terminal progress format: `table` = human-readable, `json` = machine-readable |

A pipeline config is a superset of the matrix config format. After all matrix cells complete,
the pipeline automatically writes comparison tables, a Pareto chart, and a recommendation to
`results_dir/`. This is the recommended way to run a multi-model study end-to-end.

**Examples**

```bash
# Preview the full plan
llm-bench pipeline --config configs/pipeline-example.yaml --dry-run

# Execute all cells
llm-bench pipeline --config configs/pipeline-example.yaml

# Continue past cell failures, post-process successful results
llm-bench pipeline --config configs/pipeline-example.yaml --continue-on-error
```

---

## sweep

Ramp concurrency and emit a throughput-vs-latency curve.

```
llm-bench sweep [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config PATH` | path (required) | — | Base YAML benchmark config file |
| `--concurrency-range LIST` | string (required) | — | Comma-separated concurrency levels to test, e.g. `1,2,4,8` |
| `--max-p95-ms FLOAT` | float | — | Stop the sweep early when p95 latency exceeds this threshold (ms); exits 1 |
| `--requests N` | int | — | Override requests count per level (overrides config) |
| `--output PATH` | string | `sweep_results.csv` | Path for the combined sweep CSV |

Runs the benchmark once at each concurrency level, loads the model only once, and writes a
combined CSV with one row per level plus columns for `concurrency` and `throughput_rps`. After
all levels complete it prints a summary table and annotates the **knee point** (highest
requests/second across all levels).

Use `--max-p95-ms` as a latency SLA: the sweep stops when the p95 threshold is exceeded and
exits with code 1, making it usable as a CI gate.

**Examples**

```bash
# Sweep concurrency 1 → 2 → 4 → 8
llm-bench sweep --config configs/example.yaml --concurrency-range 1,2,4,8

# Stop when p95 exceeds 5 seconds
llm-bench sweep --config configs/example.yaml --concurrency-range 1,2,4,8 --max-p95-ms 5000

# Save results to a custom path
llm-bench sweep --config configs/example.yaml --concurrency-range 1,2,4 --output results/sweep.csv
```

**Sweep summary output**

```
=== Sweep Summary ===
 Concurrency       RPS    p50 ms    p95 ms     tok/s
-----------------------------------------------------
           1     0.098    1050.0    1200.0      49.1
           2     0.183    1080.0    1350.0      91.5  <- knee
           4     0.201    1950.0    2800.0      99.8

Knee point: concurrency=2  rps=0.183  p95=1350.0 ms
```

---

## pull

Download a model from HuggingFace Hub.

```
llm-bench pull [OPTIONS] REPO_ID
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `REPO_ID` | string (required) | — | HuggingFace repository ID, e.g. `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` |
| `--quant QUANT` | string | — | GGUF quantization suffix to download (e.g. `Q4_K_M`). Required when `--backend gguf` |
| `--backend` | choice | auto | Download backend: `gguf` or `transformers`. Defaults to `gguf` when `--quant` is given, otherwise `transformers` |
| `--dest PATH` | path | `~/models/` | Destination directory for GGUF files |
| `--max-size-gb FLOAT` | float | `10.0` | Abort if the remote file exceeds this size in GB |
| `--token TOKEN` | string | `$HF_TOKEN` | HuggingFace access token for gated models |

Downloads to `~/models/` for GGUF files and to the HuggingFace cache
(`~/.cache/huggingface/hub/`) for Transformers models. If the file is already cached the
command prints "Already cached" and exits 0 without re-downloading.

Requires `huggingface-hub` (installed by default with `llm-inference-benchmark`).

**Examples**

```bash
# Download a GGUF quantisation
llm-bench pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --quant Q4_K_M

# Download a Transformers model to HF cache
llm-bench pull HuggingFaceTB/SmolLM2-360M-Instruct --backend transformers

# Limit download size and use a custom destination
llm-bench pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF --quant Q5_K_M --max-size-gb 5 --dest ~/my-models/

# Gated model — pass a HF token
llm-bench pull meta-llama/Meta-Llama-3-8B-Instruct --backend transformers --token hf_xxx
```

---

## datasets

Manage cached real-world prompt datasets for benchmarking.

```
llm-bench datasets COMMAND
```

### datasets pull

Download and cache a real-world prompt dataset.

```
llm-bench datasets pull [OPTIONS] NAME
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `NAME` | string (required) | — | Dataset name (see table below) |
| `--max-samples N` | int ≥ 1 | per-dataset | Override the maximum number of samples to download |
| `--token TOKEN` | string | `$HF_TOKEN` | HuggingFace access token (required for gated datasets) |

Supported dataset names:

| Name | Source | Description | Default samples | Gated |
|------|--------|-------------|-----------------|-------|
| `wildchat` | `allenai/WildChat-1M` | Real-world user chat turns | 500 | No |
| `lmsys-chat` | `lmsys/lmsys-chat-1m` | Real multi-turn chat (first user turn) | 500 | Yes |
| `hermes-fn` | `NousResearch/hermes-function-calling-v1` | Function-calling prompts | 200 | No |
| `long-context-4k` | `allenai/c4` (en) | ~4 k-token passages for prefill profiling | 100 | No |
| `long-context-16k` | `allenai/c4` (en) | ~16 k-token passages | 50 | No |
| `long-context-64k` | `allenai/c4` (en) | ~64 k-token passages | 10 | No |

Samples are cached as a `.jsonl` file in `~/.llm-bench/datasets/`. Pass the dataset name to
`llm-bench` with `--dataset <name>` to use it as the prompt source for a benchmark run.

Requires the `datasets` package: `uv pip install datasets`.

**Examples**

```bash
llm-bench datasets pull wildchat
llm-bench datasets pull lmsys-chat --token hf_xxx
llm-bench datasets pull long-context-4k --max-samples 20
```

### datasets list

List locally cached datasets and their sample counts.

```
llm-bench datasets list
```

No options.

```bash
llm-bench datasets list
```

```
Dataset               Samples
------------------------------
wildchat              500
long-context-4k       100
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
| `concurrency` | int ≥ 1 | `1` | Maximum number of requests in-flight at once (closed-loop mode) |
| `arrival_rate_rps` | float > 0 | — | Open-loop mode: dispatch requests at this rate (req/s) regardless of response time. Mutually exclusive with `concurrency`; set via `--arrival-rate` on the CLI |
| `warmup_requests` | int ≥ 0 | `2` | Warmup requests (excluded from metrics) |
| `repeats` | int ≥ 1 | `1` | Repeat the full benchmark loop N times; p95/tok/s become the median |
| `seed` | int | — | Random seed for stochastic sampling (`temperature > 0`). Applied per-request in `transformers` (`torch.manual_seed`) and at model load in `llama-cpp`. For `openai`, the value is forwarded as a best-effort hint — server support varies. Has no effect on greedy decoding (`temperature: 0.0`). |
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


## serve

Start the llm-bench Web API server.

```
llm-bench serve [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--host TEXT` | string | `127.0.0.1` | Bind host |
| `--port INTEGER` | int | `8080` | Bind port |

Requires the `server` optional extra:

```bash
uv pip install 'llm-inference-benchmark[server]'
llm-bench serve
llm-bench serve --host 0.0.0.0 --port 9000
```

### API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | `{"status": "ok"}` liveness check |
| `GET` | `/api/models` | List GGUF files (`~/models/**/*.gguf`) and HuggingFace cache model dirs |
| `GET` | `/api/runs` | All past runs ordered by `created_at DESC` |
| `POST` | `/api/runs` | Submit `{"config": {...}}` — returns `{"run_id": "..."}` immediately (202) |
| `GET` | `/api/runs/{run_id}` | Status (`pending`/`running`/`done`/`error`) + output when done |
| `GET` | `/api/runs/{run_id}/stream` | SSE stream of stdout lines; terminates with `data: [done:STATUS]` |

Results persist in `~/.llm-bench/results.db` (SQLite, WAL mode).
