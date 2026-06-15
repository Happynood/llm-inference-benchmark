# Benchmark Metrics

## Metric Definitions

| Metric | Unit | Description |
|--------|------|-------------|
| `request_count` | count | Number of benchmark requests (excluding warmup) |
| `p50_latency_ms` | ms | Median end-to-end latency per request |
| `p95_latency_ms` | ms | 95th-percentile latency per request |
| `tokens_per_second` | tok/s | Output tokens produced divided by total latency |
| `total_tokens` | count | Sum of input + output tokens across all requests |
| `backend` | string | Backend name (`mock`, `transformers`, `llama-cpp`, ...) |
| `model` | string | Model identifier as passed in config |
| `peak_cpu_memory_mb` | MB | Peak process RSS during the benchmark run (warmup + benchmark loop) |
| `peak_cuda_memory_mb` | MB | Peak PyTorch allocator CUDA memory during the run; blank when torch/CUDA absent |
| `peak_vram_memory_mb` | MiB | Peak GPU VRAM from `nvidia-smi` during the run; blank when `nvidia-smi` unavailable |
| `empty_output_count` | count | Completions whose stripped text is empty (whitespace-only counts as empty) |
| `min_output_chars` | count | Minimum stripped character count across all completions |
| `mean_output_chars` | count | Mean stripped character count across all completions |
| `repeated_output_count` | count | Completions whose text is identical to at least one other in the same run |
| `sanity_pass_rate` | ratio [0, 1] | Fraction of non-empty completions; 1.0 = all outputs contained text |
| `task_quality_pass_rate` | ratio [0, 1] | Fraction of completions that pass all rubric checks; blank when no `quality_file` was set |
| `task_quality_checked_count` | count | Number of completions evaluated against rubric checks; 0 when no `quality_file` was set |
| `timestamp` | ISO 8601 | UTC timestamp when the run completed |

## Memory Measurement

### CPU peak (`peak_cpu_memory_mb`)

Captured by polling `psutil.Process.memory_info().rss` in a background daemon thread every
50 ms. RSS (Resident Set Size) is the OS-level physical memory used by the process, which
includes model weights, PyTorch tensors, and all C-extension allocations — unlike
`tracemalloc`, which only tracks the Python heap.

The measurement window covers the full `run_benchmark` call: warmup requests + benchmark
requests. Because model weights are loaded in `Backend.__init__` before `run_benchmark` is
called, the RSS at the start of the window already includes the model. Peak values therefore
reflect model weights + maximum inference-time activations.

**Limitation**: at 50 ms poll interval, very short-lived spikes (<50 ms) may be missed.
For CPU inference where each request takes tens–hundreds of ms this is accurate enough.

### CUDA peak (`peak_cuda_memory_mb`)

Captured via `torch.cuda.max_memory_allocated()`. The counter is reset and MemorySampler is
started together inside `run_benchmark`, so CPU and CUDA measurement windows are co-incident
(benchmark loop only; warmup excluded from both).

- **`None` / blank in CSV**: torch not installed, or CUDA toolkit not available.
- **`0.0`**: CUDA available but inference ran on a CPU device (no GPU allocations made).
- **`>0`**: GPU inference; value reflects allocator-tracked memory only — fragmentation and
  reserved-but-unallocated pages are excluded.

**Limitation**: measures PyTorch allocator-tracked memory only — reserved-but-free pages are
excluded, and allocations made by non-PyTorch GPU backends (such as llama-cpp's own CUDA
allocator) are not captured here at all. For those cases, use `peak_vram_memory_mb`.

### Driver-level VRAM peak (`peak_vram_memory_mb`)

Captured by polling `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
in a background daemon thread every 500 ms. Reports total VRAM used on the GPU as seen by
the NVIDIA driver — model weights, KV cache, CUDA workspace, and all other GPU allocations,
regardless of which library or backend made them.

This metric is the right one for llama-cpp GPU inference: llama.cpp manages VRAM through
its own allocator rather than PyTorch's, so `peak_cuda_memory_mb` shows `0.0` even during
active GPU inference. `peak_vram_memory_mb` captures the true VRAM footprint in that case.

- **`None` / blank in CSV**: `nvidia-smi` was not found or failed on the first probe.
  No NVIDIA GPU present, driver not installed, or `nvidia-smi` not on `PATH`.
- **`>0`**: VRAM in MiB used during the benchmark loop. On machines with an NVIDIA GPU,
  includes baseline OS/driver idle usage (~10–50 MiB) even for CPU-only runs, because
  the metric reflects total driver-visible VRAM across all processes.

**Limitation**: at a 500 ms poll interval, very short-lived VRAM spikes may be missed.
The value reflects total GPU VRAM across all processes — other GPU workloads running
concurrently will inflate the reported peak.

> **Older runs**: CSVs captured before this metric was added have a blank
> `peak_vram_memory_mb` column. `llm-bench compare` treats missing or blank as `N/A`.

## Output Sanity Metrics

Five lightweight structural checks run on completion texts after every benchmark loop
(warmup excluded). They detect degenerate outputs, not semantic quality.

### `empty_output_count` / `sanity_pass_rate`

`sanity_pass_rate = (requests - empty_output_count) / requests`.
A completion is empty if its stripped text has zero characters — whitespace-only
completions count as empty.

`sanity_pass_rate = 1.0` means every output contained at least one non-whitespace
character. Values below 1.0 indicate the backend produced silent or blank completions,
which warrants investigation before trusting throughput numbers.

### `repeated_output_count`

Counts completions whose stripped text appears at least once more in the same run.
With `temperature=0.0` and a cycling prompt set, repetition is mathematically expected
(same prompt → same greedy output). Treat this as a degeneration signal rather than a
pass/fail threshold.

### `min_output_chars` / `mean_output_chars`

Character counts after stripping whitespace. These provide a quick sanity check that the
model is generating output of reasonable length, and that `max_tokens` is not silently
truncating before any real content is produced.

**Backward compatibility**: CSVs produced before v0.13 do not contain these columns.
`llm-bench compare` shows `N/A` in the `Sanity %` column for such runs.

## Task Quality Metrics (v0.16)

`Sanity %` checks structure (non-empty); `Task Q %` checks content against a per-prompt rubric.

### `task_quality_pass_rate` / `task_quality_checked_count`

When a `quality_file` is specified in the benchmark config, each non-warmup completion is
evaluated against a per-prompt-position rubric loaded from a YAML spec file.

```yaml
# Example quality spec (quality_file: data/quality/my_task.yaml)
- contains_all: ["Paris"]          # all phrases must appear (case-insensitive)
  contains_any: ["capital", "city"] # at least one of these must appear
  forbidden: ["I don't know"]       # none of these may appear (case-insensitive)
  min_chars: 10                     # minimum stripped character length
- null                              # null → skip check for this prompt position
```

Supported check types:

| Key | Type | Description |
|-----|------|-------------|
| `contains_all` | list[str] | All phrases must appear (case-insensitive substring match) |
| `contains_any` | list[str] | At least one phrase must appear (case-insensitive) |
| `forbidden` | list[str] | None of these phrases may appear (case-insensitive) |
| `regex` | str | Applied to stripped text via `re.search`; must match |
| `min_chars` | int | Stripped text must be at least this many characters |

**Cycling**: completions are matched to rubrics by `i % prompt_count`. A rubric list shorter
than the prompt file leaves later prompt positions unchecked.

`task_quality_pass_rate = passes / checked` where `checked` is the count of completions
that had a non-null rubric entry.  When nothing was checked (all null, or no `quality_file`),
the field is blank in the CSV and `N/A` in `llm-bench compare`.

**Distinction from sanity**: `sanity_pass_rate` detects structural failures (empty/blank
output); `task_quality_pass_rate` detects semantic failures (answer is present but wrong).
A run with `sanity_pass_rate=1.0` and `task_quality_pass_rate=0.0` means the model
produced text but none of it matched the expected content.

**`--min-quality` constraint**: pass `--min-quality <float>` to `llm-bench recommend` to
exclude runs whose `task_quality_pass_rate` is below threshold or unknown.  Runs without
quality data are only excluded when the constraint is active — backward-compatible.

## Computation Notes

- **p50/p95 latency**: both use C1 linear interpolation
  (`idx = (p/100) * (N-1)`, interpolated between `sorted[floor(idx)]` and `sorted[ceil(idx)]`).
  This matches NumPy's default (`np.percentile(..., interpolation='linear')`).
  For p50 with even N this is equivalent to the arithmetic mean of the two central values.
- **tokens/sec**: output-only tokens divided by sum of per-request latencies
  (`output_tokens_total / sum(latency_ms) * 1000`).
  For v0.1 sequential execution, sum-of-latencies equals wall-clock elapsed time.
  **This equivalence breaks when concurrent execution is added** — the denominator will switch
  to wall-clock elapsed time at that point.
- **p50 ≈ p95 with mock backend**: the mock produces near-constant latency by design, so
  percentiles are nearly identical. Real backends with variable scheduling jitter will separate
  these values meaningfully.
- **Warmup requests** are excluded from all metric calculations.

---

## Pareto Analysis

`llm-bench pareto results/*.csv` reads saved benchmark CSVs and classifies each run as
**optimal** or **dominated** using Pareto dominance.

### Dominance rule

Run A dominates run B when A is **no worse** on every available metric and **strictly
better** on at least one:

| Metric | Direction | Participation |
|--------|-----------|---------------|
| `p95_latency_ms` | minimise | always |
| `tokens_per_second` | maximise | always |
| `peak_vram_memory_mb` | minimise | only when both rows have a value |
| `sanity_pass_rate` | maximise | only when both rows have a value |
| `task_quality_pass_rate` | maximise | only when both rows have a value |

Optional metrics are excluded from any comparison where one or both rows carry `None`
for that field.  This means an older CSV without VRAM data is not penalised — the
comparison narrows to the metrics both runs share.

### Output

```bash
uv run llm-bench pareto results/quant-q4km.csv results/quant-q8.csv
```

```
| Backend   | Model                                  | N  | p95 (ms) | tok/s | CPU mem (MB) | VRAM mem (MB) | Sanity % | Pareto  |
|-----------|----------------------------------------|----|----------|-------|--------------|---------------|----------|---------|
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf | 10 | 904.33   | 55.3  | 1289.8       | 2361.0        | 100.0%   | optimal |
| llama-cpp | .../Llama-3.2-3B-Instruct-Q8_0.gguf   | 10 | 1185.23  | 42.2  | 1418.4       | 3697.0        | 100.0%   | -       |
```

Q4\_K\_M is strictly better than Q8\_0 on p95 latency, throughput, and VRAM, so Q4\_K\_M
is the sole Pareto-optimal configuration in this two-run comparison.

### Limitations

- **N matters**: Pareto analysis at N=10 requests inherits all limitations of the raw
  metrics — p95 at N=10 is the single worst observation, not a stable tail-latency
  estimate.
- **No quality dimension**: output quality (perplexity, task accuracy) is not a metric.
  Pareto optimality here means "efficient on speed/memory/sanity", not "best model".
- **No aggregation**: each CSV is one run.  Run-to-run variance is not considered.

---

## Constraint-based Recommender

`llm-bench recommend results/*.csv [--max-vram-mb N] [--max-p95-ms N] [--min-sanity N] [--min-quality N]`
reads saved benchmark CSVs, filters them against explicit constraints, and recommends
the best configuration.

### Constraints

| Flag | Metric filtered | Exclusion trigger |
|------|----------------|-------------------|
| `--max-vram-mb` | `peak_vram_memory_mb` | value > threshold, or value missing when flag is set |
| `--max-p95-ms` | `p95_latency_ms` | value > threshold |
| `--min-sanity` | `sanity_pass_rate` | value < threshold, or value missing when flag is set |
| `--min-quality` | `task_quality_pass_rate` | value < threshold, or value missing when flag is set |

All constraints are optional.  With no flags every run is a candidate.

### Selection logic

1. Apply all constraints.  Runs that violate any constraint are excluded with a reason.
2. Among candidates, classify by Pareto dominance (same rules as `llm-bench pareto`).
3. Among Pareto-optimal candidates, pick the one with the lowest p95 latency.
4. If no candidates pass all constraints, print the exclusion table and exit with code 1.

### Missing optional metrics

`peak_vram_memory_mb` and `sanity_pass_rate` may be absent in older CSVs.

- If **no constraint** is set for the metric → the run is still a candidate.
- If a **constraint is set** and the value is missing → the run is excluded with
  reason `VRAM unknown` or `sanity unknown`.

### Example

```
uv run llm-bench recommend results/quant-q4km.csv results/quant-q8.csv \
    --max-vram-mb 4096 --max-p95-ms 1000 --min-sanity 1.0
```

Output:

```
Recommendation
──────────────────────────────────────────
  Backend : llama-cpp
  Model   : Llama-3.2-3B-Instruct-Q4_K_M.gguf
  N       : 10
  p95     : 915.22 ms
  tok/s   : 55.3
  VRAM    : 2361.0 MB
  Sanity  : 100.0%

Why: lowest p95 among 2 candidate(s) passing all constraints; Pareto-optimal.
```

If no run satisfies the constraints:

```
No recommendation: no runs satisfy all constraints.

Excluded (2)
──────────────────────────────────────────
  llama-cpp  Q4_K_M.gguf  →  p95 latency too high (915.2 ms > 800.0 ms)
  llama-cpp  Q8_0.gguf    →  p95 latency too high (1186.8 ms > 800.0 ms)
```

Exit code is 1 in this case; 0 when a winner is found.

### Limitations

- Recommendation quality is bounded by the quality of the input CSVs.
  A recommendation from two runs is weaker than one from a systematic sweep.
- Output quality (perplexity, task accuracy) is not a constraint.
  `--min-sanity` only checks that completions are non-empty, not that they are correct.
- Run-to-run variance is not considered.  p95 from N=10 requests is a rough estimate.

---

## CI / Harness Validation

Results from the mock backend validate that the measurement pipeline is wired up correctly.
They do **not** measure model inference speed. The mock backend sleeps for a configured
`latency_ms` and returns a fixed token count — no model is loaded, no forward pass runs.

> **Rule**: never compare mock results to real backend results in the same table.
> Mock numbers belong in CI logs, not benchmark comparisons.

### Mock backend (v0.1 — harness validation only)

| metric | value | what it validates |
|--------|-------|-------------------|
| backend | mock | backend field is present |
| p50_latency_ms | ~5 ms | configured `latency_ms` is measured |
| p95_latency_ms | ~5 ms | p95 ≈ p50 for deterministic mock |
| tokens_per_second | ~10,000 (simulated) | tokens/sec formula is correct |

---

## Real Hardware Evidence

Results in this section come from real inference backends on real hardware.
Full curated reports with config, software versions, and interpretation are in
[docs/results/](results/).

> **`peak_vram_memory_mb` note**: runs in the transformers-backend subsections below were
> captured before driver-level VRAM sampling was added. Their CSVs do not contain that column
> and `llm-bench compare` will show `N/A` for it. The llama-cpp run has `peak_vram_memory_mb`
> measured via nvidia-smi; see its curated report for the value.

### `sshleifer/tiny-gpt2` — CPU (i5-11400H, float32)

> Hardware: Intel Core i5-11400H @ 2.70 GHz, 12 logical cores, CPU-only (no GPU).
> Software: Python 3.12.13, torch 2.12.0, transformers 5.12.0.
> Model: `sshleifer/tiny-gpt2` — 2-layer GPT-2 toy model, ~117 K params, ~4 MB.
> **Not representative of production models.** Validates that the harness measures real
> inference latency (actual weights, tokenizer, CPU kernels — not simulated).
> Full report: [docs/results/gpu-rtx3050-tiny-gpt2.md](results/gpu-rtx3050-tiny-gpt2.md)

Command:
```bash
uv run llm-bench --config configs/transformers-cpu.yaml --output results/cpu.csv
```

Config: `requests=10`, `warmup_requests=2`, `max_new_tokens=50`, `device=cpu`, `do_sample=false`.

| metric | value |
|--------|-------|
| backend | transformers |
| model | sshleifer/tiny-gpt2 |
| requests | 10 |
| warmup_requests | 2 |
| p50_latency_ms | 40.95 |
| p95_latency_ms | 44.67 |
| tokens_per_second | 1211.23 |
| total_tokens | 598 |
| peak_cpu_memory_mb | 721.37 |
| peak_cuda_memory_mb | 0.00 (CUDA toolkit present, inference on CPU device) |

### Interpretation

- p50 of ~41 ms for a 2-layer toy model on CPU; a full GPT-2 (12 layers, 117 M params) would
  be ~10–20× slower (~400–800 ms). Llama 3 8B would be 100–1000× slower.
- ~1211 output tokens/sec on this model is meaningless as a production throughput figure —
  production models on CPU produce 5–50 tok/s.
- p95/p50 spread (~14%) reflects OS scheduling jitter, expected on CPU without process pinning.

### `sshleifer/tiny-gpt2` — GPU (RTX 3050 Laptop, float16)

> Hardware: Intel Core i5-11400H @ 2.70 GHz, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).
> Software: Python 3.12.13, torch 2.12.0, transformers 5.12.0.
> Model: `sshleifer/tiny-gpt2` — 2-layer GPT-2 toy model.
> Full report: [docs/results/gpu-rtx3050-tiny-gpt2.md](results/gpu-rtx3050-tiny-gpt2.md)

Command:
```bash
uv run llm-bench --config configs/transformers-gpu.yaml --output results/gpu.csv --manifest results/gpu.manifest.json
```

| metric | value |
|--------|-------|
| backend | transformers |
| model | sshleifer/tiny-gpt2 |
| requests | 10 |
| warmup_requests | 2 |
| p50_latency_ms | 59.95 |
| p95_latency_ms | 61.86 |
| tokens_per_second | 829.60 |
| total_tokens | 598 |
| peak_cpu_memory_mb | 1382.92 |
| peak_cuda_memory_mb | 8.82 |

### Interpretation

- GPU (59.95 ms p50) is **slower than CPU (40.95 ms)** for this toy model. This is expected:
  `sshleifer/tiny-gpt2` has only 2 layers and 117 K parameters. At this scale, GPU kernel-launch
  overhead and CPU-GPU data transfer dominate. Real production models (Llama 3 8B+) reverse this
  decisively — GPU inference is 10–100× faster than CPU there.
- `peak_cuda_memory_mb`: 8.82 MB confirms that this tiny model fits in GPU RAM with a negligible
  footprint. A full GPT-2 small (117 M params) in float16 uses ~250 MB; Llama 3 8B uses ~8 GB+.
- The `gpu` section in the manifest correctly captured `torch_cuda_available: true` and
  `torch_cuda_device_name: "NVIDIA GeForce RTX 3050 Laptop GPU"`. The `nvidia-smi` sub-fields
  (name, driver_version, etc.) were `null` in this run — `nvidia-smi` was not accessible in
  the run environment.
- `torch_dtype: float16` reduces GPU memory usage vs float32 at identical accuracy for this model.

### `Llama-3.2-3B-Instruct-Q4_K_M` — CPU vs GPU (RTX 3050, llama.cpp)

> Hardware: Intel Core i5-11400H, NVIDIA GeForce RTX 3050 Laptop GPU (3772 MiB VRAM).
> Software: Python 3.12.13, llama-cpp-python 0.3.29 (pre-built cu124 wheel).
> Model: `Llama-3.2-3B-Instruct-Q4_K_M` — 3B parameters, Q4\_K\_M quantization, 1.9 GB GGUF, 28 layers.
> Full report: [docs/results/llama-cpp-rtx3050-llama32-3b.md](results/llama-cpp-rtx3050-llama32-3b.md)

| metric | CPU (`n_gpu_layers=0`) | GPU — RTX 3050 (`n_gpu_layers=99`) |
|--------|----------------------|-------------------------------------|
| backend | llama-cpp | llama-cpp |
| model | Llama-3.2-3B-Instruct-Q4\_K\_M | Llama-3.2-3B-Instruct-Q4\_K\_M |
| requests | 10 | 10 |
| warmup_requests | 2 | 2 |
| p50_latency_ms | 2750.56 | 931.18 |
| p95_latency_ms | 2939.79 | 939.69 |
| tokens_per_second | 18.01 | **53.71** |
| peak_vram_mib | — | 2361 (nvidia-smi polling) |

### Interpretation

- GPU is **2.95× faster** than CPU for this 28-layer production model with Q4\_K\_M quantization.
- GPU latency is highly consistent: p95/p50 = 1.009 — GPU execution is compute-bound.
- CPU shows more variance: p95/p50 = 1.069 — memory-bandwidth-bound, subject to OS jitter.
- 2361 MiB peak VRAM leaves ~1.4 GB headroom on the RTX 3050's 3772 MiB VRAM budget.
- 53.7 tok/s on GPU is real-time capable; a 50-token response completes in under 1 second.

---

## llama.cpp Backend

The `llama-cpp` backend runs GGUF quantized models via `llama-cpp-python`. It is designed
for production-size models (Llama 3 8B, Mistral 7B, Phi-2, etc.) that are too large for
the `transformers` backend's memory footprint in float32.

### Config schema

```yaml
backend: llama-cpp
model: /absolute/path/to/model.gguf   # local file, not a HF model ID
requests: 10
warmup_requests: 2
prompts_file: data/prompts/smoke.txt

llama_cpp:
  n_ctx: 2048         # context window (tokens); smaller = less VRAM for KV cache
  n_gpu_layers: 0     # 0 = CPU only; positive = layers offloaded to GPU; -1 = all
  max_tokens: 50      # max output tokens per request
  temperature: 0.0    # 0.0 = greedy/deterministic
  n_threads: null     # null = llama.cpp auto-detect; set to core count for best CPU perf
  verbose: false      # true prints llama.cpp model load progress
```

### Install

CPU-only (pure Python wheel, fast install):
```bash
uv sync --extra llama-cpp
```

GPU with CUDA (compiles from source, takes 2–5 min):
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp
```

### VRAM budget for RTX 3050 4 GB

| Model | Quantization | GGUF size | Peak VRAM (observed) | n_gpu_layers for 4 GB |
|-------|-------------|-----------|---------------------|----------------------|
| Llama 3.2 3B | Q4\_K\_M | ~1.9 GB | **2361 MiB** (measured) | 99 (all 28 layers) |
| Llama 3.2 3B | Q8\_0 | ~3.2 GB | **3697 MiB** (measured) | 99 (all 28 layers, 399 MiB headroom) |
| Llama 3 8B | Q4\_K\_M | ~4.7 GB | — | 20–24 (partial) |
| Llama 3 8B | Q4\_K\_S | ~4.3 GB | — | 24–28 (partial) |
| Mistral 7B | Q4\_K\_M | ~4.1 GB | — | 28–32 (most/all) |
| TinyLlama 1.1B | Q4\_K\_M | ~0.65 GB | — | 22 (all layers) |

Start with `n_gpu_layers: 0` to confirm the model loads, then increase until VRAM is
nearly full (leaving ~300–500 MB headroom for KV cache and OS).

### CUDA memory note

`peak_cuda_memory_mb` in the CSV will be `0.0` for llama-cpp GPU runs. llama.cpp manages
VRAM through its own allocator, not via `torch.cuda`. To measure VRAM usage, check
`nvidia-smi` while the benchmark is running, or read the `ggml_metal`/`ggml_cuda` log
lines from `verbose: true`.

### Result files

```
results/
├── llama-cpp-cpu.csv
├── llama-cpp-cpu.manifest.json
├── llama-cpp-gpu.csv
└── llama-cpp-gpu.manifest.json
```

The `results/` directory is gitignored. Curated reports go to `docs/results/` after
a run is validated.

## Run Matrix

The `matrix` subcommand runs multiple benchmark configurations sequentially from a single YAML
file and writes one CSV + manifest per run into a shared results directory.

```yaml
# configs/matrix-example.yaml
results_dir: results

runs:
  - name: mock-short-chat
    config: configs/example.yaml
    workload_profile: short_chat

  - name: mock-summarization
    config: configs/example.yaml
    workload_profile: summarization
```

Run the matrix:
```bash
llm-bench matrix --config configs/matrix-example.yaml
# Preview without running:
llm-bench matrix --config configs/matrix-example.yaml --dry-run
```

After completion, compare all runs in one table:
```bash
llm-bench compare results/*.csv --sort p95
```

**Output structure:**
```
results/
├── mock-short-chat.csv
├── mock-short-chat.manifest.json
├── mock-summarization.csv
└── mock-summarization.manifest.json
```

The `results/` directory is gitignored — only configs and fixtures live in source control.

## Workload Profiles

Named profiles standardize the prompt set across experiment runs so backend and parameter
comparisons are apples-to-apples. Select a profile with `workload_profile:` in the YAML config.

| Profile | Input | Output | Use when |
|---------|-------|--------|----------|
| `short_chat` | short | short | Benchmarking interactive latency (chat, QA) |
| `summarization` | medium | short | Comparing models on read-heavy tasks |
| `code_completion` | short | medium | Benchmarking code generation throughput |
| `long_context_smoke` | long | short | Stressing the prefill pass; context window capacity |

**Quick start:**
```bash
uv run llm-bench --config configs/profile-short-chat.yaml --output results/short-chat.csv
uv run llm-bench --config configs/profile-summarization.yaml --output results/summarization.csv
```

Backward compatibility: `prompts_file:` continues to work unchanged. If `workload_profile:` is
set, it takes precedence and points to a committed prompt fixture under `data/prompts/`.

## Reproducing a Run

Pass `--manifest <path>.json` to save a full environment snapshot alongside the CSV:

```bash
uv run llm-bench --config configs/transformers-cpu.yaml \
    --output results.csv \
    --manifest manifest.json
```

The manifest contains everything needed to reproduce or compare runs:

| Field | Description |
|-------|-------------|
| `git_commit` | SHA of HEAD at run time; `null` if not in a git repo |
| `git_dirty` | `true` if the working tree had uncommitted changes |
| `config_sha256` | SHA256 of the raw YAML config bytes |
| `prompts_sha256` | SHA256 of the raw prompts file bytes |
| `python_version` | Full `sys.version` string |
| `platform_info` | OS and kernel from `platform.platform()` |
| `cpu_model` | CPU model name from `/proc/cpuinfo` (Linux) or `platform.processor()` |
| `cpu_count` | Logical CPU count from `os.cpu_count()` |
| `package_version` | `llm-inference-benchmark` package version |
| `torch_version` | `torch` version, or `null` if not installed |
| `transformers_version` | `transformers` version, or `null` if not installed |
| `psutil_version` | `psutil` version |
| `gpu` | Nested object with GPU fingerprint, or `null` when no GPU is detected |
| `gpu.name` | GPU model name from `nvidia-smi`, or `null` |
| `gpu.driver_version` | NVIDIA driver version from `nvidia-smi`, or `null` |
| `gpu.cuda_version` | CUDA version reported by the driver, or `null` |
| `gpu.vram_total_mb` | Total VRAM in MiB from `nvidia-smi`, or `null` |
| `gpu.torch_cuda_available` | `torch.cuda.is_available()` result, or `null` if torch absent |
| `gpu.torch_cuda_device_name` | `torch.cuda.get_device_name(0)` when CUDA is available, else `null` |

All `gpu.*` fields are `null` when their source (`nvidia-smi` or `torch`) is unavailable.
The `gpu` object itself is `null` when neither source provides any information.

**To verify a reproduced run matches the original**: compare `config_sha256` and
`prompts_sha256`. If they match, the config and prompts are byte-identical.
If `git_dirty` was `true` at run time, the working tree had changes not captured
in `git_commit` — results may not be fully reproducible from that commit alone.

---

## Quantization Comparison Methodology

Comparing quantizations of the same model measures the accuracy–memory–speed trade-off
inherent in post-training quantization. This section defines the methodology used in
curated quantization comparison reports in `docs/results/`.

### What is quantization?

GGUF quantization reduces model weight precision to decrease file size and VRAM footprint.
Lower precision = less VRAM and faster inference, but potentially lower output quality.
Common GGUF quantization formats compared in this benchmark:

| Format | Bits/weight (approx) | File size (Llama 3.2 3B) | Typical use |
|--------|---------------------|--------------------------|-------------|
| Q4_K_M | ~4.5 bits | ~1.9 GB | Fast inference on 4–8 GB VRAM; good quality–speed trade-off |
| Q8_0 | 8 bits | ~3.4 GB | Near-lossless; reference baseline for quality comparisons |

Q8_0 is the practical upper bound for GGUF quality — weights are stored at 8-bit
precision with minimal quantization error. Q4_K_M uses K-quant methods that apply
different precision to different weight groups to maintain output quality while
achieving ~2× compression vs Q8_0.

### Comparison protocol

Runs are considered comparable when:

1. **Same model family and parameter count** — e.g., both `Llama-3.2-3B-Instruct`.
2. **Same prompt fixture** — same `prompts_file` or `workload_profile` in both configs,
   ensuring identical input tokens.
3. **Same generation parameters** — `max_tokens`, `temperature`, `n_ctx` identical
   across runs; `temperature: 0.0` (greedy) for deterministic output.
4. **Best stable GPU offload per quantization** — use the highest `n_gpu_layers` that
   fits within the VRAM budget for each quantization, leaving ≥300 MiB headroom.
   If both fit at `n_gpu_layers=99` (all layers), use the same value.
   If they differ, document the constraint explicitly — do not force identical
   `n_gpu_layers` if one quantization would OOM.
5. **Produced by the repository CLI** — `uv run llm-bench matrix --config ...`.

### VRAM budget estimation for llama-cpp

From the n_gpu_layers sweep evidence on RTX 3050 with Llama 3.2 3B Q4_K_M:

- CUDA-init baseline: **~655 MiB** (CUDA context + workspace; present even at `n_gpu_layers=0`)
- Per-layer VRAM (Q4_K_M, 28-layer model): **~60 MiB/layer**
- Per-layer VRAM (Q8_0, estimated): **~107 MiB/layer** (Q8_0 is ~1.78× denser than Q4_K_M)

To estimate safe `n_gpu_layers` for a new quantization on a 4 GB card:

```
available = total_vram_mib - cuda_baseline_mib - headroom_mib
             = 4096 - 655 - 300 = ~3141 MiB
max_layers = floor(available / per_layer_mib)
```

These are estimates. Probe with `n_gpu_layers=99` first; if context creation fails with
`ValueError: Failed to create llama_context`, reduce in steps of 4 until stable.

### What this comparison does and does not measure

| Measured | Not measured |
|----------|-------------|
| Inference latency (p50, p95) | Output quality / perplexity |
| Throughput (tok/s) | Accuracy on downstream tasks |
| Peak VRAM (`peak_vram_memory_mb`) | Long-context behaviour |
| Peak CPU RSS (`peak_cpu_memory_mb`) | Batch or concurrent request throughput |

Quality evaluation (perplexity, benchmark accuracy) is a planned addition to the roadmap.
The quantization comparison in this project reports speed and memory only — the implicit
assumption is that Q8_0 serves as a high-quality reference and Q4_K_M trades a small
accuracy loss for ~2× smaller VRAM footprint and proportionally faster inference.

### Run command

```bash
# Set model: path in each config, then:
CUDA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name "*.so*" \
  | xargs -I{} dirname {} | sort -u | tr '\n' ':')
LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" \
  uv run llm-bench matrix --config configs/llama-cpp-quant-compare.yaml

uv run llm-bench compare results/quant-q4km.csv results/quant-q8.csv
```
