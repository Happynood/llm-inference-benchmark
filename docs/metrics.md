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
| `peak_cuda_memory_mb` | MB | Peak CUDA memory allocated during the run; empty when no GPU |
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

**Limitation**: measures PyTorch allocator-tracked memory, not raw VRAM at the driver level.
Reserved-but-free memory pages are not reflected in this number.

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
