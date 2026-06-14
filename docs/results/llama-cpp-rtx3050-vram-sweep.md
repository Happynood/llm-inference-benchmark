# llama.cpp — RTX 3050 — n\_gpu\_layers Sweep (Llama 3.2 3B Q4\_K\_M)

Three-point sweep across `n_gpu_layers` values to quantify the latency, throughput,
and VRAM trade-off from CPU-only to full GPU offload on a 4 GB laptop GPU.

Run via the repository CLI:

```bash
CUDA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name "*.so*" \
  | xargs -I{} dirname {} | sort -u | tr '\n' ':')
LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" \
  uv run llm-bench matrix --config configs/llama-cpp-vram-sweep.yaml

uv run llm-bench compare \
  results/sweep-gpu0.csv results/sweep-gpu20.csv results/sweep-gpu99.csv
```

## Hardware

| Component | Details |
|-----------|---------|
| CPU | Intel Core i5-11400H @ 2.70 GHz, 12 logical cores |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4096 MiB total |
| Driver | CUDA 13.x (no nvcc; cu124 wheel used for CUDA support) |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.12.13 |
| llama-cpp-python | 0.3.29 (pre-built cu124 wheel) |
| nvidia-cuda-runtime-cu12 | 12.x |
| nvidia-cublas-cu12 | 12.9.2.10 |
| OS | Linux 7.0.0-22-generic x86-64 |

## Model

| Field | Value |
|-------|-------|
| Model | Llama-3.2-3B-Instruct |
| Quantization | Q4\_K\_M |
| File | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~1.9 GB) |
| Architecture | Llama 3.2 transformer, **28 layers** |
| Source | `bartowski/Llama-3.2-3B-Instruct-GGUF` on Hugging Face |

## Benchmark Configs

| Run | Config file | `n_gpu_layers` |
|-----|-------------|----------------|
| sweep-gpu0 | `configs/llama-cpp-sweep-gpu0.yaml` | 0 (CPU only) |
| sweep-gpu20 | `configs/llama-cpp-sweep-gpu20.yaml` | 20 (20/28 layers on GPU) |
| sweep-gpu99 | `configs/llama-cpp-sweep-gpu99.yaml` | 99 (all 28 layers on GPU) |

Shared parameters: `requests=10`, `warmup_requests=2`, `n_ctx=512`, `max_tokens=50`,
`temperature=0.0` (greedy/deterministic), `prompts_file=data/prompts/smoke.txt`.

## CLI Output — `llm-bench compare`

```
| Backend   | Model                                      | N  | p50 (ms) | p95 (ms) | tok/s | CPU mem (MB) | CUDA mem (MB) | VRAM mem (MB) |
|-----------|--------------------------------------------|----|----------|----------|-------|--------------|---------------|---------------|
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf     | 10 | 970.65   | 984.31   | 51.4  | 1359.3       | 0.0           | 2361.0        |
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf     | 10 | 1357.39  | 1420.41  | 36.6  | 2453.0       | 0.0           | 1829.0        |
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf     | 10 | 2836.98  | 3092.79  | 17.5  | 2862.9       | 0.0           | 655.0         |
```

The compare table is sorted by p95 (fastest first). Rows correspond to `sweep-gpu99`,
`sweep-gpu20`, and `sweep-gpu0` respectively.

`CUDA mem (MB)` is the PyTorch allocator peak — always `0.0` for llama.cpp which uses
its own CUDA allocator. `VRAM mem (MB)` is the driver-level peak from `peak_vram_memory_mb`
(nvidia-smi polled at 500 ms intervals).

## Results (ordered by `n_gpu_layers`)

| `n_gpu_layers` | Layers on GPU | p50 (ms) | p95 (ms) | tok/s | CPU mem (MB) | Peak VRAM (MiB) |
|----------------|---------------|----------|----------|-------|--------------|-----------------|
| 0 | 0 / 28 (CPU only) | 2836.98 | 3092.79 | 17.52 | 2862.9 | 655 |
| 20 | 20 / 28 (partial) | 1357.39 | 1420.41 | 36.59 | 2453.0 | 1829 |
| 99 | 28 / 28 (full) | **970.65** | **984.31** | **51.44** | 1359.3 | **2361** |

## Interpretation

### Latency and throughput

| Comparison | p50 speedup | tok/s gain |
|------------|-------------|------------|
| 0 → 20 layers | **2.09×** | 36.59 / 17.52 = 2.09× |
| 0 → 99 layers | **2.92×** | 51.44 / 17.52 = 2.94× |
| 20 → 99 layers | **1.40×** | 51.44 / 36.59 = 1.41× additional |

Partial offload (20/28 layers) delivers ~71% of the full-offload speedup at 78% of the
VRAM cost. Diminishing returns apply: moving from 20 to 28 GPU layers adds only 1.40× more
speed for an additional 532 MiB of VRAM (1829 → 2361 MiB).

Full offload remains the best choice when the model fits: 2361 / 4096 MiB = 57.6% VRAM
utilized, leaving ~1735 MiB for the KV cache and other processes.

### VRAM scaling

VRAM grows linearly with GPU layer count after an initial baseline:

| `n_gpu_layers` | Peak VRAM | vs baseline | MiB / layer |
|----------------|-----------|-------------|-------------|
| 0 (baseline) | 655 MiB | — | — |
| 20 | 1829 MiB | +1174 MiB | ~58.7 MiB |
| 28 | 2361 MiB | +1706 MiB | ~60.9 MiB |

**Baseline VRAM at n\_gpu\_layers=0 (655 MiB):** Even with no inference layers on GPU,
the CUDA-compiled llama-cpp-python wheel initializes the CUDA backend unconditionally.
This allocates a CUDA context, GGML compute-graph workspace, and scratch buffers totalling
~655 MiB. This is not CPU-only VRAM usage — it is the llama.cpp CUDA init overhead that
exists regardless of how many layers are offloaded.

The per-layer estimate (~60 MiB/layer) can be used to approximate safe `n_gpu_layers` for
any GGUF model: divide the available VRAM budget (after subtracting the ~655 MiB baseline)
by the observed per-layer size.

### CPU memory

CPU RSS decreases as more layers move to GPU:

| `n_gpu_layers` | CPU mem (MB) |
|----------------|--------------|
| 0 | 2862.9 |
| 20 | 2453.0 |
| 99 | 1359.3 |

At full GPU offload, model weights live in VRAM; CPU RSS covers the tokenizer, KV cache
data structures, and Python process overhead (~1.4 GB).

### Latency consistency

| `n_gpu_layers` | p95 / p50 | Interpretation |
|----------------|-----------|----------------|
| 0 | 1.090 (9.0%) | Memory-bandwidth-bound; OS scheduling jitter visible |
| 20 | 1.046 (4.6%) | Mixed CPU+GPU; lower variance than CPU-only |
| 99 | 1.014 (1.4%) | Compute-bound on GPU; very tight latency distribution |

Full GPU offload is not only fastest but also most consistent — the p95/p50 ratio drops
from 9% to 1.4%.

### VRAM budget summary

| Config | VRAM used | VRAM free | Fits in 4 GB? |
|--------|-----------|-----------|---------------|
| n\_gpu\_layers=0 | 655 MiB | 3441 MiB | ✓ |
| n\_gpu\_layers=20 | 1829 MiB | 2267 MiB | ✓ |
| n\_gpu\_layers=99 | 2361 MiB | 1735 MiB | ✓ |

## Infrastructure note: CLI fix applied in this branch

The matrix command had a bug where the previous run's backend was not explicitly released
before the next run started. For CUDA-backed runs, this could leave GPU memory allocated
during the subsequent context creation, causing `ValueError: Failed to create llama_context`
when the second run required more VRAM than was available after the first.

Fix: `del backend; gc.collect()` after each run's CSV and manifest are written in
`matrix_cmd` (see `src/llm_inference_benchmark/cli.py`). The sweep in this report was
produced after that fix was applied.

## Limitations

- Single machine, single session: no statistical replication across reboots or days.
- N=10 requests per config point: p95 is the worst of 10 observations, not a stable
  tail-latency estimate. Use N≥30 for reliable tail estimates.
- VRAM polled at 500 ms intervals: short-lived spikes shorter than the polling interval
  may be missed.
- VRAM reflects total driver-visible usage across all processes sharing the GPU, not the
  benchmark process alone. Other GPU workloads running concurrently will inflate the reading.
- `peak_cuda_memory_mb` is `0.0` for all llama-cpp runs (PyTorch allocator is not used).
  Use `peak_vram_memory_mb` for llama-cpp VRAM accounting.
- The 655 MiB CUDA baseline at n\_gpu\_layers=0 is specific to the CUDA-compiled cu124
  wheel. A CPU-only llama-cpp-python wheel would show ~11 MiB (idle VRAM only).
- Output quality not evaluated; throughput and latency only.
