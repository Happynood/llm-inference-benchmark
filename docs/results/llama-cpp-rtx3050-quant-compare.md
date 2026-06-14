# llama.cpp — Quantization Comparison: Q4\_K\_M vs Q8\_0 (Llama 3.2 3B, RTX 3050)

Comparison of two GGUF quantization formats for the same model, same prompts, and same
GPU offload configuration. Produced via the repository CLI:

```bash
CUDA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name "*.so*" \
  | xargs -I{} dirname {} | sort -u | tr '\n' ':')
LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" \
  uv run llm-bench matrix --config configs/llama-cpp-quant-compare.yaml

uv run llm-bench compare results/quant-q4km.csv results/quant-q8.csv
```

## Hardware

| Component | Details |
|-----------|---------|
| CPU | Intel Core i5-11400H @ 2.70 GHz, 12 logical cores |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4096 MiB total |
| Driver | CUDA 13.x (no nvcc; cu124 wheel used) |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.12.13 |
| llama-cpp-python | 0.3.29 (pre-built cu124 wheel) |
| nvidia-cuda-runtime-cu12 | 12.x |
| nvidia-cublas-cu12 | 12.9.2.10 |
| OS | Linux 7.0.0-22-generic x86-64 |

## Model

| Field | Q4\_K\_M | Q8\_0 |
|-------|----------|-------|
| Base model | Llama-3.2-3B-Instruct | Llama-3.2-3B-Instruct |
| Quantization | Q4\_K\_M (~4.5 bits/weight) | Q8\_0 (8 bits/weight) |
| GGUF file size | ~1.9 GB | ~3.2 GB |
| Architecture | 28 layers, same for both | 28 layers, same for both |
| Source | `bartowski/Llama-3.2-3B-Instruct-GGUF` | `bartowski/Llama-3.2-3B-Instruct-GGUF` |

Q8_0 stores weights at full 8-bit precision — effectively lossless vs float16 for most tasks.
Q4_K_M applies K-quant compression (~4.5 bits/weight on average) to reduce size by ~1.7×
while preserving most output quality.

## Benchmark Configs

| Run | Config file | Quantization | `n_gpu_layers` |
|-----|-------------|--------------|----------------|
| quant-q4km | `configs/llama-cpp-q4km-best.yaml` | Q4\_K\_M | 99 (all 28 layers) |
| quant-q8 | `configs/llama-cpp-q8-best.yaml` | Q8\_0 | 99 (all 28 layers) |

Shared parameters: `requests=10`, `warmup_requests=2`, `n_ctx=512`, `max_tokens=50`,
`temperature=0.0` (greedy/deterministic), `prompts_file=data/prompts/smoke.txt`.

Both quantizations fit at `n_gpu_layers=99` (all 28 layers on GPU):
- Q4\_K\_M: 2361 MiB (57.6% of 4096 MiB) — comfortable headroom
- Q8\_0: 3697 MiB (90.2% of 4096 MiB) — tight but stable; probed successful before the run

## CLI Output — `llm-bench compare`

```
| Backend   | Model                                  | N  | p50 (ms) | p95 (ms) | tok/s | CPU mem (MB) | CUDA mem (MB) | VRAM mem (MB) |
|-----------|----------------------------------------|----|----------|----------|-------|--------------|---------------|---------------|
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf | 10 | 929.77   | 960.76   | 53.6  | 1277.9       | 0.0           | 2361.0        |
| llama-cpp | .../Llama-3.2-3B-Instruct-Q8_0.gguf   | 10 | 1194.74  | 1198.34  | 41.9  | 1412.5       | 0.0           | 3697.0        |
```

`CUDA mem` is always `0.0` for llama.cpp — it uses its own CUDA allocator, not PyTorch's.
`VRAM mem` is `peak_vram_memory_mb` (nvidia-smi polling at 500 ms).

## Results

| Quantization | p50 (ms) | p95 (ms) | tok/s | CPU mem (MB) | Peak VRAM (MiB) |
|---|---|---|---|---|---|
| Q4\_K\_M (n\_gpu\_layers=99) | **929.77** | **960.76** | **53.56** | 1277.9 | **2361** |
| Q8\_0 (n\_gpu\_layers=99) | 1194.74 | 1198.34 | 41.85 | 1412.5 | 3697 |

## Interpretation

### Speed

Q4\_K\_M is **1.28× faster** than Q8\_0 at the same GPU offload configuration:

| Metric | Q4\_K\_M | Q8\_0 | Ratio (Q4K / Q8) |
|--------|----------|-------|-----------------|
| p50 latency | 929.77 ms | 1194.74 ms | **1.28× faster** |
| tok/s | 53.56 | 41.85 | **1.28× higher** |

This speedup is driven by **memory bandwidth**, not compute. At GPU inference speeds,
the bottleneck is not arithmetic — it is how fast weight data can be loaded from VRAM
into GPU compute cores per token. Q4_K_M weights are ~1.78× more compact than Q8_0,
so fewer bytes per weight are read per forward pass. The smaller transfer cost directly
translates to lower latency and higher throughput.

This is the fundamental reason quantization speeds up LLM inference on GPU: fewer bits
per weight = fewer bytes transferred per forward pass = faster.

### Memory

| Quantization | VRAM used | VRAM free | % of 4 GB |
|---|---|---|---|
| Q4\_K\_M | 2361 MiB | 1735 MiB | 57.6% |
| Q8\_0 | 3697 MiB | 399 MiB | 90.2% |

Q8\_0 leaves only 399 MiB free VRAM on the RTX 3050. This is enough for 10-request
benchmark runs at `n_ctx=512`, but tight for interactive use (longer conversations,
larger context windows, or concurrent processes competing for VRAM). Q4\_K\_M leaves
1735 MiB headroom, making it substantially more practical on a 4 GB card.

### Latency consistency

| Quantization | p95 / p50 | Interpretation |
|---|---|---|
| Q4\_K\_M | 1.033 (3.3%) | Tight; slight scheduling variance visible |
| Q8\_0 | 1.003 (0.3%) | Extremely tight; near-deterministic GPU execution |

Q8\_0 shows tighter p95/p50 consistency than Q4\_K\_M (0.3% vs 3.3%). This is likely
because Q8\_0's forward pass is more purely compute-bound (simple uniform 8-bit loads)
while Q4\_K\_M's K-quant dequantization adds a small variable-cost step. Neither
variance is operationally significant for this workload size.

### Summary

| Dimension | Winner | Magnitude |
|-----------|--------|-----------|
| Speed (tok/s) | Q4\_K\_M | 1.28× faster |
| VRAM footprint | Q4\_K\_M | 1.57× smaller (2361 vs 3697 MiB) |
| Latency consistency | Q8\_0 | marginally (0.3% vs 3.3% p95/p50) |
| Output quality (not measured) | Q8\_0 | near-lossless vs ~4.5-bit compression |

**Recommendation for a 4 GB laptop GPU**: Q4\_K\_M is the better practical choice — it
is faster, leaves more VRAM headroom for larger contexts or concurrent workloads, and
is the standard "quality/speed sweet spot" quantization. Use Q8\_0 only when output
quality is a hard requirement and 3.7 GB of dedicated VRAM is confirmed available.

## Limitations

- Single machine, single session: no statistical replication across reboots or days.
- N=10 requests: p95 is the worst of 10 observations, not a stable tail-latency estimate.
- Output quality not evaluated. The recommendation above assumes Q8\_0 is higher quality
  than Q4\_K\_M — this is the conventional expectation for K-quant vs Q8_0, but is not
  verified by this benchmark.
- `peak_cuda_memory_mb = 0.0` for all llama-cpp runs. Use `peak_vram_memory_mb`.
- Q8\_0 at 3697 MiB leaves only 399 MiB VRAM headroom. Other GPU processes running
  concurrently could cause OOM at n\_gpu\_layers=99.
- Prompts from `data/prompts/smoke.txt` are short (5 prompts, cycling). Longer prompts
  or larger `max_tokens` would increase KV cache pressure and may tighten the Q8\_0
  VRAM margin further.
