# llama.cpp — RTX 3050 — Qwen2.5-Coder 1.5B Instruct Q5\_K\_M

Benchmark of a compact code-generation model under llama.cpp on a 4 GB laptop GPU.
All layers fit comfortably in VRAM, leaving headroom for larger context windows.

## Hardware

| Component | Details |
|-----------|---------|
| CPU | Intel Core i5-11400H @ 2.70 GHz, 12 logical cores |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4096 MiB total |
| Driver | 595.71.05 · CUDA 13.2 |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.12.13 |
| llama-cpp-python | 0.3.x (pre-built cu124 wheel) |
| nvidia-cuda-runtime-cu12 | 12.x (`libcudart.so.12` compat layer) |
| nvidia-cublas-cu12 | 12.x (`libcublas.so.12` compat layer) |
| OS | Linux 7.0.0-22-generic x86-64 |

## Model

| Field | Value |
|-------|-------|
| Model | Qwen2.5-Coder-1.5B-Instruct |
| Quantization | Q5\_K\_M |
| File | `qwen2.5-coder-1.5b-instruct-q5_k_m.gguf` |
| Source | local `~/models/` |

## Benchmark Config

| Parameter | Value |
|-----------|-------|
| `requests` | 10 |
| `warmup_requests` | 2 |
| `n_ctx` | 512 |
| `max_tokens` | 100 |
| `temperature` | 0.0 (greedy / deterministic) |
| `n_gpu_layers` | 99 (all layers on GPU) |
| `seed` | 42 |

Prompts: `data/prompts/code_completion.txt` — 10 code-completion questions.

## Results

| Metric | Value |
|--------|-------|
| p50 latency | 1500.33 ms |
| p95 latency | 1510.82 ms |
| tokens/sec | **66.81** |
| decode tok/s | 66.81 |
| total tokens | 1182 |
| mean output tokens | 100 |
| mean input tokens | 18.2 |
| Peak VRAM | **1503 MiB** (36.7% of 4096 MiB) |
| Peak CPU RSS | 1294 MiB |
| Sanity pass rate | 100% |
| Model load time | 5308 ms |

## Interpretation

- **1503 MiB VRAM** — leaves 2.5 GB free on the RTX 3050; two concurrent instances or
  a much larger context window would still fit within the 4 GB budget.
- **66.8 tok/s** is fast enough for real-time code completion with 100-token outputs
  completing in ~1.5 seconds.
- **Latency is very consistent**: p95/p50 = 1510/1500 = **1.007** (0.7% spread),
  indicating the GPU is compute-bound with negligible scheduling jitter.
- Compared to Llama-3.2-3B Q4\_K\_M (full offload, 50-token runs: 53.7 tok/s, 2361 MiB),
  this 1.5B model delivers higher throughput at 64% of the VRAM cost — useful when
  VRAM is the binding constraint.

## Reproduction

```bash
SITELIB=.venv/lib/python3.12/site-packages
CUDA_LIBS=$(find "$SITELIB/nvidia" -name "*.so*" | xargs -I{} dirname {} | sort -u | tr '\n' ':')

LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" \
  uv run llm-bench \
    --config configs/llama-cpp-gpu.yaml \
    --output results/qwen25-coder-1.5b-q5km.csv \
    --seed 42
```

Set `model:` in the config to your local GGUF path and `n_gpu_layers: 99`.

## Limitations

- Single machine, one session — no statistical replication across days or reboots.
- p95 at N=10 is the worst of 10 observations; increase `requests` to 30+ for stable tail
  latency estimates.
- Peak VRAM measured via `nvidia-smi` polling (500 ms interval); very short spikes may be
  missed. `peak_cuda_memory_mb` in the CSV is 0.0 because llama-cpp uses its own allocator.
- Output quality not evaluated (no task rubric applied).
- CUDA 12 compat layer (`nvidia-cublas-cu12` alongside the CUDA 13 driver) works on this
  hardware but is not an officially supported configuration.
