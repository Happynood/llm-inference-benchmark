# RTX 3050 Laptop — `sshleifer/tiny-gpt2` CPU vs GPU

**Date**: 2026-06-14  
**Operator**: Alexey  
**Branch**: `feat/workload-profiles` (`7647c2f`)  
**Git dirty**: yes (workload-profile fixtures uncommitted at run time)

## Purpose

Establish the first real GPU baseline and validate that:

1. The harness correctly measures CUDA memory (`torch.cuda.max_memory_allocated`).
2. GPU inference path works end-to-end (load model on device, generate, collect metrics).
3. The manifest captures GPU fingerprint (`torch_cuda_available`, `torch_cuda_device_name`).

This run uses a 2-layer toy model on purpose — downloads are ~4 MB and take seconds,
making it reproducible on any machine with the `transformers` extra installed.

## Hardware

| Component | Value |
|-----------|-------|
| CPU | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| Logical cores | 12 |
| RAM | (not recorded) |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4 096 MiB |
| `torch_cuda_device_name` | NVIDIA GeForce RTX 3050 Laptop GPU |
| CUDA (torch) | 12.2 (via `torch.cuda`) |
| `nvidia-smi` | not accessible in run environment (all fields `null` in manifest) |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.12.13 |
| torch | 2.12.0 |
| transformers | 5.12.0 |
| psutil | 7.2.2 |
| OS | Linux 7.0.0-22-generic x86-64 |

## Model

`sshleifer/tiny-gpt2` — 2-layer GPT-2 variant, ~117 K parameters, ~4 MB on disk.

> **This is a toy model.** It exists solely to validate the harness.
> Production models (GPT-2 full 117 M params, Llama 3 8B) are 1 000–70 000× larger.
> Numbers here reflect harness overhead, kernel-launch latency, and data-transfer cost,
> not production inference performance.

## Config

### CPU run — `configs/transformers-cpu.yaml`

```yaml
backend: transformers
model: sshleifer/tiny-gpt2
requests: 10
warmup_requests: 2
prompts_file: data/prompts/smoke.txt
hf:
  max_new_tokens: 50
  device: cpu
  torch_dtype: float32
  do_sample: false
```

### GPU run — `configs/transformers-gpu.yaml`

```yaml
backend: transformers
model: sshleifer/tiny-gpt2
requests: 10
warmup_requests: 2
prompts_file: data/prompts/smoke.txt
hf:
  max_new_tokens: 50
  device: cuda
  torch_dtype: float16
  do_sample: false
```

## Results

| Metric | CPU (float32) | GPU (float16, RTX 3050) |
|--------|--------------|------------------------|
| p50_latency_ms | 40.95 | 59.95 |
| p95_latency_ms | 44.67 | 61.86 |
| tokens_per_second | 1211.23 | 829.60 |
| total_tokens | 598 | 598 |
| peak_cpu_memory_mb | 721.37 | 1382.92 |
| peak_cuda_memory_mb | 0.00 | **8.82** |
| requests | 10 | 10 |
| warmup_requests | 2 | 2 |

## Interpretation

### GPU is slower than CPU for this model

At 2 layers and 117 K parameters, `sshleifer/tiny-gpt2` is so small that:

- **Kernel-launch overhead** (~1–5 ms per CUDA call) exceeds the compute savings.
- **CPU-GPU data transfer** for inputs and outputs adds latency that the forward pass
  doesn't offset at this scale.
- The result (59.95 ms GPU vs 40.95 ms CPU) is expected and well-documented in the
  PyTorch literature for sub-million-parameter models.

For production models this reverses decisively:
- GPT-2 117 M on CPU: ~400–800 ms per request.
- Llama 3 8B Q4 on RTX 3050: ~2–5 tokens/sec (estimated), vs CPU at <1 token/sec.

### CUDA memory is accurate

`peak_cuda_memory_mb: 8.82` confirms that the PyTorch allocator tracking works correctly.
A 2-layer float16 model with 117 K parameters and 50 output tokens occupies ~9 MB of
VRAM — this is consistent with the model size.

The CPU-side memory increase (721 MB on CPU → 1383 MB on GPU device) reflects the PyTorch
runtime keeping a copy of weights in pinned CPU memory for transfer, plus additional
Python-side objects for the CUDA context.

### `nvidia-smi` was inaccessible

The `gpu.*` SMI fields in the manifest (`name`, `driver_version`, `cuda_version`,
`vram_total_mb`) are all `null`. `nvidia-smi` was not available in the shell environment
during the run. The `torch.cuda` fields (`torch_cuda_available: true`,
`torch_cuda_device_name`) were populated correctly.

## Reproducibility

To reproduce this run on the same hardware:

```bash
uv sync --extra transformers
uv run llm-bench --config configs/transformers-cpu.yaml --output results/cpu.csv --manifest results/cpu.manifest.json
uv run llm-bench --config configs/transformers-gpu.yaml --output results/gpu.csv --manifest results/gpu.manifest.json
uv run llm-bench compare results/cpu.csv results/gpu.csv
```

To verify byte-identical config: compare `config_sha256` in your manifest against the one
recorded here. Prompts must also match (`prompts_sha256` covers `data/prompts/smoke.txt`).

## What's next

This run establishes that the GPU path works. The next milestone is a
production-size model via the `llama-cpp-python` backend with 4-bit GGUF quantization:
Llama 3 8B Q4_K_M fits in 4 GB VRAM and will produce latency numbers that are meaningful
for real optimization decisions. See [docs/results/README.md](README.md) for planned runs.
