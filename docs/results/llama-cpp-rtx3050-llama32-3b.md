# llama.cpp — RTX 3050 — Llama 3.2 3B Instruct Q4\_K\_M

CPU vs GPU inference benchmark using llama-cpp-python on a laptop GPU with a real
production-size quantized model.

## Hardware

| Component | Details |
|-----------|---------|
| CPU | Intel Core i5-11400H @ 2.70 GHz, 12 logical cores |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 3772 MiB total |
| Driver | CUDA 13.x driver, no nvcc (no CUDA toolkit installed) |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.12.13 |
| llama-cpp-python | 0.3.29 (pre-built cu124 wheel) |
| nvidia-cuda-runtime-cu12 | 12.x (cu12 `libcudart.so.12`) |
| nvidia-cublas-cu12 | 12.9.2.10 (cu12 `libcublas.so.12`) |
| OS | Linux 7.0.0-22-generic x86-64 |

## Model

| Field | Value |
|-------|-------|
| Model | Llama-3.2-3B-Instruct |
| Quantization | Q4\_K\_M |
| File | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~1.9 GB) |
| Architecture | Llama 3.2 transformer, **28 layers** |
| Source | `bartowski/Llama-3.2-3B-Instruct-GGUF` on Hugging Face |

## Benchmark Config

| Parameter | Value |
|-----------|-------|
| `requests` | 10 |
| `warmup_requests` | 2 |
| `n_ctx` | 512 |
| `max_tokens` | 50 |
| `temperature` | 0.0 (greedy / deterministic) |
| `n_gpu_layers` (CPU run) | 0 (all layers on CPU) |
| `n_gpu_layers` (GPU run) | 99 (all 28 layers on CUDA0) |

Prompts: 10 distinct ML/AI questions (gradient descent, transformer architecture,
quantization trade-offs, attention mechanism, GGUF format, context windows, etc.).

## Results

| Metric | CPU (`n_gpu_layers=0`) | GPU — RTX 3050 (`n_gpu_layers=99`) |
|--------|----------------------|-------------------------------------|
| p50 latency | 2750.56 ms | **931.18 ms** |
| p95 latency | 2939.79 ms | **939.69 ms** |
| tokens/sec | 18.01 | **53.71** |
| requests | 10 | 10 |
| output tokens | 500 (50 × 10) | 500 (50 × 10) |
| Peak VRAM | — | **2361 MiB** |

**GPU speedup: 2.95× lower latency, 2.98× higher throughput.**

### Per-request latencies (sorted, ms)

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-----|---|---|---|---|---|---|---|---|---|---|
| CPU | 2688 | 2703 | 2712 | 2725 | 2748 | 2753 | 2778 | 2789 | 2830 | 3030 |
| GPU | 917 | 924 | 930 | 930 | 930 | 932 | 933 | 935 | 935 | 944 |

## Interpretation

- **GPU is 2.95× faster** for this 28-layer, 3B parameter model with Q4\_K\_M quantization.
  All layers fit on the RTX 3050 (28 layers assigned to CUDA0 at load time).
- **GPU latency is highly consistent**: p95/p50 = 939.69/931.18 = **1.009** (0.9% spread),
  indicating the GPU is compute-bound with minimal OS scheduling jitter between requests.
- **CPU latency shows more variance**: p95/p50 = 2939.79/2750.56 = **1.069** (6.9%), typical
  for memory-bandwidth-bound sequential CPU inference with OS scheduling effects.
- **2361 MiB peak VRAM** for a 1.9 GB model: model weights (~1900 MiB) + KV cache at
  `n_ctx=512` + CUDA workspace overhead. Leaves ~1.4 GB headroom on the 3772 MiB GPU.
- **53.7 tok/s** is practical for real-time interactive use — a 50-token response completes
  in under 1 second on GPU, versus ~2.8 seconds on CPU.
- This confirms the expected CPU→GPU speedup for production-size models. Compare with the
  `sshleifer/tiny-gpt2` result where GPU was *slower* due to kernel-launch overhead
  dominating a 2-layer, 117 K parameter toy model.

## Reproduction

### 1. Download the model

```python
import os
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    local_dir=os.path.expanduser("~/models"),
)
print(path)
```

### 2. Install llama-cpp-python with CUDA 12 support

The RTX 3050 has a CUDA 13.x driver but no nvcc (no CUDA toolkit), so the source build
fails. The pre-built cu124 wheel works via a CUDA 12 / 13 compatibility layer.

```bash
# Pre-built cu124 wheel (avoids nvcc build requirement)
uv pip install "llama-cpp-python>=0.2" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# CUDA 12 runtime and cuBLAS (required by the cu124 wheel)
uv pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

### 3. Set LD\_LIBRARY\_PATH and run

The cu124 wheel links against `libcudart.so.12` and `libcublas.so.12`. These are now
in the venv alongside the existing `libcublas.so.13` (from PyTorch cu13). Point
`LD_LIBRARY_PATH` at all `nvidia/*/lib/` directories in the venv:

```bash
SITELIB=.venv/lib/python3.12/site-packages
CUDA_LIBS=$(find "$SITELIB/nvidia" -name "*.so*" | xargs -I{} dirname {} | sort -u | tr '\n' ':')
LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" \
    uv run llm-bench --config configs/llama-cpp-gpu.yaml --output results/llama-cpp-gpu.csv
```

Edit `configs/llama-cpp-gpu.yaml` to set `model:` to your GGUF path. For this model,
set `n_gpu_layers: 99` (or `-1`) to offload all 28 layers to the GPU.

## Limitations

- **Single machine run, one session**: no statistical replication across reboots or days.
- **p95 at N=10** is the worst of 10 observations — not a stable statistical estimate.
  Increase `requests` to 30+ for more reliable tail latency estimates.
- **Peak VRAM via nvidia-smi polling** (500 ms interval): very short spikes (<500 ms)
  may be missed. The `peak_cuda_memory_mb` field in the CSV will be `0.0` because
  llama-cpp uses its own VRAM allocator, not PyTorch's.
- **Latency includes tokenization and sampling overhead**, not only GPU compute.
- **Output quality not evaluated**: only throughput and latency were measured.
- **CUDA library workaround** (`nvidia-cublas-cu12` alongside CUDA 13 driver) works on this
  hardware but is not an officially supported configuration. Verify on your own system.
- **Single batch size (1)**: sequential requests, one at a time. Concurrent inference not tested.
