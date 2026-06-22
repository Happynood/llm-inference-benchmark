# Quickstart

**Prerequisites**: Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/happynood/llm-inference-benchmark
cd llm-inference-benchmark
```

## Mock backend (no downloads, CI-safe)

```bash
uv sync
uv run llm-bench --config configs/example.yaml --output results/mock.csv
uv run llm-bench compare results/mock.csv
```

The mock backend sleeps for a configured `latency_ms` — no model is loaded.
Numbers validate that the measurement pipeline is wired correctly.

## Verify your setup

Run `llm-bench verify` after installation to confirm the harness is working and
to check which optional backends are available:

```bash
uv run llm-bench verify
```

Example output:

```
BACKEND        STATUS      LATENCY  NOTES
----------------------------------------------------
mock           OK            1.3 ms
transformers   OK               N/A  deps installed
llama-cpp      SKIP             N/A  missing: llama_cpp
openai         OK               N/A  stdlib only
onnx           SKIP             N/A  missing: optimum
vllm           SKIP             N/A  missing: vllm
```

- **OK (mock)** — the full inference pipeline ran end-to-end; the harness is wired correctly.
- **OK (others)** — the required Python packages are installed.
- **SKIP** — the optional dependency is not installed; install the relevant extra to enable it.
- **FAIL** — a dependency is present but something went wrong; check the NOTES column.

Use `--format json` for machine-readable output:

```bash
uv run llm-bench verify --format json
```

Run `verify` again after installing each new backend to confirm it is detected correctly.

## Transformers backend (CPU)

```bash
uv sync --extra transformers
uv run llm-bench --config configs/transformers-cpu.yaml --output results/hf-cpu.csv
```

Downloads `sshleifer/tiny-gpt2` (~4 MB) on first run.

## Transformers backend (GPU, CUDA)

```bash
uv sync --extra transformers
uv run llm-bench --config configs/transformers-gpu.yaml --output results/hf-gpu.csv
```

Requires a CUDA-capable GPU and a PyTorch build that matches your CUDA version.

## llama.cpp backend (CPU)

```bash
uv sync --extra llama-cpp
# Edit configs/llama-cpp-cpu.yaml: set model: /path/to/model.gguf
uv run llm-bench --config configs/llama-cpp-cpu.yaml --output results/llama-cpu.csv
```

Download GGUF models from [Hugging Face Hub](https://huggingface.co/models?library=gguf), e.g.:
```bash
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models/
```

## llama.cpp backend (GPU, pre-built CUDA wheel)

```bash
make install-llama-cpp-prebuilt   # pre-built cu124 wheel, no nvcc required
# Edit configs/llama-cpp-gpu.yaml: set model: /path/to/model.gguf and n_gpu_layers
uv run llm-bench --config configs/llama-cpp-gpu.yaml --output results/llama-gpu.csv
```

If you have the CUDA toolkit installed (`nvcc` on PATH), build from source instead:
```bash
make install-llama-cpp-cuda
```

## vLLM backend (GPU, Linux only)

vLLM provides high-throughput offline inference using its in-process LLM API.

**Prerequisites:**
- Linux — vLLM is not supported on Windows or macOS.
- CUDA-capable GPU. CPU inference is not available.
- CUDA driver ≥ 12.1 (`nvidia-smi` must report CUDA Version ≥ 12.1). No `nvcc` or CUDA toolkit needed.

```bash
uv sync --extra vllm
# Edit your config: set backend: vllm and model: <HuggingFace model id>
uv run llm-bench --config configs/vllm.yaml --output results/vllm.csv
```

Minimal config example:
```yaml
backend: vllm
model: meta-llama/Llama-3.2-3B-Instruct
requests: 20
warmup_requests: 2
prompts_file: data/prompts/smoke.txt
```

## ONNX backend (GPU, CUDA)

The default install uses the CPU build of ONNX Runtime. For GPU acceleration, replace it with
the GPU variant after syncing:

```bash
uv sync --extra onnx
uv pip install onnxruntime-gpu   # replaces CPU onnxruntime; CUDA 12 compatible
```

No `nvcc` or CUDA toolkit is required — only the CUDA driver and CUDA runtime libraries.

Set `device: cuda` in your config's `onnx:` section:

```yaml
backend: onnx
model: gpt2
requests: 20
warmup_requests: 2
prompts_file: data/prompts/smoke.txt

onnx:
  max_new_tokens: 50
  device: cuda       # use CUDAExecutionProvider
  export: true       # auto-export from HF on first run
```

Run:
```bash
uv run llm-bench --config configs/onnx-example.yaml --output results/onnx-gpu.csv
```

## OpenAI-compatible endpoint

```bash
# Start any OpenAI-compatible server first (Ollama, llama.cpp server, LM Studio, vLLM)
# Then edit configs/openai-endpoint.yaml with the server URL and model name
uv run llm-bench --config configs/openai-endpoint.yaml --output results/openai.csv
```

## Override config fields without editing YAML

Use `--set KEY=VALUE` to override any config field on the command line. Values are parsed
as YAML scalars (int, float, bool, str) and take precedence over the YAML file.

```bash
# Override latency for the mock backend
uv run llm-bench --config configs/example.yaml --set mock.latency_ms=5

# Override llama.cpp token limit and thread count in one command
uv run llm-bench --config configs/llama-cpp-cpu.yaml \
  --set llama_cpp.max_tokens=200 --set llama_cpp.n_threads=4

# Combine --set with other flags
uv run llm-bench --config configs/example.yaml --set mock.latency_ms=0 --requests 100
```

## Run matrix (multiple configs in one command)

```bash
uv run llm-bench matrix --config configs/matrix-example.yaml
uv run llm-bench matrix --config configs/matrix-example.yaml --dry-run   # preview
uv run llm-bench compare results/*.csv --sort p95

# Show only the 3 fastest configurations by throughput
uv run llm-bench compare results/*.csv --sort toks --limit 3
```

## Parameter sweep

```bash
uv run llm-bench matrix --config configs/sweep-example.yaml --dry-run
uv run llm-bench matrix --config configs/sweep-example.yaml
```

## Quantization comparison (Q4\_K\_M vs Q8\_0)

```bash
# Download both quantizations
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models/
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q8_0.gguf --local-dir ~/models/

# Set model: paths in configs/llama-cpp-q4km-best.yaml and configs/llama-cpp-q8-best.yaml
make run-quant-compare
```

## Run manifest (environment fingerprint)

```bash
uv run llm-bench --config configs/example.yaml \
  --output results/mock.csv \
  --manifest results/manifest.json
```

## Repeated trials with variance

Add `repeats: 5` to any config YAML, then run normally:
```bash
uv run llm-bench --config configs/example-repeats.yaml --output results/repeated.csv
```

Reported p95 and tok/s become the median across repeats. New CSV columns
`p95_latency_ms_std` and `tokens_per_second_std` carry the sample standard deviation.

## Full check suite

```bash
make test        # pytest -v (mock tests only, CI-safe)
make test-hf     # pytest -m integration (requires uv sync --extra transformers)
make lint        # ruff check .
make format      # ruff format .
make typecheck   # pyright
```
