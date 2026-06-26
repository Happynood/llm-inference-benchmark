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

## GPU Quick Start

Got an NVIDIA GPU and want to run the Web UI with GPU inference?

**Requirements:** NVIDIA driver ≥ 520 · CUDA 12.x compatible GPU · no CUDA toolkit needed

### One-command setup

```bash
make setup-gpu   # detects GPU, installs CUDA wheel, creates required symlinks
make webui-gpu   # starts the Web UI at http://localhost:8080
```

### Manual step-by-step

```bash
# 1. Install all backends (includes CPU llama-cpp-python)
uv sync --extra all-backends

# 2. Replace the CPU wheel with a pre-built CUDA wheel and set up runtime libs
#    (creates libcudart.so.12 / libcublas.so.12 symlinks — no LD_LIBRARY_PATH needed)
make install-llama-cpp-prebuilt

# 3. Start the Web UI
uv run llm-bench serve
```

After starting the server, open `http://localhost:8080` and check the **Capabilities** indicator.
`llama_cpp_gpu: true` confirms GPU inference is active.

> **Note:** `uv sync` may revert the CUDA wheel back to the CPU build. Re-run
> `make install-llama-cpp-prebuilt` after any `uv sync` to restore GPU support.

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
# Installs cu124 pre-built wheel + creates CUDA lib symlinks (no nvcc or LD_LIBRARY_PATH needed)
make install-llama-cpp-prebuilt
# Edit configs/llama-cpp-gpu.yaml: set model: /path/to/model.gguf and n_gpu_layers
uv run llm-bench --config configs/llama-cpp-gpu.yaml --output results/llama-gpu.csv
```

**Choosing `n_gpu_layers`** — set to `99` to offload all layers; reduce if you hit VRAM limits:

| GPU VRAM | Model             | Recommended `n_gpu_layers` |
|----------|-------------------|---------------------------|
| 4 GB     | Llama 3.2 3B Q4   | 28 (all layers)            |
| 4 GB     | Llama 3 8B Q4     | 20–24                      |
| 8 GB     | Llama 3 8B Q4     | 32 (all layers)            |

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

## Web UI

Start the browser dashboard:

```bash
uv run llm-bench serve
# or the Makefile shortcut:
make webui
```

Open <http://localhost:8080> and use the **+ New Run** form to configure and submit benchmark runs.

**Key features:**

- **Leaderboard** — sidebar button that shows the single best run for each key metric
  (Throughput, p50/p95 Latency, TTFT, VRAM, Energy) at a glance.
- **Compare bar** — appears above the run list when two or more runs are selected; use the
  Table / Chart / Trend / Pareto / CSV buttons to switch between views.

### llama-cpp GPU in the Web UI

When you select `llama-cpp` as the backend, the dashboard shows a warning banner if
`llama-cpp-python` was installed without CUDA support (which is the default).  The warning
means benchmarks will run on CPU regardless of the GPU Layers setting — this is not a
crash, just reduced performance.

To fix it, reinstall with GPU support before starting `llm-bench serve`:

```bash
# Option A — Pre-built CUDA wheel (no compiler needed)
make install-llama-cpp-prebuilt

# Option B — Build from source (requires nvcc)
make install-llama-cpp-cuda
```

Restart `llm-bench serve` after reinstalling.

### Web UI via Docker

Run the dashboard without a local Python installation:

```bash
docker compose up webui
# Open http://localhost:8080
```

Run results persist across restarts in a named Docker volume (`llm-bench-data`).
GGUF models at `~/models/` on the host are visible inside the container at `/models`.

## Full local validation

Run these steps end-to-end to verify the entire stack before a release:

```bash
# 1. Unit tests (no model downloads, CI-safe)
make test

# 2. Check which backends are detected
uv run llm-bench verify

# 3. Smoke-test the CLI pipeline with the mock backend
uv run llm-bench --config configs/example.yaml --output results/local-smoke.csv
uv run llm-bench compare results/local-smoke.csv

# 4. Smoke-test the Web UI health endpoint
uv pip install 'llm-inference-benchmark[server]'
uv run llm-bench serve &
sleep 2 && curl -sf http://localhost:8080/api/health && echo " OK"
kill %1

# 5. (Optional) Transformers backend — downloads sshleifer/tiny-gpt2 (~4 MB)
uv sync --extra transformers
uv run llm-bench --config configs/transformers-cpu.yaml --output results/local-hf.csv

# 6. (Optional) llama.cpp — requires a GGUF file on disk
uv sync --extra llama-cpp
# edit configs/llama-cpp-cpu.yaml: set model: /path/to/model.gguf
uv run llm-bench --config configs/llama-cpp-cpu.yaml --output results/local-llama.csv
```

Steps 1–4 are CI-safe and require no model downloads.  Steps 5–6 are optional but recommended
before publishing a release.
