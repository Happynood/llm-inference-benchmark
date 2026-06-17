.PHONY: install install-hf install-llama-cpp install-llama-cpp-cuda install-llama-cpp-prebuilt \
        test test-hf lint format typecheck \
        run run-hf run-gpu run-matrix \
        run-llama-cpp-cpu run-llama-cpp-gpu \
        run-quant-compare \
        clean

# ── Dependencies ─────────────────────────────────────────────────────────────

install:
	uv sync

install-hf:
	uv sync --extra transformers

install-llama-cpp:
	uv sync --extra llama-cpp

# Build llama.cpp with CUDA support from source (requires nvcc / CUDA toolkit).
install-llama-cpp-cuda:
	CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp

# Install a pre-built CUDA wheel (no nvcc required — works with driver-only setups).
# GPU driver ≥ 520 and CUDA 12.4 compatible hardware required.
install-llama-cpp-prebuilt:
	uv pip install "llama-cpp-python>=0.2" \
	  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
	uv pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12

# ── CUDA library path ────────────────────────────────────────────────────────
#
# Pre-built llama-cpp-python CUDA wheels bundle their own CUDA shared libraries
# under .venv/lib/…/site-packages/nvidia/. On systems without a CUDA toolkit
# (nvcc absent) those paths are not in LD_LIBRARY_PATH by default.
# This variable resolves them so GPU runs work out of the box.
# It expands to an empty string when the nvidia packages are not installed.

CUDA_LIB_PATH := $(shell \
  find .venv/lib -name "*.so*" -path "*/site-packages/nvidia/*" 2>/dev/null \
  | xargs -I{} dirname {} \
  | sort -u \
  | tr '\n' ':')

# ── Quality checks ───────────────────────────────────────────────────────────

test:
	uv run pytest -v

test-hf:
	uv run pytest -v -m integration

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

# ── Benchmark runs ───────────────────────────────────────────────────────────

run:
	uv run llm-bench --config configs/example.yaml --output benchmark.csv

run-hf:
	uv run llm-bench --config configs/transformers-cpu.yaml --output benchmark-hf.csv

run-gpu:
	uv run llm-bench --config configs/transformers-gpu.yaml --output benchmark-gpu.csv \
	  --manifest results/manifest-gpu.json

run-matrix:
	uv run llm-bench matrix --config configs/matrix-example.yaml

run-llama-cpp-cpu:
	uv run llm-bench --config configs/llama-cpp-cpu.yaml --output results/llama-cpp-cpu.csv \
	  --manifest results/llama-cpp-cpu.manifest.json

run-llama-cpp-gpu:
	LD_LIBRARY_PATH="$(CUDA_LIB_PATH)$$LD_LIBRARY_PATH" \
	  uv run llm-bench --config configs/llama-cpp-gpu.yaml \
	    --output results/llama-cpp-gpu.csv \
	    --manifest results/llama-cpp-gpu.manifest.json

# Run Q4_K_M vs Q8_0 quantization comparison and print the result table.
# Prerequisites: install-llama-cpp-prebuilt (or install-llama-cpp-cuda),
#                model paths set in configs/llama-cpp-q4km-best.yaml and
#                configs/llama-cpp-q8-best.yaml.
run-quant-compare:
	LD_LIBRARY_PATH="$(CUDA_LIB_PATH)$$LD_LIBRARY_PATH" \
	  uv run llm-bench matrix --config configs/llama-cpp-quant-compare.yaml
	uv run llm-bench compare results/quant-q4km.csv results/quant-q8.csv

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -f benchmark.csv benchmark-hf.csv benchmark-gpu.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
