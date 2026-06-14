.PHONY: install install-hf install-llama-cpp install-llama-cpp-cuda \
        test test-hf lint format typecheck \
        run run-hf run-gpu run-matrix run-llama-cpp-cpu run-llama-cpp-gpu clean

install:
	uv sync

install-hf:
	uv sync --extra transformers

install-llama-cpp:
	uv sync --extra llama-cpp

install-llama-cpp-cuda:
	CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp

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

run:
	uv run llm-bench --config configs/example.yaml --output benchmark.csv

run-hf:
	uv run llm-bench --config configs/transformers-cpu.yaml --output benchmark-hf.csv

run-gpu:
	uv run llm-bench --config configs/transformers-gpu.yaml --output benchmark-gpu.csv --manifest results/manifest-gpu.json

run-matrix:
	uv run llm-bench matrix --config configs/matrix-example.yaml

run-llama-cpp-cpu:
	uv run llm-bench --config configs/llama-cpp-cpu.yaml --output results/llama-cpp-cpu.csv \
	  --manifest results/llama-cpp-cpu.manifest.json

run-llama-cpp-gpu:
	uv run llm-bench --config configs/llama-cpp-gpu.yaml --output results/llama-cpp-gpu.csv \
	  --manifest results/llama-cpp-gpu.manifest.json

clean:
	rm -f benchmark.csv benchmark-hf.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
