.PHONY: install install-hf test test-hf lint format typecheck run run-hf run-gpu clean

install:
	uv sync

install-hf:
	uv sync --extra transformers

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

clean:
	rm -f benchmark.csv benchmark-hf.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
