.PHONY: install test lint format typecheck run clean

install:
	uv sync

test:
	uv run pytest -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

run:
	uv run llm-bench --config configs/example.yaml --output benchmark.csv

clean:
	rm -f benchmark.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
