# llm-inference-benchmark

## Commands
- `uv sync`
- `uv run pytest`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pyright`
- `uv run llm-bench --config configs/example.yaml --output benchmark.csv`

## Rules
- Treat `docs/project-charter.md` as the fixed product direction: a reproducible inference
  optimization lab that grows from benchmark harness to configuration comparison and
  recommendation.
- Do not commit model weights, datasets, checkpoints, or `.env`.
- Keep benchmark claims reproducible: command, config, fixture, hardware, limitations.
- Update `docs/metrics.md` when benchmark numbers change.
