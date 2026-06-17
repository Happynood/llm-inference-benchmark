# Contributing

## Development Setup

```bash
git clone https://github.com/happynood/llm-inference-benchmark
cd llm-inference-benchmark
uv sync
```

This installs the core package and all dev dependencies (pytest, ruff, pyright).
Optional backend extras (`transformers`, `llama-cpp`) are not required for the default test suite.

## Running Tests

```bash
# Full mock-only suite (CI-safe, no model downloads)
uv run pytest -v

# With coverage report
uv run pytest -v --cov=llm_inference_benchmark --cov-report=term-missing

# Integration tests (requires uv sync --extra transformers)
uv run pytest -m integration -v
```

## Code Quality

```bash
make lint       # ruff check .
make format     # ruff format .
make typecheck  # pyright
```

All three must pass before a PR is merged. CI enforces this automatically.

## Making Changes

1. **One PR per feature or fix.** Keep PRs focused; large bundled PRs are harder to review.
2. **Update `CHANGELOG.md`** — add a line under `## [Unreleased]` describing what changed.
3. **Add or update tests** — new behavior needs test coverage; bug fixes need a regression test.
4. **Keep CI green** — the suite must pass with `uv run pytest -v` using the mock backend only.
5. **No generated artifacts** — do not commit CSV results, model weights, `.venv`, caches, or secrets.

## Adding a New Backend

1. Subclass `Backend` from `src/llm_inference_benchmark/backends/base.py`.
2. Add a config block to `BenchmarkConfig` in `config.py`.
3. Wire it into `_build_backend()` in `runner.py`.
4. Add unit tests using mocks — no real model download required for CI.
5. Document the new backend in `docs/cli.md` and `docs/quickstart.md`.

## Commit Style

Use [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(backends): add onnxruntime backend
fix(cli): handle missing CSV column gracefully
docs(metrics): add TTFT computation notes
```

## Reporting Issues

Use [GitHub Issues](https://github.com/Happynood/llm-inference-benchmark/issues).
Include: OS, Python version, backend, config YAML, and the full error output.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
