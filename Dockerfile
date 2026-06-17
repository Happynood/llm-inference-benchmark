FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --extra transformers --no-dev --frozen --no-install-project

COPY src/ ./src/
COPY README.md ./
RUN uv sync --extra transformers --no-dev --frozen

ENTRYPOINT ["/app/.venv/bin/llm-bench"]
CMD ["--help"]
