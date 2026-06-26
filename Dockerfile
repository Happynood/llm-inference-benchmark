# ---- cpu: llama.cpp without CUDA, ubuntu:22.04 base ----
FROM ubuntu:22.04 AS cpu
ARG SKIP_LLAMA_CPP=0
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl git python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN if [ "$SKIP_LLAMA_CPP" = "1" ]; then \
      uv sync --no-dev --frozen --no-install-project; \
    else \
      uv sync --extra llama-cpp --no-dev --frozen --no-install-project; \
    fi

COPY src/ ./src/
COPY README.md ./

RUN if [ "$SKIP_LLAMA_CPP" = "1" ]; then \
      uv sync --no-dev --frozen; \
    else \
      uv sync --extra llama-cpp --no-dev --frozen; \
    fi

CMD ["llm-bench", "--help"]

# ---- gpu: llama.cpp with CUDA, nvidia/cuda:12.6.0-devel-ubuntu22.04 base ----
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS gpu
ARG SKIP_LLAMA_CPP=0
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl git python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN if [ "$SKIP_LLAMA_CPP" = "1" ]; then \
      uv sync --no-dev --frozen --no-install-project; \
    else \
      CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 \
        uv sync --extra llama-cpp --no-dev --frozen --no-install-project; \
    fi

COPY src/ ./src/
COPY README.md ./

RUN if [ "$SKIP_LLAMA_CPP" = "1" ]; then \
      uv sync --no-dev --frozen; \
    else \
      CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 \
        uv sync --extra llama-cpp --no-dev --frozen; \
    fi

CMD ["llm-bench", "--help"]

# ---- webui: Web API + dashboard (llm-bench serve) ----
FROM ubuntu:22.04 AS webui
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync --extra server --no-dev --frozen --no-install-project

COPY src/ ./src/
COPY README.md ./

RUN uv sync --extra server --no-dev --frozen

EXPOSE 8080
CMD ["llm-bench", "serve", "--host", "0.0.0.0", "--port", "8080"]

# ---- webui-gpu: Web API + dashboard with CUDA GPU support ----
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS webui-gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl git python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 \
    uv sync --extra server --extra llama-cpp --no-dev --frozen --no-install-project

COPY src/ ./src/
COPY README.md ./

RUN CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 \
    uv sync --extra server --extra llama-cpp --no-dev --frozen

EXPOSE 8080
CMD ["llm-bench", "serve", "--host", "0.0.0.0", "--port", "8080"]
