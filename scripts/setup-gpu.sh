#!/usr/bin/env bash
set -euo pipefail
if ! command -v nvidia-smi &>/dev/null; then
  echo "No NVIDIA GPU detected. Skipping GPU setup."
  exit 0
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Detected GPU: ${GPU_NAME}"
uv pip install "llama-cpp-python>=0.2" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
echo "GPU setup complete. Run: uv run llm-bench serve"
