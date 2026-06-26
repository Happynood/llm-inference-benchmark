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
uv pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
# Create symlinks so libggml-cuda.so can find CUDA 12 libs via its $ORIGIN rpath
# without requiring LD_LIBRARY_PATH at runtime.
uv run python -c "
import site, os, pathlib
sp = pathlib.Path(site.getsitepackages()[0])
llama_lib = sp / 'llama_cpp' / 'lib'
cuda_rt   = sp / 'nvidia' / 'cuda_runtime' / 'lib'
cublas    = sp / 'nvidia' / 'cublas' / 'lib'
pairs = [('libcudart.so.12', cuda_rt), ('libcublas.so.12', cublas), ('libcublasLt.so.12', cublas)]
for name, src_dir in pairs:
    src = src_dir / name
    dst = llama_lib / name
    if src.exists() and not dst.exists():
        os.symlink(src, dst)
        print(f'  linked {name}')
    elif dst.exists():
        print(f'  ok     {name}')
    else:
        print(f'  skip   {name} (source not found)')
"
echo "GPU setup complete. Run: make webui-gpu"
