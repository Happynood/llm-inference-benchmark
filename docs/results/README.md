# Real-Run Evidence Registry

This directory contains curated benchmark reports from real hardware runs.
Each report is a permanent record — the config, prompts, hardware context, and
limitations are documented so the numbers can be interpreted and reproduced.

**Nothing in this directory is generated output.** Generated CSVs and manifests
stay in `results/` (gitignored). Only curated documentation lives here.

## What counts as a real run

A run qualifies for this registry when:

1. It uses a real inference backend (`transformers`, `llama-cpp`, etc.) — not the mock backend.
2. The model performs actual forward passes — not simulated latency.
3. Hardware context is recorded: CPU, GPU, RAM, driver/CUDA versions.
4. The config file is committed and the prompts SHA256 is documented.
5. Limitations are stated honestly (toy model, single run, no statistical replication, etc.).

## Why mock runs are excluded

The mock backend is a deterministic timing stub used for CI and harness validation.
It produces `~5 ms` latency and `~10,000 tok/s` because `time.sleep(latency_ms / 1000)`
is the only work it does. These numbers validate that the harness measures correctly,
not that any model is fast.

See [docs/metrics.md](../metrics.md) for the full distinction between mock validation
and real hardware evidence.

## Run Registry

| Report | Hardware | Backend | Model | Date | Key result |
|--------|----------|---------|-------|------|------------|
| [RTX 3050 — 4-model backend comparison](rtx3050-4model-backend-comparison.md) | i5-11400H + RTX 3050 4 GB | llama-cpp + transformers | Llama-3B Q4\_K\_M / Q8\_0, gpt2-medium, Bonsai-8B | 2026-06-27 | 3 of 4 runs Pareto-optimal; Q4\_K\_M best llama-cpp option; gpt2-medium leads tok/J at 2.89 |
| [RTX 3050 — tiny-gpt2 CPU vs GPU](gpu-rtx3050-tiny-gpt2.md) | i5-11400H + RTX 3050 4 GB | transformers | sshleifer/tiny-gpt2 | 2026-06-14 | GPU slower than CPU for 2-layer toy model; establishes GPU baseline |
| [RTX 3050 — Llama 3.2 3B CPU vs GPU](llama-cpp-rtx3050-llama32-3b.md) | i5-11400H + RTX 3050 4 GB | llama-cpp | Llama-3.2-3B-Instruct-Q4\_K\_M | 2026-06-14 | GPU 2.95× faster; 53.7 tok/s vs 18.0 tok/s CPU; all 28 layers on CUDA0 |
| [RTX 3050 — n\_gpu\_layers sweep (0 / 20 / 99)](llama-cpp-rtx3050-vram-sweep.md) | i5-11400H + RTX 3050 4 GB | llama-cpp | Llama-3.2-3B-Instruct-Q4\_K\_M | 2026-06-14 | VRAM scales ~60 MiB/layer after 655 MiB baseline; full offload 2.92× CPU; partial offload viable |
| [RTX 3050 — Q4\_K\_M vs Q8\_0 quantization comparison](llama-cpp-rtx3050-quant-compare.md) | i5-11400H + RTX 3050 4 GB | llama-cpp | Llama-3.2-3B-Instruct (both quants) | 2026-06-14 | Q4\_K\_M 1.31× faster, 1.57× less VRAM, Pareto-optimal |
| [RTX 3050 — Qwen2.5-Coder-1.5B Q5\_K\_M full GPU](llama-cpp-rtx3050-qwen25-coder-1.5b.md) | i5-11400H + RTX 3050 4 GB | llama-cpp | Qwen2.5-Coder-1.5B-Instruct-Q5\_K\_M | 2026-06-18 | 66.8 tok/s; 1503 MiB VRAM (36.7%); real-time code completion on 4 GB budget |
