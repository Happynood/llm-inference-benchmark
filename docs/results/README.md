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
| [RTX 3050 — tiny-gpt2 CPU vs GPU](gpu-rtx3050-tiny-gpt2.md) | i5-11400H + RTX 3050 4 GB | transformers | sshleifer/tiny-gpt2 | 2026-06-14 | GPU slower than CPU for 2-layer toy model; establishes GPU baseline |

## Planned runs

These runs are planned but not yet executed. They will move to the registry table
when committed prompt fixtures, configs, and curated reports exist.

| Backend | Model | Purpose |
|---------|-------|---------|
| llama-cpp | Llama 3 8B Q4_K_M (4 GB VRAM) | First production-size model; CPU vs GPU at 4-bit quantization |
| llama-cpp | Llama 3 8B Q8_0 | Quality–speed trade-off at higher precision |
| transformers | full GPT-2 (117 M) | Intermediate model to bridge toy → production |
