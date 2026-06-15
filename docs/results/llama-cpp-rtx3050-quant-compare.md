# llama.cpp — Quantization Comparison: Q4\_K\_M vs Q8\_0 (Llama 3.2 3B, RTX 3050)

Comparison of two GGUF quantization formats for the same model, same prompts, and same
GPU offload configuration. Produced via the repository CLI (v0.13 benchmark + quality
metrics; v0.14 Pareto analysis; v0.15 constraint-based recommender; v0.16 task-quality
evaluation):

```bash
CUDA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name "*.so*" \
  | xargs -I{} dirname {} | sort -u | tr '\n' ':')
LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" \
  uv run llm-bench matrix --config configs/llama-cpp-quant-compare.yaml

uv run llm-bench compare results/quant-q4km.csv results/quant-q8.csv
uv run llm-bench pareto  results/quant-q4km.csv results/quant-q8.csv
```

## Hardware

| Component | Details |
|-----------|---------|
| CPU | Intel Core i5-11400H @ 2.70 GHz, 12 logical cores |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4096 MiB total |
| Driver | CUDA 13.x (no nvcc; cu124 wheel used) |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.12.13 |
| llama-cpp-python | 0.3.29 (pre-built cu124 wheel) |
| nvidia-cuda-runtime-cu12 | 12.x |
| nvidia-cublas-cu12 | 12.9.2.10 |
| OS | Linux 7.0.0-22-generic x86-64 |

## Model

| Field | Q4\_K\_M | Q8\_0 |
|-------|----------|-------|
| Base model | Llama-3.2-3B-Instruct | Llama-3.2-3B-Instruct |
| Quantization | Q4\_K\_M (~4.5 bits/weight) | Q8\_0 (8 bits/weight) |
| GGUF file size | ~1.9 GB | ~3.2 GB |
| Architecture | 28 layers, same for both | 28 layers, same for both |
| Source | `bartowski/Llama-3.2-3B-Instruct-GGUF` | `bartowski/Llama-3.2-3B-Instruct-GGUF` |

Q8_0 stores weights at full 8-bit precision — effectively lossless vs float16 for most tasks.
Q4_K_M applies K-quant compression (~4.5 bits/weight on average) to reduce size by ~1.7×
while preserving most output quality.

## Benchmark Configs

| Run | Config file | Quantization | `n_gpu_layers` |
|-----|-------------|--------------|----------------|
| quant-q4km | `configs/llama-cpp-q4km-best.yaml` | Q4\_K\_M | 99 (all 28 layers) |
| quant-q8 | `configs/llama-cpp-q8-best.yaml` | Q8\_0 | 99 (all 28 layers) |

Shared parameters: `requests=10`, `warmup_requests=2`, `n_ctx=512`, `max_tokens=50`,
`temperature=0.0` (greedy/deterministic), `prompts_file=data/prompts/smoke.txt`.

Both quantizations fit at `n_gpu_layers=99` (all 28 layers on GPU):
- Q4\_K\_M: 2361 MiB (57.6% of 4096 MiB) — comfortable headroom
- Q8\_0: 3697 MiB (90.2% of 4096 MiB) — tight but stable; probed successful before the run

## CLI Output — `llm-bench compare`

```
| Backend   | Model                                  | N  | p50 (ms) | p95 (ms) | tok/s | CPU mem (MB) | CUDA mem (MB) | VRAM mem (MB) | Sanity % |
|-----------|----------------------------------------|----|----------|----------|-------|--------------|---------------|---------------|----------|
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf | 10 | 904.33   | 915.22   | 55.3  | 1289.8       | 0.0           | 2361.0        | 100.0%   |
| llama-cpp | .../Llama-3.2-3B-Instruct-Q8_0.gguf   | 10 | 1185.23  | 1186.75  | 42.2  | 1418.4       | 0.0           | 3697.0        | 100.0%   |
```

`CUDA mem` is always `0.0` for llama.cpp — it uses its own CUDA allocator, not PyTorch's.
`VRAM mem` is `peak_vram_memory_mb` (nvidia-smi polling at 500 ms).
`Sanity %` is `sanity_pass_rate × 100` — the fraction of non-empty completions.

## Results

| Quantization | p50 (ms) | p95 (ms) | tok/s | CPU mem (MB) | Peak VRAM (MiB) |
|---|---|---|---|---|---|
| Q4\_K\_M (n\_gpu\_layers=99) | **904.33** | **915.22** | **55.28** | 1289.8 | **2361** |
| Q8\_0 (n\_gpu\_layers=99) | 1185.23 | 1186.75 | 42.21 | 1418.4 | 3697 |

Run date: 2026-06-14. See [Limitations](#limitations) for reproducibility caveats.

## Output Sanity Metrics (v0.13)

| Quantization | empty\_output\_count | min\_output\_chars | mean\_output\_chars | repeated\_output\_count | sanity\_pass\_rate |
|---|---|---|---|---|---|
| Q4\_K\_M | 0 | 202 | 253.4 | 10 | 1.0 |
| Q8\_0 | 0 | 226 | 260.8 | 10 | 1.0 |

**sanity\_pass\_rate = 1.0** for both: every completion contained non-whitespace text.
No empty outputs — the model generated meaningful responses to all 10 requests.

**repeated\_output\_count = 10** for both: this is expected behavior, not degeneration.
The run uses 5 unique prompts (`data/prompts/smoke.txt`) with `requests=10` and
`temperature=0.0` (greedy/deterministic). The prompt list cycles twice, so each prompt
fires exactly twice and produces the identical output both times. Every output therefore
appears exactly twice — `repeated_output_count = 10 = request_count`.

To verify: re-run with `requests=5` (one pass through all 5 prompts) and
`repeated_output_count` drops to 0.

**min\_output\_chars and mean\_output\_chars** show that Q8\_0 produces slightly longer
completions on average (260.8 vs 253.4 chars stripped). Both are well above 0 — the
model fills its `max_tokens=50` budget on all requests.

## Interpretation

### Speed

Q4\_K\_M is **1.31× faster** than Q8\_0 at the same GPU offload configuration:

| Metric | Q4\_K\_M | Q8\_0 | Ratio (Q4K / Q8) |
|--------|----------|-------|-----------------|
| p50 latency | 904.33 ms | 1185.23 ms | **1.31× faster** |
| tok/s | 55.28 | 42.21 | **1.31× higher** |

This speedup is driven by **memory bandwidth**, not compute. At GPU inference speeds,
the bottleneck is not arithmetic — it is how fast weight data can be loaded from VRAM
into GPU compute cores per token. Q4_K_M weights are ~1.78× more compact than Q8_0,
so fewer bytes per weight are read per forward pass. The smaller transfer cost directly
translates to lower latency and higher throughput.

This is the fundamental reason quantization speeds up LLM inference on GPU: fewer bits
per weight = fewer bytes transferred per forward pass = faster.

### Memory

| Quantization | VRAM used | VRAM free | % of 4 GB |
|---|---|---|---|
| Q4\_K\_M | 2361 MiB | 1735 MiB | 57.6% |
| Q8\_0 | 3697 MiB | 399 MiB | 90.2% |

Q8\_0 leaves only 399 MiB free VRAM on the RTX 3050. This is enough for 10-request
benchmark runs at `n_ctx=512`, but tight for interactive use (longer conversations,
larger context windows, or concurrent processes competing for VRAM). Q4\_K\_M leaves
1735 MiB headroom, making it substantially more practical on a 4 GB card.

### Latency consistency

| Quantization | p95 / p50 | Interpretation |
|---|---|---|
| Q4\_K\_M | 1.012 (1.2%) | Very tight; small scheduling variance |
| Q8\_0 | 1.001 (0.1%) | Near-deterministic GPU execution |

Q8\_0 shows tighter p95/p50 than Q4\_K\_M (0.1% vs 1.2%), consistent with the previous
run. Q8\_0's uniform 8-bit weight loads are more compute-bound; Q4\_K\_M's K-quant
dequantization step adds a small variable-cost component. Neither variance is
operationally significant at this workload size.

### Comparison to previous run (PR #27, 2026-05)

| Metric | Q4\_K\_M prev | Q4\_K\_M now | Q8\_0 prev | Q8\_0 now |
|--------|--------------|-------------|-----------|---------|
| p50 (ms) | 929.77 | 904.33 | 1194.74 | 1185.23 |
| tok/s | 53.56 | 55.28 | 41.85 | 42.21 |
| VRAM (MiB) | 2361 | 2361 | 3697 | 3697 |

VRAM is identical (expected — same model, same GPU). Latency is ~2–3% lower this run;
variation is within the expected run-to-run range for 10-request benchmarks on a shared
laptop GPU (OS scheduling, thermal state, background processes).

### Summary

| Dimension | Winner | Magnitude |
|-----------|--------|-----------|
| Speed (tok/s) | Q4\_K\_M | 1.31× faster |
| VRAM footprint | Q4\_K\_M | 1.57× smaller (2361 vs 3697 MiB) |
| Latency consistency | Q8\_0 | marginally (0.1% vs 1.2% p95/p50) |
| Output sanity | Tied | Both 100% pass rate, no empty outputs |
| Task quality (v0.16 rubric) | Q4\_K\_M | 100% vs 80% (Q8\_0 fails prompt 3 at max\_tokens=50) |

**Recommendation for a 4 GB laptop GPU**: Q4\_K\_M is the better practical choice — it
is faster, leaves more VRAM headroom for larger contexts or concurrent workloads, and
passes all deterministic rubric checks at max_tokens=50. Use Q8\_0 only when VRAM is not
a constraint, you increase max_tokens to ≥ 100, or output quality at longer generation
lengths is a hard requirement.

## Pareto Analysis (v0.14)

Pareto dominance identifies whether any configuration is unambiguously better: no worse
on every metric and strictly better on at least one.  Metrics ranked by optimisation
direction: lower p95 latency, higher tok/s, lower VRAM, higher sanity pass rate.

```
uv run llm-bench pareto results/quant-q4km.csv results/quant-q8.csv
```

```
| Backend   | Model                                  | N  | p95 (ms) | tok/s | CPU mem (MB) | VRAM mem (MB) | Sanity % | Pareto  |
|-----------|----------------------------------------|----|----------|-------|--------------|---------------|----------|---------|
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf | 10 | 915.22   | 55.3  | 1289.8       | 2361.0        | 100.0%   | optimal |
| llama-cpp | .../Llama-3.2-3B-Instruct-Q8_0.gguf   | 10 | 1186.75  | 42.2  | 1418.4       | 3697.0        | 100.0%   | -       |
```

**Q4\_K\_M is the sole Pareto-optimal configuration.**

Q4\_K\_M beats Q8\_0 on every measured dimension simultaneously:

| Metric | Q4\_K\_M | Q8\_0 | Direction | Winner |
|--------|----------|-------|-----------|--------|
| p95 latency | 915.22 ms | 1186.75 ms | minimize | Q4\_K\_M |
| tok/s | 55.3 | 42.2 | maximize | Q4\_K\_M |
| VRAM (MiB) | 2361 | 3697 | minimize | Q4\_K\_M |
| Sanity pass rate | 100.0% | 100.0% | maximize | Tied |

Q8\_0 is dominated: Q4\_K\_M is faster, uses less VRAM, and produces equally-valid
outputs. There is no measured trade-off that justifies Q8\_0 on this hardware configuration.

The Pareto result does not capture **output quality** (semantic accuracy, factual
correctness). Q8\_0 stores weights at near-lossless precision and may produce better
outputs on tasks where Q4\_K\_M compression degrades generation. See [Limitations](#limitations)
for the full caveat.

## Task Quality Evaluation (v0.16)

Re-ran both quantizations with a deterministic YAML rubric spec attached to each config,
using `configs/llama-cpp-quant-compare-quality.yaml`:

```bash
CUDA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name "*.so*" \
  | xargs -I{} dirname {} | sort -u | tr '\n' ':')
LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" \
  uv run llm-bench matrix --config configs/llama-cpp-quant-compare-quality.yaml

uv run llm-bench compare results/quant-q4km-quality.csv results/quant-q8-quality.csv
uv run llm-bench pareto  results/quant-q4km-quality.csv results/quant-q8-quality.csv
uv run llm-bench recommend results/quant-q4km-quality.csv results/quant-q8-quality.csv \
  --max-vram-mb 4096 --max-p95-ms 1000 --min-sanity 1.0 --min-quality 1.0
```

### Rubric spec (`configs/quality/smoke-rubric.yaml`)

5 rubrics aligned to the 5 prompts in `data/prompts/smoke.txt`.
All string checks are case-insensitive.

| Prompt | Check type | Criteria |
|--------|------------|----------|
| 0 — Capital of France | `contains_any` + `forbidden` | must contain "Paris"; must not contain hedging phrases |
| 1 — Gradient descent | `contains_any` | must contain ≥ 1 of: gradient, descent, minimize, loss, optimize, converge |
| 2 — Transformer architecture | `contains_any` | must contain ≥ 1 of: transformer, attention, encoder, decoder, feed-forward |
| 3 — BERT vs GPT | `contains_all` | must contain both "bert" AND "gpt" |
| 4 — Attention mechanism | `contains_any` | must contain ≥ 1 of: attention, query, key, value, weight, score |

**Rubric type: deterministic structural checks — not semantic judge scoring.**
A pass means the output mentioned the required terms. It does not measure accuracy, depth, or coherence.

### `llm-bench compare` (with Task Q %)

```
| Backend   | Model                                                     | N  | p50 (ms) | p95 (ms) | tok/s | CPU mem (MB) | CUDA mem (MB) | VRAM mem (MB) | Sanity % | Task Q % |
|-----------|-----------------------------------------------------------|----|----------|----------|-------|--------------|---------------|---------------|----------|----------|
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf                    | 10 | 901.01   | 915.86   | 55.5  | 1291.8       | 0.0           | 2361.0        | 100.0%   | 100.0%   |
| llama-cpp | .../Llama-3.2-3B-Instruct-Q8_0.gguf                      | 10 | 1190.62  | 1226.97  | 41.8  | 708.2        | 0.0           | 3697.0        | 100.0%   | 80.0%    |
```

### `llm-bench pareto` (with Task Q %)

```
| Backend   | Model                                                     | N  | p95 (ms) | tok/s | CPU mem (MB) | VRAM mem (MB) | Sanity % | Task Q % | Pareto  |
|-----------|-----------------------------------------------------------|----|----------|-------|--------------|---------------|----------|----------|---------|
| llama-cpp | .../Llama-3.2-3B-Instruct-Q4_K_M.gguf                    | 10 | 915.86   | 55.5  | 1291.8       | 2361.0        | 100.0%   | 100.0%   | optimal |
| llama-cpp | .../Llama-3.2-3B-Instruct-Q8_0.gguf                      | 10 | 1226.97  | 41.8  | 708.2        | 3697.0        | 100.0%   | 80.0%    | -       |
```

Q4\_K\_M remains the sole Pareto-optimal configuration, now also dominant on task quality (100% vs 80%).

### `llm-bench recommend` (all 4 constraints)

```
Recommendation
──────────────────────────────────────────
  Backend  : llama-cpp
  Model    : Llama-3.2-3B-Instruct-Q4_K_M.gguf
  N        : 10
  p95      : 915.86 ms
  tok/s    : 55.5
  VRAM     : 2361.0 MB
  Sanity   : 100.0%
  Task Q   : 100.0%

Why: lowest p95 among 1 candidate(s) passing all constraints; Pareto-optimal.

Excluded (1)
──────────────────────────────────────────
  llama-cpp  Llama-3.2-3B-Instruct-Q8_0.gguf  →  p95 latency too high (1227.0 ms > 1000.0 ms)
```

Note: Q8\_0 was excluded by `--max-p95-ms 1000` before the quality constraint could apply.
Removing the latency constraint would additionally exclude it for task quality below `--min-quality 1.0`.

### Q8\_0 task quality failure — root cause

Q8\_0 scores 80% (4/5 prompts pass). The failing prompt is **prompt 3: "What are the main
differences between BERT and GPT?"** — rubric: `contains_all: [bert, gpt]`.

At `max_tokens=50`, Q8\_0's response focuses entirely on BERT for the first ~50 tokens
without reaching "GPT". The response is grammatically valid and on-topic — it just hasn't
finished discussing both models when truncated. Q4\_K\_M generates a more compact
answer that mentions both models within the 50-token budget.

**Confirmed diagnostic** (single-prompt run at `max_tokens=100`):
Q8\_0 passes the BERT/GPT rubric at 100 tokens (task_quality=1.00; mean_output_chars=529).

**Interpretation**: this is a **response-organisation difference between quantizations**,
not quality degradation. Q8\_0 produces longer, more elaborate answers; Q4\_K\_M produces
more compact ones that fit the rubric within the token budget. Both are valid LLM behaviors.
The finding is benchmark-configuration-specific: the rubric + max_tokens combination
surfaces a real operational difference worth knowing.

## Limitations

- Single machine, single session: no statistical replication across reboots or days.
- N=10 requests: p95 is the worst of 10 observations, not a stable tail-latency estimate.
- Task quality rubric is **deterministic structural** (keyword presence), not semantic.
  Passing the rubric does not guarantee factual accuracy or output depth.
- Q8\_0 task quality at 80% is specific to `max_tokens=50`. At `max_tokens=100`, Q8\_0
  passes all 5 rubric checks. The finding reflects this workload configuration.
- Conventional expectation: Q8\_0 produces near-lossless outputs vs Q4\_K\_M compression.
  This rubric does not measure that dimension — it measures structural term coverage only.
- `peak_cuda_memory_mb = 0.0` for all llama-cpp runs. Use `peak_vram_memory_mb`.
- Q8\_0 at 3697 MiB leaves only 399 MiB VRAM headroom. Other GPU processes running
  concurrently could cause OOM at n\_gpu\_layers=99.
- Prompts from `data/prompts/smoke.txt` are short (5 prompts, cycling). Longer prompts
  or larger `max_tokens` would increase KV cache pressure and may tighten the Q8\_0
  VRAM margin further.
- `repeated_output_count = 10` is structural (cycling deterministic prompts), not
  degeneration. See [Output Sanity Metrics](#output-sanity-metrics-v013) for details.
