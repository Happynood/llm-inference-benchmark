# Benchmark Metrics

## Metric Definitions

| Metric | Unit | Description |
|--------|------|-------------|
| `request_count` | count | Number of benchmark requests (excluding warmup) |
| `p50_latency_ms` | ms | Median end-to-end latency per request |
| `p95_latency_ms` | ms | 95th-percentile latency per request |
| `tokens_per_second` | tok/s | Output tokens produced divided by total latency |
| `total_tokens` | count | Sum of input + output tokens across all requests |
| `backend` | string | Backend name (`mock`, `transformers`, `llama-cpp`, ...) |
| `model` | string | Model identifier as passed in config |
| `timestamp` | ISO 8601 | UTC timestamp when the run completed |

## Computation Notes

- **p95 latency**: linear interpolation between adjacent sorted samples
  (`idx = 0.95 * (N-1)`, interpolated between `sorted[floor(idx)]` and `sorted[ceil(idx)]`).
  This is the C1 method, matching NumPy's default interpolation.
- **tokens/sec**: output-only tokens divided by sum of per-request latencies
  (`output_tokens_total / sum(latency_ms) * 1000`).
  For v0.1 sequential execution, sum-of-latencies equals wall-clock elapsed time.
  **This equivalence breaks when concurrent execution is added** — the denominator will switch
  to wall-clock elapsed time at that point.
- **p50 ≈ p95 with mock backend**: the mock produces near-constant latency by design, so
  percentiles are nearly identical. Real backends with variable scheduling jitter will separate
  these values meaningfully.
- **Warmup requests** are excluded from all metric calculations.

## v0.1 Results (mock backend)

> Hardware: development laptop, CPU only, no actual inference.
> These numbers validate the harness, not real model performance.

| metric | value |
|--------|-------|
| backend | mock |
| model | mock-gpt2 |
| requests | 20 |
| p50_latency_ms | ~5 ms |
| p95_latency_ms | ~5 ms |
| tokens_per_second | ~10,000 (simulated) |

## v0.2 Results — `transformers` backend, CPU

> Hardware: Intel Core i5-11400H @ 2.70 GHz, 12 logical cores, CPU-only (no GPU).
> Software: Python 3.14.4, torch 2.12.0, transformers 5.12.0.
> Model: `sshleifer/tiny-gpt2` — 2-layer GPT-2 toy model, ~102 K params, ~4 MB.
> **Not representative of production models.** Used here to validate that the harness
> correctly measures end-to-end inference latency and token throughput.

Command:
```bash
uv run llm-bench --config configs/transformers-cpu.yaml --output benchmark-hf.csv
```

Config: `requests=10`, `warmup_requests=2`, `max_new_tokens=50`, `device=cpu`, `do_sample=false`.

| metric | value |
|--------|-------|
| backend | transformers |
| model | sshleifer/tiny-gpt2 |
| requests | 10 |
| warmup_requests | 2 |
| p50_latency_ms | 40.70 |
| p95_latency_ms | 46.19 |
| tokens_per_second | 1192.87 |
| total_tokens | 598 |

### Interpretation

- p50 of ~41 ms for a 2-layer toy model on CPU; a full GPT-2 (12 layers, 117 M params) would
  be ~10–20× slower (~400–800 ms). Llama 3 8B would be 100–1000× slower.
- ~1193 output tokens/sec on this model is meaningless as a production throughput figure —
  production models on CPU produce 5–50 tok/s.
- p95/p50 spread (~14%) reflects OS scheduling jitter, expected on CPU without process pinning.

Real backend results will be updated here after each new backend lands.
