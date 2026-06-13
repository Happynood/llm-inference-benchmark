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

Real backend results will be added here after each backend implementation.
Update this file and commit when benchmark numbers change (per CLAUDE.md).
