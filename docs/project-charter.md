# Project Charter

## Fixed Project Idea

`llm-inference-benchmark` is a reproducible, laptop-friendly LLM inference optimization lab.
It starts as a benchmark harness, but the fixed product direction is broader: run controlled
experiments over model, backend, precision, runtime parameters, and workload, then identify the
best configuration under explicit latency, memory, throughput, and quality constraints.

The core formula is:

```text
model x backend x precision x parameters x workload -> Pareto-useful configuration
```

The project is intentionally small and inspectable. Its value is that every result can be traced
back to a config, prompt set, backend implementation, dependency environment, hardware context,
measurement method, and eventually a lightweight quality-evaluation method.

## Positioning

This project should answer a practical optimization question:

```text
Which inference configuration should I choose for this model, hardware, workload, and SLA?
```

It should not stop at "which run was fastest." The long-term output should be a recommendation,
or at least a Pareto table, that explains the trade-off between latency, throughput, memory, and
quality retention.

The current repository is still an early scaffold. Early milestones should stay small and
CI-friendly, but every new feature should move the project toward reproducible configuration
selection rather than ad-hoc timing scripts.

## Target Audience

- Engineers comparing inference backends on ordinary local hardware.
- ML practitioners who need reproducible, scriptable optimization experiments.
- Engineers deciding between backend, quantization, batch/concurrency, and generation settings.
- Portfolio reviewers who want to see clean backend abstractions, metrics discipline, testing, and
  honest documentation.

## Goals

1. Make benchmark runs reproducible from repository files.
2. Compare multiple inference backends under the same prompts, config shape, and metric definitions.
3. Track precision and quantization metadata so runs can be compared as configurations, not just
   as backend names.
4. Report latency, throughput, peak memory, and lifecycle timings in a transparent way.
5. Add lightweight quality checks so speed and memory are not optimized in isolation.
6. Produce comparison tables first, then Pareto-style analysis and constraint-based
   recommendations.
7. Keep the implementation small, typed, tested, CI-friendly, and easy to extend.
8. Separate harness validation results from production performance claims.

## Non-Goals

- Do not claim universal backend rankings or leaderboard-style results.
- Do not require large model downloads, GPUs, or paid cloud hardware for the default workflow.
- Do not commit model weights, datasets, checkpoints, generated benchmark CSVs, caches, or secrets.
- Do not optimize every backend to state-of-the-art production settings before the measurement
  method is solid.
- Do not add broad serving-platform features unless they directly support controlled inference
  experiments.

## Design Constraints

- Every public benchmark claim must include command, config, model, backend, hardware context, and
  limitations.
- CI must remain lightweight: mock tests run without model weights, GPU, or optional heavy extras.
- Optional backend tests may be integration tests and must be skippable when dependencies are absent.
- Metrics schema changes must be deliberate and backward-compatible when practical.
- Generated outputs are artifacts, not source, except for tiny test fixtures.
- Documentation must state when a result validates the harness rather than representing production
  inference performance.
- New comparisons should preserve enough metadata to explain why a configuration won, lost, or was
  excluded by constraints.

## Roadmap Priority

1. CSV comparison table for multiple benchmark runs.
2. Run manifest and fingerprint: git commit, config hash, prompts hash, backend version, dependency
   versions, hardware, OS, and timestamp.
3. Workload profiles: short chat, summarization, code completion, and longer-context smoke tests.
4. `llama.cpp` / GGUF backend with quantization metadata.
5. Basic output sanity checks and small quality-eval hooks so speed is not reported without
   minimal correctness validation.
6. Lifecycle metrics: model load time, warmup behavior, cold-start latency, idle memory, and peak
   memory during run.
7. Parameter sweeps for batch size, concurrency, input length, output length, and generation
   settings.
8. Pareto table and constraint-based recommender.
9. ONNX Runtime, OpenAI-compatible endpoints, and vLLM after the local reproducibility story is
   strong.

## Decision Rule

Accept a feature if it strengthens at least one of these pillars:

- reproducibility
- backend comparability
- configuration optimization
- low-resource local inference
- quality/performance trade-off analysis
- measurement honesty
- portfolio-quality engineering

Delay or reject a feature if it mainly adds surface area without improving controlled experiments,
metadata quality, comparison quality, or recommendation quality.
