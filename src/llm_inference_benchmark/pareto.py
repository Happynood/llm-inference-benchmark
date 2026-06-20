"""Pareto dominance analysis for benchmark configuration selection.

Identifies which benchmark runs are Pareto-optimal: no other run is at least
as good on every metric and strictly better on at least one.

Optimisation directions:
  - p95_latency_ms          → minimise  (mandatory)
  - tokens_per_second       → maximise  (mandatory)
  - peak_vram_memory_mb     → minimise  (optional; compared only when both rows have a value)
  - sanity_pass_rate        → maximise  (optional; compared only when both rows have a value)
  - task_quality_pass_rate  → maximise  (optional; compared only when both rows have a value)
  - perplexity              → minimise  (optional; compared only when both rows have a value)
  - judge_score             → maximise  (optional; compared only when both rows have a value)

Missing optional metrics narrow the comparison set rather than crashing or
penalising either row.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from llm_inference_benchmark.compare import RunRow, load_csv

_PARETO_HEADERS = [
    "Backend",
    "Model",
    "N",
    "p95 (ms)",
    "tok/s",
    "Load (ms)",
    "TTFT p50 (ms)",
    "CPU mem (MB)",
    "VRAM mem (MB)",
    "Sanity %",
    "Task Q %",
    "PPL",
    "Judge",
    "Pareto",
]


def dominates(a: RunRow, b: RunRow) -> bool:
    """Return True if *a* Pareto-dominates *b*.

    *a* dominates *b* when *a* is no worse than *b* on every available metric
    and strictly better on at least one.  Optional metrics participate only
    when both rows carry a non-None value for that metric.
    """
    # Each entry: (a_value, b_value, minimise?)
    comparisons: list[tuple[float, float, bool]] = [
        (a.p95_latency_ms, b.p95_latency_ms, True),
        (a.tokens_per_second, b.tokens_per_second, False),
    ]
    if a.peak_vram_memory_mb is not None and b.peak_vram_memory_mb is not None:
        comparisons.append((a.peak_vram_memory_mb, b.peak_vram_memory_mb, True))
    if a.sanity_pass_rate is not None and b.sanity_pass_rate is not None:
        comparisons.append((a.sanity_pass_rate, b.sanity_pass_rate, False))
    if a.task_quality_pass_rate is not None and b.task_quality_pass_rate is not None:
        comparisons.append((a.task_quality_pass_rate, b.task_quality_pass_rate, False))
    if a.perplexity is not None and b.perplexity is not None:
        comparisons.append((a.perplexity, b.perplexity, True))
    if a.judge_score is not None and b.judge_score is not None:
        comparisons.append((a.judge_score, b.judge_score, False))
    if a.model_load_ms is not None and b.model_load_ms is not None:
        comparisons.append((a.model_load_ms, b.model_load_ms, True))
    if a.p50_ttft_ms is not None and b.p50_ttft_ms is not None:
        comparisons.append((a.p50_ttft_ms, b.p50_ttft_ms, True))

    strictly_better = False
    for a_val, b_val, minimise in comparisons:
        if minimise:
            if a_val > b_val:
                return False
            if a_val < b_val:
                strictly_better = True
        else:
            if a_val < b_val:
                return False
            if a_val > b_val:
                strictly_better = True

    return strictly_better


def pareto_classify(rows: list[RunRow]) -> list[tuple[RunRow, bool]]:
    """Classify each row as Pareto-optimal (True) or dominated (False).

    A row is optimal when no other row dominates it.  Input order is preserved.
    """
    if not rows:
        return []
    return [
        (candidate, not any(dominates(other, candidate) for j, other in enumerate(rows) if j != i))
        for i, candidate in enumerate(rows)
    ]


def render_pareto_table(classified: list[tuple[RunRow, bool]]) -> str:
    """Render classified rows as a GitHub-Flavored Markdown table."""

    def fmt_opt(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.1f}"

    def fmt_rate(v: float | None) -> str:
        return "N/A" if v is None else f"{v * 100:.1f}%"

    def fmt_pareto(is_opt: bool) -> str:
        return "optimal" if is_opt else "-"

    def fmt_ppl(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.2f}"

    data: list[list[str]] = [
        [
            r.backend,
            r.model,
            str(r.request_count),
            f"{r.p95_latency_ms:.2f}",
            f"{r.tokens_per_second:.1f}",
            fmt_opt(r.model_load_ms),
            fmt_opt(r.p50_ttft_ms),
            f"{r.peak_cpu_memory_mb:.1f}",
            fmt_opt(r.peak_vram_memory_mb),
            fmt_rate(r.sanity_pass_rate),
            fmt_rate(r.task_quality_pass_rate),
            fmt_ppl(r.perplexity),
            fmt_rate(r.judge_score),
            fmt_pareto(is_opt),
        ]
        for r, is_opt in classified
    ]

    widths = [len(h) for h in _PARETO_HEADERS]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(s: str, w: int) -> str:
        return s.ljust(w)

    header_line = (
        "| " + " | ".join(pad(h, w) for h, w in zip(_PARETO_HEADERS, widths, strict=True)) + " |"
    )
    sep_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    data_lines = [
        "| " + " | ".join(pad(c, w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in data
    ]
    return "\n".join([header_line, sep_line, *data_lines])


def render_pareto_json(classified: list[tuple[RunRow, bool]]) -> str:
    """Serialize classified rows to a JSON array string.

    Each object contains all RunRow fields plus ``"pareto": true/false``.
    """
    data = [
        {
            "backend": r.backend,
            "model": r.model,
            "request_count": r.request_count,
            "p50_latency_ms": r.p50_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "tokens_per_second": r.tokens_per_second,
            "decode_tokens_per_second": r.decode_tokens_per_second,
            "mean_input_tokens": r.mean_input_tokens,
            "mean_output_tokens": r.mean_output_tokens,
            "model_load_ms": r.model_load_ms,
            "p50_ttft_ms": r.p50_ttft_ms,
            "p95_ttft_ms": r.p95_ttft_ms,
            "peak_cpu_memory_mb": r.peak_cpu_memory_mb,
            "peak_cuda_memory_mb": r.peak_cuda_memory_mb,
            "peak_vram_memory_mb": r.peak_vram_memory_mb,
            "sanity_pass_rate": r.sanity_pass_rate,
            "task_quality_pass_rate": r.task_quality_pass_rate,
            "perplexity": r.perplexity,
            "judge_score": r.judge_score,
            "p95_latency_ms_std": r.p95_latency_ms_std,
            "tokens_per_second_std": r.tokens_per_second_std,
            "pareto": is_opt,
        }
        for r, is_opt in classified
    ]
    return json.dumps(data, indent=2)


def build_pareto_table(paths: list[str | Path]) -> str:
    """Load CSVs, classify, and return a Markdown Pareto table."""
    if not paths:
        raise ValueError("At least one CSV path is required")
    rows = [load_csv(p) for p in paths]
    return render_pareto_table(pareto_classify(rows))


def build_pareto_json(paths: list[str | Path]) -> str:
    """Load CSVs, classify, and return a JSON array string."""
    if not paths:
        raise ValueError("At least one CSV path is required")
    rows = [load_csv(p) for p in paths]
    return render_pareto_json(pareto_classify(rows))


_PARETO_CSV_FIELDS = [
    "backend",
    "model",
    "request_count",
    "p50_latency_ms",
    "p95_latency_ms",
    "tokens_per_second",
    "decode_tokens_per_second",
    "mean_input_tokens",
    "mean_output_tokens",
    "model_load_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "peak_cpu_memory_mb",
    "peak_cuda_memory_mb",
    "peak_vram_memory_mb",
    "sanity_pass_rate",
    "task_quality_pass_rate",
    "perplexity",
    "judge_score",
    "p95_latency_ms_std",
    "tokens_per_second_std",
    "pareto",
]


def _or_empty(v: float | None) -> float | str:
    return "" if v is None else v


def render_pareto_csv(classified: list[tuple[RunRow, bool]]) -> str:
    """Serialize classified rows to a CSV string with snake_case headers.

    Includes all RunRow fields plus a ``pareto`` column (``True``/``False``)
    so callers can filter in pandas without losing dominated rows.  Absent
    optional metrics are written as empty cells so ``pandas.read_csv()``
    produces ``NaN`` automatically.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_PARETO_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for r, is_opt in classified:
        writer.writerow(
            {
                "backend": r.backend,
                "model": r.model,
                "request_count": r.request_count,
                "p50_latency_ms": r.p50_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
                "tokens_per_second": r.tokens_per_second,
                "decode_tokens_per_second": _or_empty(r.decode_tokens_per_second),
                "mean_input_tokens": _or_empty(r.mean_input_tokens),
                "mean_output_tokens": _or_empty(r.mean_output_tokens),
                "model_load_ms": _or_empty(r.model_load_ms),
                "p50_ttft_ms": _or_empty(r.p50_ttft_ms),
                "p95_ttft_ms": _or_empty(r.p95_ttft_ms),
                "peak_cpu_memory_mb": r.peak_cpu_memory_mb,
                "peak_cuda_memory_mb": _or_empty(r.peak_cuda_memory_mb),
                "peak_vram_memory_mb": _or_empty(r.peak_vram_memory_mb),
                "sanity_pass_rate": _or_empty(r.sanity_pass_rate),
                "task_quality_pass_rate": _or_empty(r.task_quality_pass_rate),
                "perplexity": _or_empty(r.perplexity),
                "judge_score": _or_empty(r.judge_score),
                "p95_latency_ms_std": _or_empty(r.p95_latency_ms_std),
                "tokens_per_second_std": _or_empty(r.tokens_per_second_std),
                "pareto": is_opt,
            }
        )
    return buf.getvalue().rstrip("\n")


def build_pareto_csv(paths: list[str | Path]) -> str:
    """Load CSVs, classify, and return a CSV string with a ``pareto`` column."""
    if not paths:
        raise ValueError("At least one CSV path is required")
    rows = [load_csv(p) for p in paths]
    return render_pareto_csv(pareto_classify(rows))
