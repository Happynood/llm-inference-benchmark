"""Two-run regression diff: compare a baseline benchmark CSV to a current one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_inference_benchmark.compare import load_csv


@dataclass(frozen=True)
class _DiffRow:
    label: str
    baseline_str: str
    current_str: str
    change_str: str


def _pct_change(baseline: float, current: float) -> float | None:
    if baseline == 0.0:
        return None
    return (current - baseline) / abs(baseline) * 100.0


def _fmt_change(baseline: float | None, current: float | None, lower_is_better: bool) -> str:
    if baseline is None or current is None:
        return "N/A"
    pct = _pct_change(baseline, current)
    if pct is None:
        return "—"
    sign = "+" if pct >= 0 else ""
    s = f"{sign}{pct:.1f}%"
    if abs(pct) < 0.05:
        return s
    good = (pct < 0) == lower_is_better
    return s + (" ✓" if good else " ✗")


def _req_row(
    label: str,
    b_val: float,
    c_val: float,
    fmt: str,
    lower_is_better: bool,
) -> _DiffRow:
    return _DiffRow(
        label,
        f"{b_val:{fmt}}",
        f"{c_val:{fmt}}",
        _fmt_change(b_val, c_val, lower_is_better),
    )


def _opt_row(
    label: str,
    b_val: float | None,
    c_val: float | None,
    fmt: str,
    lower_is_better: bool,
) -> _DiffRow | None:
    if b_val is None and c_val is None:
        return None
    b_s = "N/A" if b_val is None else f"{b_val:{fmt}}"
    c_s = "N/A" if c_val is None else f"{c_val:{fmt}}"
    return _DiffRow(label, b_s, c_s, _fmt_change(b_val, c_val, lower_is_better))


def _render_table(rows: list[_DiffRow]) -> str:
    headers = ["Metric", "Baseline", "Current", "Change"]
    data = [[r.label, r.baseline_str, r.current_str, r.change_str] for r in rows]
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def lpad(s: str, w: int) -> str:
        return s.rjust(w)

    def rpad(s: str, w: int) -> str:
        return s.ljust(w)

    header_line = (
        "| "
        + rpad(headers[0], widths[0])
        + " | "
        + " | ".join(lpad(h, w) for h, w in zip(headers[1:], widths[1:], strict=True))
        + " |"
    )
    sep_line = "|:" + "-" * widths[0] + "-|" + "|".join("-" * (w + 1) + ":|" for w in widths[1:])
    data_lines = [
        "| "
        + rpad(row[0], widths[0])
        + " | "
        + " | ".join(lpad(cell, w) for cell, w in zip(row[1:], widths[1:], strict=True))
        + " |"
        for row in data
    ]
    return "\n".join([header_line, sep_line, *data_lines])


def build_diff_table(baseline_path: str | Path, current_path: str | Path) -> str:
    """Load two benchmark CSVs and return a Markdown diff table."""
    b = load_csv(baseline_path)
    c = load_csv(current_path)

    rows: list[_DiffRow] = [
        _req_row("p50 (ms)", b.p50_latency_ms, c.p50_latency_ms, ".2f", True),
        _req_row("p95 (ms)", b.p95_latency_ms, c.p95_latency_ms, ".2f", True),
        _req_row("tok/s", b.tokens_per_second, c.tokens_per_second, ".1f", False),
    ]

    for opt in [
        _opt_row(
            "Decode tok/s",
            b.decode_tokens_per_second,
            c.decode_tokens_per_second,
            ".1f",
            False,
        ),
        _opt_row("Load (ms)", b.model_load_ms, c.model_load_ms, ".2f", True),
        _opt_row("TTFT p50 (ms)", b.p50_ttft_ms, c.p50_ttft_ms, ".2f", True),
        _opt_row("TTFT p95 (ms)", b.p95_ttft_ms, c.p95_ttft_ms, ".2f", True),
        _opt_row("VRAM (MB)", b.peak_vram_memory_mb, c.peak_vram_memory_mb, ".1f", True),
    ]:
        if opt is not None:
            rows.append(opt)

    rows.append(_req_row("CPU mem (MB)", b.peak_cpu_memory_mb, c.peak_cpu_memory_mb, ".1f", True))

    for opt in [
        _opt_row("Sanity %", b.sanity_pass_rate, c.sanity_pass_rate, ".1%", False),
        _opt_row(
            "Task Q %",
            b.task_quality_pass_rate,
            c.task_quality_pass_rate,
            ".1%",
            False,
        ),
        _opt_row("PPL", b.perplexity, c.perplexity, ".2f", True),
        _opt_row("Judge", b.judge_score, c.judge_score, ".1%", False),
    ]:
        if opt is not None:
            rows.append(opt)

    bl_name = Path(baseline_path).name
    cu_name = Path(current_path).name
    header = "\n".join(
        [
            "## Benchmark Diff",
            "",
            f"Baseline : {b.backend} | {b.model} | N={b.request_count}  ({bl_name})",
            f"Current  : {c.backend} | {c.model} | N={c.request_count}  ({cu_name})",
            "",
        ]
    )
    legend = (
        "\n✓ = improvement  ✗ = regression"
        "  (lower is better for latency/VRAM/PPL; higher for tok/s, sanity, quality, judge)"
    )
    return header + _render_table(rows) + legend
