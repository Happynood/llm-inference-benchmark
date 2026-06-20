"""Two-run regression diff: compare a baseline benchmark CSV to a current one."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

from llm_inference_benchmark.compare import load_csv

_DIFF_CSV_FIELDS = ["metric", "baseline", "current", "change_pct", "direction"]

# (label, baseline_val, current_val, lower_is_better, required)
_Candidate = tuple[str, float | None, float | None, bool, bool]


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


def _regression_pct(
    baseline: float | None, current: float | None, lower_is_better: bool
) -> float | None:
    """Return regression magnitude as a positive %, or None if not a regression / not computable."""
    if baseline is None or current is None:
        return None
    pct = _pct_change(baseline, current)
    if pct is None:
        return None
    magnitude = pct if lower_is_better else -pct
    return magnitude if magnitude > 0 else None


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


def find_regressions(
    baseline_path: str | Path,
    current_path: str | Path,
    threshold_pct: float = 0.0,
) -> list[str]:
    """Return labels of metrics that regressed by more than threshold_pct percent.

    Optional metrics absent from both runs are skipped.  Pass threshold_pct=5.0
    to ignore regressions smaller than 5 %, which is useful for noisy workloads.
    """
    b = load_csv(baseline_path)
    c = load_csv(current_path)

    candidates: list[tuple[str, float | None, float | None, bool]] = [
        ("p50 (ms)", b.p50_latency_ms, c.p50_latency_ms, True),
        ("p95 (ms)", b.p95_latency_ms, c.p95_latency_ms, True),
        ("tok/s", b.tokens_per_second, c.tokens_per_second, False),
        ("Decode tok/s", b.decode_tokens_per_second, c.decode_tokens_per_second, False),
        ("Load (ms)", b.model_load_ms, c.model_load_ms, True),
        ("TTFT p50 (ms)", b.p50_ttft_ms, c.p50_ttft_ms, True),
        ("TTFT p95 (ms)", b.p95_ttft_ms, c.p95_ttft_ms, True),
        ("VRAM (MB)", b.peak_vram_memory_mb, c.peak_vram_memory_mb, True),
        ("CPU mem (MB)", b.peak_cpu_memory_mb, c.peak_cpu_memory_mb, True),
        ("Sanity %", b.sanity_pass_rate, c.sanity_pass_rate, False),
        ("Task Q %", b.task_quality_pass_rate, c.task_quality_pass_rate, False),
        ("PPL", b.perplexity, c.perplexity, True),
        ("Judge", b.judge_score, c.judge_score, False),
    ]

    regressions: list[str] = []
    for label, b_val, c_val, lower_is_better in candidates:
        if b_val is None and c_val is None:
            continue
        mag = _regression_pct(b_val, c_val, lower_is_better)
        if mag is not None and mag > threshold_pct:
            regressions.append(label)
    return regressions


def _direction(b_val: float | None, c_val: float | None, lower_is_better: bool) -> str:
    if b_val is None or c_val is None:
        return "n/a"
    pct = _pct_change(b_val, c_val)
    if pct is None:
        return "n/a"
    if abs(pct) < 0.05:
        return "neutral"
    good = (pct < 0) == lower_is_better
    return "improvement" if good else "regression"


def render_diff_csv(candidates: list[_Candidate]) -> str:
    """Serialize diff candidates to a CSV string.

    Columns: metric, baseline, current, change_pct, direction.
    Optional metrics absent from both runs are omitted.
    Absent individual values and uncomputable change_pct are empty cells so
    ``pandas.read_csv()`` produces ``NaN`` automatically.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_DIFF_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for label, b_val, c_val, lower_is_better, required in candidates:
        if not required and b_val is None and c_val is None:
            continue
        if b_val is not None and c_val is not None:
            raw = _pct_change(b_val, c_val)
            chg: float | str = "" if raw is None else round(raw, 2)
        else:
            chg = ""
        writer.writerow(
            {
                "metric": label,
                "baseline": "" if b_val is None else b_val,
                "current": "" if c_val is None else c_val,
                "change_pct": chg,
                "direction": _direction(b_val, c_val, lower_is_better),
            }
        )
    return buf.getvalue().rstrip("\n")


def build_diff_csv(baseline_path: str | Path, current_path: str | Path) -> str:
    """Load two benchmark CSVs and return a CSV diff string.

    Each row represents one metric. Columns: metric, baseline, current,
    change_pct, direction. Optional metrics absent from both runs are omitted.
    """
    b = load_csv(baseline_path)
    c = load_csv(current_path)
    candidates: list[_Candidate] = [
        ("p50 (ms)", b.p50_latency_ms, c.p50_latency_ms, True, True),
        ("p95 (ms)", b.p95_latency_ms, c.p95_latency_ms, True, True),
        ("tok/s", b.tokens_per_second, c.tokens_per_second, False, True),
        ("Decode tok/s", b.decode_tokens_per_second, c.decode_tokens_per_second, False, False),
        ("Load (ms)", b.model_load_ms, c.model_load_ms, True, False),
        ("TTFT p50 (ms)", b.p50_ttft_ms, c.p50_ttft_ms, True, False),
        ("TTFT p95 (ms)", b.p95_ttft_ms, c.p95_ttft_ms, True, False),
        ("VRAM (MB)", b.peak_vram_memory_mb, c.peak_vram_memory_mb, True, False),
        ("CPU mem (MB)", b.peak_cpu_memory_mb, c.peak_cpu_memory_mb, True, True),
        ("Sanity %", b.sanity_pass_rate, c.sanity_pass_rate, False, False),
        ("Task Q %", b.task_quality_pass_rate, c.task_quality_pass_rate, False, False),
        ("PPL", b.perplexity, c.perplexity, True, False),
        ("Judge", b.judge_score, c.judge_score, False, False),
    ]
    return render_diff_csv(candidates)


def build_diff_json(baseline_path: str | Path, current_path: str | Path) -> str:
    """Load two benchmark CSVs and return a JSON diff string.

    The returned object has three keys:
    - ``baseline`` / ``current``: run metadata (file, backend, model, request_count).
    - ``metrics``: list of per-metric objects with ``label``, ``baseline``,
      ``current``, ``change_pct`` (null when not computable), and ``direction``
      (``"improvement"``, ``"regression"``, ``"neutral"``, or ``"n/a"``).

    Optional metrics absent from *both* runs are omitted from the list,
    consistent with the Markdown table output.
    """
    b = load_csv(baseline_path)
    c = load_csv(current_path)

    def _entry(
        label: str,
        b_val: float | None,
        c_val: float | None,
        lower_is_better: bool,
    ) -> dict[str, object]:
        pct: float | None = None
        if b_val is not None and c_val is not None:
            raw = _pct_change(b_val, c_val)
            pct = round(raw, 2) if raw is not None else None
        return {
            "label": label,
            "baseline": b_val,
            "current": c_val,
            "change_pct": pct,
            "direction": _direction(b_val, c_val, lower_is_better),
        }

    # (label, baseline_val, current_val, lower_is_better, required)
    candidates: list[tuple[str, float | None, float | None, bool, bool]] = [
        ("p50 (ms)", b.p50_latency_ms, c.p50_latency_ms, True, True),
        ("p95 (ms)", b.p95_latency_ms, c.p95_latency_ms, True, True),
        ("tok/s", b.tokens_per_second, c.tokens_per_second, False, True),
        ("Decode tok/s", b.decode_tokens_per_second, c.decode_tokens_per_second, False, False),
        ("Load (ms)", b.model_load_ms, c.model_load_ms, True, False),
        ("TTFT p50 (ms)", b.p50_ttft_ms, c.p50_ttft_ms, True, False),
        ("TTFT p95 (ms)", b.p95_ttft_ms, c.p95_ttft_ms, True, False),
        ("VRAM (MB)", b.peak_vram_memory_mb, c.peak_vram_memory_mb, True, False),
        ("CPU mem (MB)", b.peak_cpu_memory_mb, c.peak_cpu_memory_mb, True, True),
        ("Sanity %", b.sanity_pass_rate, c.sanity_pass_rate, False, False),
        ("Task Q %", b.task_quality_pass_rate, c.task_quality_pass_rate, False, False),
        ("PPL", b.perplexity, c.perplexity, True, False),
        ("Judge", b.judge_score, c.judge_score, False, False),
    ]

    metrics = [
        _entry(label, b_val, c_val, lower_is_better)
        for label, b_val, c_val, lower_is_better, required in candidates
        if required or not (b_val is None and c_val is None)
    ]

    data: dict[str, object] = {
        "baseline": {
            "file": Path(baseline_path).name,
            "backend": b.backend,
            "model": b.model,
            "request_count": b.request_count,
        },
        "current": {
            "file": Path(current_path).name,
            "backend": c.backend,
            "model": c.model,
            "request_count": c.request_count,
        },
        "metrics": metrics,
    }
    return json.dumps(data, indent=2)
