"""Read benchmark CSV files and render a Markdown comparison table."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_REQUIRED_COLS = {
    "request_count",
    "backend",
    "model",
    "p50_latency_ms",
    "p95_latency_ms",
    "tokens_per_second",
    "peak_cpu_memory_mb",
}

_HEADERS = [
    "Backend",
    "Model",
    "N",
    "p50 (ms)",
    "p95 (ms)",
    "tok/s",
    "CPU mem (MB)",
    "CUDA mem (MB)",
    "VRAM mem (MB)",
    "Sanity %",
    "Task Q %",
]


@dataclass(frozen=True)
class RunRow:
    backend: str
    model: str
    request_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float
    peak_cpu_memory_mb: float
    peak_cuda_memory_mb: float | None
    peak_vram_memory_mb: float | None = None  # absent in older CSVs → None
    sanity_pass_rate: float | None = None  # absent in older CSVs → None
    task_quality_pass_rate: float | None = None  # absent when no quality_file was set
    task_quality_checked_count: int | None = None  # absent when no quality_file was set


def _parse_optional_float(row: dict[str, str], key: str, path: str | Path) -> float | None:
    if key not in row:
        return None
    raw = row[key]
    stripped = raw.strip()
    if raw and not stripped:
        raise ValueError(f"{path}: invalid {key} value: {raw!r}")
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid {key} value: {raw!r}") from exc


def load_csv(path: str | Path) -> RunRow:
    """Parse a single-run benchmark CSV into a RunRow."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    if len(rows) > 1:
        raise ValueError(f"Expected 1 data row in {path}, got {len(rows)}")
    row = rows[0]

    missing = _REQUIRED_COLS - row.keys()
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    peak_cuda = _parse_optional_float(row, "peak_cuda_memory_mb", path)
    peak_vram = _parse_optional_float(row, "peak_vram_memory_mb", path)
    sanity = _parse_optional_float(row, "sanity_pass_rate", path)
    task_quality = _parse_optional_float(row, "task_quality_pass_rate", path)
    task_quality_checked_raw = _parse_optional_float(row, "task_quality_checked_count", path)
    task_quality_checked = (
        int(task_quality_checked_raw) if task_quality_checked_raw is not None else None
    )

    return RunRow(
        backend=row["backend"],
        model=row["model"],
        request_count=int(row["request_count"]),
        p50_latency_ms=float(row["p50_latency_ms"]),
        p95_latency_ms=float(row["p95_latency_ms"]),
        tokens_per_second=float(row["tokens_per_second"]),
        peak_cpu_memory_mb=float(row["peak_cpu_memory_mb"]),
        peak_cuda_memory_mb=peak_cuda,
        peak_vram_memory_mb=peak_vram,
        sanity_pass_rate=sanity,
        task_quality_pass_rate=task_quality,
        task_quality_checked_count=task_quality_checked,
    )


def sort_rows(rows: list[RunRow], sort_by: str = "p95") -> list[RunRow]:
    """Return a new sorted list; does not mutate the input."""
    if sort_by == "backend":
        return sorted(rows, key=lambda r: (r.backend, r.model))
    if sort_by == "model":
        return sorted(rows, key=lambda r: (r.model, r.backend))
    return sorted(rows, key=lambda r: r.p95_latency_ms)  # default: p95 ascending


def render_table(rows: list[RunRow]) -> str:
    """Render RunRows as a GitHub-Flavored Markdown table string."""

    def fmt_optional(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.1f}"

    def fmt_rate(v: float | None) -> str:
        return "N/A" if v is None else f"{v * 100:.1f}%"

    data: list[list[str]] = [
        [
            r.backend,
            r.model,
            str(r.request_count),
            f"{r.p50_latency_ms:.2f}",
            f"{r.p95_latency_ms:.2f}",
            f"{r.tokens_per_second:.1f}",
            f"{r.peak_cpu_memory_mb:.1f}",
            fmt_optional(r.peak_cuda_memory_mb),
            fmt_optional(r.peak_vram_memory_mb),
            fmt_rate(r.sanity_pass_rate),
            fmt_rate(r.task_quality_pass_rate),
        ]
        for r in rows
    ]

    widths = [len(h) for h in _HEADERS]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(s: str, w: int) -> str:
        return s.ljust(w)

    header_line = "| " + " | ".join(pad(h, w) for h, w in zip(_HEADERS, widths, strict=True)) + " |"
    sep_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    data_lines = [
        "| " + " | ".join(pad(c, w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in data
    ]
    return "\n".join([header_line, sep_line, *data_lines])


def build_comparison_table(paths: list[str | Path], sort_by: str = "p95") -> str:
    """Load CSV files, sort, and return a Markdown table string."""
    if not paths:
        raise ValueError("At least one CSV path is required")
    rows = [load_csv(p) for p in paths]
    return render_table(sort_rows(rows, sort_by))
