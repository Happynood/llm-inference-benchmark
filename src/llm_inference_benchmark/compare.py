"""Read benchmark CSV files and render a Markdown comparison table."""

from __future__ import annotations

import csv
import io
import json
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
    "Out tok/s",
    "In tok",
    "Out tok",
    "Load (ms)",
    "TTFT p50 (ms)",
    "TTFT p95 (ms)",
    "TPOT p50 (ms)",
    "CPU mem (MB)",
    "CUDA mem (MB)",
    "VRAM mem (MB)",
    "Sanity %",
    "Task Q %",
    "PPL",
    "Judge",
    "Think tok",
    "Answer tok",
    "Think %",
    "Energy (J)",
    "Tok/J",
    "Throttle %",
]

# Indices into _HEADERS that are suppressed when every row shows "N/A".
# Mandatory columns (0-5 and 13) are never suppressed.
_OPTIONAL_COL_INDICES: frozenset[int] = frozenset(
    {6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
)


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
    perplexity: float | None = None  # absent unless config.measure_perplexity was set
    judge_score: float | None = None  # absent unless config.measure_judge was set
    model_load_ms: float | None = None  # absent in pre-v0.18 CSVs → None
    p50_ttft_ms: float | None = None  # absent unless backend ran with stream=True
    p95_ttft_ms: float | None = None  # absent unless backend ran with stream=True
    p50_tpot_ms: float | None = None  # absent unless TTFT available and output_tokens > 0
    tpot_stddev_ms: float | None = None  # absent unless >= 2 requests had TTFT data
    p95_latency_ms_std: float | None = None  # absent unless config.repeats > 1
    tokens_per_second_std: float | None = None  # absent unless config.repeats > 1
    mean_input_tokens: float | None = None  # absent in pre-v0.22 CSVs → None
    mean_output_tokens: float | None = None  # absent in pre-v0.22 CSVs → None
    decode_tokens_per_second: float | None = None  # absent in pre-v0.22 CSVs or zero-output runs
    # Hardware profile (v0.25) — absent in pre-v0.25 CSVs → None
    hw_cpu: str | None = None
    hw_cpu_cores: int | None = None
    hw_ram_gb: float | None = None
    hw_gpu: str | None = None
    hw_vram_gb: float | None = None
    hw_os: str | None = None
    # Reasoning token parser (v0.26) — absent when reasoning tags not configured
    mean_reasoning_tokens: float | None = None
    mean_answer_tokens: float | None = None
    reasoning_fraction: float | None = None
    # Energy efficiency (v0.27) — absent when GPU/CPU power measurement unavailable
    energy_joules: float | None = None
    tokens_per_joule: float | None = None
    # Thermal throttling index (v0.28) — absent for concurrent/short/small runs
    thermal_throttle_pct: float | None = None


def _parse_optional_str(row: dict[str, str], key: str) -> str | None:
    raw = row.get(key, "").strip()
    return raw if raw else None


def _parse_optional_int(row: dict[str, str], key: str, path: str | Path) -> int | None:
    raw = row.get(key, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid {key} value: {raw!r}") from exc


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
    perplexity = _parse_optional_float(row, "perplexity", path)
    judge_score = _parse_optional_float(row, "judge_score", path)
    model_load_ms = _parse_optional_float(row, "model_load_ms", path)
    p50_ttft_ms = _parse_optional_float(row, "p50_ttft_ms", path)
    p95_ttft_ms = _parse_optional_float(row, "p95_ttft_ms", path)
    p50_tpot_ms = _parse_optional_float(row, "p50_tpot_ms", path)
    tpot_stddev_ms = _parse_optional_float(row, "tpot_stddev_ms", path)
    p95_std = _parse_optional_float(row, "p95_latency_ms_std", path)
    toks_std = _parse_optional_float(row, "tokens_per_second_std", path)
    mean_input_tokens = _parse_optional_float(row, "mean_input_tokens", path)
    mean_output_tokens = _parse_optional_float(row, "mean_output_tokens", path)
    decode_tps = _parse_optional_float(row, "decode_tokens_per_second", path)
    hw_cpu = _parse_optional_str(row, "hw_cpu")
    hw_cpu_cores = _parse_optional_int(row, "hw_cpu_cores", path)
    hw_ram_gb = _parse_optional_float(row, "hw_ram_gb", path)
    hw_gpu = _parse_optional_str(row, "hw_gpu")
    hw_vram_gb = _parse_optional_float(row, "hw_vram_gb", path)
    hw_os = _parse_optional_str(row, "hw_os")
    mean_reasoning_tokens = _parse_optional_float(row, "mean_reasoning_tokens", path)
    mean_answer_tokens = _parse_optional_float(row, "mean_answer_tokens", path)
    reasoning_fraction = _parse_optional_float(row, "reasoning_fraction", path)
    energy_joules = _parse_optional_float(row, "energy_joules", path)
    tokens_per_joule = _parse_optional_float(row, "tokens_per_joule", path)
    thermal_throttle_pct = _parse_optional_float(row, "thermal_throttle_pct", path)

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
        perplexity=perplexity,
        judge_score=judge_score,
        model_load_ms=model_load_ms,
        p50_ttft_ms=p50_ttft_ms,
        p95_ttft_ms=p95_ttft_ms,
        p50_tpot_ms=p50_tpot_ms,
        tpot_stddev_ms=tpot_stddev_ms,
        p95_latency_ms_std=p95_std,
        tokens_per_second_std=toks_std,
        mean_input_tokens=mean_input_tokens,
        mean_output_tokens=mean_output_tokens,
        decode_tokens_per_second=decode_tps,
        hw_cpu=hw_cpu,
        hw_cpu_cores=hw_cpu_cores,
        hw_ram_gb=hw_ram_gb,
        hw_gpu=hw_gpu,
        hw_vram_gb=hw_vram_gb,
        hw_os=hw_os,
        mean_reasoning_tokens=mean_reasoning_tokens,
        mean_answer_tokens=mean_answer_tokens,
        reasoning_fraction=reasoning_fraction,
        energy_joules=energy_joules,
        tokens_per_joule=tokens_per_joule,
        thermal_throttle_pct=thermal_throttle_pct,
    )


_FILTER_FIELDS: frozenset[str] = frozenset({"backend", "model"})


def filter_rows(rows: list[RunRow], filters: list[str]) -> list[RunRow]:
    """Return rows where every FIELD=PATTERN filter matches (case-insensitive substring).

    Raises ValueError for unsupported field names.
    """
    if not filters:
        return rows

    parsed: list[tuple[str, str]] = []
    for f in filters:
        if "=" not in f:
            raise ValueError(
                f"Invalid filter {f!r}: expected FIELD=PATTERN (e.g. backend=llama_cpp)"
            )
        field, _, pattern = f.partition("=")
        if field not in _FILTER_FIELDS:
            raise ValueError(
                f"Unknown filter field {field!r}. Valid fields: {sorted(_FILTER_FIELDS)}"
            )
        parsed.append((field, pattern.lower()))

    result = []
    for row in rows:
        if all(pattern in getattr(row, field).lower() for field, pattern in parsed):
            result.append(row)
    return result


def sort_rows(rows: list[RunRow], sort_by: str = "p95") -> list[RunRow]:
    """Return a new sorted list; does not mutate the input."""
    if sort_by == "backend":
        return sorted(rows, key=lambda r: (r.backend, r.model))
    if sort_by == "model":
        return sorted(rows, key=lambda r: (r.model, r.backend))
    if sort_by == "toks":
        return sorted(rows, key=lambda r: r.tokens_per_second, reverse=True)
    if sort_by == "load":
        _inf = float("inf")
        return sorted(rows, key=lambda r: r.model_load_ms if r.model_load_ms is not None else _inf)
    if sort_by == "ttft":
        _inf = float("inf")
        return sorted(rows, key=lambda r: r.p50_ttft_ms if r.p50_ttft_ms is not None else _inf)
    return sorted(rows, key=lambda r: r.p95_latency_ms)  # default: p95 ascending


def render_table(rows: list[RunRow]) -> str:
    """Render RunRows as a GitHub-Flavored Markdown table string."""

    def fmt_optional(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.1f}"

    def fmt_rate(v: float | None) -> str:
        return "N/A" if v is None else f"{v * 100:.1f}%"

    def fmt_ppl(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.2f}"

    def fmt_p95(r: RunRow) -> str:
        if r.p95_latency_ms_std is not None:
            return f"{r.p95_latency_ms:.2f} ± {r.p95_latency_ms_std:.2f}"
        return f"{r.p95_latency_ms:.2f}"

    def fmt_toks(r: RunRow) -> str:
        if r.tokens_per_second_std is not None:
            return f"{r.tokens_per_second:.1f} ± {r.tokens_per_second_std:.1f}"
        return f"{r.tokens_per_second:.1f}"

    data: list[list[str]] = [
        [
            r.backend,
            r.model,
            str(r.request_count),
            f"{r.p50_latency_ms:.2f}",
            fmt_p95(r),
            fmt_toks(r),
            fmt_optional(r.decode_tokens_per_second),
            fmt_optional(r.mean_input_tokens),
            fmt_optional(r.mean_output_tokens),
            fmt_optional(r.model_load_ms),
            fmt_optional(r.p50_ttft_ms),
            fmt_optional(r.p95_ttft_ms),
            fmt_optional(r.p50_tpot_ms),
            f"{r.peak_cpu_memory_mb:.1f}",
            fmt_optional(r.peak_cuda_memory_mb),
            fmt_optional(r.peak_vram_memory_mb),
            fmt_rate(r.sanity_pass_rate),
            fmt_rate(r.task_quality_pass_rate),
            fmt_ppl(r.perplexity),
            fmt_rate(r.judge_score),
            fmt_optional(r.mean_reasoning_tokens),
            fmt_optional(r.mean_answer_tokens),
            fmt_rate(r.reasoning_fraction),
            fmt_optional(r.energy_joules),
            fmt_optional(r.tokens_per_joule),
            fmt_optional(r.thermal_throttle_pct),
        ]
        for r in rows
    ]

    # Suppress optional columns where every row shows "N/A".
    visible = [
        i
        for i in range(len(_HEADERS))
        if i not in _OPTIONAL_COL_INDICES or any(row[i] != "N/A" for row in data)
    ]
    headers = [_HEADERS[i] for i in visible]
    filtered_data = [[row[i] for i in visible] for row in data]

    widths = [len(h) for h in headers]
    for row in filtered_data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(s: str, w: int) -> str:
        return s.ljust(w)

    header_line = "| " + " | ".join(pad(h, w) for h, w in zip(headers, widths, strict=True)) + " |"
    sep_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    data_lines = [
        "| " + " | ".join(pad(c, w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in filtered_data
    ]
    return "\n".join([header_line, sep_line, *data_lines])


def render_json(rows: list[RunRow]) -> str:
    """Serialize RunRows to a JSON array string (all fields; None for absent values)."""
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
            "p50_tpot_ms": r.p50_tpot_ms,
            "tpot_stddev_ms": r.tpot_stddev_ms,
            "peak_cpu_memory_mb": r.peak_cpu_memory_mb,
            "peak_cuda_memory_mb": r.peak_cuda_memory_mb,
            "peak_vram_memory_mb": r.peak_vram_memory_mb,
            "sanity_pass_rate": r.sanity_pass_rate,
            "task_quality_pass_rate": r.task_quality_pass_rate,
            "perplexity": r.perplexity,
            "judge_score": r.judge_score,
            "p95_latency_ms_std": r.p95_latency_ms_std,
            "tokens_per_second_std": r.tokens_per_second_std,
            "hw_cpu": r.hw_cpu,
            "hw_cpu_cores": r.hw_cpu_cores,
            "hw_ram_gb": r.hw_ram_gb,
            "hw_gpu": r.hw_gpu,
            "hw_vram_gb": r.hw_vram_gb,
            "hw_os": r.hw_os,
            "mean_reasoning_tokens": r.mean_reasoning_tokens,
            "mean_answer_tokens": r.mean_answer_tokens,
            "reasoning_fraction": r.reasoning_fraction,
            "energy_joules": r.energy_joules,
            "tokens_per_joule": r.tokens_per_joule,
            "thermal_throttle_pct": r.thermal_throttle_pct,
        }
        for r in rows
    ]
    return json.dumps(data, indent=2)


_CSV_FIELDS = [
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
    "p50_tpot_ms",
    "tpot_stddev_ms",
    "peak_cpu_memory_mb",
    "peak_cuda_memory_mb",
    "peak_vram_memory_mb",
    "sanity_pass_rate",
    "task_quality_pass_rate",
    "perplexity",
    "judge_score",
    "p95_latency_ms_std",
    "tokens_per_second_std",
    "hw_cpu",
    "hw_cpu_cores",
    "hw_ram_gb",
    "hw_gpu",
    "hw_vram_gb",
    "hw_os",
    "mean_reasoning_tokens",
    "mean_answer_tokens",
    "reasoning_fraction",
    "energy_joules",
    "tokens_per_joule",
    "thermal_throttle_pct",
]


def render_csv(rows: list[RunRow]) -> str:
    """Serialize RunRows to a CSV string with snake_case headers.

    Absent optional metrics are written as empty cells so pandas.read_csv()
    produces NaN automatically rather than the string "N/A".
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for r in rows:
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
                "p50_tpot_ms": _or_empty(r.p50_tpot_ms),
                "tpot_stddev_ms": _or_empty(r.tpot_stddev_ms),
                "peak_cpu_memory_mb": r.peak_cpu_memory_mb,
                "peak_cuda_memory_mb": _or_empty(r.peak_cuda_memory_mb),
                "peak_vram_memory_mb": _or_empty(r.peak_vram_memory_mb),
                "sanity_pass_rate": _or_empty(r.sanity_pass_rate),
                "task_quality_pass_rate": _or_empty(r.task_quality_pass_rate),
                "perplexity": _or_empty(r.perplexity),
                "judge_score": _or_empty(r.judge_score),
                "p95_latency_ms_std": _or_empty(r.p95_latency_ms_std),
                "tokens_per_second_std": _or_empty(r.tokens_per_second_std),
                "hw_cpu": r.hw_cpu or "",
                "hw_cpu_cores": r.hw_cpu_cores if r.hw_cpu_cores is not None else "",
                "hw_ram_gb": r.hw_ram_gb if r.hw_ram_gb is not None else "",
                "hw_gpu": r.hw_gpu or "",
                "hw_vram_gb": r.hw_vram_gb if r.hw_vram_gb is not None else "",
                "hw_os": r.hw_os or "",
                "mean_reasoning_tokens": _or_empty(r.mean_reasoning_tokens),
                "mean_answer_tokens": _or_empty(r.mean_answer_tokens),
                "reasoning_fraction": _or_empty(r.reasoning_fraction),
                "energy_joules": _or_empty(r.energy_joules),
                "tokens_per_joule": _or_empty(r.tokens_per_joule),
                "thermal_throttle_pct": _or_empty(r.thermal_throttle_pct),
            }
        )
    return buf.getvalue().rstrip("\n")


def _or_empty(v: float | None) -> float | str:
    return "" if v is None else v


def build_comparison_table(paths: list[str | Path], sort_by: str = "p95") -> str:
    """Load CSV files, sort, and return a Markdown table string."""
    if not paths:
        raise ValueError("At least one CSV path is required")
    rows = [load_csv(p) for p in paths]
    return render_table(sort_rows(rows, sort_by))
