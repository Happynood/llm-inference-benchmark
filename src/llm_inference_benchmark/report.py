"""Generate self-contained HTML benchmark reports from benchmark CSV files.

Usage::

    from llm_inference_benchmark.report import build_report_html, load_runs

    rows = load_runs(["baseline.csv", "current.csv"])
    html_text = build_report_html(rows, title="Q4 vs Q8 Comparison")
    Path("report.html").write_text(html_text)
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path

from llm_inference_benchmark.compare import RunRow, load_csv
from llm_inference_benchmark.pareto import pareto_classify

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"


def load_runs(paths: list[str | Path]) -> list[RunRow]:
    """Load each CSV path into a RunRow.

    Raises ``FileNotFoundError`` when a path does not exist, and
    ``ValueError`` when a CSV is malformed or empty.
    """
    results: list[RunRow] = []
    for p in paths:
        path = Path(p)
        results.append(load_csv(path))
    return results


def _fmt_f(v: float | None, decimals: int = 1) -> str:
    return "—" if v is None else f"{v:.{decimals}f}"


def _fmt_rate(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _short_model(model: str) -> str:
    """Return the last two path-like segments of a model name / path."""
    parts = model.replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _row_label(row: RunRow) -> str:
    return f"{row.backend}/{_short_model(row.model)}"


def _metrics_table(classified: list[tuple[RunRow, bool]]) -> str:
    cols = [
        ("Run", None),
        ("tok/s", False),
        ("p50 (ms)", True),
        ("p95 (ms)", True),
        ("TTFT p50 (ms)", True),
        ("Load (ms)", True),
        ("VRAM (MB)", True),
        ("CPU mem (MB)", True),
        ("Sanity", False),
        ("Pareto", None),
    ]

    header_cells = "".join(f"<th>{_html.escape(h)}</th>" for h, _ in cols)

    rows_html: list[str] = []
    for row, is_pareto in classified:
        pareto_cell = (
            "<td><span class='badge-pareto'>optimal</span></td>" if is_pareto else "<td>—</td>"
        )
        cells = "".join(
            [
                f"<td class='run-label'>{_html.escape(_row_label(row))}</td>",
                f"<td>{_fmt_f(row.tokens_per_second)}</td>",
                f"<td>{_fmt_f(row.p50_latency_ms)}</td>",
                f"<td>{_fmt_f(row.p95_latency_ms)}</td>",
                f"<td>{_fmt_f(row.p50_ttft_ms)}</td>",
                f"<td>{_fmt_f(row.model_load_ms)}</td>",
                f"<td>{_fmt_f(row.peak_vram_memory_mb)}</td>",
                f"<td>{_fmt_f(row.peak_cpu_memory_mb)}</td>",
                f"<td>{_fmt_rate(row.sanity_pass_rate)}</td>",
                pareto_cell,
            ]
        )
        row_class = " class='pareto-row'" if is_pareto else ""
        rows_html.append(f"<tr{row_class}>{cells}</tr>")

    tbody = "".join(rows_html)
    return (
        f"<table class='metrics-table'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{tbody}</tbody>"
        f"</table>"
    )


def _chart_data(classified: list[tuple[RunRow, bool]]) -> str:
    pts = []
    for row, is_pareto in classified:
        if row.tokens_per_second is None or row.p95_latency_ms is None:
            continue
        pts.append(
            {
                "label": _row_label(row),
                "x": row.p95_latency_ms,
                "y": row.tokens_per_second,
                "pareto": is_pareto,
                "vram": row.peak_vram_memory_mb,
                "sanity": (
                    None if row.sanity_pass_rate is None else round(row.sanity_pass_rate * 100, 1)
                ),
            }
        )
    return json.dumps(pts)


def build_report_html(rows: list[RunRow], title: str = "Benchmark Report") -> str:
    """Return a self-contained HTML string for the given RunRow list.

    *rows* are sorted by throughput (descending) before display.
    Pareto-optimal runs are highlighted in the table and chart.
    When *rows* is empty an informational placeholder page is returned.
    """
    if not rows:
        return (
            "<!DOCTYPE html><html lang='en'><head>"
            "<meta charset='UTF-8'>"
            f"<title>{_html.escape(title)}</title>"
            "</head><body>"
            f"<h1>{_html.escape(title)}</h1>"
            "<p>No benchmark runs provided.</p>"
            "</body></html>"
        )

    sorted_rows = sorted(rows, key=lambda r: r.tokens_per_second, reverse=True)
    classified = pareto_classify(sorted_rows)
    table_html = _metrics_table(classified)
    chart_pts = _chart_data(classified)
    escaped_title = _html.escape(title)

    hw_lines: list[str] = []
    for row, _ in classified:
        if row.hw_cpu or row.hw_gpu:
            parts = []
            if row.hw_cpu:
                parts.append(_html.escape(row.hw_cpu))
            if row.hw_gpu:
                parts.append(_html.escape(row.hw_gpu))
            hw_lines.append(f"<li>{_html.escape(_row_label(row))}: {', '.join(parts)}</li>")
    hw_section = f"<h2>Hardware</h2><ul>{''.join(hw_lines)}</ul>" if hw_lines else ""

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <script src="{_PLOTLY_CDN}" charset="utf-8"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:system-ui,sans-serif;background:#f8fafc;color:#1e293b;padding:1.5rem}}
    h1{{font-size:1.5rem;font-weight:700;margin-bottom:.25rem}}
    .subtitle{{color:#64748b;font-size:.875rem;margin-bottom:1.5rem}}
    h2{{font-size:1.1rem;font-weight:600;margin:1.5rem 0 .5rem}}
    .metrics-table{{width:100%;border-collapse:collapse;font-size:.82rem;background:#fff;
                    border-radius:.5rem;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
    .metrics-table th{{background:#1e293b;color:#f1f5f9;padding:.5rem .75rem;text-align:left;
                       font-weight:600;white-space:nowrap}}
    .metrics-table td{{padding:.45rem .75rem;border-bottom:1px solid #e2e8f0;white-space:nowrap}}
    .metrics-table tbody tr:last-child td{{border-bottom:none}}
    .metrics-table tbody tr:hover{{background:#f1f5f9}}
    .pareto-row{{background:#f0fdf4}}
    .pareto-row:hover{{background:#dcfce7}}
    .run-label{{font-family:monospace;font-size:.78rem;color:#334155}}
    .badge-pareto{{background:#16a34a;color:#fff;border-radius:999px;
                  padding:.1rem .45rem;font-size:.72rem;font-weight:600}}
    #chart{{margin-top:.5rem;border-radius:.5rem;overflow:hidden;
            box-shadow:0 1px 3px rgba(0,0,0,.08)}}
    ul{{padding-left:1.25rem;font-size:.85rem;color:#475569}}
    li{{margin:.2rem 0}}
    .footer{{margin-top:2rem;color:#94a3b8;font-size:.78rem}}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <div class="subtitle">Generated by llm-bench &mdash; {len(rows)} run(s)</div>

  <h2>Metrics</h2>
  {table_html}

  <h2>Throughput vs Latency</h2>
  <div id="chart"></div>
  {hw_section}
  <div class="footer">llm-bench &mdash; reproducible local LLM inference benchmarks</div>

  <script>
(function(){{
  const PTS = {chart_pts};
  if (!PTS.length) return;

  const pareto = PTS.filter(p => p.pareto);
  const dominated = PTS.filter(p => !p.pareto);

  function makeTrace(pts, name, color, symbol) {{
    return {{
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      mode: 'markers+text',
      type: 'scatter',
      name: name,
      text: pts.map(p => p.label),
      textposition: 'top center',
      textfont: {{size: 10}},
      marker: {{color: color, size: 10, symbol: symbol,
                line: {{color: '#fff', width: 1}}}},
      hovertemplate: '<b>%{{text}}</b><br>p95: %{{x:.1f}} ms<br>tok/s: %{{y:.1f}}<extra></extra>',
    }};
  }}

  const traces = [];
  if (dominated.length) traces.push(makeTrace(dominated, 'Dominated', '#94a3b8', 'circle'));
  if (pareto.length)   traces.push(makeTrace(pareto,   'Pareto-optimal', '#16a34a', 'star'));

  Plotly.newPlot('chart', traces, {{
    xaxis: {{title: 'p95 Latency (ms)', zeroline: false}},
    yaxis: {{title: 'Throughput (tok/s)', zeroline: false}},
    plot_bgcolor: '#fff',
    paper_bgcolor: '#fff',
    legend: {{orientation: 'h', y: -0.15}},
    margin: {{t: 20, r: 20, b: 60, l: 60}},
  }}, {{responsive: true}});
}})();
  </script>
</body>
</html>
"""
