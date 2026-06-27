"""Generate a self-contained HTML report from concurrency-sweep results.

Usage::

    from llm_inference_benchmark.sweep_report import SweepPoint, build_sweep_html

    points = [
        SweepPoint(concurrency=1, tokens_per_second=45.0, throughput_rps=1.5,
                   p50_latency_ms=620.0, p95_latency_ms=690.0, is_knee=False),
        SweepPoint(concurrency=4, tokens_per_second=90.0, throughput_rps=3.0,
                   p50_latency_ms=1100.0, p95_latency_ms=1400.0, is_knee=True),
    ]
    html = build_sweep_html(points, title="Llama-3.2 3B Concurrency Sweep")
    Path("sweep_report.html").write_text(html, encoding="utf-8")
"""

from __future__ import annotations

import html as _html
import json
from dataclasses import dataclass

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"


@dataclass(frozen=True)
class SweepPoint:
    concurrency: int
    tokens_per_second: float
    throughput_rps: float
    p50_latency_ms: float
    p95_latency_ms: float
    is_knee: bool
    p50_ttft_ms: float | None = None


def _fmt_f(v: float | None, decimals: int = 1) -> str:
    return "—" if v is None else f"{v:.{decimals}f}"


def _sweep_table(points: list[SweepPoint]) -> str:
    headers = ["Concurrency", "tok/s", "RPS", "p50 (ms)", "p95 (ms)", "TTFT p50 (ms)", "Knee"]
    header_cells = "".join(f"<th>{h}</th>" for h in headers)

    rows_html: list[str] = []
    for pt in points:
        knee_cell = "<td><span class='badge-knee'>knee</span></td>" if pt.is_knee else "<td>—</td>"
        row_class = " class='knee-row'" if pt.is_knee else ""
        cells = "".join(
            [
                f"<td>{pt.concurrency}</td>",
                f"<td>{_fmt_f(pt.tokens_per_second)}</td>",
                f"<td>{_fmt_f(pt.throughput_rps, 3)}</td>",
                f"<td>{_fmt_f(pt.p50_latency_ms)}</td>",
                f"<td>{_fmt_f(pt.p95_latency_ms)}</td>",
                f"<td>{_fmt_f(pt.p50_ttft_ms)}</td>",
                knee_cell,
            ]
        )
        rows_html.append(f"<tr{row_class}>{cells}</tr>")

    tbody = "".join(rows_html)
    return (
        f"<table class='metrics-table'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{tbody}</tbody>"
        f"</table>"
    )


def _chart_data(points: list[SweepPoint]) -> str:
    return json.dumps(
        [
            {
                "c": pt.concurrency,
                "tps": pt.tokens_per_second,
                "p95": pt.p95_latency_ms,
                "p50": pt.p50_latency_ms,
                "knee": pt.is_knee,
            }
            for pt in points
        ]
    )


def build_sweep_html(points: list[SweepPoint], title: str = "Concurrency Sweep Report") -> str:
    """Return a self-contained HTML page for the sweep results.

    *points* should be ordered by concurrency (ascending).
    When *points* is empty an informational placeholder is returned.
    """
    if not points:
        return (
            "<!DOCTYPE html><html lang='en'><head>"
            "<meta charset='UTF-8'>"
            f"<title>{_html.escape(title)}</title>"
            "</head><body>"
            f"<h1>{_html.escape(title)}</h1>"
            "<p>No sweep data provided.</p>"
            "</body></html>"
        )

    escaped_title = _html.escape(title)
    table_html = _sweep_table(points)
    chart_json = _chart_data(points)
    knee = next((pt for pt in points if pt.is_knee), points[-1])

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
    .metrics-table td{{padding:.45rem .75rem;border-bottom:1px solid #e2e8f0;white-space:nowrap;
                       text-align:right}}
    .metrics-table td:first-child{{text-align:center;font-weight:600}}
    .metrics-table tbody tr:last-child td{{border-bottom:none}}
    .metrics-table tbody tr:hover{{background:#f1f5f9}}
    .knee-row{{background:#fef9c3}}
    .knee-row:hover{{background:#fef08a}}
    .badge-knee{{background:#ca8a04;color:#fff;border-radius:999px;
                padding:.1rem .45rem;font-size:.72rem;font-weight:600}}
    #chart{{margin-top:.5rem;border-radius:.5rem;overflow:hidden;
            box-shadow:0 1px 3px rgba(0,0,0,.08)}}
    .knee-summary{{background:#fff;border-radius:.5rem;padding:.75rem 1rem;
                   font-size:.85rem;color:#475569;margin-top:1rem;
                   box-shadow:0 1px 3px rgba(0,0,0,.08)}}
    .knee-summary strong{{color:#1e293b}}
    .footer{{margin-top:2rem;color:#94a3b8;font-size:.78rem}}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <div class="subtitle">Concurrency sweep &mdash; {len(points)} level(s) &mdash; llm-bench</div>

  <h2>Throughput &amp; Latency vs Concurrency</h2>
  <div id="chart"></div>

  <p class="knee-summary">
    Knee point: <strong>concurrency={knee.concurrency}</strong>
    &mdash; <strong>{knee.tokens_per_second:.1f} tok/s</strong>
    &mdash; p95 <strong>{knee.p95_latency_ms:.1f} ms</strong>
  </p>

  <h2>Metrics</h2>
  {table_html}

  <div class="footer">llm-bench &mdash; reproducible local LLM inference benchmarks</div>

  <script>
(function(){{
  const PTS = {chart_json};
  if (!PTS.length) return;

  const xs = PTS.map(p => p.c);

  const traceTps = {{
    x: xs,
    y: PTS.map(p => p.tps),
    name: 'tok/s',
    type: 'scatter',
    mode: 'lines+markers',
    marker: {{
      color: PTS.map(p => p.knee ? '#ca8a04' : '#3b82f6'),
      size: PTS.map(p => p.knee ? 12 : 8),
      symbol: PTS.map(p => p.knee ? 'star' : 'circle'),
      line: {{color: '#fff', width: 1}},
    }},
    line: {{color: '#3b82f6', width: 2}},
    hovertemplate: 'concurrency=%{{x}}<br>tok/s=%{{y:.1f}}<extra></extra>',
    yaxis: 'y',
  }};

  const traceP95 = {{
    x: xs,
    y: PTS.map(p => p.p95),
    name: 'p95 latency (ms)',
    type: 'scatter',
    mode: 'lines+markers',
    marker: {{color: '#f87171', size: 7, line: {{color: '#fff', width: 1}}}},
    line: {{color: '#f87171', width: 2, dash: 'dot'}},
    hovertemplate: 'concurrency=%{{x}}<br>p95=%{{y:.1f}} ms<extra></extra>',
    yaxis: 'y2',
  }};

  Plotly.newPlot('chart', [traceTps, traceP95], {{
    xaxis: {{
      title: 'Concurrency',
      tickmode: 'array',
      tickvals: xs,
      zeroline: false,
      gridcolor: '#e2e8f0',
    }},
    yaxis: {{
      title: 'Throughput (tok/s)',
      zeroline: false,
      gridcolor: '#e2e8f0',
    }},
    yaxis2: {{
      title: 'p95 Latency (ms)',
      overlaying: 'y',
      side: 'right',
      zeroline: false,
      showgrid: false,
    }},
    plot_bgcolor: '#fff',
    paper_bgcolor: '#fff',
    legend: {{orientation: 'h', y: -0.18}},
    margin: {{t: 20, r: 80, b: 70, l: 60}},
  }}, {{responsive: true}});
}})();
  </script>
</body>
</html>
"""
