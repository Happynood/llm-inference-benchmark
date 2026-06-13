from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import click

from llm_inference_benchmark.backends.base import Backend
from llm_inference_benchmark.backends.mock import MockBackend
from llm_inference_benchmark.config import BenchmarkConfig, load_config
from llm_inference_benchmark.runner import load_prompts, run_benchmark


@click.group(invoke_without_command=True)
@click.pass_context
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="YAML benchmark config file",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="CSV output path (default: stdout summary only)",
)
def main(ctx: click.Context, config_path: str | None, output_path: str | None) -> None:
    """LLM inference benchmark toolkit.

    Run without a subcommand to execute a benchmark:

        llm-bench --config configs/example.yaml --output results.csv

    Use the compare subcommand to generate a Markdown table from saved CSVs:

        llm-bench compare results_a.csv results_b.csv
    """
    if ctx.invoked_subcommand is not None:
        return

    if config_path is None:
        raise click.UsageError("--config is required when running a benchmark")

    cfg = load_config(config_path)
    backend = _build_backend(cfg)
    prompts = load_prompts(cfg.prompts_file)

    click.echo(f"Backend: {cfg.backend}  Model: {cfg.model}  Requests: {cfg.requests}")
    report = run_benchmark(backend, cfg, prompts)

    if output_path:
        row = {k: ("" if v is None else v) for k, v in asdict(report).items()}
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        click.echo(f"Results written to {output_path}")

    click.echo("\n=== Benchmark Results ===")
    for k, v in asdict(report).items():
        if isinstance(v, float):
            click.echo(f"  {k}: {v:.2f}")
        elif v is None:
            click.echo(f"  {k}: N/A")
        else:
            click.echo(f"  {k}: {v}")


@main.command("compare")
@click.argument("csv_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--sort",
    "sort_by",
    default="p95",
    show_default=True,
    type=click.Choice(["backend", "model", "p95"], case_sensitive=False),
    help="Sort rows by this column",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write Markdown to file instead of stdout",
)
def compare_cmd(csv_files: tuple[str, ...], sort_by: str, output_path: str | None) -> None:
    """Generate a Markdown comparison table from benchmark CSV files.

    Accepts one or more CSV files produced by llm-bench --output:

        llm-bench compare mock.csv transformers.csv --sort p95
    """
    from llm_inference_benchmark.compare import build_comparison_table

    table = build_comparison_table(list(csv_files), sort_by=sort_by)
    if output_path:
        Path(output_path).write_text(table + "\n")
        click.echo(f"Table written to {output_path}")
    else:
        click.echo(table)


def _build_backend(cfg: BenchmarkConfig) -> Backend:
    if cfg.backend == "mock":
        return MockBackend(
            model=cfg.model,
            latency_ms=cfg.mock.latency_ms,
            tokens_per_response=cfg.mock.tokens_per_response,
        )
    if cfg.backend == "transformers":
        from llm_inference_benchmark.backends.hf import HFBackend  # lazy: optional dep

        return HFBackend(
            model_id=cfg.model,
            max_new_tokens=cfg.hf.max_new_tokens,
            device=cfg.hf.device,
            torch_dtype=cfg.hf.torch_dtype,
            do_sample=cfg.hf.do_sample,
        )
    raise ValueError(f"Unknown backend: {cfg.backend!r}")
