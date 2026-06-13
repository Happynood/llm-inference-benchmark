from __future__ import annotations

import csv
from dataclasses import asdict

import click

from llm_inference_benchmark.backends.base import Backend
from llm_inference_benchmark.backends.mock import MockBackend
from llm_inference_benchmark.config import BenchmarkConfig, load_config
from llm_inference_benchmark.runner import load_prompts, run_benchmark


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
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
def main(config_path: str, output_path: str | None) -> None:
    """Run LLM inference benchmark."""
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
