from __future__ import annotations

import csv
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
import yaml

from llm_inference_benchmark import __version__
from llm_inference_benchmark.backends.base import Backend
from llm_inference_benchmark.backends.mock import MockBackend
from llm_inference_benchmark.config import BenchmarkConfig, load_config
from llm_inference_benchmark.runner import load_prompts, run_repeated

_DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between TCP and UDP in one sentence.",
    "Write a Python function that reverses a list without using built-ins.",
    "What are the main advantages of transformer-based language models?",
    "Summarize the theory of relativity in three sentences.",
    "How does a binary search tree maintain its ordering property?",
    "What is the role of attention in neural networks?",
    "Describe the difference between supervised and unsupervised learning.",
    "What is a REST API and how does it work?",
    "Explain what happens during the TCP three-way handshake.",
]


def _apply_set_overrides(cfg: BenchmarkConfig, set_overrides: tuple[str, ...]) -> BenchmarkConfig:
    """Parse and apply --set KEY=VALUE overrides to *cfg*, returning updated config.

    Values are parsed as YAML scalars so that ``200`` becomes int, ``true`` becomes bool,
    etc.  Unknown paths and type mismatches are raised as ``click.UsageError``.
    """
    from pydantic import ValidationError

    from llm_inference_benchmark.sweep import apply_overrides, validate_override_path

    parsed: dict[str, Any] = {}
    for kv in set_overrides:
        if "=" not in kv:
            raise click.UsageError(f"--set expects KEY=VALUE format, got {kv!r}")
        key, _, val_str = kv.partition("=")
        try:
            validate_override_path(key)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        parsed[key] = yaml.safe_load(val_str)

    updated = apply_overrides(cfg, parsed)
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dump = updated.model_dump()
        return BenchmarkConfig.model_validate(dump)
    except ValidationError as exc:
        raise click.UsageError(f"Invalid --set value: {exc}") from exc


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="llm-bench")
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
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(),
    help="JSON manifest path for environment fingerprint and reproducibility",
)
@click.option(
    "--requests",
    "requests_override",
    default=None,
    type=click.IntRange(min=1),
    metavar="N",
    help="Override config requests count",
)
@click.option(
    "--warmup-requests",
    "warmup_requests_override",
    default=None,
    type=click.IntRange(min=0),
    metavar="N",
    help="Override config warmup_requests",
)
@click.option(
    "--concurrency",
    "concurrency_override",
    default=None,
    type=click.IntRange(min=1),
    metavar="N",
    help="Override config concurrency",
)
@click.option(
    "--arrival-rate",
    "arrival_rate_override",
    default=None,
    type=float,
    metavar="RPS",
    help=(
        "Run in open-loop mode: dispatch requests at a fixed arrival rate (requests/second). "
        "Overrides --concurrency. Models constant-arrival-rate traffic to reveal queueing latency."
    ),
)
@click.option(
    "--seed",
    "seed_override",
    default=None,
    type=int,
    metavar="N",
    help="Override config seed for reproducible prompt sampling",
)
@click.option(
    "--set",
    "set_overrides",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Override any config field via dot-path, e.g. --set llama_cpp.max_tokens=200. "
        "Values are parsed as YAML scalars (int, float, bool, str). Repeatable."
    ),
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format: table=human-readable, json=machine-readable JSON",
)
@click.option(
    "--dataset",
    "dataset_name",
    default=None,
    metavar="NAME",
    help=(
        "Use a cached real-world dataset as prompt source instead of the config prompts file. "
        "Pull datasets first with: llm-bench datasets pull <name>"
    ),
)
@click.option(
    "--base-url",
    "base_url",
    default=None,
    metavar="URL",
    help=(
        "Base URL of an OpenAI-compatible endpoint (e.g. http://localhost:11434/v1). "
        "Sets backend to 'openai' and makes --config optional. "
        "Use --set model=<name> to specify the model."
    ),
)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    metavar="KEY",
    help=(
        "API key for the endpoint specified by --base-url. "
        "Omit for local servers that do not require authentication."
    ),
)
def main(
    ctx: click.Context,
    config_path: str | None,
    output_path: str | None,
    manifest_path: str | None,
    requests_override: int | None,
    warmup_requests_override: int | None,
    concurrency_override: int | None,
    arrival_rate_override: float | None,
    seed_override: int | None,
    set_overrides: tuple[str, ...],
    output_format: str,
    dataset_name: str | None,
    base_url: str | None,
    api_key: str | None,
) -> None:
    """LLM inference benchmark toolkit.

    Run without a subcommand to execute a benchmark:

        llm-bench --config configs/example.yaml --output results.csv

    Override individual config knobs without editing the YAML:

        llm-bench --config configs/example.yaml --requests 50 --concurrency 4 --seed 42

    Use --set to override any backend-specific field via dot-path:

        llm-bench --config configs/example.yaml --set llama_cpp.max_tokens=200
        llm-bench --config configs/example.yaml --set hf.max_new_tokens=256 --set hf.device=cuda

    Use the compare subcommand to generate a Markdown table from saved CSVs:

        llm-bench compare results_a.csv results_b.csv

    Use a cached real-world dataset as prompt source:

        llm-bench datasets pull lmsys-chat
        llm-bench --config configs/example.yaml --dataset lmsys-chat --requests 50

    Test any OpenAI-compatible server without a config file:

        llm-bench --base-url http://localhost:11434/v1 --set model=llama3.2:3b
        llm-bench --base-url https://api.openai.com/v1 --api-key sk-xxx --set model=gpt-4o-mini
    """
    if ctx.invoked_subcommand is not None:
        return

    if config_path is None and base_url is None:
        raise click.UsageError("--config or --base-url is required when running a benchmark")

    if config_path is not None:
        cfg = load_config(config_path)
    else:
        from llm_inference_benchmark.config import OpenAIEndpointConfig

        cfg = BenchmarkConfig(
            backend="openai",
            openai=OpenAIEndpointConfig(base_url=base_url or "http://localhost:8080/v1"),
        )

    if base_url is not None:
        updated_openai = cfg.openai.model_copy(update={"base_url": base_url})
        cfg = cfg.model_copy(update={"backend": "openai", "openai": updated_openai})

    if set_overrides:
        cfg = _apply_set_overrides(cfg, set_overrides)
    overrides: dict[str, object] = {}
    if requests_override is not None:
        overrides["requests"] = requests_override
    if warmup_requests_override is not None:
        overrides["warmup_requests"] = warmup_requests_override
    if concurrency_override is not None:
        overrides["concurrency"] = concurrency_override
    if arrival_rate_override is not None:
        if arrival_rate_override <= 0:
            raise click.UsageError("--arrival-rate must be a positive number (requests/second)")
        overrides["arrival_rate_rps"] = arrival_rate_override
    if seed_override is not None:
        overrides["seed"] = seed_override
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    _t0 = time.perf_counter()
    backend = _build_backend(cfg, api_key=api_key)
    model_load_ms = (time.perf_counter() - _t0) * 1000.0

    if dataset_name is not None:
        from llm_inference_benchmark.datasets import load_prompts as _ds_load

        try:
            prompts = _ds_load(dataset_name, n=cfg.requests, seed=cfg.seed)
        except FileNotFoundError as exc:
            raise click.UsageError(str(exc)) from exc
    elif config_path is not None:
        prompts = load_prompts(cfg.resolve_prompts_file())
    else:
        prompts = _DEFAULT_PROMPTS[: cfg.requests]

    if output_format != "json":
        header = f"Backend: {cfg.backend}  Model: {cfg.model}  Requests: {cfg.requests}"
        if cfg.seed is not None:
            header += f"  Seed: {cfg.seed}"
        click.echo(header)
    report = run_repeated(backend, cfg, prompts, model_load_ms=model_load_ms)

    if output_format == "json":
        text = json.dumps(asdict(report))
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(text + "\n")
        else:
            click.echo(text)
    else:
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            row = {k: ("" if v is None else v) for k, v in asdict(report).items()}
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            click.echo(f"Results written to {output_path}")

    if manifest_path:
        from llm_inference_benchmark.manifest import collect_manifest, write_manifest

        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        manifest = collect_manifest(config_path or "", cfg)
        write_manifest(manifest, manifest_path)
        click.echo(f"Manifest written to {manifest_path}")

    if output_format != "json":
        click.echo("\n=== Benchmark Results ===")
        _print_report(report)


@main.command("compare")
@click.argument("csv_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--sort",
    "sort_by",
    default="p95",
    show_default=True,
    type=click.Choice(["backend", "model", "p95", "toks", "load", "ttft"], case_sensitive=False),
    help=(
        "Sort column: toks=highest throughput first; load=fastest load first (N/A last); "
        "ttft=lowest TTFT p50 first (N/A last)"
    ),
)
@click.option(
    "--limit",
    "limit",
    default=None,
    type=click.IntRange(min=1),
    metavar="N",
    help="Show only the top N rows after sorting (omit to show all).",
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    help="Output format: table=Markdown, json=machine-readable JSON array, csv=comma-separated",
)
@click.option(
    "--filter",
    "filters",
    multiple=True,
    metavar="FIELD=PATTERN",
    help=(
        "Keep only rows where FIELD contains PATTERN (case-insensitive substring). "
        "Supported fields: backend, model. Repeatable; multiple filters are ANDed."
    ),
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write output to file instead of stdout",
)
def compare_cmd(
    csv_files: tuple[str, ...],
    sort_by: str,
    limit: int | None,
    output_format: str,
    filters: tuple[str, ...],
    output_path: str | None,
) -> None:
    """Generate a comparison table from benchmark CSV files.

    Accepts one or more CSV files produced by llm-bench --output:

        llm-bench compare mock.csv transformers.csv --sort p95
        llm-bench compare results/*.csv --sort toks --limit 5
        llm-bench compare results/*.csv --filter backend=llama_cpp
        llm-bench compare results/*.csv --filter backend=llama_cpp --filter model=Q4_K_M
        llm-bench compare results/*.csv --format json
        llm-bench compare results/*.csv --format csv --output summary.csv
    """
    from llm_inference_benchmark.compare import (
        filter_rows,
        load_csv,
        render_csv,
        render_json,
        render_table,
        sort_rows,
    )

    try:
        rows = filter_rows([load_csv(p) for p in csv_files], list(filters))
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    rows = sort_rows(rows, sort_by=sort_by)
    if limit is not None:
        rows = rows[:limit]

    if output_format == "json":
        text = render_json(rows)
    elif output_format == "csv":
        text = render_csv(rows)
    else:
        text = render_table(rows)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text + "\n")
        click.echo(f"Output written to {output_path}")
    else:
        click.echo(text)


@main.command("pareto")
@click.argument("csv_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write output to file instead of stdout",
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    help="Output format: table=Markdown, json=machine-readable JSON array, csv=spreadsheet/pandas",
)
@click.option(
    "--filter",
    "filters",
    multiple=True,
    metavar="FIELD=PATTERN",
    help=(
        "Keep only rows where FIELD contains PATTERN (case-insensitive substring). "
        "Supported fields: backend, model. Repeatable; multiple filters are ANDed."
    ),
)
def pareto_cmd(
    csv_files: tuple[str, ...],
    output_path: str | None,
    output_format: str,
    filters: tuple[str, ...],
) -> None:
    """Identify Pareto-optimal benchmark configurations from CSV files.

    A configuration is Pareto-optimal when no other configuration is at least
    as good on every metric and strictly better on at least one.  Metrics:
    lower p95 latency, higher tok/s, lower VRAM (when available), higher
    sanity pass rate (when available).

        llm-bench pareto results/q4km.csv results/q8.csv
        llm-bench pareto results/*.csv --format json
        llm-bench pareto results/*.csv --filter backend=llama_cpp
        llm-bench pareto results/*.csv --filter backend=llama_cpp --filter model=Q4_K_M
    """
    from llm_inference_benchmark.compare import filter_rows, load_csv
    from llm_inference_benchmark.pareto import (
        pareto_classify,
        render_pareto_csv,
        render_pareto_json,
        render_pareto_table,
    )

    try:
        rows = filter_rows([load_csv(p) for p in csv_files], list(filters))
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    classified = pareto_classify(rows)

    if output_format == "json":
        text = render_pareto_json(classified)
    elif output_format == "csv":
        text = render_pareto_csv(classified)
    else:
        text = render_pareto_table(classified)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text + "\n")
        click.echo(f"Pareto output written to {output_path}")
    else:
        click.echo(text)


@main.command("recommend")
@click.argument("csv_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--max-vram-mb", type=float, default=None, help="Maximum peak VRAM in MB")
@click.option("--max-p95-ms", type=float, default=None, help="Maximum p95 latency in ms")
@click.option("--min-sanity", type=float, default=None, help="Minimum sanity pass rate [0, 1]")
@click.option(
    "--min-quality",
    type=float,
    default=None,
    help="Minimum task quality pass rate [0, 1] (requires quality_file in config)",
)
@click.option(
    "--max-perplexity",
    type=float,
    default=None,
    help="Maximum perplexity (requires measure_perplexity in config)",
)
@click.option(
    "--min-judge",
    type=float,
    default=None,
    help="Minimum judge score [0, 1] (requires measure_judge in config)",
)
@click.option(
    "--max-load-ms",
    type=float,
    default=None,
    help="Maximum model load time in ms (requires v0.18+ benchmark run)",
)
@click.option(
    "--max-ttft-ms",
    type=float,
    default=None,
    help="Maximum time-to-first-token p50 in ms (requires stream=True benchmark run)",
)
@click.option(
    "--filter",
    "filters",
    multiple=True,
    metavar="FIELD=PATTERN",
    help=(
        "Keep only rows where FIELD contains PATTERN (case-insensitive substring). "
        "Supported fields: backend, model. Repeatable; multiple filters are ANDed."
    ),
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format: table=human-readable text, json=machine-readable JSON",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write recommendation to file instead of stdout",
)
def recommend_cmd(
    csv_files: tuple[str, ...],
    max_vram_mb: float | None,
    max_p95_ms: float | None,
    min_sanity: float | None,
    min_quality: float | None,
    max_perplexity: float | None,
    min_judge: float | None,
    max_load_ms: float | None,
    max_ttft_ms: float | None,
    filters: tuple[str, ...],
    output_format: str,
    output_path: str | None,
) -> None:
    """Recommend the best benchmark configuration under explicit constraints.

    Reads saved benchmark CSVs and returns the fastest Pareto-optimal
    configuration that satisfies all given constraints.  Runs that violate
    a constraint are listed with the reason they were excluded.

    Exits with code 1 when no run satisfies all constraints.

        llm-bench recommend results/*.csv --max-vram-mb 4096 --max-p95-ms 1000
        llm-bench recommend results/*.csv --max-p95-ms 500 --format json
        llm-bench recommend results/*.csv --filter backend=llama_cpp --max-p95-ms 1000
        llm-bench recommend results/*.csv --filter backend=llama_cpp --filter model=Q4
    """
    from llm_inference_benchmark.compare import filter_rows, load_csv
    from llm_inference_benchmark.recommend import (
        Constraints,
        build_recommendation,
        recommend,
        render_recommendation_json,
    )

    constraints = Constraints(
        max_vram_mb=max_vram_mb,
        max_p95_ms=max_p95_ms,
        min_sanity=min_sanity,
        min_quality=min_quality,
        max_perplexity=max_perplexity,
        min_judge=min_judge,
        max_load_ms=max_load_ms,
        max_ttft_ms=max_ttft_ms,
    )

    if output_format == "json":
        try:
            rows = filter_rows([load_csv(p) for p in csv_files], list(filters))
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        result = recommend(rows, constraints)
        text = render_recommendation_json(result)
        has_winner = result.winner is not None
    else:
        try:
            text, has_winner = build_recommendation(list(csv_files), constraints, list(filters))
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text + "\n")
        click.echo(f"Recommendation written to {output_path}")
    else:
        click.echo(text)
    if not has_winner:
        sys.exit(1)


@main.command("validate-config")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="YAML benchmark config file to validate",
)
@click.option(
    "--set",
    "set_overrides",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Override any config field via dot-path before validation, "
        "e.g. --set llama_cpp.max_tokens=200. Repeatable."
    ),
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format: table=human-readable, json=machine-readable JSON",
)
def validate_config_cmd(
    config_path: str, set_overrides: tuple[str, ...], output_format: str
) -> None:
    """Validate a benchmark config file and print a summary of resolved settings.

    Reads the YAML, runs full pydantic validation, resolves the effective
    prompts file, and prints a summary.  Exits 0 on success, 1 on error.

        llm-bench validate-config --config configs/example.yaml
        llm-bench validate-config --config configs/example.yaml --format json
        llm-bench validate-config --config configs/example.yaml --set llama_cpp.max_tokens=200
    """
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    if set_overrides:
        cfg = _apply_set_overrides(cfg, set_overrides)

    if output_format == "json":
        if cfg.backend == "mock":
            backend_cfg: dict[str, object] = {
                "latency_ms": cfg.mock.latency_ms,
                "tokens_per_response": cfg.mock.tokens_per_response,
            }
        elif cfg.backend == "transformers":
            backend_cfg = {
                "max_new_tokens": cfg.hf.max_new_tokens,
                "device": cfg.hf.device,
                "torch_dtype": cfg.hf.torch_dtype,
                "do_sample": cfg.hf.do_sample,
            }
        elif cfg.backend == "llama-cpp":
            backend_cfg = {
                "n_ctx": cfg.llama_cpp.n_ctx,
                "n_gpu_layers": cfg.llama_cpp.n_gpu_layers,
                "max_tokens": cfg.llama_cpp.max_tokens,
                "temperature": cfg.llama_cpp.temperature,
                "n_threads": cfg.llama_cpp.n_threads,
                "verbose": cfg.llama_cpp.verbose,
                "stream": cfg.llama_cpp.stream,
            }
        elif cfg.backend == "openai":
            backend_cfg = {
                "base_url": cfg.openai.base_url,
                "api_key_env": cfg.openai.api_key_env,
                "max_tokens": cfg.openai.max_tokens,
                "temperature": cfg.openai.temperature,
                "timeout_s": cfg.openai.timeout_s,
                "stream": cfg.openai.stream,
            }
        elif cfg.backend == "onnx":
            backend_cfg = {
                "max_new_tokens": cfg.onnx.max_new_tokens,
                "device": cfg.onnx.device,
                "do_sample": cfg.onnx.do_sample,
                "export": cfg.onnx.export,
            }
        elif cfg.backend == "vllm":
            backend_cfg = {
                "max_new_tokens": cfg.vllm.max_new_tokens,
                "temperature": cfg.vllm.temperature,
                "tensor_parallel_size": cfg.vllm.tensor_parallel_size,
                "gpu_memory_utilization": cfg.vllm.gpu_memory_utilization,
                "dtype": cfg.vllm.dtype,
            }
        else:
            backend_cfg = {}

        data: dict[str, object] = {
            "config": config_path,
            "backend": cfg.backend,
            "model": cfg.model,
            "requests": cfg.requests,
            "concurrency": cfg.concurrency,
            "arrival_rate_rps": cfg.arrival_rate_rps,
            "warmup_requests": cfg.warmup_requests,
            "repeats": cfg.repeats,
            "prompts_file": cfg.resolve_prompts_file(),
            "workload_profile": cfg.workload_profile,
            "quality_file": cfg.quality_file,
            "seed": cfg.seed,
            "measure_perplexity": cfg.measure_perplexity,
            "measure_judge": cfg.measure_judge,
            "backend_config": backend_cfg,
            "valid": True,
        }
        click.echo(json.dumps(data))
        return

    click.echo(f"Config: {config_path}")
    click.echo(f"  backend          : {cfg.backend}")
    click.echo(f"  model            : {cfg.model}")
    click.echo(f"  requests         : {cfg.requests}")
    if cfg.arrival_rate_rps is not None:
        click.echo(f"  arrival rate     : {cfg.arrival_rate_rps} rps (open-loop)")
    else:
        click.echo(f"  concurrency      : {cfg.concurrency}")
    click.echo(f"  warmup_requests  : {cfg.warmup_requests}")
    click.echo(f"  repeats          : {cfg.repeats}")
    click.echo(f"  prompts_file     : {cfg.resolve_prompts_file()}")
    if cfg.workload_profile:
        click.echo(f"  workload_profile : {cfg.workload_profile}")
    if cfg.quality_file:
        click.echo(f"  quality_file     : {cfg.quality_file}")
    if cfg.seed is not None:
        click.echo(f"  seed             : {cfg.seed}")
    if cfg.measure_perplexity:
        click.echo(f"  measure_perplexity: {cfg.measure_perplexity}")
    if cfg.measure_judge:
        click.echo(f"  measure_judge    : {cfg.measure_judge}")

    if cfg.backend == "mock":
        click.echo(f"  mock.latency_ms  : {cfg.mock.latency_ms}")
        click.echo(f"  mock.tokens_per_response: {cfg.mock.tokens_per_response}")
    elif cfg.backend == "transformers":
        click.echo(f"  hf.max_new_tokens: {cfg.hf.max_new_tokens}")
        click.echo(f"  hf.device        : {cfg.hf.device}")
        click.echo(f"  hf.torch_dtype   : {cfg.hf.torch_dtype}")
    elif cfg.backend == "llama-cpp":
        click.echo(f"  llama_cpp.n_ctx       : {cfg.llama_cpp.n_ctx}")
        click.echo(f"  llama_cpp.n_gpu_layers: {cfg.llama_cpp.n_gpu_layers}")
        click.echo(f"  llama_cpp.max_tokens  : {cfg.llama_cpp.max_tokens}")
        click.echo(f"  llama_cpp.stream      : {cfg.llama_cpp.stream}")
    elif cfg.backend == "openai":
        click.echo(f"  openai.base_url  : {cfg.openai.base_url}")
        click.echo(f"  openai.max_tokens: {cfg.openai.max_tokens}")
        click.echo(f"  openai.timeout_s : {cfg.openai.timeout_s}")
        if cfg.openai.api_key_env:
            click.echo(f"  openai.api_key_env: {cfg.openai.api_key_env}")
    elif cfg.backend == "onnx":
        click.echo(f"  onnx.max_new_tokens: {cfg.onnx.max_new_tokens}")
        click.echo(f"  onnx.device        : {cfg.onnx.device}")
        click.echo(f"  onnx.do_sample     : {cfg.onnx.do_sample}")
        click.echo(f"  onnx.export        : {cfg.onnx.export}")
    elif cfg.backend == "vllm":
        click.echo(f"  vllm.max_new_tokens        : {cfg.vllm.max_new_tokens}")
        click.echo(f"  vllm.temperature           : {cfg.vllm.temperature}")
        click.echo(f"  vllm.tensor_parallel_size  : {cfg.vllm.tensor_parallel_size}")
        click.echo(f"  vllm.gpu_memory_utilization: {cfg.vllm.gpu_memory_utilization}")
        click.echo(f"  vllm.dtype                 : {cfg.vllm.dtype}")

    click.echo("OK")


@main.command("diff")
@click.argument("baseline_csv", type=click.Path(exists=True))
@click.argument("current_csv", type=click.Path(exists=True))
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write diff to file instead of stdout",
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    help="Output format: table=Markdown, json=machine-readable JSON, csv=spreadsheet/pandas",
)
@click.option(
    "--fail-on-regression",
    "fail_threshold",
    default=None,
    type=click.FloatRange(min=0.0),
    metavar="PCT",
    help=(
        "Exit 1 if any metric regresses by more than PCT% (use 0 for any regression). "
        "Useful for CI gating."
    ),
)
def diff_cmd(
    baseline_csv: str,
    current_csv: str,
    output_path: str | None,
    output_format: str,
    fail_threshold: float | None,
) -> None:
    """Compare two benchmark CSVs and show per-metric percentage change.

    Shows how key metrics changed between a baseline run and a current run.
    Annotates each metric with ✓ (improvement) or ✗ (regression):

        llm-bench diff results/before.csv results/after.csv
        llm-bench diff baseline.csv current.csv --format json

    Use --fail-on-regression to gate CI pipelines on metric quality:

        llm-bench diff baseline.csv current.csv --fail-on-regression 5
    """
    from llm_inference_benchmark.diff import (
        build_diff_csv,
        build_diff_json,
        build_diff_table,
        find_regressions,
    )

    if output_format == "json":
        text = build_diff_json(baseline_csv, current_csv)
    elif output_format == "csv":
        text = build_diff_csv(baseline_csv, current_csv)
    else:
        text = build_diff_table(baseline_csv, current_csv)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text + "\n")
        click.echo(f"Diff written to {output_path}")
    else:
        click.echo(text)

    if fail_threshold is not None:
        regressions = find_regressions(baseline_csv, current_csv, fail_threshold)
        if regressions:
            click.echo(
                f"\n✗ Regression check failed (threshold: {fail_threshold:.1f}%): "
                + ", ".join(regressions)
            )
            sys.exit(1)


@main.command("profiles")
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format: table=human-readable, json=machine-readable JSON array",
)
def profiles_cmd(output_format: str) -> None:
    """List available workload profiles and their descriptions.

    Profiles can be referenced by name in a benchmark config YAML
    (workload_profile: short_chat) or in a matrix config run entry.

        llm-bench profiles
        llm-bench profiles --format json
    """
    from llm_inference_benchmark.profiles import list_profiles

    profiles = list_profiles()

    if output_format == "json":
        data = [
            {
                "name": p.name,
                "input_length": p.input_length,
                "output_length": p.output_length,
                "description": p.description,
            }
            for p in profiles
        ]
        click.echo(json.dumps(data))
    else:
        for profile in profiles:
            click.echo(profile.name)
            click.echo(f"  Input : {profile.input_length}  Output: {profile.output_length}")
            click.echo(f"  {profile.description}")
            click.echo()


@main.command("verify")
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format: table=human-readable, json=machine-readable JSON",
)
def verify_cmd(output_format: str) -> None:
    """Check which backends are installed and run a smoke test on mock.

    Runs a minimal inference call on the mock backend (always available) and
    checks whether optional backend dependencies are installed.  Use this to
    confirm your setup is working before running expensive benchmarks.

        llm-bench verify
        llm-bench verify --format json
    """
    from llm_inference_benchmark.verify import run_probes

    probes = run_probes()

    if output_format == "json":
        data = [
            {
                "backend": p.backend,
                "status": p.status,
                "latency_ms": p.latency_ms,
                "reason": p.reason,
            }
            for p in probes
        ]
        click.echo(json.dumps(data))
        if any(p.status == "FAIL" for p in probes):
            sys.exit(1)
        return

    click.echo(f"{'BACKEND':<14} {'STATUS':<8} {'LATENCY':>10}  NOTES")
    click.echo("-" * 52)
    any_fail = False
    for p in probes:
        lat = f"{p.latency_ms:.1f} ms" if p.latency_ms is not None else "N/A"
        click.echo(f"{p.backend:<14} {p.status:<8} {lat:>10}  {p.reason}")
        if p.status == "FAIL":
            any_fail = True

    if any_fail:
        sys.exit(1)


@main.command("env")
@click.option(
    "--format",
    "fmt",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format: table=human-readable, json=machine-readable JSON",
)
def env_cmd(fmt: str) -> None:
    """Print current Python, package, and hardware environment.

    Useful for verifying GPU detection before a run and for sharing
    reproducibility context without running a full benchmark.

        llm-bench env
        llm-bench env --format json
    """
    from llm_inference_benchmark.manifest import collect_env_info

    info = collect_env_info()

    if fmt == "json":
        click.echo(json.dumps(asdict(info), indent=2))
        return

    click.echo(f"python      : {info.python_version.splitlines()[0]}")
    click.echo(f"platform    : {info.platform_info}")
    click.echo(f"cpu         : {info.cpu_model} ({info.cpu_count} cores)")
    click.echo(f"package     : llm-inference-benchmark {info.package_version}")
    for label, ver in [
        ("torch", info.torch_version),
        ("transformers", info.transformers_version),
        ("optimum", info.optimum_version),
        ("vllm", info.vllm_version),
        ("psutil", info.psutil_version),
    ]:
        if ver is not None:
            click.echo(f"{label:<12}: {ver}")
    if info.gpu:
        parts: list[str] = []
        if info.gpu.name:
            parts.append(info.gpu.name)
        if info.gpu.driver_version:
            parts.append(f"driver {info.gpu.driver_version}")
        if info.gpu.cuda_version:
            parts.append(f"CUDA {info.gpu.cuda_version}")
        if info.gpu.vram_total_mb is not None:
            parts.append(f"{info.gpu.vram_total_mb} MB VRAM")
        click.echo(f"gpu         : {' | '.join(parts) if parts else 'detected'}")
    else:
        click.echo("gpu         : not detected")


@main.command("matrix")
@click.option(
    "--config",
    "matrix_path",
    required=True,
    type=click.Path(exists=True),
    help="YAML matrix config file",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List runs without executing them",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    default=False,
    help="Continue remaining runs after a failure; exit 1 if any run failed",
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format: table=human-readable, json=machine-readable JSON",
)
def matrix_cmd(
    matrix_path: str, dry_run: bool, continue_on_error: bool, output_format: str
) -> None:
    """Execute all benchmark runs defined in a matrix config file.

    Each run writes a CSV and a manifest into the results directory:

        llm-bench matrix --config configs/matrix-example.yaml

    Preview what would run without executing:

        llm-bench matrix --config configs/matrix-example.yaml --dry-run

    Machine-readable dry-run listing:

        llm-bench matrix --config configs/matrix-example.yaml --dry-run --format json

    Continue remaining runs even when one fails:

        llm-bench matrix --config configs/matrix-example.yaml --continue-on-error

    Machine-readable execution summary:

        llm-bench matrix --config configs/matrix-example.yaml --format json

    Compare outputs afterwards:

        llm-bench compare results/*.csv
    """
    from llm_inference_benchmark.manifest import collect_manifest, write_manifest
    from llm_inference_benchmark.matrix import load_matrix
    from llm_inference_benchmark.runner import load_prompts, run_repeated

    matrix = load_matrix(matrix_path)
    results_dir = Path(matrix.results_dir)
    n = len(matrix.runs)

    if dry_run:
        if output_format == "json":
            click.echo(
                json.dumps(
                    {
                        "matrix": matrix_path,
                        "results_dir": str(results_dir),
                        "total": n,
                        "runs": [
                            {
                                "index": idx,
                                "name": run.name,
                                "config": run.config,
                                "workload_profile": run.workload_profile,
                                "dataset": run.dataset,
                                "overrides": run.overrides or {},
                                "output": str(results_dir / f"{run.name}.csv"),
                                "manifest": str(results_dir / f"{run.name}.manifest.json"),
                            }
                            for idx, run in enumerate(matrix.runs, 1)
                        ],
                    }
                )
            )
            return
        click.echo(f"Matrix: {n} run(s) → {results_dir}/")
        for idx, run in enumerate(matrix.runs, 1):
            click.echo(f"  [{idx}/{n}] {run.name}")
            click.echo(f"        config: {run.config}")
            if run.workload_profile:
                click.echo(f"        workload_profile: {run.workload_profile}")
            if run.dataset:
                click.echo(f"        dataset: {run.dataset}")
            if run.overrides:
                overrides_str = ", ".join(f"{k}={v}" for k, v in run.overrides.items())
                click.echo(f"        overrides: {overrides_str}")
            click.echo(f"        output: {results_dir / run.name}.csv")
        return

    _json = output_format == "json"
    results_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Matrix: {n} run(s) → {results_dir}/", err=_json)

    failures: list[tuple[str, str]] = []
    json_runs: list[dict[str, object]] = []

    for idx, run in enumerate(matrix.runs, 1):
        click.echo(f"\n[{idx}/{n}] {run.name}", err=_json)
        try:
            cfg = load_config(run.config)
            if run.overrides:
                from llm_inference_benchmark.sweep import apply_overrides

                cfg = apply_overrides(cfg, run.overrides)
            if run.workload_profile is not None:
                cfg = cfg.model_copy(update={"workload_profile": run.workload_profile})

            _t0 = time.perf_counter()
            backend = _build_backend(cfg)
            model_load_ms = (time.perf_counter() - _t0) * 1000.0
            if run.dataset:
                from llm_inference_benchmark.datasets import load_prompts as _ds_load

                try:
                    prompts = _ds_load(run.dataset, n=cfg.requests, seed=cfg.seed)
                except FileNotFoundError as exc:
                    raise click.ClickException(str(exc)) from exc
            else:
                prompts = load_prompts(cfg.resolve_prompts_file())
            click.echo(
                f"  Backend: {cfg.backend}  Model: {cfg.model}  Requests: {cfg.requests}",
                err=_json,
            )
            if run.dataset:
                click.echo(f"  Dataset: {run.dataset}", err=_json)
            report = run_repeated(backend, cfg, prompts, model_load_ms=model_load_ms)

            csv_path = results_dir / f"{run.name}.csv"
            manifest_path = results_dir / f"{run.name}.manifest.json"

            # Defense-in-depth: confirm paths are contained within results_dir.
            resolved_dir = results_dir.resolve()
            if not csv_path.resolve().is_relative_to(resolved_dir):
                raise click.ClickException(
                    f"Run name {run.name!r} would write outside results directory"
                )

            row = {k: ("" if v is None else v) for k, v in asdict(report).items()}
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            manifest = collect_manifest(run.config, cfg)
            write_manifest(manifest, manifest_path)

            click.echo(f"  → {csv_path}", err=_json)
            click.echo(f"  → {manifest_path}", err=_json)

            if _json:
                json_runs.append(
                    {
                        "index": idx,
                        "name": run.name,
                        "status": "ok",
                        "output": str(csv_path),
                        "manifest": str(manifest_path),
                    }
                )

            # Release backend resources (GPU memory, file handles) before the next run.
            del backend
            gc.collect()

        except Exception as exc:  # noqa: BLE001
            if not _json and not continue_on_error:
                raise
            msg = str(exc) or type(exc).__name__
            click.echo(f"  FAILED: {msg}", err=True)
            failures.append((run.name, msg))
            if _json:
                json_runs.append({"index": idx, "name": run.name, "status": "failed", "error": msg})

    n_ok = n - len(failures)
    if _json:
        click.echo(
            json.dumps(
                {
                    "matrix": matrix_path,
                    "results_dir": str(results_dir),
                    "total": n,
                    "succeeded": n_ok,
                    "failed": len(failures),
                    "runs": json_runs,
                }
            )
        )
        if failures:
            sys.exit(1)
        return

    if failures:
        click.echo(f"\nDone: {n_ok} completed, {len(failures)} failed.")
        click.echo("Failed runs:")
        for name, msg in failures:
            click.echo(f"  {name}: {msg}")
        sys.exit(1)

    click.echo("\nDone. Compare with:")
    click.echo(f"  llm-bench compare {results_dir}/*.csv")


@main.command("pipeline")
@click.option(
    "--config",
    "pipeline_path",
    required=True,
    type=click.Path(exists=True),
    help="YAML pipeline config file",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the run plan and exit without executing",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    default=False,
    help=(
        "Continue remaining cells after a failure; run post-processing on successful "
        "CSVs; exits 1 when any cell failed"
    ),
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format for terminal progress: table=human-readable, json=machine-readable",
)
def pipeline_cmd(
    pipeline_path: str,
    dry_run: bool,
    continue_on_error: bool,
    output_format: str,
) -> None:
    """Run a full benchmark study: matrix cells followed by compare, Pareto, and recommend.

    Reads a pipeline YAML config (a superset of the matrix format), executes all
    matrix cells, then writes comparison and analysis files to results_dir/:

        llm-bench pipeline --config configs/pipeline-example.yaml

    Preview the full plan without executing:

        llm-bench pipeline --config configs/pipeline-example.yaml --dry-run

    Continue on cell errors and run post-processing on successful results:

        llm-bench pipeline --config configs/pipeline-example.yaml --continue-on-error
    """
    from llm_inference_benchmark.compare import (
        filter_rows,
        load_csv,
        sort_rows,
    )
    from llm_inference_benchmark.compare import (
        render_json as _cmp_json,
    )
    from llm_inference_benchmark.compare import (
        render_table as _cmp_table,
    )
    from llm_inference_benchmark.config import load_config
    from llm_inference_benchmark.manifest import collect_manifest, write_manifest
    from llm_inference_benchmark.pipeline import load_pipeline
    from llm_inference_benchmark.runner import load_prompts, run_repeated

    pipeline = load_pipeline(pipeline_path)
    results_dir = Path(pipeline.results_dir)
    steps = pipeline.pipeline
    runs = pipeline.runs
    n = len(runs)

    if dry_run:
        click.echo(f"Pipeline: {n} cell(s) → {results_dir}/")
        for idx, run in enumerate(runs, 1):
            click.echo(f"  [{idx}/{n}] {run.name}")
            click.echo(f"        config: {run.config}")
            if run.workload_profile:
                click.echo(f"        workload_profile: {run.workload_profile}")
            if run.overrides:
                overrides_str = ", ".join(f"{k}={v}" for k, v in run.overrides.items())
                click.echo(f"        overrides: {overrides_str}")
        click.echo("Post-processing:")
        click.echo(f"  compare  → {results_dir}/compare.md, {results_dir}/compare.json")
        if steps.pareto:
            click.echo(f"  pareto   → {results_dir}/pareto.md, {results_dir}/pareto.json")
        if steps.recommend is not None:
            click.echo(f"  recommend → {results_dir}/recommend.md, {results_dir}/recommend.json")
        return

    _json = output_format == "json"
    results_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Pipeline: {n} cell(s) → {results_dir}/", err=_json)

    failures: list[tuple[str, str]] = []
    successful_csvs: list[Path] = []
    json_runs: list[dict[str, object]] = []

    for idx, run in enumerate(runs, 1):
        click.echo(f"\n[{idx}/{n}] {run.name}", err=_json)
        try:
            cfg = load_config(run.config)
            if run.overrides:
                from llm_inference_benchmark.sweep import apply_overrides

                cfg = apply_overrides(cfg, run.overrides)
            if run.workload_profile is not None:
                cfg = cfg.model_copy(update={"workload_profile": run.workload_profile})

            _t0 = time.perf_counter()
            backend = _build_backend(cfg)
            model_load_ms = (time.perf_counter() - _t0) * 1000.0
            if run.dataset:
                from llm_inference_benchmark.datasets import load_prompts as _ds_load

                try:
                    prompts = _ds_load(run.dataset, n=cfg.requests, seed=cfg.seed)
                except FileNotFoundError as exc:
                    raise click.ClickException(str(exc)) from exc
            else:
                prompts = load_prompts(cfg.resolve_prompts_file())
            click.echo(
                f"  Backend: {cfg.backend}  Model: {cfg.model}  Requests: {cfg.requests}",
                err=_json,
            )
            if run.dataset:
                click.echo(f"  Dataset: {run.dataset}", err=_json)
            report = run_repeated(backend, cfg, prompts, model_load_ms=model_load_ms)

            csv_path = results_dir / f"{run.name}.csv"
            manifest_path = results_dir / f"{run.name}.manifest.json"

            # Defense-in-depth: confirm paths are contained within results_dir.
            resolved_dir = results_dir.resolve()
            if not csv_path.resolve().is_relative_to(resolved_dir):
                raise click.ClickException(
                    f"Run name {run.name!r} would write outside results directory"
                )

            row = {k: ("" if v is None else v) for k, v in asdict(report).items()}
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            manifest = collect_manifest(run.config, cfg)
            write_manifest(manifest, manifest_path)

            click.echo(f"  → {csv_path}", err=_json)
            click.echo(f"  → {manifest_path}", err=_json)
            successful_csvs.append(csv_path)

            if _json:
                json_runs.append(
                    {
                        "index": idx,
                        "name": run.name,
                        "status": "ok",
                        "output": str(csv_path),
                        "manifest": str(manifest_path),
                    }
                )

            del backend
            gc.collect()

        except Exception as exc:  # noqa: BLE001
            if not _json and not continue_on_error:
                raise
            msg = str(exc) or type(exc).__name__
            click.echo(f"  FAILED: {msg}", err=True)
            failures.append((run.name, msg))
            if _json:
                json_runs.append({"index": idx, "name": run.name, "status": "failed", "error": msg})

    has_recommend_winner = True
    if successful_csvs:
        rows = [load_csv(p) for p in successful_csvs]

        filtered = filter_rows(rows, steps.compare_filter)
        sorted_rows = sort_rows(filtered, sort_by=steps.compare_sort)
        if steps.compare_limit is not None:
            sorted_rows = sorted_rows[: steps.compare_limit]
        (results_dir / "compare.md").write_text(_cmp_table(sorted_rows) + "\n")
        (results_dir / "compare.json").write_text(_cmp_json(sorted_rows) + "\n")
        click.echo(
            f"\ncompare  → {results_dir}/compare.md, {results_dir}/compare.json",
            err=_json,
        )

        if steps.pareto:
            from llm_inference_benchmark.pareto import (
                pareto_classify,
                render_pareto_json,
                render_pareto_table,
            )

            classified = pareto_classify(rows)
            (results_dir / "pareto.md").write_text(render_pareto_table(classified) + "\n")
            (results_dir / "pareto.json").write_text(render_pareto_json(classified) + "\n")
            click.echo(
                f"pareto   → {results_dir}/pareto.md, {results_dir}/pareto.json",
                err=_json,
            )

        if steps.recommend is not None:
            from llm_inference_benchmark.recommend import (
                Constraints,
                recommend,
                render_recommendation,
                render_recommendation_json,
            )

            _CONSTRAINT_FIELDS = frozenset(
                {
                    "max_vram_mb",
                    "max_p95_ms",
                    "min_sanity",
                    "min_quality",
                    "max_perplexity",
                    "min_judge",
                    "max_load_ms",
                    "max_ttft_ms",
                }
            )
            constraints = Constraints(
                **{k: v for k, v in steps.recommend.items() if k in _CONSTRAINT_FIELDS}
            )
            result = recommend(rows, constraints)
            (results_dir / "recommend.md").write_text(render_recommendation(result) + "\n")
            (results_dir / "recommend.json").write_text(render_recommendation_json(result) + "\n")
            click.echo(
                f"recommend → {results_dir}/recommend.md, {results_dir}/recommend.json",
                err=_json,
            )
            has_recommend_winner = result.winner is not None

    n_ok = n - len(failures)
    if _json:
        click.echo(
            json.dumps(
                {
                    "pipeline": pipeline_path,
                    "results_dir": str(results_dir),
                    "total": n,
                    "succeeded": n_ok,
                    "failed": len(failures),
                    "runs": json_runs,
                }
            )
        )
    elif failures:
        click.echo(f"\nDone: {n_ok} completed, {len(failures)} failed.")
        click.echo("Failed runs:")
        for name, msg in failures:
            click.echo(f"  {name}: {msg}")
    else:
        click.echo(f"\nDone. Results in {results_dir}/")

    if failures or not has_recommend_winner:
        sys.exit(1)


@main.command("sweep")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Base YAML config file. concurrency is overridden at each step.",
)
@click.option(
    "--concurrency-range",
    "concurrency_range",
    required=True,
    help="Comma-separated concurrency levels, e.g. 1,2,4,8",
)
@click.option(
    "--max-p95-ms",
    "max_p95_ms",
    default=None,
    type=float,
    help="Stop early when p95 latency exceeds this threshold (ms)",
)
@click.option(
    "--requests",
    "requests_override",
    default=None,
    type=int,
    help="Number of benchmark requests per level (overrides config)",
)
@click.option(
    "--output",
    "output_path",
    default="sweep_results.csv",
    show_default=True,
    help="Path for combined sweep CSV",
)
def sweep_cmd(
    config_path: str,
    concurrency_range: str,
    max_p95_ms: float | None,
    requests_override: int | None,
    output_path: str,
) -> None:
    """Ramp concurrency and emit a throughput-vs-latency curve.

    Run the benchmark at each concurrency level listed in --concurrency-range,
    writing one combined CSV with a row per level.  Stops early when p95 latency
    exceeds --max-p95-ms; exits with code 1 in that case.

    Example:

        llm-bench sweep --config configs/example.yaml --concurrency-range 1,2,4,8
        llm-bench sweep --config configs/example.yaml --concurrency-range 1,2,4,8 --max-p95-ms 5000
    """
    try:
        levels = [int(x.strip()) for x in concurrency_range.split(",") if x.strip()]
    except ValueError:
        raise click.UsageError(
            "--concurrency-range must be comma-separated integers, e.g. 1,2,4,8"
        ) from None
    if not levels:
        raise click.UsageError("--concurrency-range must contain at least one value")
    for lvl in levels:
        if lvl < 1:
            raise click.UsageError("All concurrency values must be >= 1")

    base_cfg = load_config(config_path)
    if requests_override is not None:
        base_cfg = base_cfg.model_copy(update={"requests": requests_override})

    click.echo(f"Sweep: {len(levels)} level(s) → {output_path}")
    click.echo(f"  Config: {config_path}  Model: {base_cfg.model}  Backend: {base_cfg.backend}")
    click.echo(f"  Requests per level: {base_cfg.requests}")
    if max_p95_ms is not None:
        click.echo(f"  Max p95: {max_p95_ms:.0f} ms")
    click.echo("")

    _t0 = time.perf_counter()
    backend = _build_backend(base_cfg)
    model_load_ms = (time.perf_counter() - _t0) * 1000.0
    prompts = load_prompts(base_cfg.resolve_prompts_file())

    # Each row: (concurrency, throughput_rps, report)
    sweep_rows: list[tuple[int, float, Any]] = []
    threshold_breached = False

    try:
        for idx, concurrency in enumerate(levels, 1):
            cfg = base_cfg.model_copy(update={"concurrency": concurrency})
            click.echo(f"[{idx}/{len(levels)}] concurrency={concurrency}  requests={cfg.requests}")

            report = run_repeated(
                backend,
                cfg,
                prompts,
                model_load_ms=model_load_ms if idx == 1 else None,
            )

            # throughput_rps = request_count / wall_elapsed_s
            # tokens_per_second = total_output_tokens / wall_elapsed_s
            # mean_output_tokens = total_output_tokens / request_count
            # → throughput_rps = tokens_per_second / mean_output_tokens
            if report.mean_output_tokens > 0:
                throughput_rps = report.tokens_per_second / report.mean_output_tokens
            else:
                throughput_rps = 0.0

            sweep_rows.append((concurrency, throughput_rps, report))

            click.echo(
                f"  p50={report.p50_latency_ms:.1f} ms  p95={report.p95_latency_ms:.1f} ms"
                f"  tok/s={report.tokens_per_second:.1f}  rps={throughput_rps:.3f}"
            )

            if max_p95_ms is not None and report.p95_latency_ms > max_p95_ms:
                click.echo(
                    f"  WARNING: p95 {report.p95_latency_ms:.1f} ms exceeds"
                    f" {max_p95_ms:.0f} ms — stopping sweep.",
                    err=True,
                )
                threshold_breached = True
                break
    finally:
        del backend
        gc.collect()

    if not sweep_rows:
        raise click.ClickException("No sweep levels completed.")

    # Write combined CSV
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    first_report_dict = {k: ("" if v is None else v) for k, v in asdict(sweep_rows[0][2]).items()}
    fieldnames = ["concurrency", "throughput_rps", *first_report_dict.keys()]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for concurrency, throughput_rps, report in sweep_rows:
            row = {
                "concurrency": concurrency,
                "throughput_rps": round(throughput_rps, 4),
                **{k: ("" if v is None else v) for k, v in asdict(report).items()},
            }
            writer.writerow(row)
    click.echo(f"\nSweep results written to {out}")

    # Summary table — knee point = highest rps among completed levels
    best_idx = max(range(len(sweep_rows)), key=lambda i: sweep_rows[i][1])
    click.echo("\n=== Sweep Summary ===")
    header = f"{'Concurrency':>12}  {'RPS':>8}  {'p50 ms':>8}  {'p95 ms':>8}  {'tok/s':>8}"
    click.echo(header)
    click.echo("-" * len(header))
    for i, (concurrency, throughput_rps, report) in enumerate(sweep_rows):
        marker = "  <- knee" if i == best_idx else ""
        click.echo(
            f"{concurrency:>12}  {throughput_rps:>8.3f}  {report.p50_latency_ms:>8.1f}"
            f"  {report.p95_latency_ms:>8.1f}  {report.tokens_per_second:>8.1f}{marker}"
        )

    best_concurrency, best_rps, best_report = sweep_rows[best_idx]
    click.echo(
        f"\nKnee point: concurrency={best_concurrency}"
        f"  rps={best_rps:.3f}"
        f"  p95={best_report.p95_latency_ms:.1f} ms"
    )

    if threshold_breached:
        sys.exit(1)


@main.command("pull")
@click.argument("repo_id")
@click.option(
    "--quant",
    default=None,
    metavar="QUANT",
    help="GGUF quantization suffix to download (e.g. Q4_K_M). Required for --backend gguf.",
)
@click.option(
    "--backend",
    type=click.Choice(["gguf", "transformers"]),
    default=None,
    help="Download backend. Defaults to gguf when --quant is set, otherwise transformers.",
)
@click.option(
    "--dest",
    "dest_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Destination directory for GGUF files (default: ~/models/).",
)
@click.option(
    "--max-size-gb",
    default=10.0,
    show_default=True,
    type=float,
    help="Abort if the remote file exceeds this size in GB.",
)
@click.option(
    "--token",
    "hf_token",
    default=None,
    envvar="HF_TOKEN",
    metavar="TOKEN",
    help="HuggingFace access token (or set HF_TOKEN env var).",
)
def pull_cmd(
    repo_id: str,
    quant: str | None,
    backend: str | None,
    dest_dir: Path | None,
    max_size_gb: float,
    hf_token: str | None,
) -> None:
    """Download a model from HuggingFace Hub.

    For GGUF models, pass --quant to select the quantization variant:

    \b
        llm-bench pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --quant Q4_K_M
        llm-bench pull HuggingFaceTB/SmolLM2-360M-Instruct --backend transformers
        llm-bench pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF --quant Q5_K_M --max-size-gb 5
    """
    from llm_inference_benchmark.puller import pull_gguf, pull_transformers

    resolved_backend = backend or ("gguf" if quant else "transformers")

    try:
        if resolved_backend == "gguf":
            if not quant:
                raise click.UsageError(
                    "--quant is required when using --backend gguf. Example: --quant Q4_K_M"
                )
            result = pull_gguf(
                repo_id,
                quant,
                dest_dir=dest_dir,
                token=hf_token,
                max_size_gb=max_size_gb,
            )
        else:
            result = pull_transformers(
                repo_id,
                token=hf_token,
                max_size_gb=max_size_gb,
            )
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc

    if result.skipped:
        click.echo(f"Already cached: {result.path}")
    else:
        click.echo(f"Downloaded to: {result.path}")


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=8080, show_default=True, type=int, help="Bind port")
def serve_cmd(host: str, port: int) -> None:
    """Start the llm-bench Web API server.

    Exposes a REST API for submitting benchmark jobs, streaming progress via
    Server-Sent Events, and querying past results from a local SQLite store.

        llm-bench serve
        llm-bench serve --host 0.0.0.0 --port 8080
    """
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn is required: uv pip install 'llm-inference-benchmark[server]'"
        ) from None
    from llm_inference_benchmark.server import app

    uvicorn.run(app, host=host, port=port)


def _print_report(report: object) -> None:
    """Print benchmark results to stdout.

    For repeated runs (report.repeats > 1), fields that have a corresponding _std
    sibling are printed as "value ± std (n=N)" to show run-to-run spread.
    Single-run output is identical to pre-v0.19 format.
    """
    from dataclasses import asdict

    from llm_inference_benchmark.metrics import MetricsReport

    assert isinstance(report, MetricsReport)
    report_dict = asdict(report)
    n = report_dict.get("repeats")
    for k, v in report_dict.items():
        if k.endswith("_std"):
            continue  # displayed inline alongside its parent field
        std_key = k + "_std"
        std_v = report_dict.get(std_key)
        if n is not None and n > 1 and std_v is not None and isinstance(v, float):
            click.echo(f"  {k}: {v:.2f} ± {std_v:.2f} (n={n})")
        elif isinstance(v, float):
            click.echo(f"  {k}: {v:.2f}")
        elif v is None:
            click.echo(f"  {k}: N/A")
        else:
            click.echo(f"  {k}: {v}")


@main.group("datasets")
def datasets_group() -> None:
    """Manage cached real-world prompt datasets for benchmarking."""


@datasets_group.command("pull")
@click.argument("name")
@click.option(
    "--max-samples",
    "max_samples",
    default=None,
    type=click.IntRange(min=1),
    metavar="N",
    help="Override maximum number of samples to download (default: per-dataset limit)",
)
@click.option(
    "--token",
    "hf_token",
    default=None,
    envvar="HF_TOKEN",
    metavar="TOKEN",
    help="HuggingFace access token (or set HF_TOKEN env var)",
)
def datasets_pull(name: str, max_samples: int | None, hf_token: str | None) -> None:
    """Download and cache a real-world prompt dataset.

    Supported names: lmsys-chat, hermes-fn,
    long-context-4k, long-context-16k, long-context-64k

    Example:

        llm-bench datasets pull lmsys-chat
        llm-bench datasets pull long-context-4k
    """
    from llm_inference_benchmark.datasets import REGISTRY, pull

    if name not in REGISTRY:
        known = ", ".join(REGISTRY)
        raise click.UsageError(f"Unknown dataset {name!r}. Known datasets: {known}")
    try:
        out = pull(name, hf_token=hf_token, max_samples=max_samples)
        click.echo(f"Dataset {name!r} cached at {out}")
    except ImportError as exc:
        raise click.UsageError(str(exc)) from exc


@datasets_group.command("list")
def datasets_list() -> None:
    """List locally cached datasets and their sample counts."""
    from llm_inference_benchmark.datasets import list_cached

    rows = list_cached()
    if not rows:
        click.echo("No datasets cached. Run: llm-bench datasets pull <name>")
        return
    click.echo(f"{'Dataset':<20}  Samples")
    click.echo("-" * 30)
    for name, count in rows:
        click.echo(f"{name:<20}  {count}")


@datasets_group.command("info")
@click.argument("name")
@click.option(
    "--samples",
    "n_samples",
    default=5,
    show_default=True,
    type=click.IntRange(min=0),
    help="Number of example prompts to show (0 to skip)",
)
def datasets_info(name: str, n_samples: int) -> None:
    """Show metadata and example prompts for a dataset.

    Prints HuggingFace repo, description, sample limit, cached status,
    and up to --samples example prompts from the local cache.

    Example:

        llm-bench datasets info wildchat
        llm-bench datasets info lmsys-chat --samples 3
    """
    from llm_inference_benchmark.datasets import REGISTRY, dataset_info

    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise click.UsageError(f"Unknown dataset {name!r}. Known datasets: {known}")

    info = dataset_info(name, n_samples=n_samples)

    cached_str = f"✓  ({info['sample_count']} samples)" if info["cached"] else "✗  (not cached)"
    click.echo(f"Dataset   : {info['name']}")
    click.echo(f"HF repo   : {info['hf_repo']}")
    click.echo(f"Description: {info['description']}")
    click.echo(f"Max samples: {info['max_samples']}")
    click.echo(f"Cached    : {cached_str}")

    if not info["cached"]:
        click.echo(f"\nRun to cache:  llm-bench datasets pull {name}")
        return

    if n_samples > 0 and info["samples"]:
        click.echo(f"\nSample prompts ({len(info['samples'])} of {info['sample_count']}):")
        for i, prompt in enumerate(info["samples"], 1):
            preview = prompt[:120].replace("\n", " ")
            if len(prompt) > 120:
                preview += "…"
            click.echo(f"  [{i}] {preview}")


def _build_backend(cfg: BenchmarkConfig, api_key: str | None = None) -> Backend:
    if cfg.backend == "mock":
        return MockBackend(
            model=cfg.model,
            latency_ms=cfg.mock.latency_ms,
            tokens_per_response=cfg.mock.tokens_per_response,
            seed=cfg.seed,
        )
    if cfg.backend == "transformers":
        from llm_inference_benchmark.backends.hf import HFBackend  # lazy: optional dep

        return HFBackend(
            model_id=cfg.model,
            max_new_tokens=cfg.hf.max_new_tokens,
            device=cfg.hf.device,
            torch_dtype=cfg.hf.torch_dtype,
            do_sample=cfg.hf.do_sample,
            seed=cfg.seed,
        )
    if cfg.backend == "llama-cpp":
        from llm_inference_benchmark.backends.llama_cpp import LlamaCppBackend  # lazy: optional dep

        return LlamaCppBackend(
            model_path=cfg.model,
            n_ctx=cfg.llama_cpp.n_ctx,
            n_gpu_layers=cfg.llama_cpp.n_gpu_layers,
            max_tokens=cfg.llama_cpp.max_tokens,
            temperature=cfg.llama_cpp.temperature,
            n_threads=cfg.llama_cpp.n_threads,
            verbose=cfg.llama_cpp.verbose,
            stream=cfg.llama_cpp.stream,
            seed=cfg.seed,
        )
    if cfg.backend == "openai":
        from llm_inference_benchmark.backends.openai_endpoint import OpenAIEndpointBackend

        return OpenAIEndpointBackend(
            base_url=cfg.openai.base_url,
            model=cfg.model,
            max_tokens=cfg.openai.max_tokens,
            temperature=cfg.openai.temperature,
            timeout_s=cfg.openai.timeout_s,
            api_key_env=cfg.openai.api_key_env,
            api_key=api_key,
            stream=cfg.openai.stream,
            seed=cfg.seed,
        )
    if cfg.backend == "onnx":
        from llm_inference_benchmark.backends.onnx import OnnxBackend  # lazy: optional dep

        return OnnxBackend(
            model_id=cfg.model,
            max_new_tokens=cfg.onnx.max_new_tokens,
            device=cfg.onnx.device,
            do_sample=cfg.onnx.do_sample,
            export=cfg.onnx.export,
            seed=cfg.seed,
        )
    if cfg.backend == "vllm":
        from llm_inference_benchmark.backends.vllm_backend import VLLMBackend  # lazy: optional dep

        return VLLMBackend(
            model_id=cfg.model,
            max_new_tokens=cfg.vllm.max_new_tokens,
            temperature=cfg.vllm.temperature,
            tensor_parallel_size=cfg.vllm.tensor_parallel_size,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            dtype=cfg.vllm.dtype,
            seed=cfg.seed,
        )
    raise ValueError(f"Unknown backend: {cfg.backend!r}")
