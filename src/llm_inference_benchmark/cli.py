from __future__ import annotations

import csv
import gc
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click

from llm_inference_benchmark import __version__
from llm_inference_benchmark.backends.base import Backend
from llm_inference_benchmark.backends.mock import MockBackend
from llm_inference_benchmark.config import BenchmarkConfig, load_config
from llm_inference_benchmark.runner import load_prompts, run_repeated


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
def main(
    ctx: click.Context,
    config_path: str | None,
    output_path: str | None,
    manifest_path: str | None,
    requests_override: int | None,
    warmup_requests_override: int | None,
    concurrency_override: int | None,
) -> None:
    """LLM inference benchmark toolkit.

    Run without a subcommand to execute a benchmark:

        llm-bench --config configs/example.yaml --output results.csv

    Override individual config knobs without editing the YAML:

        llm-bench --config configs/example.yaml --requests 50 --concurrency 4

    Use the compare subcommand to generate a Markdown table from saved CSVs:

        llm-bench compare results_a.csv results_b.csv
    """
    if ctx.invoked_subcommand is not None:
        return

    if config_path is None:
        raise click.UsageError("--config is required when running a benchmark")

    cfg = load_config(config_path)
    overrides: dict[str, int] = {}
    if requests_override is not None:
        overrides["requests"] = requests_override
    if warmup_requests_override is not None:
        overrides["warmup_requests"] = warmup_requests_override
    if concurrency_override is not None:
        overrides["concurrency"] = concurrency_override
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    _t0 = time.perf_counter()
    backend = _build_backend(cfg)
    model_load_ms = (time.perf_counter() - _t0) * 1000.0
    prompts = load_prompts(cfg.resolve_prompts_file())

    click.echo(f"Backend: {cfg.backend}  Model: {cfg.model}  Requests: {cfg.requests}")
    report = run_repeated(backend, cfg, prompts, model_load_ms=model_load_ms)

    if output_path:
        row = {k: ("" if v is None else v) for k, v in asdict(report).items()}
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        click.echo(f"Results written to {output_path}")

    if manifest_path:
        from llm_inference_benchmark.manifest import collect_manifest, write_manifest

        manifest = collect_manifest(config_path, cfg)
        write_manifest(manifest, manifest_path)
        click.echo(f"Manifest written to {manifest_path}")

    click.echo("\n=== Benchmark Results ===")
    _print_report(report)


@main.command("compare")
@click.argument("csv_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--sort",
    "sort_by",
    default="p95",
    show_default=True,
    type=click.Choice(["backend", "model", "p95", "toks", "load"], case_sensitive=False),
    help="Sort column: toks=highest throughput first; load=fastest load first (N/A last)",
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


@main.command("pareto")
@click.argument("csv_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write Markdown to file instead of stdout",
)
def pareto_cmd(csv_files: tuple[str, ...], output_path: str | None) -> None:
    """Identify Pareto-optimal benchmark configurations from CSV files.

    A configuration is Pareto-optimal when no other configuration is at least
    as good on every metric and strictly better on at least one.  Metrics:
    lower p95 latency, higher tok/s, lower VRAM (when available), higher
    sanity pass rate (when available).

        llm-bench pareto results/q4km.csv results/q8.csv
    """
    from llm_inference_benchmark.pareto import build_pareto_table

    table = build_pareto_table(list(csv_files))
    if output_path:
        Path(output_path).write_text(table + "\n")
        click.echo(f"Pareto table written to {output_path}")
    else:
        click.echo(table)


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
    output_path: str | None,
) -> None:
    """Recommend the best benchmark configuration under explicit constraints.

    Reads saved benchmark CSVs and returns the fastest Pareto-optimal
    configuration that satisfies all given constraints.  Runs that violate
    a constraint are listed with the reason they were excluded.

    Exits with code 1 when no run satisfies all constraints.

        llm-bench recommend results/*.csv --max-vram-mb 4096 --max-p95-ms 1000
    """
    from llm_inference_benchmark.recommend import Constraints, build_recommendation

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
    text, has_winner = build_recommendation(list(csv_files), constraints)
    if output_path:
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
def validate_config_cmd(config_path: str) -> None:
    """Validate a benchmark config file and print a summary of resolved settings.

    Reads the YAML, runs full pydantic validation, resolves the effective
    prompts file, and prints a summary.  Exits 0 on success, 1 on error.

        llm-bench validate-config --config configs/example.yaml
    """
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Config: {config_path}")
    click.echo(f"  backend          : {cfg.backend}")
    click.echo(f"  model            : {cfg.model}")
    click.echo(f"  requests         : {cfg.requests}")
    click.echo(f"  concurrency      : {cfg.concurrency}")
    click.echo(f"  warmup_requests  : {cfg.warmup_requests}")
    click.echo(f"  repeats          : {cfg.repeats}")
    click.echo(f"  prompts_file     : {cfg.resolve_prompts_file()}")
    if cfg.workload_profile:
        click.echo(f"  workload_profile : {cfg.workload_profile}")
    if cfg.quality_file:
        click.echo(f"  quality_file     : {cfg.quality_file}")
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
def diff_cmd(baseline_csv: str, current_csv: str, output_path: str | None) -> None:
    """Compare two benchmark CSVs and show per-metric percentage change.

    Shows how key metrics changed between a baseline run and a current run.
    Annotates each metric with ✓ (improvement) or ✗ (regression):

        llm-bench diff results/before.csv results/after.csv
    """
    from llm_inference_benchmark.diff import build_diff_table

    table = build_diff_table(baseline_csv, current_csv)
    if output_path:
        Path(output_path).write_text(table + "\n")
        click.echo(f"Diff written to {output_path}")
    else:
        click.echo(table)


@main.command("profiles")
def profiles_cmd() -> None:
    """List available workload profiles and their descriptions.

    Profiles can be referenced by name in a benchmark config YAML
    (workload_profile: short_chat) or in a matrix config run entry.

        llm-bench profiles
    """
    from llm_inference_benchmark.profiles import list_profiles

    for profile in list_profiles():
        click.echo(profile.name)
        click.echo(f"  Input : {profile.input_length}  Output: {profile.output_length}")
        click.echo(f"  {profile.description}")
        click.echo()


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
def matrix_cmd(matrix_path: str, dry_run: bool, continue_on_error: bool) -> None:
    """Execute all benchmark runs defined in a matrix config file.

    Each run writes a CSV and a manifest into the results directory:

        llm-bench matrix --config configs/matrix-example.yaml

    Preview what would run without executing:

        llm-bench matrix --config configs/matrix-example.yaml --dry-run

    Continue remaining runs even when one fails:

        llm-bench matrix --config configs/matrix-example.yaml --continue-on-error

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
        click.echo(f"Matrix: {n} run(s) → {results_dir}/")
        for idx, run in enumerate(matrix.runs, 1):
            click.echo(f"  [{idx}/{n}] {run.name}")
            click.echo(f"        config: {run.config}")
            if run.workload_profile:
                click.echo(f"        workload_profile: {run.workload_profile}")
            if run.overrides:
                overrides_str = ", ".join(f"{k}={v}" for k, v in run.overrides.items())
                click.echo(f"        overrides: {overrides_str}")
            click.echo(f"        output: {results_dir / run.name}.csv")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Matrix: {n} run(s) → {results_dir}/")

    failures: list[tuple[str, str]] = []

    for idx, run in enumerate(matrix.runs, 1):
        click.echo(f"\n[{idx}/{n}] {run.name}")
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
            prompts = load_prompts(cfg.resolve_prompts_file())
            click.echo(f"  Backend: {cfg.backend}  Model: {cfg.model}  Requests: {cfg.requests}")
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

            click.echo(f"  → {csv_path}")
            click.echo(f"  → {manifest_path}")

            # Release backend resources (GPU memory, file handles) before the next run.
            del backend
            gc.collect()

        except Exception as exc:  # noqa: BLE001
            if not continue_on_error:
                raise
            msg = str(exc) or type(exc).__name__
            click.echo(f"  FAILED: {msg}", err=True)
            failures.append((run.name, msg))

    n_ok = n - len(failures)
    if failures:
        click.echo(f"\nDone: {n_ok} completed, {len(failures)} failed.")
        click.echo("Failed runs:")
        for name, msg in failures:
            click.echo(f"  {name}: {msg}")
        sys.exit(1)

    click.echo("\nDone. Compare with:")
    click.echo(f"  llm-bench compare {results_dir}/*.csv")


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
            stream=cfg.openai.stream,
        )
    raise ValueError(f"Unknown backend: {cfg.backend!r}")
