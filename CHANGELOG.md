# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versions before
1.0.0 may include breaking changes between minor releases.

## [Unreleased]

## [1.8.0] - 2026-06-26

### Added

- **`llm-bench report`**: new command that reads one or more benchmark CSV output files and
  generates a self-contained HTML report containing an interactive Plotly scatter chart
  (throughput vs p95 latency with Pareto-optimal runs highlighted) and a full metrics summary
  table.  The report can be opened in any browser without running the server — useful for sharing
  results in GitHub PRs, issues, or wikis.

  ```bash
  llm-bench report result.csv
  llm-bench report run1.csv run2.csv --output comparison.html
  llm-bench report *.csv --title "Llama-3.2 Quant Comparison"
  ```

### Fixed

- **Docker GPU image build**: `docker compose up webui-gpu` failed with
  `libcuda.so.1 not found` during llama-cpp-python CUDA compilation.
  Added `LIBRARY_PATH=/usr/local/cuda/lib64/stubs` to the `gpu` and
  `webui-gpu` Dockerfile stages so the linker finds the CUDA driver stub
  at build time (the real driver is provided by NVIDIA Container Toolkit
  at runtime).

## [1.7.0] - 2026-06-26

### Added

- **`llm-bench datasets info <name>`**: new subcommand that prints dataset metadata (HF repo,
  description, max samples, cache status) and a configurable number of example prompts from the
  local cache.  Use `--samples N` to control how many prompts are shown (default 5; 0 to skip).
  When the dataset is not cached, the command prints a pull hint instead of samples.

- **`dataset` field in matrix/pipeline run entries**: each run in a `matrix` or `pipeline`
  YAML config can now set `dataset: <name>` to load prompts from a cached real-world dataset
  (`~/.cache/llm-bench/datasets/<name>.jsonl`) instead of the config's `prompts_file`.
  This enables apples-to-apples backend comparisons on the same real-world prompt set in a
  single matrix sweep.  Pull the dataset first with `llm-bench datasets pull <name>`, then:

  ```yaml
  runs:
    - name: llama-cpp-lmsys
      config: configs/llama-cpp-cpu.yaml
      dataset: lmsys-chat
    - name: transformers-lmsys
      config: configs/transformers-cpu.yaml
      dataset: lmsys-chat
  ```

  `--dry-run` prints the dataset name alongside the run config.  Unknown or uncached datasets
  fail fast with a clear error message before any benchmark runs start.

## [1.6.0] - 2026-06-26

### Added

- **Recommend panel in the Web UI**: a new **Recommend** tab in the sidebar toolbar lets
  users specify hardware and quality constraints (max VRAM, max p95 latency, max TTFT,
  min sanity rate, min quality rate) and get an instant constraint-based recommendation
  from all stored benchmark runs. The result panel shows the winning run with key metrics,
  up to four runners-up, and a count of excluded runs. Powered by the new
  `GET /api/ui/recommend` HTMX endpoint; no new dependencies.

## [1.5.1] - 2026-06-26

### Added

- **`docs/cli.md`**: added reference sections for five previously undocumented subcommands —
  `sweep`, `pull`, `datasets pull/list`, `pipeline`, and `verify` — each with an options
  table and usage examples.
- **`docs/cli.md`**: documented the `--arrival-rate`, `--dataset`, `--base-url`, and
  `--api-key` global flags in the "Default mode" options table, with open-loop load mode
  examples.
- **`docs/cli.md`**: added `arrival_rate_rps` to the Config YAML reference table.

### Fixed

- **Concurrent runner (asyncio)**: `asyncio.run()` raised `RuntimeError` when called
  from inside a running event loop (e.g. after Playwright e2e tests or from a Jupyter
  notebook). The runner now detects an existing loop and offloads async work to a
  thread-pool executor, making `--concurrency` and `--arrival-rate` work correctly in
  all contexts.
- **CUDA capability test mock**: `test_capabilities_cuda_primary_probe_true` and the
  related GPU-detection tests were not patching `sys.modules["llama_cpp.llama_cpp"]`
  (the inner module that the capabilities endpoint imports). The mocks now cover both
  the outer and inner modules, making the tests deterministic across CPU-only and
  GPU-enabled environments.

## [1.5.0] - 2026-06-26

### Added

- **`scripts/setup-gpu.sh`**: one-command GPU setup script — detects the NVIDIA GPU via
  `nvidia-smi` and installs the CUDA-enabled llama-cpp-python wheel automatically.
- **Makefile shortcuts**: `make webui` (start Web UI), `make webui-gpu` (install CUDA wheel
  then start Web UI), and `make setup-gpu` (run the GPU setup script).
- **docs**: `docs/quickstart.md` gains a **GPU Quick Start** section (3-command path with
  CUDA wheel warning) and an updated **Web UI** section describing the Leaderboard panel and
  Compare bar. `docs/metrics.md` adds definitions for `p50_ttft_ms`, `p95_ttft_ms`, and
  `itl_stddev_ms`.
- **README**: rewritten from 716 to ~320 lines — star-attracting focus with quick install,
  GPU in 2 steps, backends table, Web UI features, CLI examples, benchmark results, and
  configuration example; verbose design rationale and roadmap moved to docs/.

### Fixed

- **Compare action bar position**: the compare bar (Table / Chart / Trend / Pareto / CSV / ✕)
  now appears above the run list so users no longer need to scroll to the bottom to access
  comparison actions.  The bar also gains `position: sticky` so it stays visible while
  scrolling through a long run list.

- **llama-cpp CUDA capability detection**: `llama_supports_gpu_offload()` was deprecated in
  llama_cpp ≥0.3.x and always returned `False` even when the library was compiled with CUDA,
  causing the Web UI to incorrectly show a "GPU unavailable" warning on CUDA-capable builds.
  The detection now probes `ggml_backend_cuda_get_device_count` (a symbol present only in
  CUDA builds) with a fallback to the legacy function for older installs.  The warning message
  in the Web UI has also been updated to recommend `make install-llama-cpp-prebuilt` as the
  simplest fix.

### Added

- **Leaderboard panel in the Web UI**: a new **Leaderboard** button in the sidebar toolbar
  (between **Runs** and **Datasets**) loads an at-a-glance panel in the main area showing the
  single best run for each key metric (Throughput, p50/p95 Latency, TTFT, VRAM, Energy).
  Each row shows the metric name, best recorded value with units, a linked short run ID, model
  name, backend, and run date.  Only completed (`status = done`) runs with a non-null value
  are ranked.  Metrics with no qualifying data are omitted automatically.  Powered by the new
  `GET /api/ui/leaderboard` HTMX endpoint; no new CDN dependencies.

- **Metric trend chart in the Web UI**: selecting 2 or more runs and clicking the new
  **Trend** button in the compare bar loads an interactive Plotly line+scatter chart in the
  main panel.  Runs are placed on the X axis in chronological order (labelled with their
  short ID and optional label), and the Y axis shows one numeric metric at a time.  Plotly
  buttons across the top of the chart let you switch between metrics (throughput, latency,
  TTFT, VRAM, energy, efficiency, etc.) without re-requesting data from the server.  Runs
  that have no value for the active metric show a gap in the line.  Plotly is lazy-loaded
  from CDN only on the first Chart or Trend click.

- **Multi-run CSV export from the compare bar**: selecting 2 or more runs now shows a **CSV**
  button in the compare bar.  Clicking it downloads a single `compare_runs.csv` file with one
  row per selected run.  Columns match the per-run CSV (`run_id`, `label`, `backend`, `model`,
  `status`, `created_at`, `finished_at`, plus all numeric metric keys).  Rows are ordered by
  selection sequence.  Powered by the new `GET /api/runs/export.csv?ids=…` endpoint.

- **Bar-chart comparison view in the Web UI**: selecting 2 or more runs and clicking the new
  **Chart** button in the compare bar loads an interactive Plotly grouped bar chart in the
  main panel.  Each metric is a group of bars (one per run), all normalised to 0–100 % so
  metrics with different units can be compared on the same axis (100 % = best in the
  selection).  Hovering a bar shows the actual measured value (e.g. `120.5 tok/s`).  Plotly
  is loaded lazily from CDN only on the first Chart click, so the dashboard page load is
  unaffected.

- **Metric comparison table in the Web UI**: selecting 2 or more runs and clicking the new
  **Table** button in the compare bar loads a side-by-side metric table in the main panel.
  Each row is a key metric (throughput, latency, TTFT, VRAM, energy, etc.) and each column
  is a selected run.  Non-reference columns show a colour-coded percentage delta (green = better,
  red = worse) relative to the first selected run.  The **Pareto** button (previously labelled
  "Compare") continues to open the interactive Pareto scatter chart in a new tab.

### Fixed

- **GHCR images now publish on every release**: the release workflow lacked a disk-space
  cleanup step, causing the GPU image build (CUDA + llama.cpp compiled from source) to
  exhaust runner disk and fail silently — leaving GHCR stuck at v1.3.0 despite the v1.4.0
  tag being present. A cleanup step matching `docker.yml` is now applied before any build,
  and a `timeout-minutes: 120` guard prevents silent hangs.

- **Docker release images now push correctly**: the release workflow derives the GHCR image
  path from a lowercased repository name so Docker/OCI tags never contain uppercase letters.
  Previously, tags like `ghcr.io/Happynood/llm-inference-benchmark:cpu-v1.4.0` were rejected
  with *"repository name must be lowercase"*, causing the v1.4.0 image push to fail.

### Added

- **`webui` and `webui-gpu` images published to GHCR on release**: the Web UI Dockerfile
  targets (`webui`, `webui-gpu`) are now built and pushed alongside `cpu` and `gpu` on every
  version tag.  Tags follow the same pattern: `webui-<version>` / `webui-latest` and
  `webui-gpu-<version>` / `webui-gpu-latest`.

- **`HF_TOKEN` forwarded to all Docker Compose services**: setting `HF_TOKEN` in the host
  environment (or in a `.env` file) now passes the token into every Compose service
  (`bench-cpu`, `bench-gpu`, `webui`, `webui-gpu`).  Required for downloading gated
  HuggingFace models inside the container.  The variable is optional — services start
  normally when it is unset.

### Added

- **Sort order for the run list**: the filter bar in the Runs sidebar now includes a sort
  select with three options — *Newest first* (default), *Oldest first*, and *Model A→Z*.
  The sort cooperates with the existing keyword search and status filter, and the
  3-second auto-refresh preserves the active sort order.

- **Run duration in the Web UI**: completed run cards in the sidebar now show the elapsed
  wall-clock time (e.g. `45s`, `1m 23s`) in the meta line alongside throughput and
  backend.  The same duration appears in the run detail panel next to the creation
  timestamp.  Pending and running runs show no duration until they finish.

- **Clone run in the Web UI**: a **Clone** button in the run detail panel opens the
  New Run modal pre-filled with the selected run's model, backend, backend-specific
  parameters (GPU layers, context size, device, etc.), and request/concurrency/warmup
  counts.  Useful for iterating on a configuration without re-entering every field.

- **Run labels in the Web UI**: each benchmark run can now have a short text label (up to
  80 characters) that you set by clicking the label area on a sidebar run card or the run
  detail panel.  Labels appear inline, are saved instantly without a page reload, and are
  included in the sidebar keyword search so you can filter by names like "baseline" or
  "experiment-1".  The CSV export gains a `label` column as well.

- **Run search and filter in the Web UI sidebar**: a compact filter bar above the run list
  lets you narrow results by keyword (model name, backend, or run ID prefix) and by status
  (All / done / error / running / pending).  The periodic 3-second auto-refresh respects
  the active filter so live runs continue to appear while the list is narrowed.

## [1.4.0] - 2026-06-25

### Added

- **VRAM-aware model suggestions in `llm-bench pull`**: when a requested GGUF file
  would exceed local GPU VRAM (estimated as file size × 1.1 for KV-cache overhead),
  `llm-bench pull` now prints a warning and lists alternative quantizations from the
  same repo that do fit.  Suggestions are sorted by descending quality (largest that
  still fits first).  The download proceeds after the warning so CPU-only use is
  unaffected.  When no GPU is detected the check is skipped silently.
  New public helper: `suggest_fitting_variants(repo_id, vram_gb)` in
  `llm_inference_benchmark.puller`.

- **Docker `webui-gpu` service**: new `webui-gpu` build target in `Dockerfile` (based on
  `nvidia/cuda:12.6.0-devel-ubuntu22.04`) and matching service in `docker-compose.yml` —
  run `docker compose up webui-gpu` to get the full Web UI dashboard with GPU-accelerated
  llama-cpp inference on port 8080.  Both the `server` and `llama-cpp` (CUDA) extras are
  bundled into the image so benchmarks run on-GPU without any additional setup.
  The service shares the same volumes as `webui` (named `llm-bench-data`, read-only `~/models`
  mount) and exposes `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.

- **Docker `webui` service**: new `webui` build target in `Dockerfile` and matching
  `webui` service in `docker-compose.yml` — run `docker compose up webui` to start the
  `llm-bench serve` dashboard on port 8080 without any local Python installation.  Run
  results persist in a named Docker volume (`llm-bench-data`); GGUF models from `~/models/`
  on the host are mounted read-only at `/models` inside the container.

- **Full local validation checklist**: `docs/quickstart.md` now includes a step-by-step
  sequence (unit tests → `verify` → CLI smoke → Web UI health check → optional backends)
  to validate the entire stack end-to-end before publishing a release.

- **Interactive Pareto chart axis selectors**: the `/runs/{id}/pareto.html` page now lets
  you pick any two metrics for the X and Y axes via dropdown menus — e.g. VRAM vs
  throughput, TTFT vs latency, or efficiency vs p95 latency.  The Pareto front is
  recomputed in the browser whenever the selection changes.  A **Download PNG** button
  exports the chart at 1200×700 px for reports and README screenshots.

- **`llm-bench pull`**: download GGUF or Transformers models from HuggingFace Hub in one
  command.  For GGUF, pass `--quant` to select the quantization variant and the file lands
  in `~/models/` (override with `--dest`).  For Transformers, omit `--quant` and the model
  snapshot is saved to the default HF cache.  Features: pre-download size guard
  (`--max-size-gb`, default 10), post-download SHA-256 verification against LFS metadata
  (corrupted downloads are deleted automatically), skip-if-already-cached, and
  `HF_TOKEN` / `--token` support for gated models.

  ```bash
  llm-bench pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --quant Q4_K_M
  llm-bench pull HuggingFaceTB/SmolLM2-360M-Instruct --backend transformers
  llm-bench pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF --quant Q5_K_M --max-size-gb 5
  ```

- **Open-loop load mode** (`--arrival-rate RPS`): dispatch requests at a constant
  arrival rate (requests/second) regardless of response time.  Unlike semaphore-based
  concurrency, this models Poisson-process traffic patterns and reveals queueing latency
  that closed-loop testing hides.  Set `arrival_rate_rps` in YAML or pass
  `--arrival-rate 5` on the CLI.  When active, `concurrency` is ignored and the CLI
  echo shows `arrival rate : X rps (open-loop)`.

- **E2E tests for multi-run comparison**: four Playwright tests cover the compare bar
  appearing when two run cards are checked, the ✕ clear button hiding the bar, the
  **Compare** button opening the Pareto page in a new tab with the correct `?ids=` query
  string, and checkbox state persisting across the automatic HTMX run-list refresh.

- **`humaneval` dataset**: `llm-bench datasets pull humaneval` downloads all 164 Python
  function-completion problems from `openai/openai_humaneval`.  Each prompt is prefixed with
  a brief instruction so the model knows to complete the function.  Use it as a code-generation
  benchmark that is reproducible across model versions.

- **Download CSV from the dashboard**: completed runs now show a **Download CSV** button
  in the run detail panel.  Clicking it fetches `GET /api/runs/{run_id}/results.csv` and
  downloads a two-row CSV (header + data) containing `run_id`, `backend`, `model`,
  `status`, timestamps, and all numeric metrics.  Missing metrics appear as empty strings.
  The endpoint returns `404` for unknown runs and `409` for runs that have not finished.

- **Multi-run comparison from the dashboard**: each run card in the sidebar now has a
  checkbox.  Checking ≥ 2 cards reveals a **Compare N runs** button in the sidebar footer
  that opens the interactive Pareto scatter page (`/runs/pareto?ids=…`) in a new tab for
  a side-by-side throughput-vs-latency view.  Checkbox state is preserved across the
  automatic 3-second run-list refresh so selections survive polling.

- **`llm-bench sweep` command**: ramp `concurrency` across a comma-separated range and
  emit a throughput-vs-latency curve in a single CSV.  Each level produces a row with
  `concurrency`, `throughput_rps`, `p50_latency_ms`, `p95_latency_ms`, `tokens_per_second`,
  and the full per-run metric columns.  `--max-p95-ms` stops the sweep early (exit 1) when
  p95 latency exceeds the threshold.  A knee-point summary table is printed on completion.

      llm-bench sweep --config configs/example.yaml --concurrency-range 1,2,4,8
      llm-bench sweep --config configs/example.yaml --concurrency-range 1,2,4,8 --max-p95-ms 5000

- **`GET /api/capabilities` endpoint**: reports runtime backend capability flags.
  Currently exposes `llama_cpp_gpu: bool`, reflecting whether the installed
  `llama-cpp-python` wheel was compiled with CUDA support.

- **Live dashboard with SSE log streaming**: the dashboard now uses a two-panel layout
  (sidebar run cards + main detail panel).  While a benchmark is running, log lines stream
  live into the detail panel via Server-Sent Events; metrics and hardware info appear
  automatically when the run finishes — no page reload needed.

- **GPU configuration in the UI**: the New Run modal now renders backend-specific fields
  dynamically.  llama-cpp exposes **GPU Layers** (default `-1` = all layers on GPU) and
  **Context Size**; transformers exposes a **Device** dropdown (`cpu` / `cuda` / `mps`)
  with an inline hint, and a **Precision** selector (`float32` / `float16` / `bfloat16`);
  vLLM and ONNX similarly expose their GPU knobs — all with defaults that favour GPU use.

- **Modular UI file layout**: the dashboard HTML, CSS, and JavaScript are now separate
  files (`ui/templates/dashboard.html`, `ui/static/app.css`, `ui/static/app.js`) instead
  of a single ~400-line embedded string in `server.py`.  The JS and CSS are lintable,
  formattable, and independently editable.

- **Web UI run deletion**: each row in the dashboard now has a **Delete** button.
  Clicking it confirms with the user and calls `DELETE /api/runs/{run_id}`, then
  refreshes the table.  Deleting a run that is still `pending` or `running` returns
  `409 Conflict` and shows an alert instead.  Completed and errored runs are
  removed from the database immediately.

- **`all-backends` extra**: install all local inference backends in one command —
  `uv sync --extra all-backends` — instead of juggling multiple `--extra` flags.
  Resolves an apparent conflict where `uv sync --extra llama-cpp` would remove ONNX
  packages installed by a prior `uv sync --extra onnx` (and vice-versa).

### Changed

- **`llm-bench verify` SKIP notes now include the install command** for each missing
  backend, e.g. `missing: llama_cpp — install: uv sync --extra llama-cpp`, so the fix
  is visible directly in the output.

### Fixed

- **`/api/capabilities` CUDA preload**: on systems where `llama-cpp-python` was installed
  from a prebuilt CUDA wheel (libs bundled under `site-packages/nvidia/*/lib/`), the
  capabilities endpoint was importing `llama_cpp` directly without calling
  `_preload_nvidia_cuda_libs()`, so `llama_supports_gpu_offload()` silently returned
  `False` and the Web UI always showed the GPU warning even when GPU inference was
  functional.  The endpoint now calls `_preload_nvidia_cuda_libs()` first, matching the
  behaviour of `backends/llama_cpp.py`.

- **Docker documentation**: corrected references to the non-existent `Dockerfile.cuda` and
  `llm-bench-cuda` service — the repository uses a single multi-stage `Dockerfile` with
  `--target cpu` / `--target gpu` targets and `bench-cpu` / `bench-gpu` compose service
  names.

- **Web UI llama-cpp GPU**: `README.md` and `docs/quickstart.md` now explain that the GPU
  warning banner shown when `llama-cpp` is selected is expected on CPU-only installs, and
  document the two commands to enable GPU acceleration (`make install-llama-cpp-prebuilt`
  or `make install-llama-cpp-cuda`).

- **llama-cpp GPU offload warning in the UI**: when the user selects the
  **llama-cpp** backend in the New Run modal, the UI now fetches `/api/capabilities`
  and — if GPU offload is unavailable — displays a visible warning explaining that
  the model will run on CPU regardless of the GPU Layers value, along with the
  commands to reinstall with CUDA support.  Previously the benchmark silently ran
  on CPU with no indication in the UI, even when GPU Layers was set to `-1`.

- **Transformers backend no longer errors on HuggingFace models**: the "New Run" modal
  was passing the local cache directory path (e.g.
  `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B`) to `from_pretrained()` instead of
  the model name (`Qwen/Qwen3-0.6B`), causing `ValueError: Unrecognized model`.  The
  model dropdown now sends the canonical `org/name` identifier.

- **Partially-downloaded HF models no longer appear in the dropdown**: models that only
  have a `refs/` directory in the HuggingFace cache (no populated `snapshots/` directory)
  are now silently skipped by `_discover_models()`.  Previously they appeared as selectable
  options but always failed to load when the benchmark started.

- **`swe-bench-pro` dataset pull no longer fails**: the registry entry used split
  `"train"`, but `ScaleAI/SWE-bench_Pro` only provides a `"test"` split.  Fixed.

- **ONNX example config now works without authentication**: `configs/onnx-example.yaml`
  previously used `optimum-internal-testing/tiny-random-GPT2Model`, a private model that
  returns 401 Unauthorized for most users on first run.  Changed `model:` to `gpt2`
  (public, no token required).

- **Output directory is created automatically**: running `llm-bench --output results/bench.csv`
  no longer fails with `FileNotFoundError` when the parent directory (`results/`) does not
  exist.  All commands that accept `--output` or `--manifest` now call
  `mkdir(parents=True, exist_ok=True)` before writing, matching the existing behaviour of
  `matrix` and `pipeline`.  Affected commands: main benchmark, `compare`, `pareto`,
  `recommend`, `diff`.

### Added

- **`--base-url` and `--api-key` CLI flags**: test any OpenAI-compatible server (Ollama,
  LM Studio, llama.cpp server, vLLM, OpenAI) without writing a YAML config file.
  `llm-bench --base-url http://localhost:11434/v1 --set model=llama3.2:3b` runs a full
  benchmark against a local endpoint; `--api-key` passes the literal key value as a Bearer
  token and takes precedence over any `openai.api_key_env` in the config.  When combined
  with `--config`, `--base-url` overrides `openai.base_url` and switches the backend to
  `openai`.  When neither `--config` nor `--dataset` is provided, a built-in set of ten
  default prompts is used so no local file path is required.

- **GPU backend requirements documented**: the `## GPU Setup` section in `README.md` and
  `docs/quickstart.md` now includes dedicated sub-sections for the **vLLM** and **ONNX
  Runtime** backends.  Key points documented: vLLM is Linux-only and has no CPU fallback;
  the default `--extra onnx` install uses the CPU build of ONNX Runtime and GPU
  acceleration requires replacing it with `onnxruntime-gpu` plus setting `device: cuda`
  in the config; neither backend requires `nvcc` or the CUDA toolkit.

- **Web UI model dropdown shortened labels**: the **Model** dropdown in the "New Run" modal
  now shows a human-readable short name instead of the raw model path.  The org prefix is
  stripped for HuggingFace models (e.g. `meta-llama/Llama-3.2-1B-Instruct` →
  `Llama-3.2-1B-Instruct`), and a backend tag is appended in parentheses
  (`(llama.cpp)` for GGUF, `(transformers)` for HF).  The full path is preserved as the
  submitted `<option value>` and is shown as a tooltip via the `title` attribute.

- **Web UI dataset selector**: the "New Run" modal now includes a **Dataset** dropdown
  populated with all currently-cached datasets (name + sample count).  Selecting a dataset
  routes benchmark prompts through the corresponding real-world prompt set instead of the
  config's synthetic prompts file.  A "Default prompts" option (always first) preserves
  the existing behaviour when no dataset is selected.  The backend `POST /api/runs` accepts
  an optional `dataset` field and passes `--dataset <name>` to the `llm-bench` subprocess.

- **ITL jitter** (`itl_stddev_ms`): new metric that measures the standard deviation of
  inter-chunk latency (ms) pooled across all streaming requests in a run.  A high value
  indicates bursty token delivery, which degrades interactive UX even when average
  throughput is acceptable.  Populated for `openai_endpoint` runs with `stream: true`;
  `None` for non-streaming backends or responses that arrive in a single chunk.  Appears
  as `itl_stddev_ms` in CSV and JSON output and as the "ITL σ (ms)" column in the
  Markdown comparison table (column is suppressed when every row is N/A).

- **Thermal throttling index** (`thermal_throttle_pct`): new metric that measures the
  percentage drop in tokens/s between the first and last 25 % of a sequential benchmark
  run.  A positive value indicates CPU/GPU frequency scaling during the run.  Populated
  automatically for sequential runs with ≥ 8 requests and ≥ 10 s elapsed time; `None`
  otherwise (concurrent runs, short runs).  Appears in CSV output, JSON output, and the
  Markdown comparison table under the "Throttle %" column.

### Fixed

- **Per-run Pareto links restored in the runs table**: each row in the dashboard now shows
  a **Pareto** link that navigates to `/runs/{id}/pareto.html`.  The toolbar Pareto button
  (for multi-run comparison) introduced in #215 is unchanged.
- **Dashboard comparison chart is now readable**: replaced the catch-all bar chart with a
  focused dual-axis chart (tokens/s on the left axis, TTFT p50 ms on the right axis).
  A **Pareto** button appears in the compare toolbar only when ≥2 runs are selected.  Clicking it opens `GET /runs/pareto?ids=…`
  which renders the Pareto scatter for exactly those runs.
- **GPU acceleration now enabled by default for llama-cpp**: `n_gpu_layers` default
  changed from `0` (CPU only) to `-1` (offload all layers; llama.cpp auto-detects GPU
  and falls back to CPU gracefully when none is present).  `n_ctx` default raised from
  `2048` to `4096` to match typical modern model requirements without warnings.
  See the new **GPU Setup** section in README for CUDA wheel installation options.
- **`long-context-*` datasets work again**: `deepmind/pg19` uses a legacy loading script
  blocked in `datasets>=2.20`.  All three `long-context-*` registry entries now use
  `allenai/c4` (Common Crawl, English subset, public domain, no gating).  The extractor
  logic is unchanged — the `"text"` field is identical in both datasets.
- **Gated datasets (e.g. `lmsys-chat`) can now be pulled via the dashboard**: the server
  reads `HF_TOKEN` from the environment and forwards it to the HuggingFace `datasets` library.
  Previously the token was never passed, causing every gated-dataset pull to fail with a 403.
- **Dataset pull errors now surfaced in the dashboard**: when a background dataset pull
  fails (network error, disk full, etc.), the error message is stored per-dataset and
  displayed inline in the Datasets table as "Pull failed: <reason>".  Previously the
  exception was silently discarded and the table showed no indication of failure.
  A successful retry clears the error.
- **Checkbox state preserved across HTMX auto-refresh**: run row checkboxes in the
  `llm-bench serve` dashboard were silently unchecked every 5 seconds when HTMX replaced
  the table body.  The UI now saves the set of checked run IDs before each swap
  (`htmx:beforeSwap`) and restores them after settle (`htmx:afterSettle`).  The
  select-all checkbox updates to checked / indeterminate / unchecked to match.
- **Web UI benchmark runs now produce output**: `POST /api/runs` previously launched
  `python -m llm_inference_benchmark.cli` which exits silently because the CLI module has
  no `__main__` guard.  The subprocess now invokes the installed `llm-bench` console-script
  entry point directly, so benchmark output streams correctly to the run log.
- E2E tests (`tests/e2e/test_ui.py`) now show **SKIPPED** instead of ERROR when Chromium
  is not installed.  The module-level `pytestmark` calls `_chromium_available()` which
  checks whether the Playwright-managed Chromium binary exists on disk.  Install the
  browser with `playwright install chromium` to re-enable the suite.
- E2E CI and local `make test-e2e` now pass on Ubuntu 26.04: the `e2e` GitHub Actions
  job and the new `make install-playwright` target both set
  `PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64` so that Playwright downloads the
  Ubuntu 24.04 Chromium binary (binary-compatible; no native 26.04 build exists yet).

### Added
- **Three new benchmark datasets**: `gsm8k` (grade-school math, `openai/gsm8k`),
  `mmlu-pro` (professional knowledge, `TIGER-Lab/MMLU-Pro`), and `swe-bench-pro`
  (real-world software issues, `ScaleAI/SWE-bench_Pro`).  All are public, ungated, and
  available via `llm-bench datasets pull <name>`.
- **Datasets panel in Web UI**: a new section in the `llm-bench serve` dashboard lists all
  registered datasets with their cached status and sample count.  A `<select>` + **Pull**
  button triggers a background download that updates automatically via HTMX.
  New REST endpoints:
  - `GET  /api/datasets` — list all REGISTRY entries with `cached` and `samples` fields
  - `POST /api/datasets/pull` — `{"name": "<dataset>"}` starts a background pull; returns
    `{"status": "started"}`
  - `GET  /api/ui/datasets-table` — HTMX HTML fragment for the datasets table
- `make install-playwright` — installs the Playwright Chromium browser with the Ubuntu 24.04
  platform override pre-applied.  Required on Ubuntu 26.04; harmless on other platforms.
- `make test-e2e` — runs the E2E Playwright test suite (`tests/e2e/`).
- README **Running E2E tests** section with platform-specific install instructions.

### Added (prior)
- Tokens-per-joule energy efficiency metric.  During the benchmark window (warmup excluded),
  energy is sampled from `nvidia-smi power.draw` (GPU, polled every 500 ms) with Intel RAPL
  `/sys/class/powercap/intel-rapl:0/energy_uj` as the CPU fallback.  Two new columns appear
  in result CSV and JSON: `energy_joules` and `tokens_per_joule`.  Both are blank when
  neither energy source is readable.  `llm-bench compare` renders them in the table,
  suppressed when all rows show N/A.  Older CSVs without these columns load without errors.

- Reasoning token parser for thinking-model outputs (`<think>…</think>` or custom tags).
  Set `reasoning_start_tag` and `reasoning_end_tag` in the benchmark config to split each
  completion into a reasoning portion and a final-answer portion.  Three new columns appear
  in result CSV and JSON output: `mean_reasoning_tokens`, `mean_answer_tokens`, and
  `reasoning_fraction` (fraction of output tokens estimated as reasoning).  Token counts
  are estimated from char-length fractions so no tokenizer dependency is added.
  `llm-bench compare` renders the new columns in the Markdown table; columns are suppressed
  when all runs show N/A (consistent with other optional columns).  Older CSVs without
  these columns load without errors.

- Hardware profile auto-detect: six new `hw_*` columns (`hw_cpu`, `hw_cpu_cores`,
  `hw_ram_gb`, `hw_gpu`, `hw_vram_gb`, `hw_os`) are embedded in every result CSV row
  and JSON object.  Detection uses `psutil` (CPU cores, RAM) and a single `nvidia-smi`
  call (GPU name, VRAM); all fields fall back gracefully on CPU-only machines or when
  detection fails.  Older CSVs without these columns load without errors.

- Three long-context dataset variants backed by `deepmind/pg19` (public-domain books):
  `long-context-4k` (≤100 samples, ~4 096-token passages), `long-context-16k` (≤50 samples,
  ~16 384-token passages), and `long-context-64k` (≤10 samples, ~65 536-token passages).
  Use them with the existing `llm-bench datasets pull` and `--dataset` flags to drive
  prefill-latency profiling at controlled context lengths:
  ```
  llm-bench datasets pull long-context-4k
  llm-bench --config cfg.yaml --dataset long-context-4k --requests 20
  ```
  Each passage is sliced from a PG19 book (front matter skipped), wrapped in a
  summarisation prompt, and cached as JSONL.  Books shorter than half the target length
  are skipped so every cached sample reaches the declared token budget.

- Web UI "+ New Run" button and modal form: select model (populated via `GET /api/models`),
  backend, requests, concurrency, warmup requests, GPU layers (llama-cpp only), and extra
  YAML config.  Submits `POST /api/runs`, auto-refreshes the runs table, and starts
  streaming the new run's log.  No external JS libraries beyond HTMX + Plotly (already
  loaded).

- `p50_tpot_ms` and `tpot_stddev_ms` fields in `MetricsReport`, CSV, JSON, and the
  `llm-bench compare` table.  TPOT (time-per-output-token) is the decode-phase latency:
  `(request_latency_ms - ttft_ms) / output_tokens`.  Both fields are `None` when TTFT
  data is unavailable; `tpot_stddev_ms` is `None` for single-request runs.  The compare
  table suppresses the column automatically when all rows are N/A (backward-compatible
  with older CSVs).

- `llm-bench datasets pull <name>`: download and cache real-world prompt samples from
  HuggingFace to `~/.cache/llm-bench/datasets/<name>.jsonl` (streaming, no full-dataset
  RAM load).  Supported datasets: `lmsys-chat` (up to 500 first-user-turn samples from
  `lmsys/lmsys-chat-1m`) and `hermes-fn` (up to 200 function-calling prompts from
  `NousResearch/hermes-function-calling-v1`).  Requires `pip install datasets`.

- `llm-bench datasets list`: print locally cached dataset names and sample counts.

- `llm-bench --dataset <name>`: use a cached dataset as the prompt source for a benchmark
  run instead of the config `prompts_file`.  Sampling is reproducible via `--seed`.

- `wildchat` dataset: `allenai/WildChat-1M` is now registered as a public real-world chat
  dataset (no gating, no Terms of Use required).  Pull and use it like any other dataset:
  ```
  llm-bench datasets pull wildchat
  llm-bench --config cfg.yaml --dataset wildchat --requests 50
  ```
  The extractor picks the first user turn from each conversation.

- `datasets>=2.0` is now a **default dependency** installed by `uv sync` without any extras.
  Users no longer need to run `uv pip install datasets` or pass `[datasets]` to install it.
  The `[datasets]` optional-extras section has been removed from `pyproject.toml`.

- `llm-bench datasets pull lmsys-chat` now prints a clear error message with a link to the
  Terms of Use page when the HuggingFace API returns an access-denied response, rather than
  surfacing the raw HTTP exception.

- `llm-bench serve [--host HOST] [--port PORT]`: start a FastAPI server with a built-in
  HTMX + Plotly dashboard.  Opening `http://localhost:8080` in a browser shows a live
  runs table (auto-refreshed every 5 s via HTMX), per-run SSE log streaming, a bar-chart
  comparison for selected runs, and a Pareto scatter page (p95 latency vs throughput)
  for each run.  REST endpoints: `GET /api/health`, `GET /api/models` (GGUF files and
  HuggingFace cache dirs), `GET /api/runs`, `POST /api/runs` (submit a config; returns
  `run_id` immediately), `GET /api/runs/{run_id}` (poll status/results),
  `GET /api/runs/{run_id}/stream` (Server-Sent Events), `GET /api/ui/runs-table`
  (HTMX HTML fragment), `GET /runs/{run_id}/pareto.html` (interactive Plotly page).
  Results are persisted in `~/.llm-bench/results.db`.  Requires the `server` extra:
  `uv pip install 'llm-inference-benchmark[server]'`.

- New `server` optional extra: `fastapi[standard]>=0.111`, `uvicorn[standard]>=0.30`,
  `huggingface-hub>=0.23`.

- Playwright E2E test suite (`tests/e2e/test_ui.py`) covering dashboard load, runs table
  population, status badges, live-log panel, compare-chart rendering, and Pareto page.

- `llm-bench recommend --filter FIELD=PATTERN`: narrow the candidate pool before
  constraint evaluation and Pareto selection.  Supported fields: `backend`, `model`
  (case-insensitive substring match; repeatable; multiple filters are ANDed).  Composes
  with all constraint flags and `--format`:
  ```
  llm-bench recommend results/*.csv --filter backend=llama_cpp --max-p95-ms 1000
  llm-bench recommend results/*.csv --filter backend=llama_cpp --filter model=Q4_K_M
  llm-bench recommend results/*.csv --filter model=Q4 --format json
  ```

- `llm-bench pareto --filter FIELD=PATTERN`: narrow the Pareto candidate pool before
  running dominance analysis.  Supported fields: `backend`, `model` (case-insensitive
  substring match; repeatable; multiple filters are ANDed).  Composes with `--format`
  and `--output`:
  ```
  llm-bench pareto results/*.csv --filter backend=llama_cpp
  llm-bench pareto results/*.csv --filter backend=llama_cpp --filter model=Q4_K_M
  llm-bench pareto results/*.csv --filter model=Q4 --format csv --output pareto.csv
  ```

- `llm-bench diff --format csv`: export the per-metric regression diff as a CSV file.
  Columns: `metric`, `baseline`, `current`, `change_pct`, `direction`.  Absent optional
  metrics are omitted; absent individual values and uncomputable `change_pct` are written
  as empty cells so `pandas.read_csv()` produces `NaN` automatically.  Composes with
  `--output` and `--fail-on-regression`:
  ```
  llm-bench diff baseline.csv current.csv --format csv
  llm-bench diff baseline.csv current.csv --format csv --output delta.csv
  df = pd.read_csv("delta.csv"); regressions = df[df["direction"] == "regression"]
  ```

- `llm-bench pareto --format csv`: export the Pareto classification table as a CSV file.
  Includes all `compare --format csv` columns plus a `pareto` column (`True`/`False`) so
  users can filter Pareto-optimal rows in pandas without losing dominated rows.  Absent
  optional metrics are written as empty cells so `pandas.read_csv()` produces `NaN`
  automatically.  Composes with `--output`:
  ```
  llm-bench pareto results/*.csv --format csv
  llm-bench pareto results/*.csv --format csv --output pareto.csv
  df = pd.read_csv("pareto.csv"); optimal = df[df["pareto"]]
  ```

- `llm-bench compare --format csv`: export the comparison table as a CSV file.
  Header row uses snake_case field names identical to JSON keys; absent optional
  metrics are written as empty cells (not `"N/A"`) so `pandas.read_csv()` produces
  `NaN` automatically. Composes naturally with `--sort`, `--limit`, `--filter`,
  and `--output`:
  ```
  llm-bench compare results/*.csv --format csv
  llm-bench compare results/*.csv --format csv --output summary.csv
  llm-bench compare results/*.csv --sort toks --limit 5 --format csv
  llm-bench compare results/*.csv --filter backend=llama_cpp --format csv
  ```

- `llm-bench pipeline --config <file>`: run a full benchmark study from a single YAML config.
  Executes all matrix cells in sequence, then writes compare, Pareto, and recommendation
  outputs to `results_dir/`. Supports `--dry-run` to preview the plan without executing,
  `--continue-on-error` to keep running after a cell failure, and `--format table|json`
  for terminal progress. The `pipeline:` block is optional — a plain matrix YAML is valid
  pipeline input. Example:
  ```
  llm-bench pipeline --config configs/pipeline-example.yaml
  llm-bench pipeline --config configs/pipeline-example.yaml --dry-run
  llm-bench pipeline --config configs/pipeline-example.yaml --continue-on-error
  ```
  Output files written to `results_dir/`: `compare.md`, `compare.json`; `pareto.md`,
  `pareto.json` (when `pipeline.pareto: true`); `recommend.md`, `recommend.json` (when
  `pipeline.recommend:` block is present). Exits 1 when any cell failed or when a
  `recommend` block finds no winner. Calls library functions directly — no subprocess
  calls to other `llm-bench` subcommands. A `configs/pipeline-example.yaml` ships with
  the repo and uses the mock backend (no model download required).

- `llm-bench compare --filter FIELD=PATTERN`: filter comparison rows before sorting and
  `--limit`. Pattern is a case-insensitive substring match. Supported fields: `backend`,
  `model`. The flag is repeatable; multiple filters are ANDed:
  ```
  llm-bench compare results/*.csv --filter backend=llama_cpp
  llm-bench compare results/*.csv --filter model=Llama-3.2 --sort toks
  llm-bench compare results/*.csv --filter backend=llama_cpp --filter model=Q4_K_M --limit 3
  llm-bench compare results/*.csv --filter backend=llama_cpp --format json
  ```
  An unsupported field name produces a clear `UsageError` listing valid fields. When no rows
  survive filtering, an empty table or empty JSON array is returned (not an error). Composes
  naturally with `--sort`, `--limit`, and `--format`.


- `llm-bench compare --limit N`: cap the output table or JSON array to the top N rows after
  sorting. Composes naturally with `--sort` to get a focused view of the best runs:
  ```
  llm-bench compare results/*.csv --sort toks --limit 5
  llm-bench compare results/*.csv --sort p95 --limit 3 --format json
  ```
  Values below 1 are rejected with a usage error. When N exceeds the number of available
  rows all rows are returned unchanged.

- `--set KEY=VALUE` flag on the main `llm-bench` run command and on `llm-bench validate-config`:
  overrides any config field via dot-path without editing the YAML. Values are parsed as YAML
  scalars (so `200` becomes `int`, `3.14` becomes `float`, `true` becomes `bool`, `cuda`
  stays `str`). The flag is repeatable:
  ```
  llm-bench --config my.yaml --set llama_cpp.max_tokens=200 --set llama_cpp.n_gpu_layers=20
  llm-bench validate-config --config my.yaml --set hf.max_new_tokens=256
  ```
  Unknown paths and type mismatches produce a clear `UsageError` with the list of valid
  fields. Named flags (`--requests`, `--seed`, etc.) take precedence over `--set` when both
  target the same field. Reuses `apply_overrides` from the existing `sweep`/`matrix`
  subsystem.

## [1.3.0] — 2026-06-19

### Added
- `llm-bench matrix --format json`: machine-readable JSON output for the matrix command.
  `--dry-run --format json` emits a single JSON object listing all planned runs with their
  index, name, config, overrides, workload profile, and expected output/manifest paths —
  no files are created. `--format json` (actual run) routes all progress messages to stderr
  so stdout remains a single parseable JSON object; always collects outcomes from all runs
  (continues past failures); emits a summary with `total`/`succeeded`/`failed` counts and
  per-run `status`, `output`, and `error` fields; exits 1 when any run failed. Makes
  `matrix` consistent with every other subcommand that supports `--format json`.

### Changed
- `llama-cpp` backend now pre-loads CUDA shared libraries bundled by `nvidia-*` pip packages
  at Python import time using `ctypes.CDLL(..., mode=RTLD_GLOBAL)`. This makes GPU runs
  work on driver-only systems (no CUDA toolkit / nvcc) without requiring `LD_LIBRARY_PATH`
  to be set manually. The preload is a silent no-op on CPU-only machines and on systems
  without `nvidia-*` packages installed.

## [1.2.0] — 2026-06-18

### Added
- `llm-bench env`: prints the current Python, package, and hardware environment without
  requiring a benchmark config or a full run. Reports Python version, OS/platform, CPU model
  and core count, `llm-inference-benchmark` version, and optional package versions
  (`torch`, `transformers`, `optimum`, `vllm`, `psutil`) when installed. GPU section shows
  device name, driver version, CUDA version, and total VRAM from `nvidia-smi`; reports
  "not detected" on CPU-only machines. Supports `--format json` for machine-readable output
  and CI integration.
- `llm-bench --seed N` CLI flag: overrides the `seed:` field in the YAML config without
  editing the file. Follows the same pattern as `--requests`, `--warmup-requests`, and
  `--concurrency`. The seed value appears in the human-readable run header when set
  (`Seed: N`). Has no effect when the backend uses greedy decoding (`temperature: 0.0`).
- `llm-bench recommend --format json`: machine-readable JSON output from the recommend command.
  Returns an object with `winner` (all RunRow fields; `null` when no run satisfies constraints),
  `is_pareto_optimal` (bool), `candidates_count` (int), and `excluded` (array of objects with
  `backend`, `model`, and `reason`). Exit code behaviour (0 = winner found, 1 = no winner) is
  unchanged. Compatible with `--output`; useful for CI pipelines and scripting without
  screen-scraping the human-readable text output. The default format remains `table`.
- `seed` field in `BenchmarkConfig`: optional random seed for stochastic sampling.
  Applied per-request via `torch.manual_seed` in the `transformers` backend and at model
  load via `Llama(seed=...)` in the `llama-cpp` backend. Forwarded as a best-effort hint
  in the `openai` backend (server support varies). Recorded in `RunManifest` to complete
  the reproducibility fingerprint. Has no effect on greedy decoding (`temperature: 0.0`).
- `llm-bench compare --format json`: machine-readable JSON output from the compare command.
  Returns a JSON array of objects — one per CSV — with all fields present (`null` for absent
  optional metrics). Compatible with `--sort` and `--output`; useful for CI dashboards and
  scripting without screen-scraping Markdown. The default format remains `table` (Markdown).
- `vllm` backend (`backend: vllm`): high-throughput GPU inference via the vLLM engine
  (`vllm>=0.4`). Install with `uv sync --extra vllm`. Supports `max_new_tokens`,
  `temperature`, `tensor_parallel_size`, `gpu_memory_utilization`, and `dtype` under
  `vllm:` in the config.

### Changed
- `llm-bench compare` now suppresses optional columns (Out tok/s, In/Out tok, Load, TTFT,
  CUDA/VRAM mem, Sanity, Task Q, PPL, Judge) when every row in the table shows N/A for that
  column. Mandatory columns (Backend, Model, N, p50, p95, tok/s, CPU mem) are always shown.
  Mixed tables (some rows with data, some without) still show the column with N/A for the
  rows that lack data.
- `llm-bench diff --format json`: machine-readable JSON output from the diff command.
  Returns an object with `baseline`/`current` run metadata and a `metrics` list — one
  entry per tracked metric — each with `label`, `baseline`, `current`, `change_pct`
  (null when not computable), and `direction` (`"improvement"`, `"regression"`,
  `"neutral"`, or `"n/a"`). Optional metrics absent from both runs are omitted,
  consistent with Markdown table output. Compatible with `--output` and
  `--fail-on-regression`; useful for CI dashboards and scripting. The default format
  remains `table` (Markdown).
- `llm-bench diff baseline.csv current.csv`: per-metric percentage-change comparison between
  two benchmark runs. Shows p50/p95 latency, tok/s, TTFT, VRAM, load time, and quality
  metrics side-by-side with ✓ (improvement) / ✗ (regression) annotations. Optional metrics
  are suppressed when absent from both runs. Accepts `--output` to write Markdown to a file.
- `llm-bench diff --fail-on-regression PCT`: exit code 1 when any tracked metric degrades
  beyond the given percentage threshold. Use `0` to fail on any regression; use a positive
  value (e.g. `5`) to tolerate noise. The diff table is always printed before the exit code
  check, so CI logs retain full regression detail. Enables `llm-bench diff` as a CI gate
  without any external tooling.

## [1.1.0] — 2026-06-17

### Added
- `--max-ttft-ms` constraint for `llm-bench recommend`: exclude runs where TTFT p50 exceeds the
  threshold. Runs without a TTFT reading are excluded when the constraint is active (requires
  `stream: true` backend run). Consistent with the `--max-load-ms` pattern added in v0.23.
- TTFT (time-to-first-token) measurement for the `llama-cpp` backend: set `stream: true`
  under `llama_cpp:` in the config to enable streaming mode. `p50_ttft_ms` and `p95_ttft_ms`
  are populated in CSV output, matching the existing `transformers` and `openai` backend
  behaviour. Input tokens are counted via the model's tokenizer; output tokens are counted
  as non-empty streaming chunks (each chunk = one decoded token). Blocking mode (`stream:
  false`, the default) is unchanged.
- TTFT (time-to-first-token) measurement for the `transformers` backend: `p50_ttft_ms` and
  `p95_ttft_ms` are now populated for HuggingFace runs alongside the existing OpenAI-endpoint
  metrics. Implemented via a `LogitsProcessor` hook — no threading, token counts unchanged.
- Docker image (mock + transformers CPU) published to GitHub Container Registry on each
  release tag; `Dockerfile` uses a two-layer uv install for optimal cache reuse.
- `.github/workflows/release.yml` — creates a GitHub Release and pushes versioned
  Docker tags (`v1.1`, `v1.1.0`, `latest`) to `ghcr.io` on every `v*.*.*` tag push.
- `llm-bench --requests N`, `--warmup-requests N`, `--concurrency N` flags let callers
  override the corresponding YAML config values from the command line without editing the
  file. Useful for quick one-off experiments, especially with the concurrent-execution
  backend (`--concurrency 4`). CLI values take precedence over YAML; all existing
  behaviour when the flags are absent is unchanged.

## [1.0.0] — 2026-06-17

### Added
- `llm-bench --version` prints the installed package version and exits 0.

### Changed
- Version bumped to 1.0.0, signalling a stable public API. The CLI commands
  (`llm-bench`, `compare`, `pareto`, `recommend`, `matrix`), YAML config schema,
  CSV output format, and manifest format are now considered stable and will not
  change in an incompatible way without a major version bump.

## [0.23.0] — 2026-06-17

### Added
- `compare` table now shows a `Load (ms)` column (`model_load_ms` from v0.18 runs; `N/A`
  for pre-v0.18 CSVs). When a run used `repeats > 1`, the `p95 (ms)` and `tok/s` cells
  show the median with its standard deviation as `value ± std`.
- `pareto` table now shows a `Load (ms)` column and includes `model_load_ms` as a
  Pareto dimension (lower is better) when both compared runs have a value.
- `recommend` displays `Load` in the winning-configuration summary and accepts a new
  `--max-load-ms` constraint: runs where load time exceeds the threshold (or where load
  time is unknown) are excluded with a clear reason.

## [0.22.0] — 2026-06-17

### Added
- OpenAI-compatible endpoint backend (`backend: openai`): benchmarks any server that
  exposes the `/v1/chat/completions` HTTP API — llama.cpp server, Ollama, LM Studio,
  vLLM, and others. Uses `urllib` (no extra dependency). Token counts from the `usage`
  field when present; word-count fallback otherwise. API key read from a named
  environment variable (`api_key_env`); never written to config, logs, or output.
  Example config at `configs/openai-endpoint.yaml`.

**Limitation:** reported latency includes network round-trip and server-side queueing
overhead and is not directly comparable to in-process backend latency.

## [0.21.0] — 2026-06-16

### Added
- LLM-as-judge quality score: `measure_judge: true` reports the mean P(yes) from a fixed
  self-judge yes/no question asked about each of a backend's own generated completions.
  `transformers` backend only; `None` elsewhere. `Judge` column in `compare`/`pareto` tables;
  `--min-judge` constraint in `llm-bench recommend`.

## [0.20.0] — 2026-06-16

### Added
- Self-perplexity quality metric: `measure_perplexity: true` reports corpus-level perplexity
  of a backend's own generated completions via teacher forcing. `transformers` backend only;
  `None` elsewhere. `PPL` column in `compare`/`pareto` tables; `--max-perplexity` constraint
  in `llm-bench recommend`.

## [0.19.0] — 2026-06-16

### Added
- Repeated-trial variance reporting: `repeats: N` in a config runs the benchmark loop N
  times; reported p95 latency and tokens/sec become the median across repeats.
  `p95_latency_ms_std` and `tokens_per_second_std` CSV columns carry the sample standard
  deviation. Single-run CSVs (`repeats: 1`, the default) are unchanged.

## [0.18.0] — 2026-06-16

### Added
- Lifecycle metrics: model load time (`model_load_ms`) and warmup latency
  (`warmup_p50_latency_ms`), wired into CSV output and the CLI.

## [0.17.0] — 2026-06-15

### Added
- Parameter sweep matrix: `base_config:` + `sweep:` in a matrix YAML expands a cartesian
  product of overrides (including dot-path nested fields such as `llama_cpp.n_gpu_layers`
  or `hf.max_new_tokens`) into deterministic per-combination run names, with a `--dry-run`
  preview.

## [0.16.0] — 2026-06-15

### Added
- Task-quality evaluation: optional `quality_file:` in a config points to a YAML rubric
  spec (`contains_all`, `contains_any`, `forbidden`, `regex`, `min_chars` per prompt).
  `task_quality_pass_rate` / `task_quality_checked_count` in CSV output; `Task Q %` column
  in `compare`/`pareto`; `--min-quality` constraint in `llm-bench recommend`.

## [0.15.0] — 2026-06-14

### Added
- Constraint-based recommender: `llm-bench recommend` filters benchmark CSVs by
  `--max-vram-mb`, `--max-p95-ms`, and `--min-sanity`, returning the Pareto-optimal winner
  or exiting with code 1 and a clear exclusion table.

## [0.14.0] — 2026-06-14

### Added
- Pareto analysis: `llm-bench pareto` classifies configurations across saved CSVs as
  optimal or dominated on p95 latency, throughput, VRAM, and sanity, narrowing the
  comparison rather than crashing when optional metrics are missing.

## [0.13.0] — 2026-06-14

### Added
- Output sanity checks: `empty_output_count`, `min_output_chars`, `mean_output_chars`,
  `repeated_output_count`, and `sanity_pass_rate` computed per run; surfaced as `Sanity %`
  in `compare` tables.

## [0.12.0] — 2026-06-14

### Added
- Quantization comparison evidence: Q4_K_M vs Q8_0 for Llama 3.2 3B Instruct on an RTX 3050
  (llama.cpp) — curated report documenting a 1.31× speed and 1.57× VRAM difference.
- `n_gpu_layers` sweep (0 / 20 / 99) via `llm-bench matrix`, quantifying VRAM and latency
  scaling from CPU-only to full GPU offload.

## [0.11.0] — 2026-06-14

### Added
- First real production-size model benchmark: Llama 3.2 3B Instruct Q4_K_M on an RTX 3050
  Laptop GPU via the llama.cpp backend — curated report in `docs/results/`.

## [0.10.0] — 2026-06-14

### Added
- `llama-cpp` backend: GGUF quantized inference via `llama-cpp-python`, with
  `n_gpu_layers` for partial GPU offload on constrained VRAM budgets (optional extra).

## [0.9.0] — 2026-06-14

### Fixed
- Path-traversal hardening in run/matrix names (`matrix.py` validator, CLI containment check).

### Changed
- Documentation restructured to explicitly separate CI/harness validation (mock backend)
  from real hardware benchmark evidence (`docs/results/` registry).

## [0.8.0] — 2026-06-14

### Added
- Run matrix: `llm-bench matrix` executes multiple experiment configs from one YAML file
  sequentially, writing one CSV and manifest per run.

## [0.7.0] — 2026-06-14

### Added
- Workload profiles: named, reproducible prompt sets (`short_chat`, `summarization`,
  `code_completion`, `long_context_smoke`).
- First GPU benchmark baseline (RTX 3050 Laptop, `sshleifer/tiny-gpt2`).

## [0.6.0] — 2026-06-14

### Added
- Optional NVIDIA GPU fingerprint (`nvidia-smi` + `torch.cuda`) in the run manifest.

## [0.5.0] — 2026-06-14

### Added
- JSON run manifest (`--manifest`): git commit/dirty flag, config and prompts SHA256,
  Python/OS/CPU info, and dependency versions, for full run reproducibility.

## [0.4.0] — 2026-06-14

### Added
- `llm-bench compare`: Markdown comparison table across multiple saved benchmark CSVs.

## [0.3.0] — 2026-06-13

### Added
- Peak memory reporting: CPU RSS via `psutil`, PyTorch CUDA allocator peak via `torch.cuda`.

## [0.2.0] — 2026-06-13

### Added
- `transformers` backend: real CPU/GPU inference via `AutoModelForCausalLM` (optional extra).

## [0.1.0] — 2026-06-13

### Added
- Initial vertical slice: YAML-driven `BenchmarkConfig`, pluggable `Backend` ABC, zero-dependency
  `MockBackend`, `llm-bench` CLI, p50/p95 latency and tokens/sec metrics, GitHub Actions CI.
