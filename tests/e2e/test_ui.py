"""E2E tests for the llm-bench Web UI dashboard.

Run:
    uv run pytest tests/e2e/test_ui.py -v

Prerequisites:
    playwright install chromium
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator

import httpx
import pytest
from playwright.sync_api import Dialog, Page, expect

# ── Server fixture ─────────────────────────────────────────────────────────────


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Start the FastAPI app on a free port for the whole session."""
    import uvicorn

    from llm_inference_benchmark.server import _set_db_path, app

    db_path = tmp_path_factory.mktemp("e2e_db") / "test.db"
    _set_db_path(db_path)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{base}/api/health", timeout=1).raise_for_status()
            break
        except Exception:
            time.sleep(0.1)

    yield base
    server.should_exit = True


@pytest.fixture(autouse=True)
def seed_db(live_server: str) -> None:
    """Reset DB and insert two fake completed runs before each test."""
    from llm_inference_benchmark.server import _buffers, _get_db, _now_iso

    _buffers.clear()
    db = _get_db()
    db.execute("DELETE FROM runs")
    db.commit()

    runs = [
        ("mock", "gpt2-mock", 120.5, 45.0, 210.0),
        ("llama-cpp", "llama-q4.gguf", 85.2, None, 310.0),
    ]
    for i, (backend, model, toks, ttft, p95) in enumerate(runs):
        run_id = f"test-run-{i:04d}-" + "a" * 28
        lines = [
            f"Backend: {backend}  Model: {model}  Requests: 10",
            "=== Benchmark Results ===",
            f"  tokens_per_second: {toks:.2f}",
            f"  p95_latency_ms: {p95:.2f}",
        ]
        if ttft is not None:
            lines.append(f"  p50_ttft_ms: {ttft:.2f}")
        else:
            lines.append("  p50_ttft_ms: N/A")
        db.execute(
            "INSERT INTO runs (id, status, config, output, created_at, finished_at)"
            " VALUES (?,?,?,?,?,?)",
            (
                run_id,
                "done",
                f'{{"backend":"{backend}","model":"{model}"}}',
                "\n".join(lines),
                _now_iso(),
                _now_iso(),
            ),
        )
    db.commit()


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_dashboard_loads(page: Page, live_server: str) -> None:
    page.goto(live_server)
    expect(page.locator("h1")).to_contain_text("llm-bench")


def test_runs_table_shows_two_rows(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-cb", timeout=8000)
    rows = page.locator(".run-cb").count()
    assert rows == 2


def test_runs_table_shows_status_badges(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".badge-done", timeout=8000)
    expect(page.locator(".badge-done").first).to_be_visible()


def test_runs_table_shows_throughput(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-cb", timeout=8000)
    # First seeded run has 120.5 tok/s → displayed as "120.5"
    expect(page.locator("#runs-tbody")).to_contain_text("120.5")


def test_log_button_shows_log_section(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("button:has-text('Log')", timeout=8000)
    page.locator("button:has-text('Log')").first.click()
    expect(page.locator("#log-section")).to_be_visible()
    expect(page.locator("#log-run-id")).to_be_visible()


def test_compare_without_selection_shows_alert(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-cb", timeout=8000)
    dismissed: list[str] = []

    def _handle_dialog(d: Dialog) -> None:
        dismissed.append(d.message)
        d.dismiss()

    page.on("dialog", _handle_dialog)
    page.locator("button:has-text('Compare Selected')").click()
    page.wait_for_timeout(500)
    assert any("2" in msg or "least" in msg.lower() for msg in dismissed)


def test_compare_chart_renders_with_two_runs(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#select-all", timeout=8000)
    page.locator("#select-all").click()
    page.locator("button:has-text('Compare Selected')").click()
    expect(page.locator("#chart-section")).to_be_visible()
    expect(page.locator("#comparison-chart")).to_be_visible()


def test_pareto_page_loads(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("a:has-text('Pareto')", timeout=8000)
    first_href = page.locator("a:has-text('Pareto')").first.get_attribute("href")
    assert first_href is not None and "/pareto.html" in first_href
    page.goto(live_server + first_href)
    expect(page.locator("h2")).to_contain_text("Pareto")
    expect(page.locator("#chart")).to_be_visible()


def test_pareto_page_has_plotly_chart(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("a:has-text('Pareto')", timeout=8000)
    first_href = page.locator("a:has-text('Pareto')").first.get_attribute("href")
    assert first_href is not None
    page.goto(live_server + first_href)
    page.wait_for_function(
        "() => document.querySelector('#chart .main-svg') !== null", timeout=8000
    )
    svg = page.locator("#chart .main-svg").first
    expect(svg).to_be_visible()


def test_dashboard_link_on_pareto_page(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("a:has-text('Pareto')", timeout=8000)
    first_href = page.locator("a:has-text('Pareto')").first.get_attribute("href")
    assert first_href is not None
    page.goto(live_server + first_href)
    expect(page.locator("a:has-text('Dashboard')")).to_be_visible()


def test_new_run_button_is_visible(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#new-run-btn", timeout=8000)
    expect(page.locator("#new-run-btn")).to_be_visible()
    expect(page.locator("#new-run-btn")).to_contain_text("New Run")


def test_new_run_modal_opens_on_click(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#new-run-btn", timeout=8000)
    # Modal should be hidden initially
    expect(page.locator("#run-modal")).not_to_have_class("open")
    page.locator("#new-run-btn").click()
    expect(page.locator("#run-modal")).to_have_class("modal-backdrop open")


def test_new_run_modal_closes_on_cancel(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#new-run-btn", timeout=8000)
    page.locator("#new-run-btn").click()
    expect(page.locator("#run-modal")).to_have_class("modal-backdrop open")
    page.locator("button:has-text('Cancel')").click()
    expect(page.locator("#run-modal")).not_to_have_class("open")


def test_new_run_modal_has_required_fields(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#new-run-btn", timeout=8000)
    page.locator("#new-run-btn").click()
    expect(page.locator("#f-model")).to_be_visible()
    expect(page.locator("#f-backend")).to_be_visible()
    expect(page.locator("#f-requests")).to_be_visible()
    expect(page.locator("#f-concurrency")).to_be_visible()
    expect(page.locator("#f-warmup")).to_be_visible()


def test_gpu_layers_hidden_for_non_llama_cpp(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#new-run-btn", timeout=8000)
    page.locator("#new-run-btn").click()
    # Default backend is 'mock' — GPU layers row must be hidden
    expect(page.locator("#f-gpu-row")).to_be_hidden()
    # Switch to llama-cpp — GPU layers row must appear
    page.locator("#f-backend").select_option("llama-cpp")
    expect(page.locator("#f-gpu-row")).to_be_visible()
    # Switch back — hidden again
    page.locator("#f-backend").select_option("mock")
    expect(page.locator("#f-gpu-row")).to_be_hidden()
