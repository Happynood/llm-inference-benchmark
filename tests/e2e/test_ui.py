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
from playwright.sync_api import BrowserContext, Page, expect


def _chromium_available() -> bool:
    try:
        import os

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            return os.path.isfile(p.chromium.executable_path)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _chromium_available(),
    reason="Chromium not available (run: playwright install chromium)",
)

# ── Plotly CDN stub ────────────────────────────────────────────────────────────

_PLOTLY_STUB = (
    "window.Plotly={"
    "_render:function(el){"
    "var c=typeof el==='string'?document.getElementById(el):el;"
    "if(!c)return;"
    "if(!c.querySelector('.main-svg')){"
    "var s=document.createElementNS('http://www.w3.org/2000/svg','svg');"
    "s.setAttribute('class','main-svg');"
    "s.style.display='block';"
    "c.appendChild(s);"
    "}"
    "},"
    "newPlot:function(el){this._render(el);},"
    "react:function(el){this._render(el);},"
    "downloadImage:function(){}"
    "};"
)


@pytest.fixture(autouse=True)
def stub_plotly_cdn(context: BrowserContext) -> None:
    """Intercept Plotly CDN requests to avoid external network dependency in tests."""
    context.route(
        "**plotly*",
        lambda route: route.fulfill(
            body=_PLOTLY_STUB,
            content_type="application/javascript",
        ),
    )


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
    from llm_inference_benchmark.server import _buffers, _get_db, _now_iso, _pull_errors

    _buffers.clear()
    _pull_errors.clear()
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
    expect(page.locator(".brand")).to_contain_text("llm-bench")


def test_runs_table_shows_two_rows(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    rows = page.locator(".run-card").count()
    assert rows == 2


def test_runs_table_shows_status_badges(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".badge-done", timeout=8000)
    expect(page.locator(".badge-done").first).to_be_visible()


def test_runs_table_shows_throughput(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    # First seeded run has 120.5 tok/s → displayed in sidebar card meta
    expect(page.locator("#run-list")).to_contain_text("120.5")


def test_log_button_shows_log_section(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    # Click a run card to load the detail panel with the log output
    page.locator(".run-card").first.click()
    page.wait_for_selector(".log-section", timeout=8000)
    expect(page.locator(".log-section")).to_be_visible()
    expect(page.locator("#log-output")).to_be_visible()


def test_run_card_click_shows_detail_panel(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    page.locator(".run-card").first.click()
    page.wait_for_selector("#detail-inner", timeout=8000)
    expect(page.locator("#detail-inner")).to_be_visible()


def test_run_detail_has_delete_button(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    page.locator(".run-card").first.click()
    page.wait_for_selector("#detail-inner", timeout=8000)
    expect(page.locator("button:has-text('Delete')")).to_be_visible()


def test_pareto_page_loads(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    page.locator(".run-card").first.click()
    page.wait_for_selector("a:has-text('Pareto Chart')", timeout=8000)
    first_href = page.locator("a:has-text('Pareto Chart')").first.get_attribute("href")
    assert first_href is not None and "/pareto.html" in first_href
    page.goto(live_server + first_href)
    expect(page.locator("h2")).to_contain_text("Pareto")
    expect(page.locator("#chart")).to_be_visible()


def test_pareto_page_has_plotly_chart(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    page.locator(".run-card").first.click()
    page.wait_for_selector("a:has-text('Pareto Chart')", timeout=8000)
    first_href = page.locator("a:has-text('Pareto Chart')").first.get_attribute("href")
    assert first_href is not None
    page.goto(live_server + first_href)
    page.wait_for_function(
        "() => document.querySelector('#chart .main-svg') !== null", timeout=8000
    )
    svg = page.locator("#chart .main-svg").first
    expect(svg).to_be_visible()


def test_dashboard_link_on_pareto_page(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    page.locator(".run-card").first.click()
    page.wait_for_selector("a:has-text('Pareto Chart')", timeout=8000)
    first_href = page.locator("a:has-text('Pareto Chart')").first.get_attribute("href")
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
    # Modal is hidden initially via the HTML hidden attribute
    expect(page.locator("#modal-overlay")).to_be_hidden()
    page.locator("#new-run-btn").click()
    expect(page.locator("#modal-overlay")).to_be_visible()


def test_new_run_modal_closes_on_cancel(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#new-run-btn", timeout=8000)
    page.locator("#new-run-btn").click()
    expect(page.locator("#modal-overlay")).to_be_visible()
    page.locator("button:has-text('Cancel')").click()
    expect(page.locator("#modal-overlay")).to_be_hidden()


def test_new_run_modal_has_required_fields(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#new-run-btn", timeout=8000)
    page.locator("#new-run-btn").click()
    expect(page.locator("#f-model")).to_be_visible()
    expect(page.locator("#f-backend")).to_be_visible()
    expect(page.locator("#f-requests")).to_be_visible()
    expect(page.locator("#f-concurrency")).to_be_visible()
    expect(page.locator("#f-warmup")).to_be_visible()


def test_run_card_stays_selected_after_htmx_refresh(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    first_card = page.locator(".run-card").first
    first_run_id = first_card.get_attribute("data-run-id")
    first_card.click()
    # Confirm selected class was applied
    page.wait_for_function(
        f"() => !!document.querySelector('.run-card[data-run-id=\"{first_run_id}\"]')"
        f"?.classList.contains('selected')",
        timeout=4000,
    )
    # Trigger HTMX refresh without waiting 3 s
    page.evaluate("htmx.trigger('#run-list', 'load')")
    page.wait_for_selector(".run-card", timeout=8000)
    # After re-render, the card must still carry the selected class
    page.wait_for_function(
        f"() => !!document.querySelector('.run-card[data-run-id=\"{first_run_id}\"]')"
        f"?.classList.contains('selected')",
        timeout=8000,
    )


def test_detail_panel_loads_after_card_click(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector(".run-card", timeout=8000)
    page.locator(".run-card").first.click()
    # HTMX loads the run detail fragment; detail-inner is rendered by the server
    page.wait_for_selector("#detail-inner", timeout=8000)
    expect(page.locator("#detail-inner")).to_be_visible()


def test_dataset_pull_error_shown_in_table(page: Page, live_server: str) -> None:
    from unittest.mock import patch

    from llm_inference_benchmark import datasets as _datasets_mod
    from llm_inference_benchmark.server import _pull_errors

    _pull_errors["lmsys-chat"] = "Simulated network error"
    try:
        # Patch list_cached so lmsys-chat appears uncached regardless of local state,
        # allowing the pull error to be surfaced in the UI.
        with patch.object(_datasets_mod, "list_cached", return_value=[]):
            page.goto(live_server)
            page.evaluate("showTab('datasets')")
            page.wait_for_selector("#datasets-tbody", timeout=8000)
            page.evaluate(
                "htmx.ajax('GET', '/api/ui/datasets-table',"
                " {target: '#datasets-tbody', swap: 'innerHTML'})"
            )
            page.wait_for_function(
                "() => document.querySelector('.ds-pull-error') !== null",
                timeout=8000,
            )
            expect(page.locator(".ds-pull-error").first).to_contain_text("Pull failed:")
            expect(page.locator(".ds-pull-error").first).to_contain_text("Simulated network error")
    finally:
        _pull_errors.pop("lmsys-chat", None)


def test_gpu_layers_hidden_for_non_llama_cpp(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.wait_for_selector("#new-run-btn", timeout=8000)
    page.locator("#new-run-btn").click()
    # Default backend is 'mock' — GPU layers field must not be present
    expect(page.locator("#f-llama-gpu")).to_have_count(0)
    # Switch to llama-cpp — GPU layers field appears in backend-fields
    page.locator("#f-backend").select_option("llama-cpp")
    expect(page.locator("#f-llama-gpu")).to_be_visible()
    # Switch back to mock — field disappears again
    page.locator("#f-backend").select_option("mock")
    expect(page.locator("#f-llama-gpu")).to_have_count(0)


# ── Multi-run comparison tests ─────────────────────────────────────────────────


def _check_two_cards(page: Page) -> tuple[str, str]:
    """Check the first two run cards and return their run IDs."""
    page.wait_for_selector(".run-card", timeout=8000)
    cards = page.locator(".run-card")
    rid0 = cards.nth(0).get_attribute("data-run-id")
    rid1 = cards.nth(1).get_attribute("data-run-id")
    assert rid0 and rid1
    page.locator(f'.compare-cb[value="{rid0}"]').click()
    page.locator(f'.compare-cb[value="{rid1}"]').click()
    return rid0, rid1


def test_compare_bar_appears_when_two_runs_checked(page: Page, live_server: str) -> None:
    page.goto(live_server)
    _check_two_cards(page)
    bar = page.locator("#compare-bar")
    expect(bar).to_be_visible()
    expect(page.locator("#compare-count")).to_contain_text("2 runs selected")


def test_compare_bar_clears_on_x_button(page: Page, live_server: str) -> None:
    page.goto(live_server)
    _check_two_cards(page)
    expect(page.locator("#compare-bar")).to_be_visible()
    page.locator('button[aria-label="Clear selection"]').click()
    expect(page.locator("#compare-bar")).to_be_hidden()
    checked = page.eval_on_selector_all(".compare-cb", "els => els.filter(e => e.checked).length")
    assert checked == 0


def test_compare_opens_pareto_tab(page: Page, live_server: str) -> None:
    page.goto(live_server)
    rid0, rid1 = _check_two_cards(page)
    expect(page.locator("#compare-bar")).to_be_visible()
    with page.context.expect_page() as new_page_info:
        page.locator('button:has-text("Pareto")').click()
    new_page = new_page_info.value
    new_page.wait_for_load_state("domcontentloaded")
    url = new_page.url
    assert "/runs/pareto" in url
    assert rid0 in url
    assert rid1 in url


def test_compare_table_loads_in_main_panel(page: Page, live_server: str) -> None:
    page.goto(live_server)
    _check_two_cards(page)
    expect(page.locator("#compare-bar")).to_be_visible()
    page.locator('button:has-text("Table")').click()
    page.wait_for_selector(".compare-table-wrap", timeout=8000)
    expect(page.locator(".compare-table-wrap")).to_be_visible()
    expect(page.locator(".cmp-table")).to_be_visible()
    expect(page.locator(".detail-title")).to_contain_text("Metric Comparison")


def test_compare_chart_loads_in_main_panel(page: Page, live_server: str) -> None:
    page.goto(live_server)
    _check_two_cards(page)
    expect(page.locator("#compare-bar")).to_be_visible()
    page.locator('button:has-text("Chart")').click()
    page.wait_for_selector("#cmp-chart-div", timeout=8000)
    expect(page.locator("#cmp-chart-div")).to_be_visible()
    expect(page.locator(".detail-title")).to_contain_text("Metric Chart")


def test_compare_csv_button_visible_when_two_runs_selected(page: Page, live_server: str) -> None:
    page.goto(live_server)
    _check_two_cards(page)
    expect(page.locator("#compare-bar")).to_be_visible()
    expect(page.locator('button:has-text("CSV")')).to_be_visible()


def test_compare_trend_button_visible_when_two_runs_selected(page: Page, live_server: str) -> None:
    page.goto(live_server)
    _check_two_cards(page)
    expect(page.locator("#compare-bar")).to_be_visible()
    expect(page.locator('button:has-text("Trend")')).to_be_visible()


def test_compare_trend_loads_in_main_panel(page: Page, live_server: str) -> None:
    page.goto(live_server)
    _check_two_cards(page)
    expect(page.locator("#compare-bar")).to_be_visible()
    page.locator('button:has-text("Trend")').click()
    page.wait_for_selector("#cmp-trend-div", timeout=8000)
    expect(page.locator("#cmp-trend-div")).to_be_visible()
    expect(page.locator(".detail-title")).to_contain_text("Metric Trend")


def test_compare_checkboxes_survive_htmx_refresh(page: Page, live_server: str) -> None:
    page.goto(live_server)
    rid0, rid1 = _check_two_cards(page)
    expect(page.locator("#compare-bar")).to_be_visible()
    page.evaluate("htmx.trigger('#run-list', 'load')")
    page.wait_for_selector(".run-card", timeout=8000)
    page.wait_for_function(
        f"() => !!document.querySelector('.compare-cb[value=\"{rid0}\"]')?.checked",
        timeout=6000,
    )
    page.wait_for_function(
        f"() => !!document.querySelector('.compare-cb[value=\"{rid1}\"]')?.checked",
        timeout=6000,
    )
    expect(page.locator("#compare-bar")).to_be_visible()


def test_leaderboard_button_visible_in_sidebar(page: Page, live_server: str) -> None:
    page.goto(live_server)
    expect(page.locator("#tab-btn-leaderboard")).to_be_visible()
    expect(page.locator("#tab-btn-leaderboard")).to_contain_text("Leaderboard")


def test_leaderboard_panel_loads_on_click(page: Page, live_server: str) -> None:
    page.goto(live_server)
    page.locator("#tab-btn-leaderboard").click()
    page.wait_for_selector(".detail-title", timeout=8000)
    expect(page.locator(".detail-title")).to_contain_text("Leaderboard")
