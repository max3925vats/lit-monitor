"""The live-progress pane is hidden at load (no run active).

In a real browser, after the page loads and the HTMX controls fragment settles,
`#live-progress-wrap` must still carry the `hidden` attribute on both dashboards
(nothing is running in the isolated live_server). This proves the pane no longer
shows always — it is run-gated. We intentionally NEVER click Run/Start/Dry-run
(those spawn real pipeline subprocesses); asserting the idle/hidden state needs
no clicking.
"""
import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.visual


def _assert_wrap_hidden_at_load(base: str, path: str, stream_glob: str) -> None:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page()

        def abort_sse(route, request):
            # The SSE stream never closes; abort it so the page settles. The pane
            # is hidden anyway — we only need the controls fragment to swap in.
            route.abort()

        pg.route(stream_glob, abort_sse)
        # 'load' (not networkidle): the SSE connection would keep networkidle from
        # ever settling. The controls fragment loads via hx-trigger=load.
        pg.goto(f"{base}{path}", wait_until="load")
        pg.wait_for_selector("#live-progress-wrap", state="attached", timeout=10000)
        # Give the controls poll/afterSettle a beat to run the toggle JS.
        pg.wait_for_timeout(800)
        is_hidden = pg.eval_on_selector(
            "#live-progress-wrap", "el => el.hasAttribute('hidden')"
        )
        assert is_hidden, f"{path}: live-progress-wrap should be hidden when idle"
        b.close()


def test_brain_build_live_progress_hidden_when_idle(live_server):
    _assert_wrap_hidden_at_load(
        live_server, "/brain-build", "**/api/brain-build/stream"
    )


def test_discovery_live_progress_hidden_when_idle(live_server):
    _assert_wrap_hidden_at_load(live_server, "/discovery", "**/api/discovery/stream")
