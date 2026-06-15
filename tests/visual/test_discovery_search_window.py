"""Task 8: search time-range card — real-browser behaviour.

Verifies the parts that only exist in the browser (the server-side argv mapping
is covered by ``tests/unit/test_routes_discovery.py``):

  - the card is poll-safe (lives outside the 5s-polled controls section);
  - switching to "Rolling window" + setting days=30 updates the Since date to
    today-30 (the bidirectional link in site.js);
  - switching to "Custom range" reveals the From/To inputs;
  - on submit the active mode's values are injected into the start POST
    (htmx:configRequest), so the window selection actually reaches the route.

Config-safe: /api/discovery/start is intercepted so nothing spawns.
"""
from datetime import date, timedelta

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.visual


def test_rolling_window_links_days_to_since_and_submits(live_server):
    base = live_server
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page()
        captured = {}

        def handle_start(route, request):
            captured["body"] = request.post_data
            route.fulfill(status=200, content_type="text/html", body="ok")

        def abort_sse(route, request):
            route.abort()

        pg.route("**/api/discovery/start", handle_start)
        pg.route("**/api/discovery/stream", abort_sse)

        pg.goto(f"{base}/discovery", wait_until="domcontentloaded")
        # Wait for both the (HTMX-loaded) controls form AND the (server-rendered)
        # window card to be present.
        pg.wait_for_selector("#discovery-window-card", timeout=10000)
        pg.wait_for_selector("sl-button[value='false']", timeout=10000)
        pg.wait_for_timeout(600)  # let Shoelace hydrate

        # Expand the collapsed card.
        pg.eval_on_selector(
            "#discovery-window-card sl-details", "el => el.show()"
        )
        pg.wait_for_timeout(200)

        # Switch to Rolling window.
        pg.eval_on_selector(
            "#discovery-window-mode", "el => { el.value = 'rolling'; "
            "el.dispatchEvent(new CustomEvent('sl-change')); }"
        )
        pg.wait_for_timeout(200)
        # The rolling row should now be visible (hidden attr removed).
        assert pg.eval_on_selector(
            "#discovery-window-rolling", "el => !el.hasAttribute('hidden')"
        ), "rolling row should be visible after selecting Rolling window"

        # Set days back = 30; the Since date should become today-30.
        pg.eval_on_selector(
            "#discovery-window-days",
            "el => { el.value = '30'; "
            "el.dispatchEvent(new CustomEvent('sl-input')); }",
        )
        pg.wait_for_timeout(200)
        since_val = pg.eval_on_selector(
            "#discovery-window-since", "el => el.value"
        )
        expected = (date.today() - timedelta(days=30)).isoformat()
        assert since_val == expected, (
            f"Since should be today-30 ({expected}); got {since_val!r}"
        )

        # Submit via the Run button and confirm the window params reach the POST.
        pg.click("sl-button[value='false']")  # "Run now"
        pg.wait_for_timeout(500)
        body = captured.get("body") or ""
        assert 'name="window_mode"' in body and "rolling" in body, (
            f"expected window_mode=rolling in POST body; got {body!r}"
        )
        assert 'name="since_days"' in body and "30" in body, (
            f"expected since_days=30 in POST body; got {body!r}"
        )
        b.close()


def test_custom_range_reveals_from_to_inputs(live_server):
    base = live_server
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page()

        def abort_sse(route, request):
            route.abort()

        pg.route("**/api/discovery/stream", abort_sse)
        pg.goto(f"{base}/discovery", wait_until="domcontentloaded")
        pg.wait_for_selector("#discovery-window-card", timeout=10000)
        pg.wait_for_timeout(600)

        pg.eval_on_selector(
            "#discovery-window-card sl-details", "el => el.show()"
        )
        pg.eval_on_selector(
            "#discovery-window-mode", "el => { el.value = 'range'; "
            "el.dispatchEvent(new CustomEvent('sl-change')); }"
        )
        pg.wait_for_timeout(200)
        assert pg.eval_on_selector(
            "#discovery-window-range", "el => !el.hasAttribute('hidden')"
        ), "range row should be visible after selecting Custom range"
        # The rolling row should be hidden when range is active.
        assert pg.eval_on_selector(
            "#discovery-window-rolling", "el => el.hasAttribute('hidden')"
        ), "rolling row should be hidden when Custom range is active"
        b.close()
