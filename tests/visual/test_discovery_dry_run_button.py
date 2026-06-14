"""P3: dry_run is carried by the clicked submit button (sl-button name/value),
not a hidden input. Verify the click→POST chain in a real browser, intercepting
/api/discovery/start so nothing spawns (config-safe)."""
import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.visual


def test_dry_run_button_submits_dry_run_true(live_server):
    base = live_server
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page()
        captured = {}

        def handle_start(route, request):
            captured["body"] = request.post_data
            # canned 200 (NOT abort) so HTMX swaps a normal response, not error markup
            route.fulfill(status=200, content_type="text/html", body="ok")

        def abort_sse(route, request):
            # The SSE stream never closes and would block wait_until="networkidle".
            # Aborting it is safe — the test only needs the controls fragment.
            route.abort()

        pg.route("**/api/discovery/start", handle_start)
        pg.route("**/api/discovery/stream", abort_sse)

        # Use domcontentloaded (not networkidle) because the SSE stream is a
        # persistent connection that prevents networkidle from ever settling.
        pg.goto(f"{base}/discovery", wait_until="domcontentloaded")
        # Wait for the HTMX-loaded controls fragment to swap in the dry-run button.
        pg.wait_for_selector("sl-button[value='true']", timeout=10000)
        pg.wait_for_timeout(500)  # let Shoelace hydrate
        pg.click("sl-button[value='true']")  # "Dry-run (preview only)"
        pg.wait_for_timeout(500)
        body = captured.get("body") or ""
        # The form uses hx-encoding="multipart/form-data", so dry_run arrives as a
        # multipart part, not URL-encoded. The full part looks like:
        #   Content-Disposition: form-data; name="dry_run"\r\n\r\ntrue\r\n
        # Assert the field name and value "true" are both present and adjacent —
        # confirming the sl-button carried its name/value into the submission.
        assert 'name="dry_run"' in body, (
            f"expected dry_run field in POST body; got {body!r}"
        )
        # The multipart value for the dry_run field must be "true" (not "false").
        assert 'name="dry_run"\r\n\r\ntrue\r\n' in body, (
            f"expected dry_run=true in POST body (multipart); got {body!r}"
        )
        b.close()
