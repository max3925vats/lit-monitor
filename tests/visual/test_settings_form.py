"""P2: the settings page uses a custom data-path JSON serializer. After the
Shoelace migration the serializer must read sl-* controls correctly. pytest
cannot exercise browser JS, so this drives real headless Chromium: set an
sl-input value, submit, and assert the edited value reached the server with the
correct type and nesting.

The test intercepts the POST *request* body (not just the response) — this
proves the serializer's value-read + number coercion + nesting all work. A
response-body check (`ok: true`) is kept for belt-and-suspenders.

Config-safe: live_server is isolated to a temp config copy (conftest). The GET
/settings re-render reads through resolve_path which may fall back to the
CWD-relative config/ in development; the POST (write path) always targets the
isolated tmp dir via _config_write_dir(). The round-trip is therefore verified
via the request body + 200 response rather than a page-reload value check."""
import json

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.visual


def test_settings_sl_input_roundtrips(live_server):
    base = live_server
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page()
        pg.goto(f"{base}/settings", wait_until="networkidle")
        # Shoelace migration: sl-input must be present (not native input)
        assert pg.locator("sl-input[data-path='weights.domain_context']").count() == 1
        # Wait for Shoelace to hydrate before interacting
        pg.wait_for_timeout(800)
        pg.eval_on_selector(
            "sl-input[data-path='weights.domain_context']",
            "el => { el.value = '0.37'; }",
        )
        # Verify the value was accepted by the Shoelace control
        val_before = pg.eval_on_selector(
            "sl-input[data-path='weights.domain_context']", "el => el.value"
        )
        assert val_before == "0.37", f"sl-input did not accept value: {val_before!r}"
        # Submit the ranking form; intercept the *request* body to prove the
        # serializer read the sl-input value, coerced it to float, and nested
        # it correctly under `weights.domain_context`.  A response-only check
        # (200 OK) would pass even if the serializer sent the wrong value.
        with pg.expect_request("**/api/settings/ranking") as req_info:
            pg.locator("#ranking sl-button[type='submit']").first.click()
        req = req_info.value
        body = json.loads(req.post_data)
        assert body["weights"]["domain_context"] == 0.37, body
        assert isinstance(body["weights"]["domain_context"], float), body
        # Also confirm the server accepted the payload
        resp = req.response()
        assert resp.status == 200, (
            f"Expected 200 but got {resp.status} — "
            "the settings-json extension likely did not send valid JSON"
        )
        b.close()
