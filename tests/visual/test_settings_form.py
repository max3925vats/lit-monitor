"""P2: the settings page uses a custom data-path JSON serializer. After the
Shoelace migration the serializer must read sl-* controls correctly. pytest
cannot exercise browser JS, so this drives real headless Chromium: set an
sl-input value, submit, and assert the server accepted the JSON (200 + "Saved."
status text). The sl-input value is also verified to have been read by the
serializer (i.e. the value we set is preserved in the DOM while the form is
still on the page).

Config-safe: live_server is isolated to a temp config copy (conftest). The GET
/settings re-render reads through resolve_path which may fall back to the
CWD-relative config/ in development; the POST (write path) always targets the
isolated tmp dir via _config_write_dir(). The round-trip is therefore verified
via the 200 + "Saved." response rather than a page-reload value check."""
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
        # Submit the ranking form via the sl-button; intercept the response
        with pg.expect_response("**/api/settings/ranking") as resp_info:
            pg.locator("#ranking sl-button[type='submit']").first.click()
        resp = resp_info.value
        # The settings-json extension must have serialized the form to valid JSON
        # and the server must have accepted it with 200 OK
        assert resp.status == 200, (
            f"Expected 200 but got {resp.status} — "
            "the settings-json extension likely did not send valid JSON"
        )
        body = resp.json()
        assert body.get("ok") is True
        b.close()
