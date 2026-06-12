import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright


def test_sl_input_serializes_through_htmx_post(live_server):
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(f"{live_server}/dev/sl-probe", wait_until="networkidle")
        pg.wait_for_timeout(700)                      # let Shoelace hydrate
        pg.fill("sl-input[name=probe] >> input", "spikeval")
        pg.click("sl-button[type=submit]")
        pg.wait_for_selector("#echo", timeout=5000)
        echo = pg.inner_text("#echo")
        assert "probe=spikeval" in echo, f"sl-input did not serialize: {echo!r}"
        assert "flag=on" in echo, f"sl-checkbox did not serialize: {echo!r}"
        b.close()
