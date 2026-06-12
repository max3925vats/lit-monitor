import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_shell_chrome_present(live_server, theme):
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 900})
        if theme == "light":
            ctx.add_init_script("try{localStorage.setItem('lit_theme','light')}catch(e){}")
        pg = ctx.new_page()
        pg.goto(f"{live_server}/", wait_until="networkidle")
        pg.wait_for_timeout(700)
        assert pg.locator(".app-sidebar").count() == 1
        assert pg.locator(".app-topbar .app-ask-btn").count() == 1
        assert pg.evaluate("!!customElements.get('sl-button')"), "Shoelace did not hydrate"
        ctx.close()
        b.close()
