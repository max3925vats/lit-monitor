"""Brain-build collection switcher (2026-06-13 user-reported bug fix).

The user reported the "Change collection" control did not actually switch the
collection. Root cause: Shoelace silently rewrites ``<sl-option value="…">``
when the value contains a space (``"Molecular Simulations"`` -> the DOM value
becomes ``"Molecular_Simulations"``). The form therefore submitted the
underscored token, and the backend persisted that corrupted name to
``paths.yaml`` — so the active collection no longer matched the real Zotero
collection.

This test drives real headless Chromium: it loads /brain-build, picks a
collection whose name contains a space, clicks "Switch", and asserts the
ISOLATED ``paths.yaml`` now holds the REAL name (with the space intact), not the
underscored corruption.

Config-safe: ``live_server`` (conftest) isolates the config dir to a temp copy
via ``LIT_MONITOR_ROOT``. Clicking "Switch" only rewrites that temp
``paths.yaml``; the real tracked ``config/`` is never touched. We resolve the
isolated file the same way the app does — through ``config_dir()`` (which honors
``LIT_MONITOR_ROOT`` set by the fixture).
"""
from __future__ import annotations

import pytest
import yaml
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.visual


def _isolated_paths_file():
    """Resolve the isolated paths.yaml the same way the app does.

    ``config_dir()`` honors the ``LIT_MONITOR_ROOT`` env var the live_server
    fixture sets, so this points at the TEMP copy, never the user's real config.
    """
    from lit_monitor.core.config import config_dir

    return config_dir() / "paths.yaml"


def test_switch_persists_collection_name_with_spaces(live_server) -> None:
    base = live_server
    paths_file = _isolated_paths_file()

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page()

        # The SSE stream is a persistent connection that prevents networkidle
        # from ever settling — abort it (the test only needs the form).
        pg.route("**/stream", lambda r, req: r.abort())
        # DEFENSIVE: block the brain-build START endpoint so an accidental click
        # on the dashboard's "Start" button can never spawn a real subprocess.
        # This test only exercises the collection-switch POST.
        pg.route(
            "**/api/brain-build/start",
            lambda r, req: r.fulfill(status=200, body="blocked-by-test"),
        )

        pg.goto(f"{base}/brain-build", wait_until="load")

        # The dropdown is HTMX-loaded into #collection-field. It may render as a
        # full <sl-select> (Zotero creds resolve → real collection list) or fall
        # back to an <sl-input> (creds missing). Handle both: prefer the select.
        pg.wait_for_selector(
            "#collection-field sl-select, #collection-field sl-input",
            state="attached",
            timeout=10000,
        )
        pg.wait_for_timeout(800)  # let Shoelace hydrate

        target = "Molecular Simulations"  # a collection name WITH a space
        has_select = pg.locator("#collection-field sl-select").count() > 0

        if has_select:
            # Find an <sl-option> whose visible label contains a space, and set
            # the select's value to that option's DOM value (which Shoelace has
            # already underscored). Selecting via the option's own .value is what
            # a real user click does — it reproduces the corruption path exactly.
            picked = pg.eval_on_selector_all(
                "#collection-field sl-option",
                """opts => {
                    for (const o of opts) {
                        const label = o.textContent.trim();
                        if (label.includes(' ')) {
                            return {value: o.value, label: label};
                        }
                    }
                    return null;
                }""",
            )
            assert picked is not None, (
                "expected at least one collection name containing a space"
            )
            target = picked["label"]
            pg.eval_on_selector(
                "#collection-field sl-select",
                f"el => {{ el.value = {picked['value']!r}; }}",
            )
        else:
            # Fallback sl-input path: type the real name (no corruption here).
            pg.eval_on_selector(
                "#collection-field sl-input",
                f"el => {{ el.value = {target!r}; }}",
            )
        pg.wait_for_timeout(300)

        # Submit ONLY the collection-switch form (scoped id — never the Start
        # button). HX-Refresh:true on success would reload the page; that's fine.
        with pg.expect_response("**/setup/api/paths/collection") as resp_info:
            pg.click("#collection-switch-form sl-button[type='submit']")
        assert resp_info.value.status == 200
        pg.wait_for_timeout(500)
        b.close()

    # The isolated paths.yaml must now hold the REAL name (space intact), NOT the
    # underscored corruption Shoelace produced in the DOM value.
    data = yaml.safe_load(paths_file.read_text())
    saved = (data.get("zotero") or {}).get("collection_name")
    assert saved == target, (
        f"expected paths.yaml zotero.collection_name == {target!r} "
        f"(spaces intact), got {saved!r} — the space→underscore corruption "
        "was persisted"
    )
