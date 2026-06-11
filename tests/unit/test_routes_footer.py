"""Footer rendering tests (Task #73 — Phase E).

The site footer (GitHub / Report a bug / README links) is defined in
``scripts/server/templates/base.html`` and must render on every route that
extends ``base.html``. These tests assert it appears on three representative
pages: the landing page, the Setup wizard landing, and the Brain-Build
dashboard.

The footer must NEVER expose an email address — the user explicitly chose
the GitHub Issues link as the sole contact channel.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server import runtime as runtime_mod
from lit_monitor.server.app import create_app


def _assert_footer(html: str) -> None:
    """Shared assertions for the footer block across pages."""
    assert 'class="site-footer"' in html
    assert "github.com/max3925vats/lit-monitor" in html
    assert "github.com/max3925vats/lit-monitor/issues" in html
    # Safety: no email leak. The user chose Issues-only as the contact path.
    assert "mailto:" not in html


@pytest.mark.unit
def test_footer_appears_on_landing() -> None:
    """GET / renders the site footer with all three GitHub links."""
    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    _assert_footer(resp.text)


@pytest.mark.unit
def test_footer_appears_on_setup() -> None:
    """GET /setup renders the site footer with all three GitHub links."""
    client = TestClient(create_app())
    # /setup may redirect to /setup/ — follow_redirects defaults to True
    # in TestClient, so a single GET handles both shapes.
    resp = client.get("/setup")
    assert resp.status_code == 200
    _assert_footer(resp.text)


@pytest.mark.unit
def test_footer_appears_on_brain_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /brain-build renders the site footer with all three GitHub links.

    Forces the "DB unavailable" branch so the route renders without needing
    a real state DB on disk — the footer is part of base.html regardless.
    """
    runtime_mod.reset_runtime()

    def _boom() -> None:
        raise RuntimeError("no runtime in test")

    from lit_monitor.server.routes import brain_build as bb_route

    monkeypatch.setattr(bb_route, "get_runtime", _boom)

    client = TestClient(create_app())
    resp = client.get("/brain-build")
    assert resp.status_code == 200
    _assert_footer(resp.text)
