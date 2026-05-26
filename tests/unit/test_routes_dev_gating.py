"""Unit tests for the LIT_MONITOR_DEV flag-gating in create_app (Task #66 D1).

Two invariants:

1. With no ``LIT_MONITOR_DEV`` env var, GET /dev returns 404 — the router
   is NOT mounted.
2. With ``LIT_MONITOR_DEV=1``, GET /dev returns 200 and the body advertises
   the dev surface.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_dev_route_404_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the dev flag is unset, the /dev route must not exist."""
    # Make sure no leftover env var from another test leaks into this one.
    monkeypatch.delenv("LIT_MONITOR_DEV", raising=False)

    from scripts.server.app import create_app

    client = TestClient(create_app())
    resp = client.get("/dev")

    assert resp.status_code == 404, (
        "Without LIT_MONITOR_DEV=1, /dev must not be mounted."
    )


@pytest.mark.unit
def test_dev_route_200_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """When LIT_MONITOR_DEV=1, /dev is mounted and serves the placeholder."""
    monkeypatch.setenv("LIT_MONITOR_DEV", "1")

    from scripts.server.app import create_app

    app = create_app()
    # app.state.dev_mode is the contract used by templates for the banner.
    assert app.state.dev_mode is True

    client = TestClient(app)
    resp = client.get("/dev")

    assert resp.status_code == 200
    # Loose substring check — body wording can evolve in #67+ without breaking this.
    assert "dev" in resp.text.lower()
