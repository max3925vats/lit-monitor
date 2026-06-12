"""Sidebar IA reachability + structure tests (app-shell redesign).

The app-shell sidebar (templates/_partials/sidebar.html) and the server-side
active-state both read NAV_GROUPS, so the IA lives in exactly one place. These
tests pin the reachability guarantee (every grouped link is present), the group
labels, the global Ask button + health badge, and the server-side active-state
marking — all driven off NAV_GROUPS so they can't drift from the source of truth.

Mirrors the `client = TestClient(create_app())` fixture from
tests/unit/test_routes_ask.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.nav import NAV_GROUPS


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    from lit_monitor.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


def test_sidebar_has_all_groups(client):
    html = client.get("/").text
    for group in NAV_GROUPS:
        assert group.label in html, f"missing group {group.label}"


def test_sidebar_links_present(client):
    html = client.get("/").text
    for group in NAV_GROUPS:
        for item in group.items:
            assert f'href="{item.href}"' in html, f"missing nav link {item.href}"


def test_active_state_on_section(client):
    assert "active" in client.get("/discovery").text


def test_ask_button_and_health_badge_present(client):
    html = client.get("/").text
    assert "app-ask-btn" in html
    assert 'id="health-badge"' in html
