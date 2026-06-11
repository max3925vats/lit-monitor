"""Nav/IA redesign: grouped dropdown topnav reachability + structure tests.

The flat nav previously orphaned 5 pages (Themes, Trending, Insights, Domain,
Settings). The redesign surfaces every page through grouped CSS-only dropdowns
(Run / Explore / Tune) plus top-level Ask + Setup links. These tests pin the
reachability guarantee, the group labels, the preserved health-badge, and the
server-side active-state marking.

Mirrors the `client = TestClient(create_app())` fixture from
tests/unit/test_routes_ask.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    from lit_monitor.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


# Every page the topnav must reach. The 5 formerly-orphaned pages are included
# here — this is the reachability guarantee.
ALL_NAV_HREFS = [
    "/discovery",
    "/brain-build",
    "/schedule",
    "/ask",
    "/themes",
    "/trending",
    "/domain",
    "/insights",
    "/settings",
    "/setup",
]


class TestNavReachability:
    def test_nav_surfaces_all_pages(self, client):
        r = client.get("/")
        assert r.status_code == 200
        for href in ALL_NAV_HREFS:
            assert f'href="{href}"' in r.text, f"nav is missing a link to {href}"

    def test_nav_has_group_labels(self, client):
        r = client.get("/")
        assert r.status_code == 200
        for label in ("Run", "Explore", "Tune"):
            assert label in r.text, f"group label {label!r} missing from nav"

    def test_health_badge_preserved(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert 'id="health-badge"' in r.text


class TestNavActiveState:
    def test_active_state_marks_current_section(self, client):
        r = client.get("/themes")
        assert r.status_code == 200
        # The Explore group (which owns /themes) or the Themes link itself must
        # be marked active so the user can see where they are.
        assert "active" in r.text
