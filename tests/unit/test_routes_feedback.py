"""Bundle H: HTTP endpoint + HTML page tests for /feedback and /api/feedback/*.

Covers:
- POST /api/feedback — records event, returns {ok: True}
- POST /api/feedback — bad signal_type → 422
- POST /api/feedback — rating without signal_type='rated' → 422
- POST /api/feedback — signal_type='rated' without rating → 422
- POST /api/feedback — signal_type='rated' with valid rating → 200
- GET  /api/feedback/recent — empty list when no events
- GET  /api/feedback/recent — returns events after seeding
- GET  /api/feedback/summary — returns counts per signal_type
- GET  /feedback — HTML page renders
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from lit_monitor.server.routes.feedback import router as feedback_router
from lit_monitor.server.runtime import reset_runtime

TEMPLATES_DIR = Path(__file__).parents[2] / "lit_monitor" / "server" / "templates"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture()
def real_db(tmp_path):
    from lit_monitor.core.state_db import StateDB
    return StateDB(tmp_path / "state.db")


@pytest.fixture()
def client(real_db, monkeypatch):
    """TestClient with feedback router + patched runtime pointing at real_db."""
    import lit_monitor.server.app as app_mod

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app_mod.configure_templates(templates)
    # monkeypatch auto-restores app.templates after the test so we don't leak
    # this test-only templates object into sibling tests (order-dependent flakiness).
    monkeypatch.setattr(app_mod, "templates", templates)

    rt = MagicMock()
    rt.state_db = real_db

    monkeypatch.setattr(
        "lit_monitor.server.routes.feedback.get_runtime", lambda: rt
    )

    app = FastAPI()
    app.include_router(feedback_router)
    app.state.dev_mode = False
    app.state.version = "test"
    return TestClient(app), real_db


# ---------------------------------------------------------------------------
# POST /api/feedback
# ---------------------------------------------------------------------------

class TestFeedbackPost:
    def test_records_event_returns_ok(self, client):
        c, db = client
        r = c.post(
            "/api/feedback",
            json={"doi": "10.1/test", "signal_type": "saved", "source": "discovery"},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        # Verify row was written.
        events = db.list_feedback_events()
        assert len(events) == 1
        assert events[0]["signal_type"] == "saved"

    def test_bad_signal_type_422(self, client):
        c, _ = client
        r = c.post(
            "/api/feedback",
            json={"doi": "10.1/test", "signal_type": "unknown_signal"},
        )
        assert r.status_code == 422

    def test_rating_without_rated_signal_422(self, client):
        c, _ = client
        r = c.post(
            "/api/feedback",
            json={"doi": "10.1/test", "signal_type": "saved", "rating": 4},
        )
        assert r.status_code == 422

    def test_rated_signal_without_rating_422(self, client):
        c, _ = client
        r = c.post(
            "/api/feedback",
            json={"doi": "10.1/test", "signal_type": "rated"},
        )
        assert r.status_code == 422

    def test_rated_signal_with_valid_rating_200(self, client):
        c, db = client
        r = c.post(
            "/api/feedback",
            json={"doi": "10.1/test", "signal_type": "rated", "rating": 4},
        )
        assert r.status_code == 200
        events = db.list_feedback_events()
        assert events[0]["rating"] == 4

    def test_thumbs_up_200(self, client):
        c, _ = client
        r = c.post(
            "/api/feedback",
            json={"doi": "10.1/test", "signal_type": "thumbs_up"},
        )
        assert r.status_code == 200

    def test_dismissed_has_negative_weight(self, client):
        c, db = client
        r = c.post(
            "/api/feedback",
            json={"doi": "10.1/test", "signal_type": "dismissed"},
        )
        assert r.status_code == 200
        events = db.list_feedback_events()
        assert events[0]["weight"] < 0


# ---------------------------------------------------------------------------
# GET /api/feedback/recent
# ---------------------------------------------------------------------------

class TestFeedbackRecent:
    def test_empty_returns_list(self, client):
        c, _ = client
        r = c.get("/api/feedback/recent")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_events_after_seeding(self, client):
        c, db = client
        db.record_feedback_event("10.1/a", "saved")
        db.record_feedback_event("10.1/b", "thumbs_up")
        r = c.get("/api/feedback/recent?limit=10")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2

    def test_limit_respected(self, client):
        c, db = client
        for i in range(5):
            db.record_feedback_event(f"10.1/{i}", "opened")
        r = c.get("/api/feedback/recent?limit=3")
        assert r.status_code == 200
        assert len(r.json()) == 3


# ---------------------------------------------------------------------------
# GET /api/feedback/summary
# ---------------------------------------------------------------------------

class TestFeedbackSummary:
    def test_empty_returns_zeroed_dict(self, client):
        c, _ = client
        r = c.get("/api/feedback/summary")
        assert r.status_code == 200
        data = r.json()
        # All signal types present with 0.
        assert "saved" in data
        assert data["saved"] == 0

    def test_counts_per_signal(self, client):
        c, db = client
        db.record_feedback_event("10.1/a", "saved")
        db.record_feedback_event("10.1/b", "saved")
        db.record_feedback_event("10.1/c", "dismissed")
        r = c.get("/api/feedback/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["saved"] == 2
        assert data["dismissed"] == 1


# ---------------------------------------------------------------------------
# GET /feedback — redirects to the renamed /insights page (FI-2)
# ---------------------------------------------------------------------------

class TestFeedbackPage:
    def test_redirects_to_insights(self, client):
        # FI-2 renamed the /feedback page to /insights; /feedback now 307s.
        c, _ = client
        r = c.get("/feedback", follow_redirects=False)
        assert r.status_code in (302, 307)
        assert r.headers["location"].endswith("/insights")
