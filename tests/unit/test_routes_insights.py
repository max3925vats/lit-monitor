"""FI-2: HTTP + HTML tests for the read-only /insights page.

Covers:
- GET  /insights                     — page renders (learning state, clusters, signal mix)
- GET  /feedback                     — redirects to /insights (page rename)
- POST /api/feedback                 — capture endpoint still works (rename-safe)
- learning-state card rendering (available / first-run)
- cluster-weights table rendering (populated / empty)
- nav exposes /insights in the Tune group
- no Chart.js / CDN / <canvas> leaks into the rendered page

Mirrors the fixture pattern in ``test_routes_feedback.py`` but yields the bare
TestClient. The route-module seams ``_learning_state`` / ``_cluster_weights`` are
patched per-test to exercise the template without a live engine.
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from scripts.server.routes.feedback import router as feedback_router
from scripts.server.runtime import reset_runtime

TEMPLATES_DIR = Path(__file__).parents[2] / "scripts" / "server" / "templates"


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
    from scripts.core.state_db import StateDB

    return StateDB(tmp_path / "state.db")


@pytest.fixture()
def client(real_db, monkeypatch):
    """Bare TestClient with the feedback router + patched runtime."""
    import scripts.server.app as app_mod

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["fromjson"] = _json.loads
    monkeypatch.setattr(app_mod, "templates", templates)

    rt = MagicMock()
    rt.state_db = real_db
    monkeypatch.setattr(
        "scripts.server.routes.feedback.get_runtime", lambda: rt
    )

    app = FastAPI()
    app.include_router(feedback_router)
    app.state.dev_mode = False
    app.state.version = "test"
    return TestClient(app)


# ---------------------------------------------------------------------------
# Page render + redirect + capture-survival
# ---------------------------------------------------------------------------

def test_get_insights_page_renders(client):
    r = client.get("/insights")
    assert r.status_code == 200 and "Insights" in r.text


def test_feedback_redirects_to_insights(client):
    r = client.get("/feedback", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"].endswith("/insights")


def test_capture_post_api_feedback_still_works(client):
    # the capture endpoint must NOT be broken by the rename.
    # Real contract (see routes/feedback.py POST /api/feedback): JSON body,
    # 200 + {"ok": True} on success.
    r = client.post(
        "/api/feedback",
        json={"doi": "10.1/a", "signal_type": "saved", "source": "discovery"},
    )
    assert r.status_code in (200, 204)
    assert r.json() == {"ok": True}


# ---------------------------------------------------------------------------
# Learning-state card
# ---------------------------------------------------------------------------

def test_insights_learning_state_card(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._learning_state",
        lambda: {
            "available": True,
            "n_events": 42,
            "soft_gate": 0.58,
            "inert": False,
            "computed_at": "2026-06-01",
            "dim": 384,
            "top_papers": [],
        },
    )
    r = client.get("/insights")
    assert "42" in r.text and "0.58" in r.text


def test_insights_first_run_no_vector(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._learning_state",
        lambda: {"available": False},
    )
    r = client.get("/insights")
    assert r.status_code == 200 and (
        "feedback" in r.text.lower() or "hasn't learned" in r.text.lower()
    )


# ---------------------------------------------------------------------------
# Cluster-weights table
# ---------------------------------------------------------------------------

def test_insights_cluster_weights_table(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._cluster_weights",
        lambda: {
            "floor": 0.1,
            "clusters": [
                {"id": 1, "name": "Chromatography", "weight": 0.42, "at_floor": False}
            ],
        },
    )
    r = client.get("/insights")
    assert "Chromatography" in r.text and "0.42" in r.text


def test_insights_cluster_weights_empty_first_run(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._cluster_weights",
        lambda: {"floor": 0.1, "clusters": []},
    )
    r = client.get("/insights")
    assert r.status_code == 200 and (
        "no themes" in r.text.lower() or "brain-build" in r.text.lower()
    )


# ---------------------------------------------------------------------------
# Nav + no-CDN guards
# ---------------------------------------------------------------------------

def test_nav_tune_has_insights(client):
    r = client.get("/insights")
    assert 'href="/insights"' in r.text and "Insights" in r.text


def test_no_chartjs_or_cdn(client):
    r = client.get("/insights")
    assert (
        "chart.js" not in r.text.lower()
        and "cdn.jsdelivr" not in r.text.lower()
        and "<canvas" not in r.text.lower()
    )
