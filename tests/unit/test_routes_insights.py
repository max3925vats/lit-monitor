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


def test_recent_event_empty_rating_renders_em_dash(client):
    # FI-2 double-escape guard: a recent event with no rating/source must render
    # an em-dash placeholder (&mdash; -> "—"), NOT the double-escaped literal
    # "&amp;mdash;" that Jinja autoescape produces for {{ ... or "&mdash;" }}.
    r = client.post(
        "/api/feedback",
        json={"doi": "10.2/no-rating", "signal_type": "saved"},
    )
    assert r.status_code in (200, 204)

    page = client.get("/insights")
    assert page.status_code == 200
    assert "&amp;mdash;" not in page.text  # no double-escaped literal
    assert "—" in page.text           # em-dash actually rendered


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


# ---------------------------------------------------------------------------
# FI-3: lazy feedback-timeline fragment
# ---------------------------------------------------------------------------

def test_insights_timeline_fragment(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._timeline",
        lambda granularity, weeks, dimension: [
            {"period": "2026-W22", "signal_type": "saved", "count": 5},
            {"period": "2026-W22", "signal_type": "dismissed", "count": 2},
        ],
    )
    r = client.get("/insights/timeline?granularity=week")
    assert r.status_code == 200 and "2026-W22" in r.text and "saved" in r.text


def test_insights_timeline_empty_notice(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._timeline",
        lambda granularity, weeks, dimension: [],
    )
    r = client.get("/insights/timeline")
    assert r.status_code == 200 and "no feedback" in r.text.lower()


def test_insights_timeline_granularity_toggle(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "scripts.server.routes.feedback._timeline",
        lambda granularity, weeks, dimension: seen.update(g=granularity, w=weeks) or [],
    )
    client.get("/insights/timeline?granularity=day")
    assert seen["g"] == "day" and seen["w"] is not None  # a default lookback is passed


def test_insights_timeline_no_leak(client, monkeypatch, caplog):
    import logging

    def _boom(granularity, weeks, dimension):
        raise RuntimeError("db://secret/path")

    monkeypatch.setattr("scripts.server.routes.feedback._timeline", _boom)
    with caplog.at_level(logging.ERROR, logger="scripts.server.routes.feedback"):
        r = client.get("/insights/timeline")
    assert r.status_code == 200 and "db://secret" not in r.text
    assert any("db://secret" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# FI-3: lazy top-aligned-papers fragment
# ---------------------------------------------------------------------------

def test_insights_top_papers_fragment(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._learning_state",
        lambda: {
            "available": True,
            "n_events": 12,
            "soft_gate": 0.3,
            "inert": False,
            "computed_at": "x",
            "dim": 384,
            "top_papers": [{"doi": "10.1/a", "title": "Carta 2009", "cosine": 0.77}],
        },
    )
    r = client.get("/insights/top-papers")
    assert "Carta 2009" in r.text and "0.77" in r.text and 'href="/corpus/10.1/a"' in r.text


def test_insights_top_papers_empty_notice(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._learning_state",
        lambda: {
            "available": True,
            "n_events": 12,
            "soft_gate": 0.3,
            "inert": False,
            "computed_at": "x",
            "dim": 384,
            "top_papers": [],
        },
    )
    r = client.get("/insights/top-papers")
    assert r.status_code == 200 and ("no " in r.text.lower())  # graceful empty


def test_insights_top_papers_no_vector_notice(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._learning_state",
        lambda: {"available": False},
    )
    r = client.get("/insights/top-papers")
    assert r.status_code == 200 and (
        "feedback" in r.text.lower() or "hasn't learned" in r.text.lower()
    )


def test_insights_top_papers_no_leak(client, monkeypatch, caplog):
    import logging
    def _boom():
        raise RuntimeError("vec://secret/path")
    monkeypatch.setattr("scripts.server.routes.feedback._learning_state", _boom)
    with caplog.at_level(logging.ERROR, logger="scripts.server.routes.feedback"):
        r = client.get("/insights/top-papers")
    assert r.status_code == 200 and "vec://secret" not in r.text
    assert any("vec://secret" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# FI-6: feedback-by-source aggregate section on the page
# ---------------------------------------------------------------------------

def test_insights_by_source_section(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._summary_by_source",
        lambda: {"discovery": 7, "themes": 3, "implicit_zotero_save": 2},
    )
    monkeypatch.setattr(
        "scripts.server.routes.feedback._learning_state", lambda: {"available": False}
    )
    monkeypatch.setattr(
        "scripts.server.routes.feedback._cluster_weights",
        lambda: {"floor": 0.1, "clusters": []},
    )
    r = client.get("/insights")
    assert (
        r.status_code == 200
        and "discovery" in r.text
        and "implicit_zotero_save" in r.text
        and "By source" in r.text
    )


def test_insights_by_source_empty_first_run(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.feedback._summary_by_source", lambda: {}
    )
    monkeypatch.setattr(
        "scripts.server.routes.feedback._learning_state", lambda: {"available": False}
    )
    monkeypatch.setattr(
        "scripts.server.routes.feedback._cluster_weights",
        lambda: {"floor": 0.1, "clusters": []},
    )
    r = client.get("/insights")
    assert r.status_code == 200 and (
        "no feedback" in r.text.lower() or "no source" in r.text.lower()
    )


# ---------------------------------------------------------------------------
# FI-6: timeline dimension (by signal / by source) toggle passthrough
# ---------------------------------------------------------------------------

def test_timeline_dimension_source_passthrough(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "scripts.server.routes.feedback._timeline",
        lambda granularity, weeks, dimension: seen.update(g=granularity, d=dimension)
        or [{"period": "2026-W22", "source": "discovery", "count": 4}],
    )
    r = client.get("/insights/timeline?granularity=week&dimension=source")
    assert seen["d"] == "source" and "discovery" in r.text


def test_timeline_dimension_defaults_to_signal(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "scripts.server.routes.feedback._timeline",
        lambda granularity, weeks, dimension: seen.update(d=dimension)
        or [{"period": "2026-W22", "signal_type": "saved", "count": 4}],
    )
    client.get("/insights/timeline?granularity=week")
    assert seen["d"] == "signal"  # default


def test_timeline_invalid_dimension_falls_back(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "scripts.server.routes.feedback._timeline",
        lambda granularity, weeks, dimension: seen.update(d=dimension) or [],
    )
    client.get("/insights/timeline?dimension=bogus")
    assert seen["d"] == "signal"  # route whitelists before passing to backend
