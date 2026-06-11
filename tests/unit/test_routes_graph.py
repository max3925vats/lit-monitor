"""FG-4: GET /graph Knowledge Graph overview page route tests.

The graph-overview assembly is patched at the single seam
``scripts.server.routes.graph._get_graph_overview`` — no live KuzuDB / SQLite.
Mirrors the ``client`` fixture from tests/unit/test_routes_corpus.py.
"""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    from lit_monitor.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


def test_get_graph_page_renders(client):
    r = client.get("/graph")
    assert r.status_code == 200 and "Knowledge Graph" in r.text


def test_graph_overview_fragment_renders_stats(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.graph._get_graph_overview",
        lambda: {
            "available": True,
            "paper_count": 10,
            "indexed": 8,
            "total": 10,
            "entity_count": 42,
            "by_type": {"Method": 5},
            "by_predicate": {"MENTIONS": 30},
            "by_source": {"schema": 20, "biobert": 15, "llm_cloud": 7},
            "last_indexed": "2026-06-01",
        },
    )
    r = client.get("/graph/overview")
    assert r.status_code == 200
    assert (
        "Method" in r.text
        and "biobert" in r.text
        and "30" in r.text
        and "8" in r.text
    )


def test_graph_overview_no_graph_notice(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.graph._get_graph_overview",
        lambda: {"available": False},
    )
    r = client.get("/graph/overview")
    assert r.status_code == 200
    assert "graph" in r.text.lower() and "backfill" in r.text.lower()


def test_graph_overview_no_leak(client, monkeypatch, caplog):
    def _boom():
        raise RuntimeError("kuzu://secret/path")

    monkeypatch.setattr(
        "scripts.server.routes.graph._get_graph_overview", _boom
    )
    with caplog.at_level(logging.ERROR, logger="scripts.server.routes.graph"):
        r = client.get("/graph/overview")
    assert r.status_code == 200 and "kuzu://secret" not in r.text
    assert any(
        "kuzu://secret" in rec.getMessage() for rec in caplog.records
    )


def test_nav_explore_has_knowledge_graph(client):
    r = client.get("/")
    assert 'href="/graph"' in r.text and "Knowledge Graph" in r.text
