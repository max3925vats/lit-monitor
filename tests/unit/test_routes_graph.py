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
        "lit_monitor.server.routes.graph._get_graph_overview",
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
        "lit_monitor.server.routes.graph._get_graph_overview",
        lambda: {"available": False},
    )
    r = client.get("/graph/overview")
    assert r.status_code == 200
    assert "graph" in r.text.lower() and "backfill" in r.text.lower()


def test_graph_overview_no_leak(client, monkeypatch, caplog):
    def _boom():
        raise RuntimeError("kuzu://secret/path")

    monkeypatch.setattr(
        "lit_monitor.server.routes.graph._get_graph_overview", _boom
    )
    with caplog.at_level(logging.ERROR, logger="lit_monitor.server.routes.graph"):
        r = client.get("/graph/overview")
    assert r.status_code == 200 and "kuzu://secret" not in r.text
    assert any(
        "kuzu://secret" in rec.getMessage() for rec in caplog.records
    )


def test_nav_explore_has_knowledge_graph(client):
    r = client.get("/")
    # The Knowledge-graph link is surfaced in the app-shell sidebar.
    assert 'href="/graph"' in r.text


# ---------------------------------------------------------------------------
# P4 (Explore): Shoelace + rubric migration of the graph overview fragment.
# ---------------------------------------------------------------------------

def _graph_present(**over):
    """A graph-present overview payload (with sensible defaults)."""
    payload = {
        "available": True,
        "paper_count": 10,
        "indexed": 8,
        "total": 10,
        "entity_count": 42,
        "by_type": {"Method": 5, "Protein": 3},
        "by_predicate": {"MENTIONS": 30, "EXTENDS": 4},
        "by_source": {"schema": 20, "biobert": 15, "llm_cloud": 7},
        "last_indexed": "2026-06-01",
    }
    payload.update(over)
    return payload


def test_graph_overview_uses_sl_details_with_open_entities(client, monkeypatch):
    """Entities-by-type is the ONE open sl-details; predicate/source collapsed."""
    monkeypatch.setattr(
        "lit_monitor.server.routes.graph._get_graph_overview", _graph_present
    )
    body = client.get("/graph/overview").text

    # Three collapsibles, each labelled with its count in the summary.
    assert 'summary="Entities by type (2)"' in body
    assert 'summary="Edges by predicate (2)"' in body
    assert 'summary="Mentions by source (3)"' in body

    # The entities sl-details carries `open`; the other two do NOT.
    import re

    details = re.findall(r"<sl-details[^>]*>", body)
    entities = [d for d in details if "Entities by type" in d]
    predicate = [d for d in details if "Edges by predicate" in d]
    source = [d for d in details if "Mentions by source" in d]
    assert entities and " open" in entities[0]
    assert predicate and " open" not in predicate[0]
    assert source and " open" not in source[0]


def test_graph_overview_tables_stay_html(client, monkeypatch):
    """Tables remain HTML `papers-table`s inside the sl-details (no web table)."""
    monkeypatch.setattr(
        "lit_monitor.server.routes.graph._get_graph_overview", _graph_present
    )
    body = client.get("/graph/overview").text
    assert '<table class="papers-table">' in body


def test_graph_absent_single_consolidated_empty_state(client, monkeypatch):
    """Graph-absent renders ONE consolidated empty-state card, not three
    per-table empty messages."""
    monkeypatch.setattr(
        "lit_monitor.server.routes.graph._get_graph_overview",
        lambda: {"available": False},
    )
    body = client.get("/graph/overview").text

    # The three OLD per-table empty strings must be gone.
    assert "No entities in the graph yet." not in body
    assert "No edges in the graph yet." not in body
    assert "No mention edges in the graph yet." not in body

    # Exactly one consolidated empty-state card, with the build hint.
    assert body.count("No knowledge graph yet") == 1
    assert "graph build" in body or "graph backfill" in body
