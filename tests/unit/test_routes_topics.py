"""Bundle E (v0.9) + Audit-2 F5: GET /api/topics/{name}/expansion-suggestions.

Audit-2 F5 background
---------------------
``topics._safe_graph_db()`` reads ``get_runtime().graph_db``. Before the F5
fix ``ServerRuntime`` had no ``graph_db`` property, so ``getattr(rt,
"graph_db", None)`` ALWAYS returned None and the endpoint ALWAYS returned
``{"suggestions": []}`` regardless of graph state — a permanently dead route.

These tests pin the fixed contract:
- A runtime whose ``graph_db`` resolves to a graph with entities matching the
  topic → endpoint returns the real suggestions (non-empty).
- ``ServerRuntime`` actually exposes a ``graph_db`` attribute now (regression
  guard against the property being dropped again).
- The graph-unavailable path (graph_db is None) still returns empty.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app
from lit_monitor.server.runtime import ServerRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_db_with_entity_and_coentities(
    entity_cid: str | None,
    co_entities: list[tuple[str, int]],  # list of (canonical_id, co_count)
    *,
    top_k: int | None = None,
) -> MagicMock:
    """Fake graph_db whose ``_conn.execute`` drives suggest_expansions().

    Mirrors the fixture in tests/unit/test_query_expansion.py: the first
    execute() resolves the topic query to ``entity_cid``; the second returns
    the co-occurring entities (truncated to top_k to simulate Kuzu LIMIT $k).
    """
    graph_db = MagicMock()

    if entity_cid is None:
        entity_result = MagicMock()
        entity_result.has_next.return_value = False
        graph_db._conn.execute.return_value = entity_result
    else:
        entity_result = MagicMock()
        entity_result.has_next.side_effect = [True, False]
        entity_result.get_next.return_value = (entity_cid,)

        limited = co_entities[:top_k] if top_k is not None else co_entities

        co_result = MagicMock()
        co_result.has_next.side_effect = [True] * len(limited) + [False]
        co_result.get_next.side_effect = [(cid, cnt) for cid, cnt in limited]

        graph_db._conn.execute.side_effect = [entity_result, co_result]

    return graph_db


def _runtime_with_graph(graph_db) -> ServerRuntime:
    """A ServerRuntime whose ``graph_db`` property resolves to ``graph_db``.

    Pre-seeds the lazy cache + resolution sentinel so the real property
    returns our fake without touching kuzu or the config.
    """
    rt = ServerRuntime()
    rt._graph_db = graph_db
    rt._graph_db_resolved = True
    return rt


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# F5: ServerRuntime must expose graph_db (the dropped property)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRuntimeGraphDbProperty:
    def test_runtime_has_graph_db_attribute(self):
        """Regression: ServerRuntime must define a graph_db property.

        Before F5 this attribute did not exist, so the topics route's
        ``getattr(rt, "graph_db", None)`` silently returned None forever.
        """
        assert hasattr(ServerRuntime, "graph_db")

    def test_graph_db_resolves_via_safe_graph_db(self):
        """The property delegates to safe_graph_db() and caches the result."""
        fake = MagicMock(name="fake_graph_db")
        with patch(
            "lit_monitor.graph.import_citations.safe_graph_db", return_value=fake
        ) as m:
            rt = ServerRuntime()
            assert rt.graph_db is fake
            # Second access is cached — safe_graph_db not called again.
            assert rt.graph_db is fake
        assert m.call_count == 1

    def test_graph_db_caches_none(self):
        """A None result (no [graph] extra) is cached, not re-probed."""
        with patch(
            "lit_monitor.graph.import_citations.safe_graph_db", return_value=None
        ) as m:
            rt = ServerRuntime()
            assert rt.graph_db is None
            assert rt.graph_db is None
        assert m.call_count == 1


# ---------------------------------------------------------------------------
# F5: GET /api/topics/{name}/expansion-suggestions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExpansionSuggestions:
    def test_returns_real_suggestions_when_graph_has_matching_entities(self, client):
        """The core F5 fix: a runtime whose graph_db has entities matching the
        topic must yield real (non-empty) suggestions, not the old []."""
        graph_db = _make_graph_db_with_entity_and_coentities(
            "chromatography",
            [("protein_a", 12), ("cation_exchange", 8), ("hcp", 5)],
            top_k=3,
        )
        rt = _runtime_with_graph(graph_db)

        with patch("lit_monitor.server.routes.topics.get_runtime", return_value=rt):
            r = client.get("/api/topics/chromatography/expansion-suggestions?top_k=3")

        assert r.status_code == 200
        body = r.json()
        assert body["topic_name"] == "chromatography"
        # Not empty — this is exactly what the dead route could never return.
        assert body["suggestions"] == ["protein_a", "cation_exchange", "hcp"]
        assert body["top_k_requested"] == 3

    def test_respects_top_k(self, client):
        graph_db = _make_graph_db_with_entity_and_coentities(
            "filtration",
            [("tff", 9), ("nanofiltration", 7)],
            top_k=1,
        )
        rt = _runtime_with_graph(graph_db)

        with patch("lit_monitor.server.routes.topics.get_runtime", return_value=rt):
            r = client.get("/api/topics/filtration/expansion-suggestions?top_k=1")

        assert r.status_code == 200
        assert r.json()["suggestions"] == ["tff"]

    def test_empty_when_graph_unavailable(self, client):
        """graph_db None (no [graph] extra) → empty suggestions, 200."""
        rt = _runtime_with_graph(None)

        with patch("lit_monitor.server.routes.topics.get_runtime", return_value=rt):
            r = client.get("/api/topics/chromatography/expansion-suggestions")

        assert r.status_code == 200
        assert r.json()["suggestions"] == []

    def test_empty_when_no_matching_entity(self, client):
        """Graph present but no entity matches the topic → empty suggestions."""
        graph_db = _make_graph_db_with_entity_and_coentities(None, [])
        rt = _runtime_with_graph(graph_db)

        with patch("lit_monitor.server.routes.topics.get_runtime", return_value=rt):
            r = client.get("/api/topics/unknown-topic/expansion-suggestions")

        assert r.status_code == 200
        assert r.json()["suggestions"] == []

    def test_blank_topic_name_422(self, client):
        """Whitespace-only topic name → 422."""
        rt = _runtime_with_graph(MagicMock())
        with patch("lit_monitor.server.routes.topics.get_runtime", return_value=rt):
            r = client.get("/api/topics/%20/expansion-suggestions")
        assert r.status_code == 422
