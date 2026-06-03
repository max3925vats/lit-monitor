"""H6: GET /api/entities — entity neighborhood + listing endpoint tests.

Covers:
- Known entity → 200 with correct neighborhood shape.
- Unknown entity (empty papers + co_entities) → 404.
- Listing with type filter → 200.
- Bad type value → 422 (Pydantic Literal validation).
- top_k overflow → 422.
- Missing type parameter → 422.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_GRAPH_DB = MagicMock(name="fake_graph_db")


@pytest.fixture()
def client() -> TestClient:
    """TestClient backed by a real create_app().

    Individual tests monkeypatch safe_graph_db and query functions on the
    entities route module so the real kuzu backend is never touched.
    """
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# H6: GET /api/entities/{canonical_id:path} — entity neighborhood
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityNeighborhood:
    def test_known_entity_200(self, client):
        """H6: known entity returns 200 with neighborhood shape."""
        hood = {
            "canonical_id": "crispr",
            "papers": [{"doi": "10.1/a", "title": "Paper A", "year": 2022}],
            "co_entities": [],
        }
        with (
            patch("scripts.server.routes.entities.safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch("scripts.server.routes.entities.get_entity_neighborhood", return_value=hood),
        ):
            r = client.get("/api/entities/crispr")

        assert r.status_code == 200
        assert r.json()["canonical_id"] == "crispr"
        assert isinstance(r.json()["papers"], list)

    def test_known_entity_with_slashes(self, client):
        """H6: canonical_id containing slashes captured by {canonical_id:path}."""
        hood = {
            "canonical_id": "author/smith-j",
            "papers": [{"doi": "10.1/b", "title": "B", "year": 2021}],
            "co_entities": [{"canonical_id": "rna", "type": "material", "co_count": 3}],
        }
        with (
            patch("scripts.server.routes.entities.safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch("scripts.server.routes.entities.get_entity_neighborhood", return_value=hood),
        ):
            r = client.get("/api/entities/author/smith-j")

        assert r.status_code == 200

    def test_unknown_entity_404(self, client):
        """H6: empty papers AND co_entities → 404."""
        empty = {"canonical_id": "none", "papers": [], "co_entities": []}
        with (
            patch("scripts.server.routes.entities.safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch("scripts.server.routes.entities.get_entity_neighborhood", return_value=empty),
        ):
            r = client.get("/api/entities/none")

        assert r.status_code == 404

    def test_graph_unavailable_503(self, client):
        """H6: safe_graph_db returns None → 503."""
        with patch("scripts.server.routes.entities.safe_graph_db", return_value=None):
            r = client.get("/api/entities/crispr")

        assert r.status_code == 503


# ---------------------------------------------------------------------------
# H6: GET /api/entities — entity listing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityListing:
    def test_list_200(self, client):
        """H6: valid type + top_k → 200 with list response."""
        with (
            patch("scripts.server.routes.entities.safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch("scripts.server.routes.entities.list_entities", return_value=[]),
        ):
            r = client.get("/api/entities?type=method&top_k=5")

        assert r.status_code == 200
        assert r.json() == []

    def test_list_passes_type_and_top_k(self, client):
        """H6: type and top_k are forwarded as kwargs to list_entities."""
        with (
            patch("scripts.server.routes.entities.safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch("scripts.server.routes.entities.list_entities", return_value=[]) as m,
        ):
            client.get("/api/entities?type=topic&top_k=10")

        # Route calls list_entities(type, top_k, graph_db) positionally — assert the
        # query params are actually forwarded, not merely that the mock was called.
        m.assert_called_once()
        call_args = m.call_args
        assert call_args.args[0] == "topic", (
            f"entity type not forwarded; got {call_args.args[0]!r}"
        )
        assert call_args.args[1] == 10, (
            f"top_k not forwarded; got {call_args.args[1]!r}"
        )

    def test_bad_type_422(self, client):
        """H6: unknown type value → 422 (Pydantic Literal validation)."""
        r = client.get("/api/entities?type=bogus")
        assert r.status_code == 422

    def test_top_k_overflow_422(self, client):
        """H6: top_k=500 exceeds upper bound of 200 → 422."""
        r = client.get("/api/entities?type=method&top_k=500")
        assert r.status_code == 422

    def test_top_k_zero_422(self, client):
        """H6: top_k=0 below lower bound of 1 → 422."""
        r = client.get("/api/entities?type=method&top_k=0")
        assert r.status_code == 422

    def test_missing_type_422(self, client):
        """H6: missing required type parameter → 422."""
        r = client.get("/api/entities")
        assert r.status_code == 422

    def test_graph_unavailable_503(self, client):
        """H6: safe_graph_db returns None → 503 for listing route."""
        with patch("scripts.server.routes.entities.safe_graph_db", return_value=None):
            r = client.get("/api/entities?type=method")

        assert r.status_code == 503
