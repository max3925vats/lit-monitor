"""H10: POST /api/search free-text paper retrieval endpoint tests.

Covers:
- Pydantic validation: bad mode → 422, empty query → 422, k out-of-range → 422
- Mode pass-through: vector, graph, hybrid → correct mode arg forwarded
- Shared function: get_papers_by_query is the single callsite in the route
- Empty results: 200 + []
- Result shape: list[{doi, title, score}] forwarded unchanged from shared fn
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    """TestClient backed by a real create_app() with no real backends."""
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_bad_mode_returns_422(self, client: TestClient) -> None:
        """Pydantic Literal rejects unknown mode strings."""
        r = client.post("/api/search", json={"query": "antibody", "mode": "bogus", "k": 10})
        assert r.status_code in (400, 422)

    def test_empty_query_422(self, client: TestClient) -> None:
        """Empty string query fails validator."""
        r = client.post("/api/search", json={"query": "", "mode": "vector", "k": 10})
        assert r.status_code == 422

    def test_whitespace_only_query_422(self, client: TestClient) -> None:
        """Whitespace-only string query fails validator."""
        r = client.post("/api/search", json={"query": "   ", "mode": "vector", "k": 10})
        assert r.status_code == 422

    def test_k_too_large_422(self, client: TestClient) -> None:
        """k > 100 fails validator."""
        r = client.post("/api/search", json={"query": "x", "mode": "vector", "k": 500})
        assert r.status_code == 422

    def test_k_zero_422(self, client: TestClient) -> None:
        """k = 0 fails validator."""
        r = client.post("/api/search", json={"query": "x", "mode": "vector", "k": 0})
        assert r.status_code == 422

    def test_k_negative_422(self, client: TestClient) -> None:
        """k < 0 fails validator."""
        r = client.post("/api/search", json={"query": "x", "mode": "vector", "k": -1})
        assert r.status_code == 422

    def test_missing_query_422(self, client: TestClient) -> None:
        """Missing query field → Pydantic required-field error."""
        r = client.post("/api/search", json={"mode": "vector", "k": 10})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Mode pass-through tests
# ---------------------------------------------------------------------------

class TestModes:
    def test_vector_mode_calls_shared_function(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Vector mode: correct args forwarded; result shape returned unchanged."""
        captured: dict = {}

        def fake_gpbq(query: str, *, mode: str, k: int, **kw):  # type: ignore[override]
            captured.update({"query": query, "mode": mode, "k": k})
            return [{"doi": "10.1/a", "title": "Test Paper", "score": 0.9}]

        monkeypatch.setattr("lit_monitor.server.routes.search.get_papers_by_query", fake_gpbq)
        r = client.post("/api/search", json={"query": "antibody", "mode": "vector", "k": 5})
        assert r.status_code == 200
        assert captured["query"] == "antibody"
        assert captured["mode"] == "vector"
        assert captured["k"] == 5
        body = r.json()
        assert isinstance(body, list)
        assert body[0]["doi"] == "10.1/a"
        assert body[0]["title"] == "Test Paper"
        assert body[0]["score"] == pytest.approx(0.9)

    def test_graph_mode_passes_through(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Graph mode: mode='graph' forwarded to shared function."""
        captured: dict = {}

        def fake_gpbq(query: str, *, mode: str, k: int, **kw):  # type: ignore[override]
            captured["mode"] = mode
            return []

        monkeypatch.setattr("lit_monitor.server.routes.search.get_papers_by_query", fake_gpbq)
        client.post("/api/search", json={"query": "BRCA1", "mode": "graph", "k": 10})
        assert captured["mode"] == "graph"

    def test_hybrid_mode_passes_through(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hybrid mode: mode='hybrid' forwarded to shared function."""
        captured: dict = {}

        def fake_gpbq(query: str, *, mode: str, k: int, **kw):  # type: ignore[override]
            captured["mode"] = mode
            return []

        monkeypatch.setattr("lit_monitor.server.routes.search.get_papers_by_query", fake_gpbq)
        client.post("/api/search", json={"query": "BRCA1", "mode": "hybrid", "k": 10})
        assert captured["mode"] == "hybrid"

    def test_default_mode_is_vector(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default mode when omitted is 'vector' (SearchRequest default)."""
        captured: dict = {}

        def fake_gpbq(query: str, *, mode: str, k: int, **kw):  # type: ignore[override]
            captured["mode"] = mode
            return []

        monkeypatch.setattr("lit_monitor.server.routes.search.get_papers_by_query", fake_gpbq)
        client.post("/api/search", json={"query": "x"})
        assert captured["mode"] == "vector"

    def test_default_k_is_20(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default k when omitted is 20."""
        captured: dict = {}

        def fake_gpbq(query: str, *, mode: str, k: int, **kw):  # type: ignore[override]
            captured["k"] = k
            return []

        monkeypatch.setattr("lit_monitor.server.routes.search.get_papers_by_query", fake_gpbq)
        client.post("/api/search", json={"query": "x", "mode": "graph"})
        assert captured["k"] == 20


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------

class TestEmptyResults:
    def test_no_matches_returns_200_empty_list(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zero results: 200 + [] (not 404)."""
        monkeypatch.setattr(
            "lit_monitor.server.routes.search.get_papers_by_query",
            lambda q, *, mode, k, **kw: [],
        )
        r = client.post("/api/search", json={"query": "unicorn", "mode": "vector", "k": 10})
        assert r.status_code == 200
        assert r.json() == []


# ---------------------------------------------------------------------------
# k boundary values
# ---------------------------------------------------------------------------

class TestKBoundaries:
    def test_k_1_accepted(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """k=1 is the minimum valid value."""
        captured: dict = {}

        def fake_gpbq(query: str, *, mode: str, k: int, **kw):  # type: ignore[override]
            captured["k"] = k
            return []

        monkeypatch.setattr("lit_monitor.server.routes.search.get_papers_by_query", fake_gpbq)
        r = client.post("/api/search", json={"query": "x", "mode": "vector", "k": 1})
        assert r.status_code == 200
        assert captured["k"] == 1

    def test_k_100_accepted(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """k=100 is the maximum valid value."""
        captured: dict = {}

        def fake_gpbq(query: str, *, mode: str, k: int, **kw):  # type: ignore[override]
            captured["k"] = k
            return []

        monkeypatch.setattr("lit_monitor.server.routes.search.get_papers_by_query", fake_gpbq)
        r = client.post("/api/search", json={"query": "x", "mode": "vector", "k": 100})
        assert r.status_code == 200
        assert captured["k"] == 100
