"""H4: GET /api/papers/{doi} paper snapshot endpoint tests.

Covers:
- Known DOI → 200 + correct snapshot shape
- Unknown DOI (empty metadata) → 404
- Malformed DOI (fails regex) → 422
- Graph backend unavailable (safe_graph_db returns None) → 503
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GOOD_SNAPSHOT = {
    "metadata": {"doi": "10.1234/ok", "title": "Test Paper"},
    "entities_by_type": {"GENE": ["BRCA1"]},
    "relationships_in": [],
    "relationships_out": [{"rel": "cites", "target": "10.1234/other"}],
}

_EMPTY_SNAPSHOT = {
    "metadata": {},
    "entities_by_type": {},
    "relationships_in": [],
    "relationships_out": [],
}

# A minimal MagicMock that stands in for a real GraphDB instance.
_FAKE_GRAPH_DB = MagicMock(name="fake_graph_db")


@pytest.fixture()
def client() -> TestClient:
    """TestClient backed by a real create_app().

    Individual tests monkeypatch safe_graph_db and get_paper_snapshot on the
    papers route module so the real kuzu backend is never touched.
    """
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# H4: 200 — known DOI
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPaperSnapshot:
    def test_known_doi_200(self, client):
        """H4: known DOI → 200 with correct snapshot body."""
        import scripts.server.routes.papers as papers_route

        with (
            patch.object(papers_route, "safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch.object(papers_route, "get_paper_snapshot", return_value=_GOOD_SNAPSHOT),
        ):
            r = client.get("/api/papers/10.1234/ok")

        assert r.status_code == 200
        body = r.json()
        assert body["metadata"]["doi"] == "10.1234/ok"
        assert body["metadata"]["title"] == "Test Paper"

    def test_known_doi_shape_has_all_keys(self, client):
        """H4: response includes all four top-level snapshot keys."""
        import scripts.server.routes.papers as papers_route

        with (
            patch.object(papers_route, "safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch.object(papers_route, "get_paper_snapshot", return_value=_GOOD_SNAPSHOT),
        ):
            r = client.get("/api/papers/10.1234/shape")

        assert r.status_code == 200
        body = r.json()
        for key in ("metadata", "entities_by_type", "relationships_in", "relationships_out"):
            assert key in body, f"Expected key {key!r} missing from snapshot response"

    def test_doi_with_slashes_accepted(self, client):
        """H4: DOIs containing multiple slashes are captured by {doi:path}."""
        deep_snapshot = {
            "metadata": {"doi": "10.1016/j.cell.2024.01.001", "title": "Cell Paper"},
            "entities_by_type": {},
            "relationships_in": [],
            "relationships_out": [],
        }
        import scripts.server.routes.papers as papers_route

        with (
            patch.object(papers_route, "safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch.object(papers_route, "get_paper_snapshot", return_value=deep_snapshot),
        ):
            r = client.get("/api/papers/10.1016/j.cell.2024.01.001")

        assert r.status_code == 200


# ---------------------------------------------------------------------------
# H4: 404 — unknown DOI (empty metadata)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPaperSnapshotNotFound:
    def test_unknown_doi_404(self, client):
        """H4: get_paper_snapshot returns empty metadata → 404."""
        import scripts.server.routes.papers as papers_route

        with (
            patch.object(papers_route, "safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch.object(papers_route, "get_paper_snapshot", return_value=_EMPTY_SNAPSHOT),
        ):
            r = client.get("/api/papers/10.1234/missing")

        assert r.status_code == 404

    def test_unknown_doi_error_detail(self, client):
        """H4: 404 detail message references the DOI."""
        import scripts.server.routes.papers as papers_route

        with (
            patch.object(papers_route, "safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch.object(papers_route, "get_paper_snapshot", return_value=_EMPTY_SNAPSHOT),
        ):
            r = client.get("/api/papers/10.1234/ghost")

        assert r.status_code == 404
        assert "10.1234/ghost" in r.json()["detail"]


# ---------------------------------------------------------------------------
# H4: 422 — malformed DOI
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPaperSnapshotMalformedDOI:
    def test_non_doi_string_422(self, client):
        """H4: path segment that is not a DOI → 422 (regex guard fires before DB)."""
        r = client.get("/api/papers/not-a-doi")
        assert r.status_code == 422

    def test_doi_missing_10_prefix_422(self, client):
        """H4: DOI must start with '10.' followed by 4-9 digits."""
        r = client.get("/api/papers/20.1234/x")
        assert r.status_code == 422

    def test_doi_too_few_digits_422(self, client):
        """H4: '10.12/x' — only 2 digits after dot → 422."""
        r = client.get("/api/papers/10.12/x")
        assert r.status_code == 422

    def test_valid_doi_format_not_422(self, client):
        """H4: sanity check — well-formed DOI passes the regex guard (may 404 or 200)."""
        import scripts.server.routes.papers as papers_route

        with (
            patch.object(papers_route, "safe_graph_db", return_value=_FAKE_GRAPH_DB),
            patch.object(papers_route, "get_paper_snapshot", return_value=_EMPTY_SNAPSHOT),
        ):
            r = client.get("/api/papers/10.1234/valid")

        # Valid DOI → regex passes; empty snapshot → 404 (not 422)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# H4: 503 — graph backend unavailable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPaperSnapshotBackendUnavailable:
    def test_graph_db_none_503(self, client):
        """H4: safe_graph_db() returns None → 503 (backend not installed)."""
        import scripts.server.routes.papers as papers_route

        with patch.object(papers_route, "safe_graph_db", return_value=None):
            r = client.get("/api/papers/10.1234/x")

        assert r.status_code == 503

    def test_graph_db_none_does_not_call_snapshot(self, client):
        """H4: when graph is None, get_paper_snapshot must NOT be called."""
        import scripts.server.routes.papers as papers_route

        mock_snapshot = MagicMock(return_value=_GOOD_SNAPSHOT)
        with (
            patch.object(papers_route, "safe_graph_db", return_value=None),
            patch.object(papers_route, "get_paper_snapshot", mock_snapshot),
        ):
            client.get("/api/papers/10.1234/x")

        mock_snapshot.assert_not_called()


# ---------------------------------------------------------------------------
# H5: GET /api/papers/{doi}/related
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRelatedPapers:
    def test_defaults_to_vector_k20(self, client):
        """H5: default mode=vector, k=20 are forwarded to get_related_papers."""
        from unittest.mock import patch

        with patch(
            "scripts.server.routes.papers.get_related_papers",
            return_value=[{"doi": "10.1/r", "score": 0.9}],
        ) as m:
            r = client.get("/api/papers/10.1234/ok/related")

        assert r.status_code == 200
        assert isinstance(r.json(), list)
        # Confirm defaults passed through as keyword args.
        kwargs = m.call_args.kwargs
        assert kwargs.get("mode") == "vector"
        assert kwargs.get("k") == 20

    def test_mode_graph_passes_through(self, client):
        """H5: explicit mode=graph and k=5 forwarded correctly."""
        from unittest.mock import patch

        with patch(
            "scripts.server.routes.papers.get_related_papers",
            return_value=[],
        ) as m:
            r = client.get("/api/papers/10.1234/ok/related?mode=graph&k=5")

        assert r.status_code == 200
        assert m.call_args.kwargs["mode"] == "graph"
        assert m.call_args.kwargs["k"] == 5

    def test_bad_mode_422(self, client):
        """H5: unknown mode value → 422 (Pydantic Literal validation)."""
        r = client.get("/api/papers/10.1234/ok/related?mode=bogus")
        assert r.status_code == 422

    def test_k_too_large_422(self, client):
        """H5: k=200 exceeds upper bound of 100 → 422."""
        r = client.get("/api/papers/10.1234/ok/related?k=200")
        assert r.status_code == 422

    def test_k_too_small_422(self, client):
        """H5: k=0 below lower bound of 1 → 422."""
        r = client.get("/api/papers/10.1234/ok/related?k=0")
        assert r.status_code == 422

    def test_malformed_doi_422(self, client):
        """H5: non-DOI path segment before /related → 422."""
        r = client.get("/api/papers/not-a-doi/related")
        assert r.status_code == 422

    def test_unknown_doi_404_when_returns_none(self, client):
        """H5: get_related_papers returns None → 404."""
        from unittest.mock import patch

        with patch(
            "scripts.server.routes.papers.get_related_papers",
            return_value=None,
        ):
            r = client.get("/api/papers/10.1234/missing/related")

        assert r.status_code == 404
