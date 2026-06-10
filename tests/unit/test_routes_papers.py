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

    def test_related_endpoint_closes_handle(self, client, monkeypatch):
        """AR-2: the /related route must close its GraphDB handle on the happy path.

        The leak fix wraps get_related_papers in try/finally: graph_db.close().
        We hand the route a fake handle that counts close() calls and assert it
        was closed exactly once.
        """
        closed = {"n": 0}

        class _FakeDB:
            def close(self):
                closed["n"] += 1

        monkeypatch.setattr(
            "scripts.server.routes.papers.safe_graph_db",
            lambda *a, **k: _FakeDB(),
        )
        monkeypatch.setattr(
            "scripts.server.routes.papers.get_related_papers",
            lambda *a, **k: [{"doi": "10.1/x", "score": 1.0}],
        )
        r = client.get("/api/papers/10.1234/abc/related")
        assert r.status_code == 200
        assert closed["n"] == 1

    def test_related_endpoint_closes_handle_on_error(self, monkeypatch):
        """AR-2: the handle is closed even when get_related_papers raises.

        The real route has no 500-guard, so the exception propagates to FastAPI
        and surfaces as a 500.  We use raise_server_exceptions=False so the
        TestClient turns the propagated error into a response we can inspect,
        and assert the finally-clause still closed the handle.
        """
        closed = {"n": 0}

        class _FakeDB:
            def close(self):
                closed["n"] += 1

        monkeypatch.setattr(
            "scripts.server.routes.papers.safe_graph_db",
            lambda *a, **k: _FakeDB(),
        )

        def _boom(*a, **k):
            raise RuntimeError("kuzu boom")

        monkeypatch.setattr(
            "scripts.server.routes.papers.get_related_papers", _boom
        )
        # The route does not wrap the error in a 500-guard; the exception
        # propagates.  raise_server_exceptions=False lets us observe the 500
        # response instead of the exception bubbling out of the test client.
        local_client = TestClient(create_app(), raise_server_exceptions=False)
        r = local_client.get("/api/papers/10.1234/abc/related")
        assert closed["n"] == 1  # handle closed even on error
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# H7: POST /api/papers/{doi}/relink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRelink:
    def test_relink_happy_200(self, client, monkeypatch):
        """H7: valid DOI + known paper + successful tool → 200 with ok status."""
        from unittest.mock import patch

        # _doi_exists must return True for the route to invoke the tool
        monkeypatch.setattr("scripts.server.routes.papers._doi_exists", lambda doi: True)
        with patch(
            "scripts.server.routes.papers._invoke_relink",
            return_value={"relinked": 3, "added": 2},
        ) as m:
            r = client.post("/api/papers/10.1234/ok/relink")

        assert r.status_code == 200
        body = r.json()
        assert body["doi"] == "10.1234/ok"
        assert body["status"] == "ok"
        assert body["summary"]["relinked"] == 3
        m.assert_called_once_with("10.1234/ok")

    def test_relink_unknown_doi_404(self, client, monkeypatch):
        """H7: DOI not found in state.db → 404."""
        monkeypatch.setattr("scripts.server.routes.papers._doi_exists", lambda doi: False)
        r = client.post("/api/papers/10.1234/missing/relink")
        assert r.status_code == 404

    def test_relink_malformed_doi_422(self, client):
        """H7: path segment that is not a DOI → 422 (regex guard fires first)."""
        r = client.post("/api/papers/not-a-doi/relink")
        assert r.status_code == 422

    def test_relink_tool_exception_returns_generic_500(self, client, monkeypatch, caplog):
        """P2.2: tool raises → 500 with a GENERIC detail; no exception text or
        filesystem path leaks to the client, and the failure is logged.
        """
        import logging
        from unittest.mock import patch

        monkeypatch.setattr("scripts.server.routes.papers._doi_exists", lambda doi: True)
        # FileNotFoundError stringifies to include the offending path — exactly
        # the kind of detail that must never reach the client.
        secret_path = "/Users/secret/vault/notes/10.1234-ok.md"
        with caplog.at_level(logging.ERROR, logger="scripts.server.routes.papers"):
            with patch(
                "scripts.server.routes.papers._invoke_relink",
                side_effect=FileNotFoundError(secret_path),
            ):
                r = client.post("/api/papers/10.1234/ok/relink")

        assert r.status_code == 500
        body = r.json()
        assert body["status"] == "error"
        assert body["detail"] == "Internal error"
        # P2.2 info-leak guard: no exception text, path, or traceback in the body.
        serialized = str(body)
        assert secret_path not in serialized
        assert "FileNotFoundError" not in serialized
        assert "Traceback" not in serialized
        assert "File " not in serialized
        # The real error must still be logged server-side (with the path).
        assert any(secret_path in rec.getMessage() or rec.exc_info for rec in caplog.records)


# ---------------------------------------------------------------------------
# H7: POST /api/papers/{doi}/re-extract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReExtract:
    def test_re_extract_happy_200(self, client, monkeypatch):
        """H7: valid DOI + known paper + successful tool → 200 with ok status."""
        from unittest.mock import patch

        monkeypatch.setattr("scripts.server.routes.papers._doi_exists", lambda doi: True)
        with patch(
            "scripts.server.routes.papers._invoke_re_extract",
            return_value={"phase1": "ok", "phase2": "ok"},
        ):
            r = client.post("/api/papers/10.1234/ok/re-extract")

        assert r.status_code == 200
        body = r.json()
        assert body["doi"] == "10.1234/ok"
        assert body["status"] == "ok"

    def test_re_extract_unknown_404(self, client, monkeypatch):
        """H7: DOI not found in state.db → 404."""
        monkeypatch.setattr("scripts.server.routes.papers._doi_exists", lambda doi: False)
        r = client.post("/api/papers/10.1234/missing/re-extract")
        assert r.status_code == 404

    def test_re_extract_malformed_doi_422(self, client):
        """H7: path segment that is not a DOI → 422."""
        r = client.post("/api/papers/not-a-doi/re-extract")
        assert r.status_code == 422

    def test_re_extract_tool_error_generic_500(self, client, monkeypatch, caplog):
        """P2.2: tool raises → 500 with a GENERIC detail; no exception text
        leaks to the client, and the failure is logged server-side.
        """
        import logging
        from unittest.mock import patch

        monkeypatch.setattr("scripts.server.routes.papers._doi_exists", lambda doi: True)
        with caplog.at_level(logging.ERROR, logger="scripts.server.routes.papers"):
            with patch(
                "scripts.server.routes.papers._invoke_re_extract",
                side_effect=RuntimeError("LLM timeout at /tmp/llm/cache.db"),
            ):
                r = client.post("/api/papers/10.1234/ok/re-extract")

        assert r.status_code == 500
        body = r.json()
        assert body["status"] == "error"
        assert body["detail"] == "Internal error"
        # P2.2 info-leak guard: exception text / path must not reach the client.
        serialized = str(body)
        assert "LLM timeout" not in serialized
        assert "/tmp/llm/cache.db" not in serialized
        assert "Traceback" not in serialized
        # Logged server-side.
        assert any(rec.exc_info for rec in caplog.records)


# ---------------------------------------------------------------------------
# Bundle B: GET /api/papers/{doi}/score-breakdown
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_db(tmp_path) -> TestClient:
    """TestClient backed by a real create_app() with a real (empty) StateDB."""
    from scripts.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")

    # Monkeypatch _get_score_breakdown_db so the route uses our temp DB
    import scripts.server.routes.papers as _papers_mod
    _orig = getattr(_papers_mod, "_get_score_breakdown_db", None)

    app = create_app()
    client = TestClient(app)
    # Patch the helper used by the endpoint to return our StateDB
    with patch.object(_papers_mod, "_get_score_breakdown_db", return_value=db):
        yield client


@pytest.fixture()
def client_with_seeded_run(tmp_path) -> TestClient:
    """TestClient with a StateDB that has one run + one paper with score_breakdown."""

    import scripts.server.routes.papers as _papers_mod
    from scripts.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    run_id = db.start_discovery_run(run_params=None)
    db.add_discovery_paper(
        run_id=run_id,
        doi="10.1234/seeded",
        title="Seeded Paper",
        score=0.85,
        rationale="Important.",
        ingested=False,
        score_breakdown={"vector": 0.7, "domain_context": 0.15},
    )

    app = create_app()
    client = TestClient(app)
    with patch.object(_papers_mod, "_get_score_breakdown_db", return_value=db):
        yield client


@pytest.mark.unit
class TestScoreBreakdownEndpoint:
    """GET /api/papers/{doi}/score-breakdown — Bundle B HTTP endpoint."""

    def test_returns_200_for_seeded_paper(self, client_with_seeded_run):
        """Known DOI with stored breakdown → 200."""
        r = client_with_seeded_run.get("/api/papers/10.1234/seeded/score-breakdown")
        assert r.status_code == 200

    def test_response_has_expected_keys(self, client_with_seeded_run):
        """Response body has doi, run_id, breakdown, computed_at keys."""
        r = client_with_seeded_run.get("/api/papers/10.1234/seeded/score-breakdown")
        body = r.json()
        for key in ("doi", "run_id", "breakdown"):
            assert key in body, f"Expected key {key!r} missing from response"

    def test_doi_echoed_back(self, client_with_seeded_run):
        """doi in response matches the requested DOI."""
        r = client_with_seeded_run.get("/api/papers/10.1234/seeded/score-breakdown")
        assert r.json()["doi"] == "10.1234/seeded"

    def test_breakdown_values_numeric(self, client_with_seeded_run):
        """breakdown values are numeric (int or float)."""
        r = client_with_seeded_run.get("/api/papers/10.1234/seeded/score-breakdown")
        breakdown = r.json()["breakdown"]
        for key, val in breakdown.items():
            assert isinstance(val, (int, float)), (
                f"breakdown[{key!r}] = {val!r} is not numeric"
            )

    def test_breakdown_has_vector_and_domain_context(self, client_with_seeded_run):
        """breakdown contains both vector and domain_context signals."""
        r = client_with_seeded_run.get("/api/papers/10.1234/seeded/score-breakdown")
        breakdown = r.json()["breakdown"]
        assert "vector" in breakdown
        assert "domain_context" in breakdown

    def test_404_for_unknown_doi(self, client_with_db):
        """Unknown DOI → 404."""
        r = client_with_db.get("/api/papers/10.9999/missing/score-breakdown")
        assert r.status_code == 404

    def test_404_detail_mentions_doi(self, client_with_db):
        """404 detail message references the DOI."""
        r = client_with_db.get("/api/papers/10.9999/ghost/score-breakdown")
        assert r.status_code == 404
        assert "10.9999/ghost" in r.json()["detail"]

    def test_response_is_valid_json(self, client_with_seeded_run):
        """Response Content-Type is JSON and body is parseable."""
        r = client_with_seeded_run.get("/api/papers/10.1234/seeded/score-breakdown")
        assert r.status_code == 200
        # json() would have raised if not parseable; double-check content-type
        assert "application/json" in r.headers.get("content-type", "")

    def test_malformed_doi_422(self, client_with_db):
        """Non-DOI path → 422 (regex guard)."""
        r = client_with_db.get("/api/papers/not-a-doi/score-breakdown")
        assert r.status_code == 422
