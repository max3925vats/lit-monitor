"""P5: HTTP read endpoints for discovery runs.

Tests cover:
  - GET /api/discovery/runs (list, pagination, validation)
  - GET /api/discovery/runs/{run_id} (detail + papers, 404)
  - GET /api/discovery/runs/{run_id}/papers (sorted, top_k validation)
  - Direct shared-query-layer functions from scripts.api.queries
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app
from scripts.server.runtime import reset_runtime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


def _make_fake_runtime(state_db) -> MagicMock:
    """Return a MagicMock runtime whose .state_db is the supplied real StateDB."""
    rt = MagicMock()
    rt.state_db = state_db
    return rt


@pytest.fixture
def empty_db(tmp_path):
    """Real StateDB with no data."""
    from scripts.core.state_db import StateDB

    return StateDB(tmp_path / "state.db")


@pytest.fixture
def seeded_db(tmp_path):
    """Real StateDB seeded with one discovery run and two paper results.

    Returns (db, run_id) so callers can reference the run id.
    """
    from scripts.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    run_id = db.start_discovery_run({"topics": ["x"]})
    db.add_discovery_paper(
        run_id, doi="10.0/a", title="Alpha", score=0.9, rationale="r1", ingested=True
    )
    db.add_discovery_paper(
        run_id, doi="10.0/b", title="Beta", score=0.7, rationale="r2", ingested=False
    )
    db.finish_discovery_run(run_id, status="success", total_found=2, total_ingested=1)
    return db, run_id


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# GET /api/discovery/runs  (list)
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_returns_zero_total(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs")
        assert r.status_code == 200
        body = r.json()
        assert body == {"runs": [], "total": 0}

    def test_with_data_returns_run(self, client, seeded_db):
        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs?limit=5")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert len(body["runs"]) == 1
        assert body["runs"][0]["id"] == run_id
        assert body["runs"][0]["status"] == "success"

    def test_run_has_expected_keys(self, client, seeded_db):
        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs")
        run = r.json()["runs"][0]
        for key in ("id", "started_at", "finished_at", "status", "total_found", "total_ingested"):
            assert key in run, f"key {key!r} missing from run dict"

    def test_limit_overflow_422(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs?limit=500")
        assert r.status_code == 422

    def test_limit_zero_422(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs?limit=0")
        assert r.status_code == 422

    def test_offset_negative_422(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs?offset=-1")
        assert r.status_code == 422

    def test_default_limit_accepted(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs")
        assert r.status_code == 200

    def test_limit_100_accepted(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs?limit=100")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/discovery/runs/{run_id}  (detail)
# ---------------------------------------------------------------------------


class TestRunDetail:
    def test_known_run_200(self, client, seeded_db):
        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get(f"/api/discovery/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == run_id
        assert body["status"] == "success"
        assert "papers" in body
        assert len(body["papers"]) == 2

    def test_papers_sorted_by_score_desc_in_detail(self, client, seeded_db):
        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get(f"/api/discovery/runs/{run_id}")
        papers = r.json()["papers"]
        scores = [p["score"] for p in papers]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_returns_404(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs/99999")
        assert r.status_code == 404

    def test_detail_json_serializable(self, client, seeded_db):
        """Ensure the response has no non-JSON types (no Row tuples, etc.)."""
        import json

        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get(f"/api/discovery/runs/{run_id}")
        # TestClient already parses via json, but this confirms no TypeError
        assert json.dumps(r.json()) is not None


# ---------------------------------------------------------------------------
# GET /api/discovery/runs/{run_id}/papers  (papers-only endpoint)
# ---------------------------------------------------------------------------


class TestRunPapers:
    def test_papers_sorted_by_score(self, client, seeded_db):
        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get(f"/api/discovery/runs/{run_id}/papers?top_k=5")
        assert r.status_code == 200
        papers = r.json()["papers"]
        assert len(papers) == 2
        assert papers[0]["doi"] == "10.0/a"  # score 0.9
        assert papers[1]["doi"] == "10.0/b"  # score 0.7

    def test_top_k_overflow_422(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs/1/papers?top_k=500")
        assert r.status_code == 422

    def test_top_k_zero_422(self, client, empty_db):
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs/1/papers?top_k=0")
        assert r.status_code == 422

    def test_top_k_100_accepted(self, client, seeded_db):
        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get(f"/api/discovery/runs/{run_id}/papers?top_k=100")
        assert r.status_code == 200

    def test_top_k_limits_results(self, client, seeded_db):
        """top_k=1 should return only the highest-scored paper."""
        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get(f"/api/discovery/runs/{run_id}/papers?top_k=1")
        papers = r.json()["papers"]
        assert len(papers) == 1
        assert papers[0]["doi"] == "10.0/a"  # highest score

    def test_unknown_run_returns_empty_papers(self, client, empty_db):
        """Papers endpoint for an unknown run_id returns empty list (not 404)."""
        rt = _make_fake_runtime(empty_db)
        with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
            r = client.get("/api/discovery/runs/99999/papers")
        assert r.status_code == 200
        assert r.json()["papers"] == []


# ---------------------------------------------------------------------------
# Direct shared-query-layer tests (no HTTP)
# ---------------------------------------------------------------------------


class TestQueriesShared:
    def test_get_discovery_runs_empty(self, tmp_path):
        from scripts.api.queries import get_discovery_runs
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        result = get_discovery_runs(db, limit=20, offset=0)
        assert result == {"runs": [], "total": 0}

    def test_get_discovery_runs_returns_expected_keys(self, tmp_path):
        from scripts.api.queries import get_discovery_runs
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.finish_discovery_run(rid, "success", 3, 2)
        result = get_discovery_runs(db, limit=20, offset=0)
        assert result["total"] == 1
        run = result["runs"][0]
        for key in ("id", "started_at", "finished_at", "status", "total_found", "total_ingested"):
            assert key in run, f"missing key {key!r}"

    def test_get_discovery_runs_with_data(self, tmp_path):
        from scripts.api.queries import get_discovery_runs
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.finish_discovery_run(rid, "success", 3, 2)
        result = get_discovery_runs(db, limit=20, offset=0)
        assert result["runs"][0]["status"] == "success"
        assert result["runs"][0]["total_ingested"] == 2

    def test_get_discovery_run_not_found(self, tmp_path):
        from scripts.api.queries import get_discovery_run
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        assert get_discovery_run(db, 99999) is None

    def test_get_discovery_run_found(self, tmp_path):
        from scripts.api.queries import get_discovery_run
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({"topics": ["x"]})
        db.finish_discovery_run(rid, "success", 5, 3)
        run = get_discovery_run(db, rid)
        assert run is not None
        assert run["id"] == rid
        assert run["status"] == "success"

    def test_get_discovery_run_papers_sorted(self, tmp_path):
        from scripts.api.queries import get_discovery_run_papers
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.add_discovery_paper(rid, doi="10/lo", title="Lo", score=0.1, rationale="", ingested=False)
        db.add_discovery_paper(rid, doi="10/hi", title="Hi", score=0.9, rationale="", ingested=True)
        papers = get_discovery_run_papers(db, rid, top_k=5)
        assert [p["doi"] for p in papers] == ["10/hi", "10/lo"]

    def test_get_discovery_run_papers_top_k(self, tmp_path):
        from scripts.api.queries import get_discovery_run_papers
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        for i in range(5):
            db.add_discovery_paper(
                rid, doi=f"10/p{i}", title=f"P{i}", score=float(i) / 10,
                rationale="", ingested=False,
            )
        papers = get_discovery_run_papers(db, rid, top_k=3)
        assert len(papers) == 3
        # Highest scores are 0.4, 0.3, 0.2
        assert papers[0]["score"] > papers[1]["score"] > papers[2]["score"]

    def test_get_discovery_run_papers_json_serializable(self, tmp_path):
        """All values in paper dicts must be JSON-native (no Row tuples)."""
        import json

        from scripts.api.queries import get_discovery_run_papers
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.add_discovery_paper(rid, doi="10/a", title="A", score=0.5, rationale="r", ingested=True)
        papers = get_discovery_run_papers(db, rid, top_k=10)
        assert json.dumps(papers) is not None
