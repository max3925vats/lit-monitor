"""P5: HTTP read endpoints for discovery runs.
P8: HTML /discovery index enhancements + /discovery/{run_id} detail page.

Tests cover:
  - GET /api/discovery/runs (list, pagination, validation)
  - GET /api/discovery/runs/{run_id} (detail + papers, 404)
  - GET /api/discovery/runs/{run_id}/papers (sorted, top_k validation)
  - Direct shared-query-layer functions from scripts.api.queries
  - P8: /discovery index with run history / latest-run section
  - P8: /discovery/{run_id} HTML detail page (200 + 404 + action buttons)
  - P8: route-ordering safety (/discovery/notify-handler not shadowed)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app
from lit_monitor.server.runtime import reset_runtime

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
    from lit_monitor.core.state_db import StateDB

    return StateDB(tmp_path / "state.db")


@pytest.fixture
def seeded_db(tmp_path):
    """Real StateDB seeded with one discovery run and two paper results.

    Returns (db, run_id) so callers can reference the run id.
    """
    from lit_monitor.core.state_db import StateDB

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
        from lit_monitor.api.queries import get_discovery_runs
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        result = get_discovery_runs(db, limit=20, offset=0)
        assert result == {"runs": [], "total": 0}

    def test_get_discovery_runs_returns_expected_keys(self, tmp_path):
        from lit_monitor.api.queries import get_discovery_runs
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.finish_discovery_run(rid, "success", 3, 2)
        result = get_discovery_runs(db, limit=20, offset=0)
        assert result["total"] == 1
        run = result["runs"][0]
        for key in ("id", "started_at", "finished_at", "status", "total_found", "total_ingested"):
            assert key in run, f"missing key {key!r}"

    def test_get_discovery_runs_with_data(self, tmp_path):
        from lit_monitor.api.queries import get_discovery_runs
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.finish_discovery_run(rid, "success", 3, 2)
        result = get_discovery_runs(db, limit=20, offset=0)
        assert result["runs"][0]["status"] == "success"
        assert result["runs"][0]["total_ingested"] == 2

    def test_get_discovery_run_not_found(self, tmp_path):
        from lit_monitor.api.queries import get_discovery_run
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        assert get_discovery_run(db, 99999) is None

    def test_get_discovery_run_found(self, tmp_path):
        from lit_monitor.api.queries import get_discovery_run
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({"topics": ["x"]})
        db.finish_discovery_run(rid, "success", 5, 3)
        run = get_discovery_run(db, rid)
        assert run is not None
        assert run["id"] == rid
        assert run["status"] == "success"

    def test_get_discovery_run_papers_sorted(self, tmp_path):
        from lit_monitor.api.queries import get_discovery_run_papers
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.add_discovery_paper(rid, doi="10/lo", title="Lo", score=0.1, rationale="", ingested=False)
        db.add_discovery_paper(rid, doi="10/hi", title="Hi", score=0.9, rationale="", ingested=True)
        papers = get_discovery_run_papers(db, rid, top_k=5)
        assert [p["doi"] for p in papers] == ["10/hi", "10/lo"]

    def test_get_discovery_run_papers_top_k(self, tmp_path):
        from lit_monitor.api.queries import get_discovery_run_papers
        from lit_monitor.core.state_db import StateDB

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

        from lit_monitor.api.queries import get_discovery_run_papers
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        rid = db.start_discovery_run({})
        db.add_discovery_paper(rid, doi="10/a", title="A", score=0.5, rationale="r", ingested=True)
        papers = get_discovery_run_papers(db, rid, top_k=10)
        assert json.dumps(papers) is not None


# ---------------------------------------------------------------------------
# P8: Additional fixtures for HTML page tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_db(empty_db):
    """TestClient with get_runtime() returning a real empty StateDB."""
    rt = _make_fake_runtime(empty_db)
    with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
        yield TestClient(create_app())


@pytest.fixture
def client_with_seeded_run(seeded_db):
    """TestClient with get_runtime() returning a seeded StateDB; also yields run_id."""
    db, run_id = seeded_db
    rt = _make_fake_runtime(db)
    with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
        yield TestClient(create_app()), run_id


# ---------------------------------------------------------------------------
# P8: GET /discovery (existing dashboard must survive + new sections)
# ---------------------------------------------------------------------------


class TestDiscoveryIndexEnhancements:
    def test_existing_dashboard_still_renders(self, client_with_db):
        """P8: existing /discovery dashboard must not be broken."""
        r = client_with_db.get("/discovery")
        assert r.status_code == 200
        body = r.text.lower()
        assert "discovery" in body

    def test_existing_controls_section_present(self, client_with_db):
        """SSE / controls section must still be in the page."""
        r = client_with_db.get("/discovery")
        assert r.status_code == 200
        body = r.text.lower()
        # The HTMX controls fragment loader and SSE pane live in index.html
        assert "hx-get" in body or "api/discovery/controls" in body or "controls" in body

    def test_run_history_links_to_run_detail(self, client_with_seeded_run):
        """Run history table must link rows to /discovery/{run_id}."""
        client, run_id = client_with_seeded_run
        r = client.get("/discovery")
        assert r.status_code == 200
        body = r.text
        assert f"/discovery/{run_id}" in body

    def test_latest_run_section_present_with_data(self, client_with_seeded_run):
        """P8: /discovery page should reference the seeded run_id somewhere."""
        client, run_id = client_with_seeded_run
        r = client.get("/discovery")
        assert r.status_code == 200
        body = r.text
        assert str(run_id) in body


# ---------------------------------------------------------------------------
# P8: GET /discovery/{run_id} — HTML detail page
# ---------------------------------------------------------------------------


class TestDiscoveryRunDetailPage:
    def test_known_run_200(self, client_with_seeded_run):
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200

    def test_page_contains_seeded_paper_title(self, client_with_seeded_run):
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        assert "Alpha" in r.text

    def test_page_contains_doi(self, client_with_seeded_run):
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        assert "10.0/a" in r.text

    def test_unknown_run_404(self, client_with_db):
        r = client_with_db.get("/discovery/99999")
        assert r.status_code == 404

    def test_relink_button_present(self, client_with_seeded_run):
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        assert "relink" in r.text.lower()

    def test_re_extract_button_present(self, client_with_seeded_run):
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        body = r.text.lower()
        assert "re-extract" in body or "reextract" in body

    def test_relink_targets_h7_endpoint(self, client_with_seeded_run):
        """Relink button must hx-post to /api/papers/{doi}/relink (literal DOI)."""
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        assert "/api/papers/10.0/a/relink" in r.text

    def test_re_extract_targets_h7_endpoint(self, client_with_seeded_run):
        """Re-extract button must hx-post to /api/papers/{doi}/re-extract (literal DOI)."""
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        assert "/api/papers/10.0/a/re-extract" in r.text

    def test_back_link_present(self, client_with_seeded_run):
        """Page must provide a link back to /discovery."""
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        assert "/discovery" in r.text

    def test_feedback_buttons_render_without_explicit_config(
        self, client_with_seeded_run
    ):
        """P4 Part A: feedback buttons appear by default (no web_ui config set).

        The run-detail page passes show_feedback_buttons into paper_card.html,
        and the route now defaults that flag True. With no explicit
        web_ui.show_feedback_buttons in config, the Save/Dismiss/thumbs row
        (an hx-post to /api/feedback) must render.
        """
        client, run_id = client_with_seeded_run
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        # The feedback row only renders when show_feedback_buttons is truthy.
        assert "/api/feedback" in r.text
        assert '"signal_type": "saved"' in r.text

    def test_feedback_buttons_suppressed_when_explicitly_disabled(
        self, client_with_seeded_run, monkeypatch
    ):
        """P4 Part A (M2): an explicit web_ui.show_feedback_buttons = false must
        still suppress the feedback row, even though the default is now True."""
        client, run_id = client_with_seeded_run
        monkeypatch.setattr(
            "scripts.server.config_io.load_config",
            lambda which: {"web_ui": {"show_feedback_buttons": False}},
        )
        r = client.get(f"/discovery/{run_id}")
        assert r.status_code == 200
        assert "/api/feedback" not in r.text


# ---------------------------------------------------------------------------
# P8: route-ordering safety
# ---------------------------------------------------------------------------


class TestDiscoveryLastRunKvSemantics:
    """AR-7: the 'Last run' key/value block is a <dl class="kv">, not a
    <th>-in-<tbody> table. Labels + the run_id value must survive verbatim."""

    def test_last_run_renders_as_definition_list(self, client):
        # The 'Last run' block is driven by get_recent_runs_by_type → last_run.
        fake_db = MagicMock()
        fake_db.get_recent_runs_by_type.return_value = [
            {"run_id": "abc123", "started_at": "2026-06-01", "status": "complete",
             "papers_processed": 5, "papers_skipped": 1, "papers_failed": 0,
             "finished_at": "2026-06-01 12:00:00"},
        ]
        fake_rt = MagicMock()
        fake_rt.state_db = fake_db
        fake_rt.config.obsidian.vault_path = ""
        with patch("scripts.server.routes.discovery.get_runtime", return_value=fake_rt):
            r = client.get("/discovery")
        assert r.status_code == 200
        assert '<dl class="kv">' in r.text
        assert "<dt>Run ID</dt>" in r.text
        assert "<dt>Status</dt>" in r.text
        assert "<dt>Papers processed</dt>" in r.text
        # The run_id value must survive the table→dl conversion verbatim.
        assert "abc123" in r.text


@pytest.fixture
def seeded_many_runs_db(tmp_path):
    """Real StateDB with 25 discovery_runs so the runs table spans >1 page (20)."""
    from lit_monitor.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    ids = []
    for i in range(25):
        rid = db.start_discovery_run({"topics": [f"t{i}"]})
        db.finish_discovery_run(rid, status="success", total_found=i, total_ingested=i)
        ids.append(rid)
    return db, ids


@pytest.fixture
def client_with_many_runs(seeded_many_runs_db):
    db, ids = seeded_many_runs_db
    rt = _make_fake_runtime(db)
    with patch("scripts.server.routes.discovery.get_runtime", return_value=rt):
        yield TestClient(create_app()), ids


class TestDiscoveryRunsPagination:
    """AR-7: corpus-style prev/next pagination on the discovery-runs table."""

    def test_page_one_has_next_no_prev(self, client_with_many_runs):
        client, ids = client_with_many_runs
        r = client.get("/discovery")
        assert r.status_code == 200
        body = r.text
        assert "runs_offset=20" in body       # Next → page 2
        assert "runs_offset=0" not in body     # no Prev on page 1
        assert "of 25" in body                 # total surfaced

    def test_page_two_renders_next_slice_and_prev(self, client_with_many_runs):
        client, ids = client_with_many_runs
        # Newest-first (id DESC): page 1 = ids[24..5], page 2 = ids[4..0].
        oldest_run = ids[0]
        # Match the exact table link (trailing quote) so /discovery/1 does not
        # spuriously match /discovery/19, /discovery/10, etc.
        oldest_link = f'/discovery/{oldest_run}"'
        # The oldest run's table link is on page 2 only (not page 1).
        page1 = client.get("/discovery").text
        assert oldest_link not in page1

        r = client.get("/discovery?runs_offset=20")
        assert r.status_code == 200
        body = r.text
        assert oldest_link in body
        # Last page: a Prev link, no Next.
        assert "runs_offset=0" in body          # Prev → page 1
        assert "runs_offset=40" not in body     # no Next past the end


class TestRouteOrderingSafety:
    def test_notify_handler_not_shadowed_by_run_detail(self, client_with_db):
        """P3 /discovery/notify-handler must still be reachable after P8 ships.

        FastAPI matches routes in registration order. If P8's
        /discovery/{run_id:int} is registered BEFORE P3's
        /discovery/notify-handler, FastAPI will try to coerce "notify-handler"
        to int, fail, and return 422 WITHOUT falling through to P3's handler.
        A 422 response here is the bug signal — not an acceptable pass.
        """
        # Pass ?run_id=1 so the notify-handler's own required query param is
        # satisfied. This ensures a 422 here can only come from route shadowing
        # (FastAPI tried to coerce "notify-handler" as the {run_id:int} path
        # param and failed) rather than the handler's own validation (missing
        # query param). Without ?run_id the handler itself returns 422 for a
        # different reason — making the two failure modes indistinguishable.
        r = client_with_db.get("/discovery/notify-handler?run_id=1")
        # 422 means P3's handler is shadowed by P8's {run_id:int} pattern —
        # that is exactly the regression R1 guards against. Not a pass.
        assert r.status_code != 422, (
            "P3 /discovery/notify-handler is shadowed by P8's "
            "/discovery/{run_id:int}. Ensure discovery_notify_router is "
            "registered BEFORE discovery_router in scripts/server/app.py."
        )
        # P3 handler returns 200 (chooser HTML), 302/307 (redirect to viewer),
        # or 204 (no-content for preferred_viewer='none'). 404 is not expected
        # here either but is a distinct failure mode from 422.
        assert r.status_code in (200, 204, 302, 303, 307, 308), (
            f"unexpected status {r.status_code} from /discovery/notify-handler"
        )
