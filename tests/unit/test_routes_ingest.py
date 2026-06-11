"""H2: POST /api/ingest endpoint tests.

Covers:
- Happy path: 202 + {status: queued, paper_id: doi}
- Bad DOI: 422 (Pydantic validator)
- Missing title: 422
- Duplicate DOI: 409
- R28 hardening: _process_paper failure → 500 + queue row marked 'failed', NOT 'queued'
- StateDB migration: ingest_queue table created additively
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from lit_monitor.core.state_db import StateDB
from lit_monitor.server.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_runtime(state_db: StateDB) -> MagicMock:
    """Return a minimal MagicMock runtime that serves the given StateDB."""
    runtime = MagicMock()
    runtime.state_db = state_db
    return runtime


@pytest.fixture()
def db(tmp_path) -> StateDB:
    """Isolated StateDB in tmp_path for each test."""
    return StateDB(tmp_path / "state.db")


@pytest.fixture()
def client(db, monkeypatch) -> TestClient:
    """TestClient with runtime.state_db wired to an isolated StateDB.

    Monkeypatches get_runtime on the ingest route module (where it's
    imported) — same pattern as test_brain_build_dashboard.py.
    """
    from lit_monitor.server.routes import ingest as ingest_route

    fake_runtime = _make_fake_runtime(db)
    monkeypatch.setattr(ingest_route, "get_runtime", lambda: fake_runtime)
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHappyPath:
    def test_returns_202_and_queued_status(self, client, db, monkeypatch):
        """H2: valid request returns 202 with status=queued and paper_id=doi."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        r = client.post(
            "/api/ingest",
            json={"doi": "10.1234/test.001", "title": "Test Paper"},
        )
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "queued"
        assert body["paper_id"] == "10.1234/test.001"

    def test_queue_row_marked_done_after_success(self, client, db, monkeypatch):
        """H2: after _process_paper succeeds, ingest_queue row has status='done'."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        client.post(
            "/api/ingest",
            json={"doi": "10.1234/done.001", "title": "Done Paper"},
        )

        with db._connect() as conn:
            row = conn.execute(
                "SELECT status, completed_at FROM ingest_queue WHERE doi = ?",
                ("10.1234/done.001",),
            ).fetchone()

        assert row is not None
        assert row[0] == "done"
        assert row[1] is not None  # completed_at was stamped

    def test_optional_fields_accepted(self, client, db, monkeypatch):
        """H2: optional fields (authors, year, abstract, zotero_key) are accepted."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        r = client.post(
            "/api/ingest",
            json={
                "doi": "10.1234/full.001",
                "title": "Full Paper",
                "authors": ["Alice", "Bob"],
                "year": 2024,
                "abstract": "An abstract.",
                "zotero_key": "ZOT123",
            },
        )
        assert r.status_code == 202


# ---------------------------------------------------------------------------
# Validation (422)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    def test_bad_doi_returns_422(self, client):
        """H2: malformed DOI (no 10. prefix) → 422."""
        r = client.post("/api/ingest", json={"doi": "not-a-doi", "title": "T"})
        assert r.status_code == 422

    def test_doi_without_10_prefix_returns_422(self, client):
        """H2: DOI must start with '10.' → anything else is 422."""
        r = client.post("/api/ingest", json={"doi": "12.1234/x", "title": "T"})
        assert r.status_code == 422

    def test_missing_title_returns_422(self, client):
        """H2: title is required — missing key → 422."""
        r = client.post("/api/ingest", json={"doi": "10.1234/x"})
        assert r.status_code == 422

    def test_empty_title_returns_422(self, client):
        """H2: empty-string title → 422 (validator rejects blank)."""
        r = client.post("/api/ingest", json={"doi": "10.1234/x", "title": ""})
        assert r.status_code == 422

    def test_whitespace_only_title_returns_422(self, client):
        """H2: whitespace-only title → 422."""
        r = client.post("/api/ingest", json={"doi": "10.1234/x", "title": "   "})
        assert r.status_code == 422

    def test_valid_doi_formats_accepted(self, client, monkeypatch):
        """H2: DOIs with long suffixes and slashes are accepted."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        r = client.post(
            "/api/ingest",
            json={"doi": "10.1016/j.biomaterials.2006.09.036", "title": "T"},
        )
        assert r.status_code == 202

    def test_oversized_title_returns_422(self, client):
        """P2.3: a title longer than the cap (500) → 422 (validator rejects)."""
        from lit_monitor.server.routes.ingest import _MAX_TITLE_LEN

        oversized = "x" * (_MAX_TITLE_LEN + 100)  # 600 chars
        r = client.post(
            "/api/ingest", json={"doi": "10.1234/x", "title": oversized}
        )
        assert r.status_code == 422

    def test_normal_length_title_accepted(self, client, monkeypatch):
        """P2.3: a normal-length title still passes (regression guard)."""
        from lit_monitor.server.routes import ingest as ingest_route
        from lit_monitor.server.routes.ingest import _MAX_TITLE_LEN

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        # A long-but-legitimate title right at the cap must still be accepted.
        at_cap = "A" * _MAX_TITLE_LEN
        r = client.post(
            "/api/ingest", json={"doi": "10.1234/at-cap", "title": at_cap}
        )
        assert r.status_code == 202

    def test_oversized_doi_returns_422(self, client):
        """Q3.5: a DOI longer than the cap (255) → 422 (validator rejects).

        _DOI_RE's \\S+ suffix is unbounded, so without the length cap this
        regex-valid but absurdly long DOI would slip through.
        """
        from lit_monitor.server.routes.ingest import _MAX_DOI_LEN

        # Regex-valid prefix + an oversized suffix pushing past the cap.
        oversized = "10.1234/" + ("x" * _MAX_DOI_LEN)
        r = client.post(
            "/api/ingest", json={"doi": oversized, "title": "T"}
        )
        assert r.status_code == 422

    def test_normal_doi_accepted(self, client, monkeypatch):
        """Q3.5: a normal-length DOI still passes (regression guard)."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        r = client.post(
            "/api/ingest", json={"doi": "10.1234/normal.doi", "title": "T"}
        )
        assert r.status_code == 202


# ---------------------------------------------------------------------------
# Duplicate DOI (409)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDuplicateDOI:
    def test_duplicate_doi_returns_409(self, client, db, monkeypatch):
        """H2: second POST with same DOI returns 409."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        # First ingest — should succeed.
        r1 = client.post(
            "/api/ingest",
            json={"doi": "10.1234/dup.001", "title": "T"},
        )
        assert r1.status_code == 202

        # Second ingest with the same DOI — must 409.
        r2 = client.post(
            "/api/ingest",
            json={"doi": "10.1234/dup.001", "title": "T"},
        )
        assert r2.status_code == 409

    def test_duplicate_response_body(self, client, db, monkeypatch):
        """H2: 409 body contains status=duplicate and the paper_id."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        client.post("/api/ingest", json={"doi": "10.1234/dup.002", "title": "T"})
        r2 = client.post(
            "/api/ingest",
            json={"doi": "10.1234/dup.002", "title": "T"},
        )
        body = r2.json()
        assert body.get("paper_id") == "10.1234/dup.002"

    def test_concurrent_insert_race_returns_409_not_500(self, client, db, monkeypatch):
        """A3-6: a same-DOI row appearing BETWEEN the SELECT and the INSERT must
        yield a 409 (duplicate), not a 500 IntegrityError.

        Deterministic race simulation: wrap state_db._connect so that just
        before the route's INSERT connection is opened, a 'concurrent' request
        inserts the queue row. The route's own INSERT then hits the PK/UNIQUE
        constraint — which must be caught and mapped to the same 409 the SELECT
        path returns, NOT propagated as a 500.
        """
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))

        doi = "10.1234/race.001"
        real_connect = db._connect
        call_count = {"n": 0}

        def racing_connect(*a, **kw):
            # The route opens _connect() twice: 1st for the SELECT, 2nd for the
            # INSERT. Before the 2nd open, slip in the conflicting row via a
            # fresh real connection — exactly the "another request won" race.
            call_count["n"] += 1
            if call_count["n"] == 2:
                with real_connect() as conn:
                    conn.execute(
                        "INSERT INTO ingest_queue (doi, status, queued_at) "
                        "VALUES (?, ?, ?)",
                        (doi, "queued", "2020-01-01T00:00:00+00:00"),
                    )
            return real_connect(*a, **kw)

        monkeypatch.setattr(db, "_connect", racing_connect)

        r = client.post("/api/ingest", json={"doi": doi, "title": "T"})

        assert r.status_code == 409, (
            f"Concurrent INSERT race must return 409, got {r.status_code}: {r.text}"
        )
        body = r.json()
        # Same shape as the SELECT-path duplicate response.
        assert body.get("status") == "duplicate"
        assert body.get("paper_id") == doi


# ---------------------------------------------------------------------------
# R28 hardening — pipeline failure must mark queue row 'failed'
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestR28FailureSemantics:
    """H2: if _process_paper raises, the queue row MUST be marked 'failed',
    not left orphaned in 'queued'. This is the R28 hardening the Opus reviewer
    will specifically check.
    """

    def test_process_paper_failure_still_returns_202(self, client, db, monkeypatch):
        """A1: the POST is accepted (202) even when ingestion later fails.

        Under the BackgroundTasks design the HTTP request returns immediately
        after queueing — it does NOT block on extraction, so a pipeline
        exception cannot surface as a synchronous 500. The failure is recorded
        on the queue row (see sibling tests) and surfaced via the H3 status
        endpoint. The endpoint itself stays 202.
        """
        from lit_monitor.server.routes import ingest as ingest_route

        def boom(*a, **kw):
            raise RuntimeError("simulated pipeline failure")

        monkeypatch.setattr(ingest_route, "_process_paper", boom)

        r = client.post(
            "/api/ingest",
            json={"doi": "10.1234/fail.001", "title": "T"},
        )
        assert r.status_code == 202, f"expected 202, got {r.status_code}: {r.text}"

    def test_process_paper_failure_marks_queue_row_failed(self, client, db, monkeypatch):
        """H2: after _process_paper raises, ingest_queue.status='failed' (not 'queued').

        R28 invariant: the queue must reflect ground truth — a 'queued' row
        that never transitions to 'done' or 'failed' is invisible corruption
        that H3's queue listing would present as an active job.
        """
        from lit_monitor.server.routes import ingest as ingest_route

        def boom(*a, **kw):
            raise RuntimeError("simulated pipeline failure")

        monkeypatch.setattr(ingest_route, "_process_paper", boom)

        client.post(
            "/api/ingest",
            json={"doi": "10.1234/fail.002", "title": "T"},
        )

        with db._connect() as conn:
            row = conn.execute(
                "SELECT status, error FROM ingest_queue WHERE doi = ?",
                ("10.1234/fail.002",),
            ).fetchone()

        assert row is not None, "ingest_queue row must exist"
        assert row[0] == "failed", (
            f"Expected status='failed', got {row[0]!r}. "
            "Orphaned 'queued' rows are the R28 reviewer trap."
        )

    def test_process_paper_failure_records_error_text(self, client, db, monkeypatch):
        """H2: the error column must contain the exception message."""
        from lit_monitor.server.routes import ingest as ingest_route

        def boom(*a, **kw):
            raise RuntimeError("simulated pipeline failure")

        monkeypatch.setattr(ingest_route, "_process_paper", boom)

        client.post(
            "/api/ingest",
            json={"doi": "10.1234/fail.003", "title": "T"},
        )

        with db._connect() as conn:
            row = conn.execute(
                "SELECT error FROM ingest_queue WHERE doi = ?",
                ("10.1234/fail.003",),
            ).fetchone()

        assert row is not None
        assert row[0] and "simulated pipeline failure" in row[0]

    def test_queue_row_not_orphaned_in_queued_on_failure(self, client, db, monkeypatch):
        """H2: status is never left as 'queued' after _process_paper raises.

        Explicitly asserts the negation (not 'queued') separate from the
        positive assertion (is 'failed') so test failures are maximally clear.
        """
        from lit_monitor.server.routes import ingest as ingest_route

        def boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(ingest_route, "_process_paper", boom)

        client.post(
            "/api/ingest",
            json={"doi": "10.1234/fail.004", "title": "T"},
        )

        with db._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingest_queue WHERE doi = ?",
                ("10.1234/fail.004",),
            ).fetchone()

        assert row is not None
        assert row[0] != "queued", (
            "Queue row is STILL 'queued' after _process_paper raised — "
            "this is the orphaned-row bug the R28 hardening must prevent."
        )


# ---------------------------------------------------------------------------
# StateDB migration tests (ingest_queue additive creation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStateDBMigration:
    def test_ingest_queue_table_created_on_fresh_db(self, tmp_path):
        """H2: StateDB init creates the ingest_queue table additively."""
        StateDB(tmp_path / "state.db")

        conn = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ingest_queue'"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "ingest_queue table was not created"

    def test_ingest_queue_schema_columns(self, tmp_path):
        """H2: ingest_queue has the required columns (doi, status, queued_at, completed_at, error)."""
        StateDB(tmp_path / "state.db")

        conn = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            cols = {
                row[1]: {"type": row[2], "notnull": row[3], "dflt": row[4]}
                for row in conn.execute("PRAGMA table_info(ingest_queue)").fetchall()
            }
        finally:
            conn.close()

        assert "doi" in cols, "ingest_queue missing 'doi' column"
        assert "status" in cols, "ingest_queue missing 'status' column"
        assert "queued_at" in cols, "ingest_queue missing 'queued_at' column"
        assert "completed_at" in cols, "ingest_queue missing 'completed_at' column"
        assert "error" in cols, "ingest_queue missing 'error' column"

        # doi is the PK — notnull must be enforced by PK constraint
        # status has a NOT NULL default
        assert cols["status"]["notnull"] == 1, "status must be NOT NULL"
        assert cols["queued_at"]["notnull"] == 1, "queued_at must be NOT NULL"
        # completed_at and error are nullable
        assert cols["completed_at"]["notnull"] == 0, "completed_at must be nullable"
        assert cols["error"]["notnull"] == 0, "error must be nullable"

    def test_existing_db_migration_is_additive(self, tmp_path):
        """H2: opening an existing DB without ingest_queue adds it without destroying tables."""
        legacy_path = tmp_path / "legacy.db"

        # Simulate a pre-H2 state DB with papers but no ingest_queue.
        conn = sqlite3.connect(str(legacy_path))
        conn.execute(
            "CREATE TABLE papers (doi TEXT PRIMARY KEY, title TEXT)"
        )
        conn.execute("INSERT INTO papers (doi, title) VALUES ('10.0/legacy', 'Old Paper')")
        conn.commit()
        conn.close()

        # Open via StateDB — migration must add ingest_queue without touching papers.
        StateDB(legacy_path)

        conn2 = sqlite3.connect(str(legacy_path))
        try:
            # ingest_queue must now exist.
            tbl = conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ingest_queue'"
            ).fetchone()
            assert tbl is not None, "ingest_queue not added by additive migration"

            # Legacy data must be preserved.
            row = conn2.execute(
                "SELECT title FROM papers WHERE doi='10.0/legacy'"
            ).fetchone()
            assert row is not None, "Legacy papers row was destroyed by migration"
            assert row[0] == "Old Paper"
        finally:
            conn2.close()

    def test_migration_is_idempotent(self, tmp_path):
        """H2: re-opening a DB that already has ingest_queue doesn't raise or duplicate."""
        db_path = tmp_path / "state.db"
        StateDB(db_path)  # first open — creates ingest_queue
        StateDB(db_path)  # second open — must not fail or duplicate table

        conn = sqlite3.connect(str(db_path))
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='ingest_queue'"
            ).fetchone()[0]
        finally:
            conn.close()

        assert count == 1, f"Expected exactly 1 ingest_queue table, got {count}"


# ---------------------------------------------------------------------------
# H3: GET /api/ingest/queue
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueueListing:
    def test_empty_queue_returns_empty_list(self, client):
        """H3: no rows in ingest_queue → 200 with empty list."""
        r = client.get("/api/ingest/queue")
        assert r.status_code == 200
        assert r.json() == []

    def test_two_items_ordered_descending(self, client, monkeypatch):
        """H3: two ingests → list ordered newest-first (queued_at DESC)."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))
        client.post("/api/ingest", json={"doi": "10.1234/a", "title": "A"})
        client.post("/api/ingest", json={"doi": "10.1234/b", "title": "B"})

        r = client.get("/api/ingest/queue")
        assert r.status_code == 200
        items = r.json()
        assert len(items) == 2
        # B was added after A — DESC ordering puts B first.
        assert items[0]["doi"] == "10.1234/b"
        assert items[1]["doi"] == "10.1234/a"

    def test_response_shape(self, client, monkeypatch):
        """H3: each item in the queue list has the expected keys."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))
        client.post("/api/ingest", json={"doi": "10.1234/shape", "title": "S"})

        r = client.get("/api/ingest/queue")
        assert r.status_code == 200
        item = r.json()[0]
        for key in ("doi", "status", "queued_at", "completed_at", "error"):
            assert key in item, f"Expected key {key!r} missing from queue item"


# ---------------------------------------------------------------------------
# H3: GET /api/ingest/{doi:path}/status
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDoiStatus:
    def test_unknown_doi_404(self, client):
        """H3: DOI not in queue → 404."""
        r = client.get("/api/ingest/10.1234/missing/status")
        assert r.status_code == 404

    def test_known_doi_returns_row(self, client, monkeypatch):
        """H3: known DOI → 200 with correct doi and status."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))
        client.post("/api/ingest", json={"doi": "10.1234/known", "title": "K"})

        r = client.get("/api/ingest/10.1234/known/status")
        assert r.status_code == 200
        body = r.json()
        assert body["doi"] == "10.1234/known"
        assert body["status"] == "done"

    def test_known_doi_response_has_all_fields(self, client, monkeypatch):
        """H3: status response includes all five expected keys."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(ingest_route, "_process_paper", lambda *a, **kw: (True, []))
        client.post("/api/ingest", json={"doi": "10.1234/fields", "title": "F"})

        r = client.get("/api/ingest/10.1234/fields/status")
        assert r.status_code == 200
        body = r.json()
        for key in ("doi", "status", "queued_at", "completed_at", "error"):
            assert key in body, f"Expected key {key!r} missing from status response"


# ---------------------------------------------------------------------------
# H3: routing disambiguation — 'queue' must not match as {doi:path}
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueueRouteDoesNotMatchAsDoi:
    def test_queue_path_resolves_to_listing_not_doi_status(self, client):
        """H3 routing trap: 'queue' must not be captured by {doi:path}/status.

        FastAPI routes are matched in registration order.  The explicit
        /api/ingest/queue endpoint is registered before the wildcard
        /api/ingest/{doi:path}/status route, so 'queue' should hit the
        listing handler (200 + list), not the status handler (404 "DOI not
        in queue").
        """
        r = client.get("/api/ingest/queue")
        assert r.status_code == 200, (
            f"Got {r.status_code} — 'queue' was matched as a DOI path segment. "
            "Ensure the /queue route is registered BEFORE {doi:path}/status."
        )
        assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# A1: async BackgroundTasks ingestion — queue-status transitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncIngestTransitions:
    """A1: the background task wraps _process_paper and transitions the queue
    row to a terminal status based on its return value / raised exception.

    Starlette's TestClient runs background tasks synchronously after the
    response, so a monkeypatched _process_paper drives these transitions
    deterministically within the POST call.
    """

    def test_processed_true_yields_done(self, client, db, monkeypatch):
        """A1: _process_paper returning (True, [...]) → queue status 'done'."""
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(
            ingest_route, "_process_paper", lambda *a, **kw: (True, ["topic"])
        )

        client.post("/api/ingest", json={"doi": "10.1234/a1.done", "title": "T"})

        with db._connect() as conn:
            row = conn.execute(
                "SELECT status, completed_at FROM ingest_queue WHERE doi = ?",
                ("10.1234/a1.done",),
            ).fetchone()
        assert row is not None
        assert row[0] == "done"
        assert row[1] is not None  # completed_at stamped

    def test_processed_false_yields_no_markdown(self, client, db, monkeypatch):
        """A1: _process_paper returning (False, []) → queue status 'no_markdown'.

        This is the EXPECTED path when the Zotero item has no .md attachment
        yet — it is NOT an error and must not be marked 'failed'.
        """
        from lit_monitor.server.routes import ingest as ingest_route

        monkeypatch.setattr(
            ingest_route, "_process_paper", lambda *a, **kw: (False, [])
        )

        client.post("/api/ingest", json={"doi": "10.1234/a1.nomd", "title": "T"})

        with db._connect() as conn:
            row = conn.execute(
                "SELECT status, error FROM ingest_queue WHERE doi = ?",
                ("10.1234/a1.nomd",),
            ).fetchone()
        assert row is not None
        assert row[0] == "no_markdown", (
            f"Expected 'no_markdown', got {row[0]!r}. A missing .md attachment "
            "is not an error and must not be marked 'failed'."
        )
        assert row[1] is None  # no error text for the no_markdown path

    def test_raising_yields_failed_with_error_text(self, client, db, monkeypatch):
        """A1: _process_paper raising → queue status 'failed' + error text."""
        from lit_monitor.server.routes import ingest as ingest_route

        def boom(*a, **kw):
            raise RuntimeError("a1 simulated failure")

        monkeypatch.setattr(ingest_route, "_process_paper", boom)

        client.post("/api/ingest", json={"doi": "10.1234/a1.fail", "title": "T"})

        with db._connect() as conn:
            row = conn.execute(
                "SELECT status, error, completed_at FROM ingest_queue WHERE doi = ?",
                ("10.1234/a1.fail",),
            ).fetchone()
        assert row is not None
        assert row[0] == "failed"
        assert row[1] and "a1 simulated failure" in row[1]
        assert row[2] is not None  # completed_at stamped on failure


@pytest.mark.unit
class TestSynthesizedZoteroItem:
    """A1: the background task builds a Zotero-shaped item dict from the
    IngestRequest and feeds it to _process_paper. Verify the shape by
    capturing the kwargs the (monkeypatched) _process_paper receives.
    """

    def test_item_dict_shape_satisfies_process_paper_reads(
        self, client, db, monkeypatch
    ):
        """A1: synthesized item carries every field _process_paper reads.

        _process_paper reads item['key'] and item['data'] with
        title / creators / date / abstractNote / DOI. We capture the actual
        item passed and assert the shape, including that ZoteroClient
        .extract_authors round-trips the request authors.
        """
        from lit_monitor.core.zotero_client import ZoteroClient
        from lit_monitor.server.routes import ingest as ingest_route

        captured: dict = {}

        def capture(doi, item, *a, **kw):
            captured["doi"] = doi
            captured["item"] = item
            return (True, [])

        monkeypatch.setattr(ingest_route, "_process_paper", capture)

        client.post(
            "/api/ingest",
            json={
                "doi": "10.1234/a1.shape",
                "title": "Shape Paper",
                "authors": ["Smith", "Doe"],
                "year": 2023,
                "abstract": "An abstract.",
                "zotero_key": "ZKEY42",
            },
        )

        assert captured["doi"] == "10.1234/a1.shape"
        item = captured["item"]
        assert item["key"] == "ZKEY42"
        data = item["data"]
        assert data["title"] == "Shape Paper"
        assert data["abstractNote"] == "An abstract."
        assert data["DOI"] == "10.1234/a1.shape"
        # _parse_year scans the date string for a 4-digit year.
        assert "2023" in data["date"]
        # extract_authors must round-trip the supplied display names.
        assert ZoteroClient.extract_authors(data) == ["Smith", "Doe"]
