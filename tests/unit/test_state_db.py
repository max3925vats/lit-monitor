"""
Unit tests for StateDB G16 additions:
  - papers.last_insight_run nullable TIMESTAMP column (additive migration)
  - StateDB.get_papers_without_insight_run() helper for v0.8+ insight discovery
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from scripts.core.state_db import StateDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _column_info(db_path: str, table: str) -> dict[str, dict]:
    """Return column metadata from PRAGMA table_info as {name: {type, notnull, ...}}."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    # Each row: (cid, name, type, notnull, dflt_value, pk)
    return {r[1]: {"type": r[2], "notnull": r[3], "dflt_value": r[4]} for r in rows}


def _set_last_insight_run(db_path: str, doi: str, value: str | None) -> None:
    """Directly update last_insight_run for a DOI via raw sqlite3 (bypasses StateDB)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE papers SET last_insight_run = ? WHERE doi = ?",
            (value, doi),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# G16 — Column migration tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLastInsightRunColumn:
    """G16: papers.last_insight_run nullable TIMESTAMP column via additive migration."""

    def test_papers_has_last_insight_run_column(self, tmp_path):
        """G16: papers gains a nullable TIMESTAMP for v0.8+ insight tracking."""
        db_path = str(tmp_path / "state.db")
        StateDB(db_path)

        cols = _column_info(db_path, "papers")
        assert "last_insight_run" in cols, (
            "papers.last_insight_run missing — additive migration did not run"
        )

    def test_last_insight_run_is_nullable(self, tmp_path):
        """G16: last_insight_run must be nullable (notnull=0 in PRAGMA table_info)."""
        db_path = str(tmp_path / "state.db")
        StateDB(db_path)

        cols = _column_info(db_path, "papers")
        col = cols["last_insight_run"]
        assert col["notnull"] == 0, (
            f"last_insight_run must be nullable (notnull=0), got notnull={col['notnull']!r}"
        )

    def test_last_insight_run_migration_is_idempotent(self, tmp_path):
        """G16: re-opening a DB that already has the column doesn't raise."""
        db_path = str(tmp_path / "state.db")
        db1 = StateDB(db_path)
        db1.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        # Re-open: additive migration should skip because _column_exists returns True.
        StateDB(db_path)
        # Exactly one column, no duplicate.
        col_names = [r[1] for r in sqlite3.connect(db_path).execute(
            "PRAGMA table_info(papers)"
        ).fetchall()]
        assert col_names.count("last_insight_run") == 1, (
            "last_insight_run must appear exactly once — no duplicate after re-open"
        )

    def test_last_insight_run_added_to_existing_db_without_column(self, tmp_path):
        """G16: a state.db built before G16 gets the column added on next open."""
        db_path = str(tmp_path / "state.db")

        # Build papers table WITHOUT last_insight_run (simulates a pre-G16 state.db).
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE papers ("
            "doi TEXT PRIMARY KEY, title TEXT, year INTEGER, source_type TEXT"
            ")"
        )
        conn.execute("INSERT INTO papers VALUES ('10.0/a', 'A', 2024, 'zotero')")
        conn.commit()
        conn.close()

        # Open via StateDB — _init_schema runs CREATE TABLE IF NOT EXISTS (no-op)
        # then additive_migrations should add last_insight_run.
        StateDB(db_path)

        cols = _column_info(db_path, "papers")
        assert "last_insight_run" in cols, (
            "Additive migration failed to add last_insight_run to a pre-existing table"
        )

        # Existing row must be preserved with NULL in the new column.
        conn2 = sqlite3.connect(db_path)
        row = conn2.execute(
            "SELECT doi, last_insight_run FROM papers WHERE doi = '10.0/a'"
        ).fetchone()
        conn2.close()
        assert row[0] == "10.0/a"
        assert row[1] is None, (
            f"Existing row must have last_insight_run=NULL after migration, got {row[1]!r}"
        )

    def test_upsert_paper_signature_unchanged(self, tmp_path):
        """G16: upsert_paper still accepts the same dict-based API, no new required args."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        # This must not raise — signature not changed.
        db.upsert_paper({"doi": "10.0/sig-test", "title": "Sig Test", "year": 2024})
        row = db.get_paper("10.0/sig-test")
        assert row is not None


# ---------------------------------------------------------------------------
# G16 — get_papers_without_insight_run() helper tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetPapersWithoutInsightRun:
    """G16: StateDB.get_papers_without_insight_run() for v0.8+ insight queue."""

    def test_returns_unseen_papers(self, tmp_path):
        """G16: helper returns papers with NULL last_insight_run."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/b", "title": "B", "year": 2024, "source_type": "zotero"})

        # Mark /a as already seen.
        _set_last_insight_run(db_path, "10.0/a", "2026-05-29T12:00:00")

        unseen = db.get_papers_without_insight_run()
        assert len(unseen) == 1
        assert unseen[0]["doi"] == "10.0/b"

    def test_returns_all_when_none_have_run(self, tmp_path):
        """G16: if no paper has been processed, all are returned."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/x", "title": "X", "year": 2024, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/y", "title": "Y", "year": 2024, "source_type": "zotero"})

        unseen = db.get_papers_without_insight_run()
        assert len(unseen) == 2

    def test_returns_empty_when_all_seen(self, tmp_path):
        """G16: if all papers have been processed, returns empty list."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/x", "title": "X", "year": 2024, "source_type": "zotero"})
        _set_last_insight_run(db_path, "10.0/x", "2026-05-29T12:00:00")

        unseen = db.get_papers_without_insight_run()
        assert unseen == []

    def test_respects_since_filter(self, tmp_path):
        """G16: helper returns papers whose last_insight_run is NULL OR older than 'since'."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/b", "title": "B", "year": 2024, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/c", "title": "C", "year": 2024, "source_type": "zotero"})

        # /a is fresh (2026-06-01), /b is stale (2026-01-01), /c never ran (NULL).
        _set_last_insight_run(db_path, "10.0/a", "2026-06-01T12:00:00")
        _set_last_insight_run(db_path, "10.0/b", "2026-01-01T12:00:00")

        since = datetime(2026, 5, 1)
        unseen = db.get_papers_without_insight_run(since=since)

        # Expect /b (stale) and /c (never) — not /a (fresh).
        dois = {p["doi"] for p in unseen}
        assert dois == {"10.0/b", "10.0/c"}

    def test_since_none_does_not_return_processed_papers(self, tmp_path):
        """G16: with since=None, recently-processed papers are excluded."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/p", "title": "P", "year": 2024, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/q", "title": "Q", "year": 2024, "source_type": "zotero"})

        _set_last_insight_run(db_path, "10.0/p", "2026-05-29T12:00:00")

        unseen = db.get_papers_without_insight_run(since=None)
        dois = {p["doi"] for p in unseen}
        assert dois == {"10.0/q"}

    def test_returns_list_of_dicts(self, tmp_path):
        """G16: helper returns list[dict], not list of Row objects."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/r", "title": "R", "year": 2024, "source_type": "zotero"})

        result = db.get_papers_without_insight_run()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "doi" in result[0]


# ---------------------------------------------------------------------------
# G6 — set_graph_indexed flips the R28 dual-write column
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSetGraphIndexed:
    """G6: set_graph_indexed(doi, val) toggles papers.graph_indexed."""

    def test_default_is_zero(self, tmp_path):
        """G1 migration default: graph_indexed = 0 on fresh insert."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT graph_indexed FROM papers WHERE doi = '10.0/a'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == 0

    def test_set_graph_indexed_updates_value(self, tmp_path):
        """G6: set_graph_indexed(doi, 1) flips the column to 1."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.set_graph_indexed("10.0/a", 1)
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT graph_indexed FROM papers WHERE doi = '10.0/a'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == 1

    def test_set_graph_indexed_idempotent(self, tmp_path):
        """G6: calling set_graph_indexed twice with same value is safe."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.set_graph_indexed("10.0/a", 1)
        db.set_graph_indexed("10.0/a", 1)
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT graph_indexed FROM papers WHERE doi = '10.0/a'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == 1

    def test_set_graph_indexed_can_clear(self, tmp_path):
        """G6: set_graph_indexed(doi, 0) clears the flag (e.g. for rebuilds)."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.set_graph_indexed("10.0/a", 1)
        db.set_graph_indexed("10.0/a", 0)
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT graph_indexed FROM papers WHERE doi = '10.0/a'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == 0

    def test_set_graph_indexed_unknown_doi_is_noop(self, tmp_path):
        """G6: set_graph_indexed on a missing DOI updates 0 rows but doesn't raise."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        # Must not raise — UPDATE matching no rows is allowed.
        db.set_graph_indexed("10.0/nope", 1)
