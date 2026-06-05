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


# ---------------------------------------------------------------------------
# N5 — ner_processed_at column + set_ner_processed_at + get_papers_for_ner_backfill
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNerBackfillHelpers:
    """N5: StateDB helpers for NER backfill tracking."""

    def test_ner_processed_at_column_exists(self, tmp_path):
        """N5: papers.ner_processed_at column is created by additive migration."""
        StateDB(tmp_path / "state.db")  # triggers schema creation
        info = _column_info(str(tmp_path / "state.db"), "papers")
        assert "ner_processed_at" in info
        # Must be nullable (notnull=0) with NULL default.
        assert info["ner_processed_at"]["notnull"] == 0
        assert info["ner_processed_at"]["dflt_value"] is None

    def test_ner_processed_at_default_is_null(self, tmp_path):
        """N5: freshly inserted papers have NULL ner_processed_at."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        conn = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            row = conn.execute(
                "SELECT ner_processed_at FROM papers WHERE doi = '10.0/a'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] is None

    def test_set_ner_processed_at(self, tmp_path):
        """N5: set_ner_processed_at stamps the column for the given DOI."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.set_ner_processed_at("10.0/a", "2026-06-01T00:00:00")
        with db._connect() as conn:
            row = conn.execute(
                "SELECT ner_processed_at FROM papers WHERE doi = '10.0/a'"
            ).fetchone()
        assert row[0] == "2026-06-01T00:00:00"

    def test_set_ner_processed_at_unknown_doi_is_noop(self, tmp_path):
        """N5: stamping a missing DOI updates 0 rows but doesn't raise."""
        db = StateDB(tmp_path / "state.db")
        # Must not raise.
        db.set_ner_processed_at("10.0/nope", "2026-06-01T00:00:00")

    def test_get_papers_for_ner_backfill_returns_only_unprocessed(self, tmp_path):
        """N5: only papers with ner_processed_at IS NULL are returned."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/b", "title": "B", "year": 2024, "source_type": "zotero"})
        db.set_ner_processed_at("10.0/a", "2026-06-01T00:00:00")

        candidates = db.get_papers_for_ner_backfill()
        dois = {p["doi"] for p in candidates}
        assert dois == {"10.0/b"}

    def test_get_papers_for_ner_backfill_with_only_unprocessed_false(self, tmp_path):
        """N5: only_unprocessed=False returns all papers regardless of stamp."""
        db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        db.set_ner_processed_at("10.0/a", "2026-06-01T00:00:00")

        candidates = db.get_papers_for_ner_backfill(only_unprocessed=False)
        dois = {p["doi"] for p in candidates}
        assert dois == {"10.0/a", "10.0/b"}

    def test_get_papers_for_ner_backfill_since_filter(self, tmp_path):
        """N5: since= restricts candidates by papers.last_updated."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/old", "title": "O", "year": 2020, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/new", "title": "N", "year": 2024, "source_type": "zotero"})
        # Backdate the old paper.
        with db._connect() as conn:
            conn.execute(
                "UPDATE papers SET last_updated = ? WHERE doi = ?",
                ("2020-01-01T00:00:00", "10.0/old"),
            )
            conn.commit()

        candidates = db.get_papers_for_ner_backfill(since=datetime(2023, 1, 1))
        dois = {p["doi"] for p in candidates}
        # Only /new (current timestamp >= 2023) should be included.
        assert "10.0/new" in dois
        assert "10.0/old" not in dois

    def test_get_papers_for_ner_backfill_limit(self, tmp_path):
        """N5: limit= caps the result count."""
        db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b", "10.0/c"):
            db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})

        candidates = db.get_papers_for_ner_backfill(limit=2)
        assert len(candidates) == 2

    def test_get_papers_for_ner_backfill_returns_list_of_dicts(self, tmp_path):
        """N5: result is list[dict], not list of Row objects."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})

        result = db.get_papers_for_ner_backfill()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "doi" in result[0]


# ---------------------------------------------------------------------------
# R5 — rel_processed_at column + set_rel_processed_at + get_papers_for_rel_backfill
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRelBackfillHelpers:
    """R5: StateDB helpers for relationship backfill tracking."""

    def test_rel_processed_at_column_exists(self, tmp_path):
        """R5: papers.rel_processed_at column is created by additive migration."""
        StateDB(tmp_path / "state.db")  # triggers schema creation
        info = _column_info(str(tmp_path / "state.db"), "papers")
        assert "rel_processed_at" in info
        # Must be nullable (notnull=0) with NULL default.
        assert info["rel_processed_at"]["notnull"] == 0
        assert info["rel_processed_at"]["dflt_value"] is None

    def test_rel_processed_at_default_is_null(self, tmp_path):
        """R5: freshly inserted papers have NULL rel_processed_at."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        conn = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            row = conn.execute(
                "SELECT rel_processed_at FROM papers WHERE doi = '10.0/a'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] is None

    def test_set_rel_processed_at(self, tmp_path):
        """R5: set_rel_processed_at stamps the column for the given DOI."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.set_rel_processed_at("10.0/a", "2026-06-01T00:00:00")
        with db._connect() as conn:
            row = conn.execute(
                "SELECT rel_processed_at FROM papers WHERE doi = '10.0/a'"
            ).fetchone()
        assert row[0] == "2026-06-01T00:00:00"

    def test_set_rel_processed_at_unknown_doi_is_noop(self, tmp_path):
        """R5: stamping a missing DOI updates 0 rows but doesn't raise."""
        db = StateDB(tmp_path / "state.db")
        # Must not raise.
        db.set_rel_processed_at("10.0/nope", "2026-06-01T00:00:00")

    def test_get_papers_for_rel_backfill_returns_only_unprocessed(self, tmp_path):
        """R5: only papers with rel_processed_at IS NULL are returned."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/b", "title": "B", "year": 2024, "source_type": "zotero"})
        db.set_rel_processed_at("10.0/a", "2026-06-01T00:00:00")

        candidates = db.get_papers_for_rel_backfill()
        dois = {p["doi"] for p in candidates}
        assert dois == {"10.0/b"}

    def test_get_papers_for_rel_backfill_with_only_unprocessed_false(self, tmp_path):
        """R5: only_unprocessed=False returns all papers regardless of stamp."""
        db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        db.set_rel_processed_at("10.0/a", "2026-06-01T00:00:00")

        candidates = db.get_papers_for_rel_backfill(only_unprocessed=False)
        dois = {p["doi"] for p in candidates}
        assert dois == {"10.0/a", "10.0/b"}

    def test_get_papers_for_rel_backfill_since_filter(self, tmp_path):
        """R5: since= restricts candidates by papers.last_updated."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/old", "title": "O", "year": 2020, "source_type": "zotero"})
        db.upsert_paper({"doi": "10.0/new", "title": "N", "year": 2024, "source_type": "zotero"})
        # Backdate the old paper.
        with db._connect() as conn:
            conn.execute(
                "UPDATE papers SET last_updated = ? WHERE doi = ?",
                ("2020-01-01T00:00:00", "10.0/old"),
            )
            conn.commit()

        candidates = db.get_papers_for_rel_backfill(since=datetime(2023, 1, 1))
        dois = {p["doi"] for p in candidates}
        # Only /new (current timestamp >= 2023) should be included.
        assert "10.0/new" in dois
        assert "10.0/old" not in dois

    def test_get_papers_for_rel_backfill_limit(self, tmp_path):
        """R5: limit= caps the result count."""
        db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b", "10.0/c"):
            db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})

        candidates = db.get_papers_for_rel_backfill(limit=2)
        assert len(candidates) == 2

    def test_get_papers_for_rel_backfill_returns_list_of_dicts(self, tmp_path):
        """R5: result is list[dict], not list of Row objects."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})

        result = db.get_papers_for_rel_backfill()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "doi" in result[0]


# ===========================================================================
# P1: discovery_runs + discovery_paper_results tables and helpers
# ===========================================================================


class TestDiscoveryRunsTable:
    def test_discovery_runs_table_exists(self, tmp_path):
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='discovery_runs'"
            ).fetchall()
        assert rows, "discovery_runs table missing"

    def test_discovery_paper_results_table_exists(self, tmp_path):
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='discovery_paper_results'"
            ).fetchall()
        assert rows, "discovery_paper_results table missing"


class TestDiscoveryRunHelpers:
    def test_start_run_returns_int(self, tmp_path):
        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run({"topics": ["antibody"]})
        assert isinstance(run_id, int) and run_id > 0

    def test_start_run_persists_params(self, tmp_path):
        import json as _json
        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run({"topics": ["x"], "since_days": 7})
        with db._connect() as conn:
            row = conn.execute(
                "SELECT status, run_params_json FROM discovery_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        assert row[0] == "running"
        params = _json.loads(row[1])
        assert params["topics"] == ["x"]
        assert params["since_days"] == 7

    def test_finish_run_updates_status(self, tmp_path):
        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run({})
        db.finish_discovery_run(run_id, status="success", total_found=5, total_ingested=2)
        with db._connect() as conn:
            row = conn.execute(
                "SELECT status, total_found, total_ingested, finished_at "
                "FROM discovery_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        assert row[0] == "success"
        assert row[1] == 5
        assert row[2] == 2
        assert row[3] is not None  # finished_at set

    def test_add_discovery_paper(self, tmp_path):
        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run({})
        db.add_discovery_paper(
            run_id,
            doi="10.1/abc",
            title="Test Paper",
            score=0.87,
            rationale="High relevance",
            ingested=True,
        )
        with db._connect() as conn:
            row = conn.execute(
                "SELECT doi, title, score, rationale, ingested, ingested_at "
                "FROM discovery_paper_results WHERE run_id=?",
                (run_id,),
            ).fetchone()
        assert row[0] == "10.1/abc"
        assert row[1] == "Test Paper"
        assert abs(row[2] - 0.87) < 0.001
        assert row[3] == "High relevance"
        assert row[4] == 1
        assert row[5] is not None  # ingested_at set when ingested=True

    def test_add_discovery_paper_not_ingested_leaves_timestamp_null(self, tmp_path):
        db = StateDB(tmp_path / "state.db")
        run_id = db.start_discovery_run({})
        db.add_discovery_paper(
            run_id,
            doi="10.1/xyz",
            title="X",
            score=0.5,
            rationale="",
            ingested=False,
        )
        with db._connect() as conn:
            row = conn.execute(
                "SELECT ingested, ingested_at FROM discovery_paper_results WHERE doi='10.1/xyz'",
            ).fetchone()
        assert row[0] == 0
        assert row[1] is None

    def test_existing_db_migration_additive(self, tmp_path):
        """P1: opening an existing state.db without discovery_runs adds it without dropping anything."""
        import sqlite3 as _sqlite3
        legacy = tmp_path / "legacy.db"
        conn = _sqlite3.connect(str(legacy))
        conn.execute("CREATE TABLE papers (doi TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO papers (doi) VALUES ('10.0/legacy')")
        conn.commit()
        conn.close()
        db = StateDB(legacy)
        with db._connect() as conn:
            # New tables created
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='discovery_runs'"
            ).fetchone()
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='discovery_paper_results'"
            ).fetchone()
            # Legacy data preserved
            row = conn.execute(
                "SELECT doi FROM papers WHERE doi='10.0/legacy'"
            ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# Bundle H: feedback_events table + helper tests
# ---------------------------------------------------------------------------

class TestFeedbackEvents:
    """Bundle H: feedback_events table, record/list/summary helpers."""

    @pytest.fixture()
    def db(self, tmp_path):
        return StateDB(tmp_path / "state.db")

    def test_feedback_events_table_exists(self, db):
        with db._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback_events'"
            ).fetchone()
        assert exists is not None

    def test_feedback_events_indexes_exist(self, db):
        with db._connect() as conn:
            idx_names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND tbl_name='feedback_events'"
                ).fetchall()
            ]
        assert "idx_feedback_doi" in idx_names
        assert "idx_feedback_created" in idx_names

    def test_record_and_list(self, db):
        db.record_feedback_event("10.1/test", "saved", source="discovery")
        events = db.list_feedback_events()
        assert len(events) == 1
        assert events[0]["doi"] == "10.1/test"
        assert events[0]["signal_type"] == "saved"
        assert events[0]["weight"] == 1.0
        assert events[0]["source"] == "discovery"

    def test_rated_signal_stores_rating(self, db):
        db.record_feedback_event("10.1/test", "rated", rating=5)
        events = db.list_feedback_events()
        assert events[0]["rating"] == 5
        # Weight for rated=5: 0.5 * (5-3) * 0.5 = 0.5
        assert abs(events[0]["weight"] - 0.5) < 1e-9

    def test_negative_weight_for_dismissed(self, db):
        db.record_feedback_event("10.1/test", "dismissed")
        events = db.list_feedback_events()
        assert events[0]["weight"] < 0

    def test_invalid_signal_type_raises(self, db):
        with pytest.raises(ValueError, match="invalid signal_type"):
            db.record_feedback_event("10.1/test", "totally_wrong")

    def test_rated_without_rating_raises(self, db):
        with pytest.raises(ValueError):
            db.record_feedback_event("10.1/test", "rated")


# ---------------------------------------------------------------------------
# P4 Part C: implicit-save support helpers
# ---------------------------------------------------------------------------

class TestImplicitSaveHelpers:
    """P4 Part C: was_surfaced_in_discovery + has_implicit_save_feedback."""

    @pytest.fixture()
    def db(self, tmp_path):
        return StateDB(tmp_path / "state.db")

    def test_was_surfaced_true_when_in_discovery_run(self, db):
        run_id = db.start_discovery_run({})
        db.add_discovery_paper(
            run_id, "10.1/surfaced", "A surfaced paper", 0.9, "", ingested=False
        )
        assert db.was_surfaced_in_discovery("10.1/surfaced") is True

    def test_was_surfaced_false_when_never_in_discovery(self, db):
        assert db.was_surfaced_in_discovery("10.1/never") is False

    def test_was_surfaced_false_for_empty_doi(self, db):
        run_id = db.start_discovery_run({})
        # A candidate with an empty DOI is recorded but must never match.
        db.add_discovery_paper(run_id, "", "No DOI", 0.5, "", ingested=False)
        assert db.was_surfaced_in_discovery("") is False

    def test_has_implicit_save_false_initially(self, db):
        assert db.has_implicit_save_feedback("10.1/x") is False

    def test_has_implicit_save_true_after_recording(self, db):
        db.record_feedback_event(
            "10.1/x", "saved", source="implicit_zotero_save"
        )
        assert db.has_implicit_save_feedback("10.1/x") is True

    def test_has_implicit_save_ignores_explicit_saved(self, db):
        """An explicit web/CLI 'saved' must NOT count as an implicit save."""
        db.record_feedback_event("10.1/x", "saved", source="discovery")
        assert db.has_implicit_save_feedback("10.1/x") is False

    def test_rating_without_rated_raises(self, db):
        with pytest.raises(ValueError):
            db.record_feedback_event("10.1/test", "saved", rating=4)

    def test_summary_returns_all_signal_types(self, db):
        summary = db.feedback_summary()
        expected = {"opened", "saved", "dismissed", "rated", "thumbs_up", "thumbs_down"}
        assert set(summary.keys()) == expected

    def test_summary_counts(self, db):
        db.record_feedback_event("10.1/a", "saved")
        db.record_feedback_event("10.1/b", "saved")
        db.record_feedback_event("10.1/c", "dismissed")
        summary = db.feedback_summary()
        assert summary["saved"] == 2
        assert summary["dismissed"] == 1
        assert summary["thumbs_up"] == 0

    def test_list_limit(self, db):
        for i in range(10):
            db.record_feedback_event(f"10.1/{i}", "opened")
        events = db.list_feedback_events(limit=3)
        assert len(events) == 3

    def test_list_since_filter(self, db):
        db.record_feedback_event("10.1/old", "opened")
        # Insert a row directly with a future created_at timestamp.
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO feedback_events (doi, signal_type, weight, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("10.1/new", "saved", 1.0, "2099-01-01 00:00:00"),
            )
        events = db.list_feedback_events(since="2050-01-01")
        assert len(events) == 1
        assert events[0]["doi"] == "10.1/new"


# ===========================================================================
# B4: reset_extractions re-queues papers as 'pending' (get_pending visibility)
# ===========================================================================


class TestResetExtractionsRequeue:
    """B4 Fix 1: reset_extractions must set status='pending' so reset papers
    re-enter the get_pending queue. Under the old status=NULL behavior they
    were invisible to get_pending and never re-processed."""

    def test_reset_extractions_makes_papers_pending_and_visible(self, tmp_path):
        """B4: a completed paper, after reset, is returned by get_pending."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({
            "doi": "10.1/done",
            "title": "Done Paper",
            "source_type": "paper",
            "status": "extraction_complete",
            "extraction_json": '{"summary": "x"}',
        })
        # Pre-reset: not pending (status is complete).
        assert db.get_pending("paper") == []

        n = db.reset_extractions()
        assert n == 1

        # Post-reset: extraction cleared AND paper re-queued as pending.
        row = db.get_paper("10.1/done")
        assert row["extraction_json"] is None
        assert row["status"] == "pending"

        pending = db.get_pending("paper")
        pending_dois = {p["doi"] for p in pending}
        assert "10.1/done" in pending_dois, (
            "reset paper must be visible to get_pending — this fails under the "
            "old status=NULL behavior"
        )


# ===========================================================================
# B4: upsert_paper fails loud on unknown / typo / flag-column keys
# ===========================================================================


class TestUpsertPaperUnknownKeys:
    """B4 Fix 2: upsert_paper must raise ValueError on keys that are not real
    upsert-able papers columns, instead of silently dropping them."""

    def test_valid_full_upsert_still_works(self, tmp_path):
        """B4: a normal upsert with only real columns succeeds and round-trips."""
        db = StateDB(tmp_path / "state.db")
        db.upsert_paper({
            "doi": "10.1/ok",
            "title": "T",
            "authors": ["A", "B"],
            "year": 2024,
            "journal": "J",
            "status": "pending",
            "source_type": "paper",
            "extraction_json": '{"k": 1}',
        })
        row = db.get_paper("10.1/ok")
        assert row["title"] == "T"
        assert row["status"] == "pending"

    def test_typo_key_raises_valueerror_naming_key(self, tmp_path):
        """B4: an unknown/typo key raises ValueError naming the offending key."""
        db = StateDB(tmp_path / "state.db")
        with pytest.raises(ValueError, match="bogus_col"):
            db.upsert_paper({"doi": "10.1/x", "title": "T", "bogus_col": 1})

    def test_flag_column_raises_with_setter_redirect(self, tmp_path):
        """B4: a flag column managed by a dedicated setter is rejected with a
        redirect message, not silently written through upsert_paper."""
        db = StateDB(tmp_path / "state.db")
        with pytest.raises(ValueError, match="set_graph_indexed"):
            db.upsert_paper({"doi": "10.1/y", "title": "T", "graph_indexed": 1})


# ---------------------------------------------------------------------------
# P1 / Item A: _init_schema silent-failure stragglers
#
# These blocks run during StateDB(...) construction. We inject a failure into
# a specific in-schema statement and assert it is no longer swallowed silently:
#  - default mode  -> a WARNING is logged (and startup still succeeds)
#  - strict mode   -> the failure is re-raised (fatal)
# Failure is injected by wrapping sqlite3.Connection.execute so it raises only
# when the SQL contains a target fragment; every other statement (table_info,
# CREATE/ALTER migrations) runs untouched so construction reaches the target.
# ---------------------------------------------------------------------------

import logging  # noqa: E402

from scripts.core.strict_mode import set_strict  # noqa: E402


def _fail_execute_on(monkeypatch, fragment: str, message: str):
    """Make connections opened by StateDB raise on any execute() whose SQL
    contains ``fragment`` (verbatim substring); everything else runs unchanged.

    sqlite3.Connection is an immutable type, so we cannot patch its ``execute``
    directly.  Instead we install a Connection subclass as the connection
    ``factory`` via a patched ``sqlite3.connect`` inside the state_db module.
    """
    import scripts.core.state_db as _sdb

    class _FailingConn(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if isinstance(sql, str) and fragment in sql:
                raise sqlite3.OperationalError(message)
            return super().execute(sql, *args, **kwargs)

    real_connect = sqlite3.connect

    def fake_connect(*args, **kwargs):
        kwargs["factory"] = _FailingConn
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(_sdb.sqlite3, "connect", fake_connect)


class TestInitSchemaSilentStragglers:
    """P1 Item A: per-statement failures inside _init_schema must not pass
    silently — they warn by default and escalate under strict mode."""

    # -- M3 phase backfill (was: except Exception: pass — fully silent) --
    def test_m3_backfill_failure_warns_in_default_mode(self, tmp_path, monkeypatch, caplog):
        _fail_execute_on(
            monkeypatch,
            "simple_complete = CASE WHEN pass1_complete",
            "boom-m3-backfill",
        )
        with caplog.at_level(logging.WARNING, logger="scripts.core.state_db"):
            db = StateDB(tmp_path / "state.db")  # must NOT raise in default mode
        assert db is not None
        assert any(
            "boom-m3-backfill" in r.message or "backfill" in r.message.lower()
            for r in caplog.records
        ), f"expected M3-backfill WARNING; got {[r.message for r in caplog.records]}"

    def test_m3_backfill_failure_raises_in_strict_mode(self, tmp_path, monkeypatch):
        set_strict(True)
        _fail_execute_on(
            monkeypatch,
            "simple_complete = CASE WHEN pass1_complete",
            "boom-m3-backfill",
        )
        with pytest.raises(Exception, match="boom-m3-backfill"):
            StateDB(tmp_path / "state.db")

    # -- N7 stale-row cleanup (was: warn-only, no strict escalation) --
    def test_n7_cleanup_failure_warns_in_default_mode(self, tmp_path, monkeypatch, caplog):
        _fail_execute_on(
            monkeypatch,
            "DELETE FROM papers ",
            "boom-n7-cleanup",
        )
        with caplog.at_level(logging.WARNING, logger="scripts.core.state_db"):
            db = StateDB(tmp_path / "state.db")
        assert db is not None
        assert any(
            "N7" in r.message or "boom-n7-cleanup" in r.message
            for r in caplog.records
        ), f"expected N7 WARNING; got {[r.message for r in caplog.records]}"

    def test_n7_cleanup_failure_raises_in_strict_mode(self, tmp_path, monkeypatch):
        set_strict(True)
        _fail_execute_on(
            monkeypatch,
            "DELETE FROM papers ",
            "boom-n7-cleanup",
        )
        with pytest.raises(Exception, match="boom-n7-cleanup"):
            StateDB(tmp_path / "state.db")

    # -- weekly->discovery rename (was: warn-only, no strict escalation) --
    def test_rename_failure_warns_in_default_mode(self, tmp_path, monkeypatch, caplog):
        _fail_execute_on(
            monkeypatch,
            "UPDATE run_log SET run_type = 'discovery'",
            "boom-rename",
        )
        with caplog.at_level(logging.WARNING, logger="scripts.core.state_db"):
            db = StateDB(tmp_path / "state.db")
        assert db is not None
        assert any(
            "discovery" in r.message.lower() or "boom-rename" in r.message
            for r in caplog.records
        ), f"expected rename WARNING; got {[r.message for r in caplog.records]}"

    def test_rename_failure_raises_in_strict_mode(self, tmp_path, monkeypatch):
        set_strict(True)
        _fail_execute_on(
            monkeypatch,
            "UPDATE run_log SET run_type = 'discovery'",
            "boom-rename",
        )
        with pytest.raises(Exception, match="boom-rename"):
            StateDB(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# P1 / Item B (B3-F8): upsert_paper column-list drift guard
#
# upsert_paper writes a hand-maintained `cols` list. This guard ensures that
# list cannot silently desync from the live `papers` schema:
#  (1) every column upsert_paper writes must exist in the table, and
#  (2) every real `papers` column must be EITHER written by upsert_paper OR
#      explicitly listed in the intentionally-not-upserted allowlist
#      (_UPSERT_FLAG_COLUMNS + a small set of setter-managed timestamp/flag
#      columns), so a NEW schema column trips a clear assertion.
# ---------------------------------------------------------------------------

from scripts.core.state_db import (  # noqa: E402
    assert_upsert_columns_consistent,
    upsert_writable_columns,
)


class TestUpsertColumnDriftGuard:
    """B3-F8: drift guard between upsert_paper's column list and papers schema."""

    def test_every_written_column_exists_in_schema(self, tmp_path):
        """The guard passes on a freshly-built schema (no drift)."""
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            rows = conn.execute("PRAGMA table_info(papers)").fetchall()
        schema_cols = {r[1] for r in rows}
        for col in upsert_writable_columns():
            assert col in schema_cols, (
                f"upsert_paper writes {col!r} which is not a real papers column"
            )

    def test_assert_helper_passes_on_live_schema(self, tmp_path):
        """assert_upsert_columns_consistent must not raise on the live schema."""
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            assert_upsert_columns_consistent(conn)  # should not raise

    def test_guard_fails_when_written_column_missing_from_schema(self, tmp_path):
        """If upsert's writable list contained a column the table lacks
        (a renamed/removed column), the guard must raise."""
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            with pytest.raises(AssertionError, match="not_a_real_column"):
                assert_upsert_columns_consistent(
                    conn, written_cols=["doi", "title", "not_a_real_column"]
                )

    def test_guard_fails_when_schema_column_unaccounted(self, tmp_path):
        """A NEW papers column that is neither written nor in the allowlist must
        trip the guard, forcing the developer to decide where it is written."""
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            conn.execute("ALTER TABLE papers ADD COLUMN brand_new_col TEXT")
            with pytest.raises(AssertionError, match="brand_new_col"):
                assert_upsert_columns_consistent(conn)

    def test_allowlisted_columns_do_not_trip_guard(self, tmp_path):
        """Setter-managed columns (graph_indexed, etc.) exist in the schema but
        are intentionally not upserted — they must NOT trip the guard."""
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            # All five dedicated-setter flag columns are present in the schema
            rows = conn.execute("PRAGMA table_info(papers)").fetchall()
            schema_cols = {r[1] for r in rows}
            for flag in ("graph_indexed", "chunks_indexed", "notes_synced",
                         "ner_processed_at", "rel_processed_at"):
                assert flag in schema_cols
            # Guard still passes despite those columns not being upserted.
            assert_upsert_columns_consistent(conn)

    def test_guard_fails_when_flag_column_in_writable_list(self, tmp_path):
        """A setter-managed flag column wrongly added to the writable list must
        trip the guard — writing it via the COALESCE upsert would clobber R28
        state. This direction is invisible to the phantom/unaccounted checks (the
        column is a real schema column AND allowlisted), so it needs its own
        assertion (check 0)."""
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            with pytest.raises(AssertionError, match="graph_indexed"):
                assert_upsert_columns_consistent(
                    conn,
                    written_cols=list(upsert_writable_columns()) + ["graph_indexed"],
                )
