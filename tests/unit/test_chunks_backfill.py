"""CB1: chunks backfill tests — closes the R28 chunks retry gap."""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest


class TestSchemaMigration:
    def test_chunks_indexed_column_present(self, tmp_path):
        """CB1: StateDB schema includes chunks_indexed column."""
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()]
        assert "chunks_indexed" in cols

    def test_existing_db_migration_idempotent(self, tmp_path):
        """CB1: migration adds chunks_indexed to a legacy papers table without disturbing data."""
        legacy = tmp_path / "legacy.db"
        conn = sqlite3.connect(legacy)
        conn.execute(
            "CREATE TABLE papers (doi TEXT PRIMARY KEY, fully_complete INTEGER DEFAULT 0)"
        )
        conn.execute(
            "INSERT INTO papers (doi, fully_complete) VALUES ('10.0/x', 1)"
        )
        conn.commit()
        conn.close()

        from lit_monitor.core.state_db import StateDB

        db = StateDB(legacy)
        with db._connect() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()]
            row = conn.execute(
                "SELECT doi, fully_complete, chunks_indexed FROM papers WHERE doi='10.0/x'"
            ).fetchone()
        assert "chunks_indexed" in cols
        # Legacy data preserved; new column defaults to 0.
        # Convert sqlite3.Row to tuple for comparison.
        assert tuple(row) == ("10.0/x", 1, 0)


class TestSetChunksIndexed:
    def test_method_updates_value(self, tmp_path):
        """CB1: set_chunks_indexed flips the flag to 1."""
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "s.db")
        with db._connect() as conn:
            conn.execute("INSERT INTO papers (doi) VALUES ('10.0/a')")
            conn.commit()

        db.set_chunks_indexed("10.0/a", 1)

        with db._connect() as conn:
            row = conn.execute(
                "SELECT chunks_indexed FROM papers WHERE doi='10.0/a'"
            ).fetchone()
        assert row[0] == 1

    def test_method_resets_to_zero(self, tmp_path):
        """CB1: set_chunks_indexed can clear the flag back to 0."""
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "s.db")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO papers (doi, chunks_indexed) VALUES ('10.0/a', 1)"
            )
            conn.commit()

        db.set_chunks_indexed("10.0/a", 0)

        with db._connect() as conn:
            row = conn.execute(
                "SELECT chunks_indexed FROM papers WHERE doi='10.0/a'"
            ).fetchone()
        assert row[0] == 0

    def test_missing_doi_is_silent(self, tmp_path):
        """CB1: set_chunks_indexed on a non-existent DOI does not raise."""
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "s.db")
        # Should not raise even when doi is absent.
        db.set_chunks_indexed("10.0/missing", 1)


class TestBackfillCore:
    @pytest.fixture
    def seeded_db(self, tmp_path):
        """Three papers for backfill tests.

        A: embeddings_indexed=1, chunks_indexed=0 — eligible for backfill.
        B: embeddings_indexed=1, chunks_indexed=1 — already done; skipped.
        C: embeddings_indexed=0, chunks_indexed=0 — not yet embedded; skipped.

        Uses upsert_paper to go through the real schema, then direct SQL
        to set test-specific column values.
        """
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "seeded.db")

        # Seed fulltext for paper A (the only one whose path needs to exist).
        (tmp_path / "a.md").write_text(
            "# Paper A\n\nThis is the body of paper A.\n\n## Methods\n\nMethods text here.\n"
        )

        # Insert papers via upsert_paper to respect the full schema.
        for paper_doi in ("10.0/a", "10.0/b", "10.0/c"):
            db.upsert_paper({
                "doi": paper_doi,
                "title": f"Paper {paper_doi}",
                "year": 2024,
                "source_type": "zotero",
            })

        # Set test state: embeddings_indexed, chunks_indexed, extraction_json.
        with db._connect() as conn:
            conn.execute(
                "UPDATE papers SET embeddings_indexed=1, chunks_indexed=0, extraction_json=? "
                "WHERE doi='10.0/a'",
                (json.dumps({"fulltext_path": str(tmp_path / "a.md")}),),
            )
            conn.execute(
                "UPDATE papers SET embeddings_indexed=1, chunks_indexed=1, extraction_json=? "
                "WHERE doi='10.0/b'",
                (json.dumps({"fulltext_path": str(tmp_path / "b.md")}),),
            )
            conn.execute(
                "UPDATE papers SET embeddings_indexed=0, chunks_indexed=0, extraction_json=? "
                "WHERE doi='10.0/c'",
                (json.dumps({"fulltext_path": str(tmp_path / "c.md")}),),
            )
            conn.commit()
        return db

    def test_backfill_picks_only_complete_and_unindexed(self, seeded_db, monkeypatch):
        """CB1: only papers with embeddings_indexed=1 AND chunks_indexed=0 are processed."""
        from lit_monitor.pipelines.chunks_backfill import backfill_chunks

        edb = MagicMock()
        edb.add_chunks.return_value = None
        result = backfill_chunks(seeded_db, edb)

        # Only paper A qualifies: complete + chunks_indexed=0.
        assert result["processed"] == 1
        assert result["succeeded"] == 1
        assert result["failed"] == 0

        # add_chunks called once with doi='10.0/a'.
        edb.add_chunks.assert_called_once()
        call_args = edb.add_chunks.call_args
        assert "10.0/a" in str(call_args)

    def test_backfill_marks_chunks_indexed_on_success(self, seeded_db):
        """CB1: chunks_indexed=1 is set after successful add_chunks."""
        from lit_monitor.pipelines.chunks_backfill import backfill_chunks

        edb = MagicMock()
        edb.add_chunks.return_value = None
        backfill_chunks(seeded_db, edb)

        with seeded_db._connect() as conn:
            row = conn.execute(
                "SELECT chunks_indexed FROM papers WHERE doi='10.0/a'"
            ).fetchone()
        assert row[0] == 1

    def test_backfill_per_paper_failure_continues(self, seeded_db):
        """CB1: per-paper failure is WARN-and-skip; flag stays 0."""
        from lit_monitor.pipelines.chunks_backfill import backfill_chunks

        edb = MagicMock()
        edb.add_chunks.side_effect = RuntimeError("chroma down")
        result = backfill_chunks(seeded_db, edb)

        assert result["failed"] == 1
        assert result["succeeded"] == 0

        # Flag stays 0 on failure (R28 invariant).
        with seeded_db._connect() as conn:
            row = conn.execute(
                "SELECT chunks_indexed FROM papers WHERE doi='10.0/a'"
            ).fetchone()
        assert row[0] == 0

    def test_backfill_doi_filter(self, seeded_db):
        """CB1: --doi filters to the specified paper; already-indexed DOI yields 0 processed."""
        from lit_monitor.pipelines.chunks_backfill import backfill_chunks

        edb = MagicMock()
        # Paper B already has chunks_indexed=1 — should be skipped.
        result = backfill_chunks(seeded_db, edb, doi="10.0/b")
        assert result["processed"] == 0

    def test_backfill_limit(self, seeded_db, tmp_path):
        """CB1: --limit caps the number of papers processed."""
        # Add two more eligible papers (embeddings_indexed=1, chunks_indexed=0)
        # so we have A, D, E all eligible.

        for d in ("10.0/d", "10.0/e"):
            fname = d.replace("/", "_") + ".md"
            (tmp_path / fname).write_text("# Body\n\nContent.\n")
            seeded_db.upsert_paper({
                "doi": d, "title": d, "year": 2024, "source_type": "zotero",
            })
            with seeded_db._connect() as conn:
                conn.execute(
                    "UPDATE papers SET embeddings_indexed=1, chunks_indexed=0, "
                    "extraction_json=? WHERE doi=?",
                    (json.dumps({"fulltext_path": str(tmp_path / fname)}), d),
                )
                conn.commit()

        from lit_monitor.pipelines.chunks_backfill import backfill_chunks

        edb = MagicMock()
        edb.add_chunks.return_value = None
        result = backfill_chunks(seeded_db, edb, limit=2)
        assert result["processed"] == 2

    def test_backfill_missing_fulltext_counts_as_failure(self, seeded_db):
        """CB1: missing fulltext_path on disk is a per-paper failure, not a crash."""
        from lit_monitor.pipelines.chunks_backfill import backfill_chunks

        # Overwrite paper A's extraction_json so fulltext_path points nowhere.
        with seeded_db._connect() as conn:
            conn.execute(
                "UPDATE papers SET extraction_json = ? WHERE doi = '10.0/a'",
                (json.dumps({"fulltext_path": "/nonexistent/path.md"}),),
            )
            conn.commit()

        edb = MagicMock()
        result = backfill_chunks(seeded_db, edb)
        assert result["failed"] == 1
        assert result["succeeded"] == 0
        edb.add_chunks.assert_not_called()
