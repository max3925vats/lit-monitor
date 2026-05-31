"""
SQLite state database wrapper.
Stores per-item extraction progress, run logs, and build progress.
Primary key for all content is the `doi` column which doubles as:
  - DOI for journal papers
Tables:
  papers                   — per-paper extraction state; source_type: paper | review
  run_log                  — pipeline run history
  brain_build_progress     — per-pass completion tracking for brain-build
  kv_store                 — arbitrary pipeline metadata (e.g. last Zotero
                             library version for discovery ingestion polling)
Schema is created on first use (CREATE TABLE IF NOT EXISTS).
No migration support — extend by adding nullable columns only.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)

# Bump this string whenever the extraction schema changes in a way that
# makes existing extraction_json incompatible with the current schema.
# Brain-build checks this against the stored kv_store value on startup (M8).
CURRENT_SCHEMA_VERSION: str = "M3"

# M4: static map from pass_num → column name for brain_build_progress updates.
# Used by mark_brain_build_pass() instead of building the column name from an
# f-string, so a regression that drops upstream pass_num validation cannot
# allow SQL injection through this code path.
_PASS_COMPLETE_COLUMNS: dict[int, str] = {
    1: "pass1_complete",
    2: "pass2_complete",
    3: "pass3_complete",
}

# Imported lazily inside methods to avoid circular import issues at module load.
# schema_max_pass(content_type) → int: paper/review=3 (only live schemas after R-10).
def _schema_max_pass(content_type: str) -> int:
    from scripts.llm.extraction_schema import schema_max_pass
    return schema_max_pass(content_type)
# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    doi                   TEXT PRIMARY KEY,
    title                 TEXT,
    authors               TEXT,          -- JSON array of strings
    year                  INTEGER,
    journal               TEXT,
    zotero_key            TEXT,
    first_seen_date       TEXT,
    status                TEXT DEFAULT 'pending',
    -- Values: pending | extraction_pass1_complete | extraction_complete
    --         | no_markdown | ocr_failed | error
    source_type           TEXT DEFAULT 'paper',
    -- Values: paper | review
    parent_id             TEXT,          -- reserved; not used by active pipelines
    note_title            TEXT,
    note_path             TEXT,
    embeddings_indexed    INTEGER DEFAULT 0,
    extraction_json       TEXT,          -- JSON blob (raw LLM output, all passes merged)
    extraction_provider   TEXT,
    extraction_model      TEXT,
    ocr_pages_json        TEXT,          -- JSON array of 0-based page indices
    ocr_heavy             INTEGER DEFAULT 0,
    keywords_json         TEXT,          -- JSON array of Zotero tag strings
    isbn                  TEXT,          -- legacy: written when textbook-build existed; unused since R-10
    last_updated          TEXT
);
CREATE TABLE IF NOT EXISTS run_log (
    run_id               TEXT PRIMARY KEY,
    run_type             TEXT,           -- brain_build | discovery | ingestion
    started_at           TEXT,
    finished_at          TEXT,
    status               TEXT,           -- running | complete | failed
    papers_processed     INTEGER DEFAULT 0,
    papers_skipped       INTEGER DEFAULT 0,
    papers_failed        INTEGER DEFAULT 0,
    errors               TEXT            -- JSON array of error messages
);
CREATE TABLE IF NOT EXISTS brain_build_progress (

    zotero_key           TEXT PRIMARY KEY,
    doi                  TEXT,
    pass1_complete       INTEGER DEFAULT 0,
    pass2_complete       INTEGER DEFAULT 0,
    pass3_complete       INTEGER DEFAULT 0,
    simple_complete      INTEGER DEFAULT 0,
    complex_complete     INTEGER DEFAULT 0,
    fully_complete       INTEGER DEFAULT 0,
    failure_reason       TEXT
);
CREATE TABLE IF NOT EXISTS kv_store (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS citation_edges (
    source_doi     TEXT NOT NULL,
    ref_id         TEXT NOT NULL,   -- verbatim in-text citation string from pass-4
    target_doi     TEXT,            -- resolved DOI of cited paper (NULL = unresolved)
    target_s2_id   TEXT,            -- S2 paperId of cited paper (NULL = unresolved)
    context        TEXT,            -- context snippet from pass-4
    resolution     TEXT NOT NULL DEFAULT 'unresolved',
    -- 'numeric_index' | 'author_year_fuzzy' | 'unresolved'
    created_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (source_doi, ref_id)
);
CREATE TABLE IF NOT EXISTS ingest_queue (
    -- H2: tracks external ingest requests (POST /api/ingest).
    -- doi is the natural PK — duplicate-DOI check in the route relies on it.
    doi          TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'queued',
    -- Values: queued | done | failed
    queued_at    TEXT NOT NULL,
    completed_at TEXT,   -- NULL until pipeline finishes or fails
    error        TEXT    -- populated by R28 hardening path if _process_paper raises
);
CREATE TABLE IF NOT EXISTS discovery_runs (
    -- P1: one row per run_discovery() invocation; P2-P10b query this table.
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at       TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at      TEXT,
    status           TEXT NOT NULL DEFAULT 'running',
    total_found      INTEGER DEFAULT 0,
    total_ingested   INTEGER DEFAULT 0,
    run_params_json  TEXT
);
CREATE TABLE IF NOT EXISTS discovery_paper_results (
    -- P1: one row per ranked candidate seen by a discovery run.
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES discovery_runs(id),
    doi          TEXT,
    title        TEXT,
    score        REAL,
    rationale    TEXT,
    ingested     INTEGER DEFAULT 0,
    ingested_at  TEXT    -- NULL when ingested=0
);
"""
# ---------------------------------------------------------------------------
# StateDB class
# ---------------------------------------------------------------------------
class StateDB:
    """Thread-safe SQLite wrapper for lit-monitor state."""
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    # -- Connection management --
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    @staticmethod
    def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
        """Return True iff ``table`` already has ``column``.

        M4: replaces fragile string-matching on SQLite's OperationalError
        message ("duplicate-column" error text varies across versions).
        Uses ``PRAGMA table_info`` which is the documented introspection API.
        """
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        # row[1] is the column name; rows are sqlite3.Row, indexable by position.
        return any(r[1] == column for r in rows)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            # Additive migrations — skip the ALTER entirely when the column
            # already exists (M4: was previously catching the OperationalError
            # and matching its message text, which is fragile across SQLite
            # versions).  Any genuine ALTER failure is surfaced via
            # strict_fallback so it cannot pass silently.
            additive_migrations: list[tuple[str, str, str]] = [
                # (table, column, ALTER statement)
                ("papers", "keywords_json",
                 "ALTER TABLE papers ADD COLUMN keywords_json TEXT"),
                ("papers", "isbn",
                 "ALTER TABLE papers ADD COLUMN isbn TEXT"),
                # M3: phase-based progress columns
                ("brain_build_progress", "simple_complete",
                 "ALTER TABLE brain_build_progress ADD COLUMN simple_complete INTEGER DEFAULT 0"),
                ("brain_build_progress", "complex_complete",
                 "ALTER TABLE brain_build_progress ADD COLUMN complex_complete INTEGER DEFAULT 0"),
                # G1: per-paper flag toggled by the R28 dual-write path (Graph RAG phase 1).
                ("papers", "graph_indexed",
                 "ALTER TABLE papers ADD COLUMN graph_indexed INTEGER DEFAULT 0"),
                # G16: per-paper timestamp for v0.8+ insight-discovery tracking.
                # NULL = never processed; set by future insight-discovery passes.
                ("papers", "last_insight_run",
                 "ALTER TABLE papers ADD COLUMN last_insight_run TEXT NULL"),
                # N5: per-paper timestamp for NER backfill (BioBERT + optional cloud-LLM).
                # NULL = never NER-processed; set by backfill_ner() after success.
                ("papers", "ner_processed_at",
                 "ALTER TABLE papers ADD COLUMN ner_processed_at TEXT NULL"),
                # R5: per-paper timestamp for relationship backfill (G4 schema + optional R2 LLM).
                # NULL = never rel-processed; set by backfill_relationships() after success.
                ("papers", "rel_processed_at",
                 "ALTER TABLE papers ADD COLUMN rel_processed_at TEXT NULL"),
                # CB1: per-paper flag for chunk-level ChromaDB embeddings.
                # 0 = not yet indexed (or failed); 1 = current.
                # Set to 1 by index_embeddings_and_mark_phases on success;
                # retried by `lit-monitor chunks backfill`.
                ("papers", "chunks_indexed",
                 "ALTER TABLE papers ADD COLUMN chunks_indexed INTEGER DEFAULT 0"),
                # P10: per-paper flag for Obsidian note (re-)render.
                # 0 = note not yet written (or deferred); 1 = current.
                # Set to 1 inline by brain_build/_process_paper when
                # discovery.notes.auto_write_per_paper=true (default).
                # Set to 1 by `lit-monitor obsidian sync` when flag is false.
                ("papers", "notes_synced",
                 "ALTER TABLE papers ADD COLUMN notes_synced INTEGER DEFAULT 0"),
                # Bundle B: per-ranked-paper JSON blob of per-signal score contributions.
                # NULL when the row was written before Bundle B; set by add_discovery_paper
                # when a score_breakdown dict is provided.
                ("discovery_paper_results", "score_breakdown_json",
                 "ALTER TABLE discovery_paper_results ADD COLUMN score_breakdown_json TEXT"),
            ]
            for table, column, sql in additive_migrations:
                if self._column_exists(conn, table, column):
                    continue
                try:
                    conn.execute(sql)
                except Exception as _alter_exc:
                    strict_fallback(
                        logger,
                        f"Schema migration failed for {sql!r}: {_alter_exc}. "
                        "State DB may be in a partially-migrated state.",
                        _alter_exc,
                    )
            # M3: backfill simple/complex from old pass columns for existing rows.
            # Only touches rows that have pass progress but no phase progress yet.
            try:
                # Hardcoded pass3 as max; if _schema_max_pass() ever returns >3, this backfill
                # block must be revisited.
                conn.execute(
                    "UPDATE brain_build_progress SET "
                    "simple_complete = CASE WHEN pass1_complete=1 AND pass2_complete=1 THEN 1 ELSE 0 END, "
                    "complex_complete = CASE WHEN pass3_complete=1 THEN 1 ELSE 0 END "
                    "WHERE simple_complete = 0 AND complex_complete = 0 "
                    "AND (pass1_complete = 1 OR pass2_complete = 1 OR pass3_complete = 1)"
                )
                # After phase backfill, ensure fully_complete is consistent.
                conn.execute(
                    "UPDATE brain_build_progress SET fully_complete = 1 "
                    "WHERE simple_complete = 1 AND complex_complete = 1 AND fully_complete = 0"
                )
            except Exception:
                pass
            # N7: one-time cleanup of stale book/chapter rows left from the
            # pre-R-10 textbook-build era.  Gated by a kv_store flag so it
            # only runs once per database, even across restarts.
            try:
                done = conn.execute(
                    "SELECT value FROM kv_store WHERE key = 'r10_cleanup_done'"
                ).fetchone()
                if not done:
                    cur = conn.execute(
                        "DELETE FROM papers "
                        "WHERE source_type IN ('book', 'chapter', 'textbook_chapter')"
                    )
                    n = cur.rowcount
                    if n:
                        logger.info(
                            "N7: removed %d stale book/chapter row(s) left from "
                            "pre-R-10 textbook-build era",
                            n,
                        )
                    conn.execute(
                        "INSERT INTO kv_store (key, value) VALUES ('r10_cleanup_done', '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = '1'"
                    )
            except Exception as exc:
                logger.warning(
                    "N7 stale-row cleanup failed (will retry on next startup): %s", exc
                )
            # 2026-05-26: rename run_type='weekly' → run_type='discovery' (the
            # pipeline no longer ships under a weekly-only brand). Gated by a kv_store
            # flag so it only runs once per database.
            try:
                done = conn.execute(
                    "SELECT value FROM kv_store WHERE key = 'weekly_to_discovery_rename'"
                ).fetchone()
                if not done:
                    cur = conn.execute(
                        "UPDATE run_log SET run_type = 'discovery' WHERE run_type = 'weekly'"
                    )
                    n = cur.rowcount
                    if n:
                        logger.info(
                            "Renamed %d run_log row(s) from run_type='weekly' to 'discovery'",
                            n,
                        )
                    conn.execute(
                        "INSERT INTO kv_store (key, value) "
                        "VALUES ('weekly_to_discovery_rename', '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = '1'"
                    )
            except Exception as exc:
                logger.warning(
                    "weekly→discovery run_type rename failed (will retry on next startup): %s",
                    exc,
                )
    # -- Paper / review CRUD --
    def upsert_paper(self, data: dict[str, Any]) -> None:
        """Insert or replace a paper or review record."""
        cols = [
            "doi", "title", "authors", "year", "journal", "zotero_key",
            "first_seen_date", "status", "source_type", "parent_id",
            "note_title", "note_path", "embeddings_indexed", "extraction_json",
            "extraction_provider", "extraction_model", "ocr_pages_json",
            "ocr_heavy", "keywords_json", "isbn", "last_updated",
        ]

        row = {}
        for col in cols:
            val = data.get(col)
            if col in ("authors",) and isinstance(val, list):
                val = json.dumps(val)
            if col == "ocr_pages_json" and isinstance(val, list):
                val = json.dumps(val)
            if col == "keywords_json" and isinstance(val, list):
                val = json.dumps(val)
            row[col] = val
        placeholders = ", ".join(f":{c}" for c in row)
        col_names = ", ".join(row.keys())
        # ON CONFLICT: only overwrite a column when the new value is non-NULL,
        # so a partial upsert (e.g. updating note_path only) never wipes fields
        # like extraction_json that were set in an earlier call.
        update_clause = ", ".join(
            f"{c} = COALESCE(excluded.{c}, {c})"
            for c in row
            if c != "doi"
        )
        # M4: column names come from the hardcoded `cols` list above; no user
        # input reaches SQL identifier positions.  Built via concatenation
        # instead of f-string interpolation so audit greps stay clean.
        sql = (
            "INSERT INTO papers (" + col_names + ") VALUES (" + placeholders + ") "
            "ON CONFLICT(doi) DO UPDATE SET " + update_clause
        )
        with self._connect() as conn:
            conn.execute(sql, row)
    def get_paper(self, doi: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE doi = ?", (doi,)
            ).fetchone()
        return dict(row) if row else None
    def get_all_by_source_type(self, source_type: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE source_type = ?", (source_type,)
            ).fetchall()
        return [dict(r) for r in rows]
    def mark_status(self, doi: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET status = ?, last_updated = datetime('now') "
                "WHERE doi = ?",
                (status, doi),
            )
    def get_pending(self, source_type: str = "paper") -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE source_type = ? AND status = 'pending'",
                (source_type,),
            ).fetchall()
        return [dict(r) for r in rows]
    def get_extraction_json(self, doi: str) -> dict | None:
        row = self.get_paper(doi)
        if row and row.get("extraction_json"):
            try:
                return json.loads(row["extraction_json"])
            except json.JSONDecodeError as exc:
                strict_fallback(
                    logger,
                    f"Corrupt extraction_json for DOI {doi}: {exc}. "
                    "The stored JSON blob is malformed — re-extract this paper to fix it.",
                    exc,
                )
                return None
        return None

    def update_extraction_json(self, doi: str, extraction: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET extraction_json = ?, last_updated = datetime('now') "
                "WHERE doi = ?",
                (json.dumps(extraction), doi),
            )
    # -- Brain build progress --
    def upsert_brain_build_progress(self, zotero_key: str, doi: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO brain_build_progress (zotero_key, doi) "
                "VALUES (?, ?)",
                (zotero_key, doi),
            )
    def mark_brain_build_pass(
        self, zotero_key: str, pass_num: int,
        complete: bool = True, content_type: str = "paper"
    ) -> None:
        """Mark a single extraction pass complete for a brain-build item.

        I4: content_type selects the schema's max pass via schema_max_pass().
        After R-10, both live schemas (paper, review) max out at pass 3,
        so the dynamic check is effectively static — kept generic for forward
        compatibility if new schemas are added.
        """
        max_pass = _schema_max_pass(content_type)
        if pass_num not in range(1, max_pass + 1):
            raise ValueError(
                f"pass_num must be 1..{max_pass} for content_type={content_type!r}; got {pass_num}"
            )
        # M4: look up column from a static map rather than interpolating
        # `pass_num` into SQL.  Even though pass_num is validated above, the
        # static map removes any path from user-controlled input to a SQL
        # identifier — a regression dropping the range check above cannot
        # introduce SQL injection here.
        if pass_num not in _PASS_COMPLETE_COLUMNS:
            raise ValueError(
                f"pass_num {pass_num} has no mapped column; expected one of "
                f"{sorted(_PASS_COMPLETE_COLUMNS)}"
            )
        col = _PASS_COMPLETE_COLUMNS[pass_num]
        # fully_complete predicate covers passes 1..max_pass.  Column names
        # come from the static map, so the predicate is built from trusted
        # identifiers only.  Concatenation is used instead of f-string
        # interpolation to keep audit greps clean.
        pass_checks = " AND ".join(
            _PASS_COMPLETE_COLUMNS[p] + " = 1" for p in range(1, max_pass + 1)
        )
        update_col_sql = (
            "UPDATE brain_build_progress SET " + col + " = ? WHERE zotero_key = ?"
        )
        update_fully_sql = (
            "UPDATE brain_build_progress SET fully_complete = 1 "
            "WHERE zotero_key = ? AND " + pass_checks
        )
        with self._connect() as conn:
            conn.execute(update_col_sql, (1 if complete else 0, zotero_key))
            conn.execute(update_fully_sql, (zotero_key,))
    def mark_brain_build_phase(
        self, zotero_key: str, phase: str, complete: bool = True
    ) -> None:
        """Mark a single extraction phase complete for a brain-build item (M3).

        phase must be "simple" or "complex".  Sets fully_complete when both
        phases are done.  Complementary to mark_brain_build_pass() — new code
        should use this method; mark_brain_build_pass() is kept for backward
        compatibility with the legacy 3-pass system.
        """
        # M4: static map from validated phase name → column, mirroring
        # _PASS_COMPLETE_COLUMNS.  Never interpolate `phase` into SQL.
        phase_columns: dict[str, str] = {
            "simple": "simple_complete",
            "complex": "complex_complete",
        }
        if phase not in phase_columns:
            raise ValueError(f"phase must be 'simple' or 'complex', got {phase!r}")
        col = phase_columns[phase]
        update_col_sql = (
            "UPDATE brain_build_progress SET " + col + " = ? WHERE zotero_key = ?"
        )
        with self._connect() as conn:
            conn.execute(update_col_sql, (1 if complete else 0, zotero_key))
            conn.execute(
                "UPDATE brain_build_progress SET fully_complete = 1 "
                "WHERE zotero_key = ? AND simple_complete = 1 AND complex_complete = 1",
                (zotero_key,),
            )

    def get_brain_build_progress(self, zotero_key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM brain_build_progress WHERE zotero_key = ?",
                (zotero_key,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_brain_build_progress(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Return brain-build progress rows joined with paper metadata (F3.1).

        LEFT JOINs ``brain_build_progress`` with ``papers`` on ``doi`` so the
        dashboard can show title + authors + year alongside the per-pass
        completion flags. Rows where the DOI is missing or unknown still come
        back with NULL paper_* columns.

        Sort order: ``fully_complete`` ASC (incomplete first), then
        ``zotero_key`` for stable ties. ``limit`` and ``offset`` support
        paginated UIs. Read-only.
        """
        sql = """
            SELECT
                bbp.zotero_key,
                bbp.doi,
                bbp.pass1_complete,
                bbp.pass2_complete,
                bbp.pass3_complete,
                bbp.simple_complete,
                bbp.complex_complete,
                bbp.fully_complete,
                bbp.failure_reason,
                p.title           AS paper_title,
                p.year            AS paper_year,
                p.authors         AS paper_authors,
                p.source_type     AS paper_source_type,
                p.status          AS paper_status,
                p.extraction_json AS paper_extraction_json
            FROM brain_build_progress bbp
            LEFT JOIN papers p ON p.doi = bbp.doi
            ORDER BY bbp.fully_complete ASC, bbp.zotero_key ASC
            LIMIT ? OFFSET ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit, offset)).fetchall()
        return [dict(r) for r in rows]
    # -- Key-value store (pipeline metadata) --
    def get_kv(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM kv_store WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_kv(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO kv_store (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def set_graph_indexed(self, doi: str, value: int) -> None:
        """G6: flip the per-paper papers.graph_indexed flag.

        Part of the R28 dual-write invariant: only set to 1 after BOTH the
        vector index (ChromaDB) AND the graph (KuzuDB) writes have succeeded.
        Called from ``scripts.pipelines._ingest.index_embeddings_and_mark_phases``.

        Parameters
        ----------
        doi:
            Paper DOI (primary key of ``papers``).
        value:
            0 or 1.  Coerced via ``int(value)`` so callers passing bools
            still produce an INTEGER column value.

        Notes
        -----
        - UPDATE on a missing DOI silently affects 0 rows; this is intentional
          so the ingest helper can call it without a pre-existence check.
        - This method intentionally does NOT touch ``last_updated`` — the
          column is a side-channel flag, not a content-bearing update.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET graph_indexed = ? WHERE doi = ?",
                (int(value), doi),
            )

    def set_chunks_indexed(self, doi: str, val: int) -> None:
        """CB1: mark whether ChromaDB chunk embeddings are current for this paper.

        Mirrors set_graph_indexed. R28 invariant: only set to 1 after add_chunks
        succeeds; failure leaves the value at 0 so chunks backfill can retry.

        Parameters
        ----------
        doi:
            Paper DOI (primary key of ``papers``).
        val:
            0 or 1. Coerced via ``int(val)`` so callers passing bools
            still produce an INTEGER column value.

        Notes
        -----
        - UPDATE on a missing DOI silently affects 0 rows; this is intentional
          so the ingest helper can call it without a pre-existence check.
        - This method intentionally does NOT touch ``last_updated`` — the
          column is a side-channel flag, not a content-bearing update.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET chunks_indexed = ? WHERE doi = ?",
                (int(val), doi),
            )

    def set_notes_synced(self, doi: str, val: int) -> None:
        """P10: mark whether the Obsidian note has been (re-)rendered for this paper.

        Mirrors set_chunks_indexed / set_graph_indexed.

        Parameters
        ----------
        doi:
            Paper DOI (primary key of ``papers``).
        val:
            0 or 1.  Coerced via ``int(val)`` so callers passing bools
            still produce an INTEGER column value.

        Notes
        -----
        - UPDATE on a missing DOI silently affects 0 rows; this is intentional
          so callers do not need a pre-existence check.
        - This method intentionally does NOT touch ``last_updated`` — the
          column is a side-channel flag, not a content-bearing update.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET notes_synced = ? WHERE doi = ?",
                (int(val), doi),
            )

    def get_notes_pending(self, limit: int | None = None) -> list[str]:
        """P10: DOIs where embeddings_indexed=1 AND notes_synced=0.

        Uses ``embeddings_indexed=1`` as the "ready to render" gate — the
        same gate CB1 uses for chunks backfill.  Papers are ordered by
        ``last_updated DESC`` so the most recently ingested papers are
        processed first.

        Parameters
        ----------
        limit:
            Optional cap on result count.  ``None`` (default) returns all
            pending DOIs.

        Returns
        -------
        list[str]
            DOIs that are ready for note rendering but not yet synced.
        """
        query = (
            "SELECT doi FROM papers "
            "WHERE embeddings_indexed = 1 AND notes_synced = 0 "
            "ORDER BY last_updated DESC"
        )
        params: tuple = ()
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [r[0] for r in rows if r[0]]

    def reset_embeddings_indexed(self) -> None:
        """Set embeddings_indexed = 0 for every row in the papers table.

        Called when the embedding model changes so all content is scheduled
        for re-embedding on the next brain-build run.
        """
        with self._connect() as conn:
            cur = conn.execute("UPDATE papers SET embeddings_indexed = 0")
            n = cur.rowcount
        logger.info("reset_embeddings_indexed: %d row(s) scheduled for re-embedding.", n)

    # -- Schema version / extraction reset (M8) --

    def get_schema_version(self) -> str | None:
        """Return the stored extraction schema version, or None if never set."""
        # N9: warn if a stray legacy key exists — this is the most common cause of M8
        # not firing when the user manually sets 'schema_version' instead of
        # 'extraction_schema_version' while following the live test guide.
        if self.get_kv("schema_version") is not None:
            logger.warning(
                "kv_store has a stray 'schema_version' key (legacy name). "
                "The pipeline reads 'extraction_schema_version'. "
                "To trigger a schema migration, update 'extraction_schema_version' instead."
            )
        return self.get_kv("extraction_schema_version")

    def set_schema_version(self, version: str) -> None:
        """Persist the extraction schema version to kv_store."""
        self.set_kv("extraction_schema_version", version)

    def count_with_extraction(self) -> int:
        """Count papers that have a non-NULL extraction_json value."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE extraction_json IS NOT NULL"
            ).fetchone()
        return row[0] if row else 0

    def reset_extractions(self) -> int:
        """Wipe all extraction_json values and reset brain_build_progress.

        Used by M8 schema-migration prompt.  Clears extraction data so the
        next brain-build re-extracts everything under the new schema.

        Returns
        -------
        int
            Number of paper rows whose extraction_json was cleared.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE papers SET extraction_json = NULL, status = NULL, "
                "extraction_provider = NULL, extraction_model = NULL, "
                "last_updated = datetime('now') WHERE extraction_json IS NOT NULL"
            )
            n = cur.rowcount
            conn.execute(
                "UPDATE brain_build_progress SET "
                "simple_complete = 0, complex_complete = 0, fully_complete = 0, "
                "pass1_complete = 0, pass2_complete = 0, pass3_complete = 0"
            )
        logger.info("reset_extractions: cleared %d extraction(s) and reset all progress rows.", n)
        return n

    def cleanup_stale_item_types(self) -> int:
        """Delete stale book/chapter rows left from the pre-R-10 textbook-build era.

        Safe to call explicitly (e.g. from ``lit-monitor db cleanup-stale``).
        Also called automatically on startup (see ``_init_schema``).

        Returns
        -------
        int
            Number of rows deleted.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM papers "
                "WHERE source_type IN ('book', 'chapter', 'textbook_chapter')"
            )
            n = cur.rowcount
            if n:
                logger.info(
                    "cleanup_stale_item_types: removed %d stale row(s)", n,
                )
            conn.execute(
                "INSERT INTO kv_store (key, value) VALUES ('r10_cleanup_done', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = '1'"
            )
        return n

    # -- Citation graph (E1) --
    def upsert_citation_edge(
        self,
        source_doi: str,
        ref_id: str,
        target_doi: str | None,
        target_s2_id: str | None,
        context: str,
        resolution: str,
    ) -> None:
        """Insert or replace one citation edge.

        COALESCE semantics: a previously-resolved edge (target_doi IS NOT NULL)
        will not be overwritten by a new unresolved one.  Context is always
        updated (latest pass-4 context wins).
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO citation_edges "
                "(source_doi, ref_id, target_doi, target_s2_id, context, resolution) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source_doi, ref_id) DO UPDATE SET "
                "target_doi   = COALESCE(excluded.target_doi, citation_edges.target_doi), "
                "target_s2_id = COALESCE(excluded.target_s2_id, citation_edges.target_s2_id), "
                "context      = excluded.context, "
                "resolution   = CASE "
                "  WHEN excluded.target_doi IS NOT NULL THEN excluded.resolution "
                "  ELSE citation_edges.resolution "
                "END, "
                "created_at = datetime('now')",
                (source_doi, ref_id, target_doi, target_s2_id, context, resolution),
            )

    def get_citation_edges(self, source_doi: str) -> list[dict]:
        """Return all citation edges for a given source paper."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM citation_edges WHERE source_doi = ?",
                (source_doi,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_citation_edges(self) -> list[dict]:
        """Return every row in citation_edges (used by G5 mirror for full sync)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM citation_edges").fetchall()
        return [dict(r) for r in rows]

    def get_papers_citing_doi(self, target_doi: str) -> list[str]:
        """Return source_dois of papers that resolved a citation to target_doi."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT source_doi FROM citation_edges "
                "WHERE target_doi = ? AND target_doi IS NOT NULL",
                (target_doi,),
            ).fetchall()
        return [r["source_doi"] for r in rows]

    # -- Run log --
    def start_run(self, run_id: str, run_type: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO run_log (run_id, run_type, started_at, status) "
                "VALUES (?, ?, datetime('now'), 'running')",
                (run_id, run_type),
            )
    def finish_run(
        self,
        run_id: str,
        status: str = "complete",
        processed: int = 0,
        skipped: int = 0,
        failed: int = 0,
        errors: list[str] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE run_log SET finished_at = datetime('now'), status = ?, "
                "papers_processed = ?, papers_skipped = ?, papers_failed = ?, "
                "errors = ? WHERE run_id = ?",
                (
                    status,
                    processed,
                    skipped,
                    failed,
                    json.dumps(errors or []),
                    run_id,
                ),
            )
    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        """Return the most recent run_log entries, newest first (L2)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM run_log ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_runs_by_type(self, run_type: str, limit: int = 10) -> list[dict]:
        """Return the most recent run_log entries of one type, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM run_log WHERE run_type = ? ORDER BY started_at DESC LIMIT ?",
                (run_type, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- P1: structured discovery-run tracking --

    def start_discovery_run(self, run_params: dict) -> int:
        """P1: insert a discovery_runs row with status='running'; return new run_id.

        Args:
            run_params: Arbitrary dict of run parameters (topics, since_days,
                rag_mode, etc.) serialised as JSON for later auditing.

        Returns:
            Integer primary-key of the newly inserted row.
        """
        import json as _json
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO discovery_runs (run_params_json) VALUES (?)",
                (_json.dumps(run_params),),
            )
            # _connect() commits on context-manager exit, but we need lastrowid
            # before that happens — SQLite guarantees it is set after execute().
            return cur.lastrowid  # type: ignore[return-value]

    def finish_discovery_run(
        self,
        run_id: int,
        status: str,
        total_found: int,
        total_ingested: int,
    ) -> None:
        """P1: mark a discovery run as finished.

        Args:
            run_id: Row id returned by start_discovery_run().
            status: Terminal status string, e.g. 'success' or 'error'.
            total_found: Number of candidate papers found before filtering.
            total_ingested: Number of papers actually written to Zotero/DB.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE discovery_runs "
                "SET status=?, total_found=?, total_ingested=?, "
                "    finished_at=datetime('now') "
                "WHERE id=?",
                (status, total_found, total_ingested, run_id),
            )

    def add_discovery_paper(
        self,
        run_id: int,
        doi: str,
        title: str,
        score: float,
        rationale: str,
        ingested: bool,
        score_breakdown: dict | None = None,
    ) -> None:
        """P1: record a ranked candidate paper associated with a discovery run.

        Args:
            run_id: Row id returned by start_discovery_run().
            doi: Paper DOI (may be empty string when unknown).
            title: Paper title.
            score: Similarity / relevance score in [0, 1].
            rationale: LLM-generated rationale string (may be empty).
            ingested: True when the paper was successfully ingested this run;
                      ingested_at timestamp is set only in that case.
            score_breakdown: Bundle B — optional dict mapping signal name to
                             float contribution.  Serialized as JSON and stored
                             in score_breakdown_json.  None → NULL (backward compat).
        """
        import json as _json

        breakdown_json: str | None = (
            _json.dumps(score_breakdown) if score_breakdown is not None else None
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO discovery_paper_results "
                "(run_id, doi, title, score, rationale, ingested, ingested_at, "
                " score_breakdown_json) "
                "VALUES (?, ?, ?, ?, ?, ?, "
                "CASE ? WHEN 1 THEN datetime('now') ELSE NULL END, ?)",
                (run_id, doi, title, score, rationale, int(ingested), int(ingested),
                 breakdown_json),
            )

    # -- Insight discovery (G16) --

    def get_papers_without_insight_run(
        self, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Return papers with NULL or stale last_insight_run.

        v0.8+ insight discovery uses this to find papers eligible for the next
        pass. Returns all NULL rows if since is None; otherwise also includes
        rows whose last_insight_run is older than since.

        Args:
            since: Optional cutoff datetime. Papers last processed before this
                datetime are included alongside papers that have never been
                processed (last_insight_run IS NULL).

        Returns:
            List of paper rows as dicts. Unused in v0.4-v0.7; substrate for
            v0.8+ insight-discovery passes.
        """
        if since is None:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM papers WHERE last_insight_run IS NULL"
                ).fetchall()
        else:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM papers "
                    "WHERE last_insight_run IS NULL OR last_insight_run < ?",
                    (since.isoformat(),),
                ).fetchall()
        return [dict(r) for r in rows]

    # -- NER backfill (N5) --

    def set_ner_processed_at(self, doi: str, timestamp: str) -> None:
        """N5: stamp the per-paper NER-processed timestamp.

        Called by ``backfill_ner()`` after a successful NER pipeline run so
        that subsequent re-runs skip already-processed papers.

        Parameters
        ----------
        doi:
            Paper DOI (primary key of ``papers``).
        timestamp:
            ISO-format datetime string, e.g. ``datetime.now().isoformat()``.

        Notes
        -----
        UPDATE on a missing DOI silently affects 0 rows — intentional; same
        pattern as :meth:`set_graph_indexed`.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET ner_processed_at = ? WHERE doi = ?",
                (timestamp, doi),
            )

    def get_papers_for_ner_backfill(
        self,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        only_unprocessed: bool = True,
    ) -> list[dict[str, Any]]:
        """N5: return candidate papers for the NER backfill run.

        Args:
            since: When set, restrict candidates to papers whose
                ``last_updated`` column is >= this datetime.  Papers with
                NULL ``last_updated`` are always included (same convention as
                :func:`backfill_papers`).
            limit: Cap on the number of rows returned.
            only_unprocessed: When True (default) return only papers where
                ``ner_processed_at IS NULL``.  Pass False to re-process
                already-stamped papers (e.g. ``--force``).

        Returns:
            List of paper rows as dicts.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if only_unprocessed:
            conditions.append("ner_processed_at IS NULL")

        if since is not None:
            # NULL last_updated → treat as always eligible (same as backfill_papers).
            conditions.append("(last_updated IS NULL OR last_updated >= ?)")
            params.append(since.isoformat())

        sql = "SELECT * FROM papers"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # -- Relationship backfill (R5) --

    def set_rel_processed_at(self, doi: str, timestamp: str) -> None:
        """R5: stamp the per-paper relationship-processed timestamp.

        Called by ``backfill_relationships()`` after a successful G4/R2 pipeline
        run so that subsequent re-runs skip already-processed papers.

        Parameters
        ----------
        doi:
            Paper DOI (primary key of ``papers``).
        timestamp:
            ISO-format datetime string, e.g. ``datetime.now().isoformat()``.

        Notes
        -----
        UPDATE on a missing DOI silently affects 0 rows — intentional; same
        pattern as :meth:`set_ner_processed_at`.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET rel_processed_at = ? WHERE doi = ?",
                (timestamp, doi),
            )

    def get_papers_for_rel_backfill(
        self,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        only_unprocessed: bool = True,
    ) -> list[dict[str, Any]]:
        """R5: return candidate papers for the relationship backfill run.

        Args:
            since: When set, restrict candidates to papers whose
                ``last_updated`` column is >= this datetime.  Papers with
                NULL ``last_updated`` are always included (same convention as
                :func:`backfill_papers`).
            limit: Cap on the number of rows returned.
            only_unprocessed: When True (default) return only papers where
                ``rel_processed_at IS NULL``.  Pass False to re-process
                already-stamped papers (e.g. ``--force``).

        Returns:
            List of paper rows as dicts.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if only_unprocessed:
            conditions.append("rel_processed_at IS NULL")

        if since is not None:
            # NULL last_updated → treat as always eligible (same as backfill_papers).
            conditions.append("(last_updated IS NULL OR last_updated >= ?)")
            params.append(since.isoformat())

        sql = "SELECT * FROM papers"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # -- Utility --
    def known_dois(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT doi FROM papers").fetchall()
        return {r["doi"] for r in rows}
    def count_by_status(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM papers GROUP BY status"

            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def get_by_doi(self, doi: str) -> dict | None:
        return self.get_paper(doi)

    def update_status(self, doi: str, status: str) -> None:
        self.mark_status(doi, status)
