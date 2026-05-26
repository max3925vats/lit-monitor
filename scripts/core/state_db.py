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
from pathlib import Path
from typing import Any

from scripts.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)

# Bump this string whenever the extraction schema changes in a way that
# makes existing extraction_json incompatible with the current schema.
# Brain-build checks this against the stored kv_store value on startup (M8).
CURRENT_SCHEMA_VERSION: str = "M3"

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
    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            # Additive migrations — silently ignored if column already exists
            # (duplicate-column errors are normal on an already-migrated DB)
            for sql in [
                "ALTER TABLE papers ADD COLUMN keywords_json TEXT",
                "ALTER TABLE papers ADD COLUMN isbn TEXT",
                # M3: phase-based progress columns
                "ALTER TABLE brain_build_progress ADD COLUMN simple_complete INTEGER DEFAULT 0",
                "ALTER TABLE brain_build_progress ADD COLUMN complex_complete INTEGER DEFAULT 0",
            ]:
                try:
                    conn.execute(sql)
                except Exception as _alter_exc:
                    # "duplicate column name" is the expected SQLite error when the
                    # column already exists — treat it as a no-op.  Any *other* error
                    # is surfaced via strict_fallback so it doesn't pass silently.
                    if "duplicate column name" not in str(_alter_exc).lower():
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
        sql = (
            f"INSERT INTO papers ({col_names}) VALUES ({placeholders}) "
            f"ON CONFLICT(doi) DO UPDATE SET {update_clause}"
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
        col = f"pass{pass_num}_complete"
        # Build dynamic fully_complete condition based on schema max pass (I4)
        pass_checks = " AND ".join(f"pass{p}_complete = 1" for p in range(1, max_pass + 1))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE brain_build_progress SET {col} = ? WHERE zotero_key = ?",
                (1 if complete else 0, zotero_key),
            )
            conn.execute(
                f"UPDATE brain_build_progress SET fully_complete = 1 "
                f"WHERE zotero_key = ? AND {pass_checks}",
                (zotero_key,),
            )
    def mark_brain_build_phase(
        self, zotero_key: str, phase: str, complete: bool = True
    ) -> None:
        """Mark a single extraction phase complete for a brain-build item (M3).

        phase must be "simple" or "complex".  Sets fully_complete when both
        phases are done.  Complementary to mark_brain_build_pass() — new code
        should use this method; mark_brain_build_pass() is kept for backward
        compatibility with the legacy 3-pass system.
        """
        if phase not in ("simple", "complex"):
            raise ValueError(f"phase must be 'simple' or 'complex', got {phase!r}")
        col = f"{phase}_complete"
        with self._connect() as conn:
            conn.execute(
                f"UPDATE brain_build_progress SET {col} = ? WHERE zotero_key = ?",
                (1 if complete else 0, zotero_key),
            )
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
