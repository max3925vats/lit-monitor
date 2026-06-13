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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lit_monitor.core.strict_mode import strict_fallback

if TYPE_CHECKING:
    import numpy as np  # type-only: numpy is imported lazily inside helpers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewClusterSpec:
    """B3: one cluster to persist via StateDB.replace_clusters().

    inherited_id is the existing cluster id to reuse (stable-ID mapping) or
    None to insert a fresh row. display_name None → 'Cluster <id>' fallback is
    applied INSIDE the same transaction once the row id is known.
    """
    inherited_id: int | None
    display_name: str | None
    n_papers: int
    cohesion_score: float | None
    centroid_blob: bytes

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
    from lit_monitor.llm.extraction_schema import schema_max_pass
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
    run_params_json  TEXT,
    run_log_id       TEXT,         -- FK → run_log.run_id (uuid); NULL for pre-FK historical rows
    trigger          TEXT NOT NULL DEFAULT 'manual'  -- 'manual' (web/CLI) | 'scheduled' (OS scheduler)
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
CREATE TABLE IF NOT EXISTS embedding_provenance (
    -- Bundle F: tracks every ChromaDB collection's provider/model/dim provenance.
    -- ChromaDB collections are dim-locked once created; this table lets EmbeddingsDB
    -- detect when constructor args would mismatch an existing collection and override them.
    collection_name TEXT PRIMARY KEY,
    provider        TEXT NOT NULL,          -- 'ollama' | 'litellm'
    model           TEXT NOT NULL,          -- e.g. 'mxbai-embed-large', 'text-embedding-3-large'
    dim             INTEGER NOT NULL,       -- embedding dimension (1024, 3072, etc.)
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    is_current      INTEGER NOT NULL DEFAULT 1  -- 1 for the active collection
);
CREATE TABLE IF NOT EXISTS domain_focus_extracted (
    -- Bundle G (v0.9): structured focus areas extracted by LLM from the user's
    -- free-text domain_context paragraph. One row per item. Field type follows a
    -- closed vocabulary: topic | method | material | adjacent_field | exclusion.
    -- Consumed by the ranker's entity-overlap signal and by the trending-concept
    -- suggester. REPLACE semantics on save_domain_extraction — the table is
    -- wiped before a new extraction lands, never merged.
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    field_type          TEXT NOT NULL,    -- topic|method|material|adjacent_field|exclusion
    value               TEXT NOT NULL,
    confidence          REAL,
    user_confirmed      INTEGER NOT NULL DEFAULT 0,
    last_analyzed_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_dfe_field_type ON domain_focus_extracted(field_type);

-- Bundle C (v0.9): theme-clustered library centroids.
-- clusters stores k-means results; cluster_assignments maps each paper to its
-- nearest centroid. archived=1 rows are soft-deleted at the next recompute —
-- kept for audit trail and to allow stable-ID mapping across recomputes.
CREATE TABLE IF NOT EXISTS clusters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name    TEXT NOT NULL,
    n_papers        INTEGER NOT NULL DEFAULT 0,
    cohesion_score  REAL,
    centroid_blob   BLOB,           -- np.float32 bytes, shape (dim,)
    computed_at     TEXT NOT NULL DEFAULT (datetime('now')),
    archived        INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cluster_assignments (
    doi                  TEXT NOT NULL,
    cluster_id           INTEGER NOT NULL,
    distance_to_centroid REAL,
    assigned_at          TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (doi, cluster_id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(id),
    FOREIGN KEY (doi)        REFERENCES papers(doi)
);
CREATE INDEX IF NOT EXISTS idx_cluster_assignments_cluster ON cluster_assignments(cluster_id);

-- Bundle E (v0.9): "trending"-concept suggestions from graph mention-count growth.
-- "Trending" here = library-activity recency: growth is measured by bucketing
-- MENTIONS edges on extracted_at (INGEST time), so it reflects concepts recently
-- ADDED to the library, not field-level publication trends.
-- One row per concept+detection-run. user_action: pending|accepted|dismissed.
-- Cooldown is enforced by querying action_at < now - cooldown_days.
CREATE TABLE IF NOT EXISTS trending_concepts_suggested (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_text    TEXT NOT NULL,
    concept_type    TEXT NOT NULL,    -- topic|method|material|keyword|author
    n_mentions_new  INTEGER NOT NULL,
    n_mentions_prev INTEGER NOT NULL,
    growth_rate     REAL NOT NULL,
    suggested_at    TEXT NOT NULL DEFAULT (datetime('now')),
    user_action     TEXT,             -- accepted|dismissed|pending
    action_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_trending_action ON trending_concepts_suggested(user_action);

-- Bundle H (v0.9): user feedback events for active-learning (Bundle J substrate).
-- signal_type is a closed vocabulary: opened|saved|dismissed|rated|thumbs_up|thumbs_down
-- weight is pre-computed at insert time using the Rocchio-style map below.
-- rating is NULL unless signal_type='rated'.
CREATE TABLE IF NOT EXISTS feedback_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doi          TEXT NOT NULL,
    signal_type  TEXT NOT NULL,
    weight       REAL NOT NULL,
    rating       INTEGER,
    source       TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (doi) REFERENCES papers(doi)
);
CREATE INDEX IF NOT EXISTS idx_feedback_doi     ON feedback_events(doi);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback_events(created_at);

-- Bundle J (v1.0 active-learning CORE, no version bump): computed Rocchio
-- interest vectors keyed by scope ('global' or a per-cluster scope tag).
-- vector is np.float32 bytes, shape (dim,) — SAME blob encoding as
-- clusters.centroid_blob (np.ndarray.astype(float32).tobytes()).
-- n_events records how many feedback events contributed to this vector, so
-- the rank-time soft-gate (Bundle J2) can decide whether to apply it at all.
-- INERT by construction: J1 only computes + stores; nothing reads this yet.
CREATE TABLE IF NOT EXISTS interest_vectors (
    scope        TEXT PRIMARY KEY,
    vector       BLOB,
    n_events     INTEGER NOT NULL DEFAULT 0,
    computed_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS corpus_stat_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at   TEXT NOT NULL DEFAULT (datetime('now')),
    papers        INTEGER NOT NULL DEFAULT 0,
    graph_nodes   INTEGER NOT NULL DEFAULT 0,
    themes        INTEGER NOT NULL DEFAULT 0,
    run_type      TEXT,
    run_id        TEXT
);
"""
# ---------------------------------------------------------------------------
# StateDB class
# ---------------------------------------------------------------------------

# B4: papers columns that ARE real columns but are intentionally managed by
# dedicated setters (set_graph_indexed / set_chunks_indexed / set_notes_synced /
# set_ner_processed_at / set_rel_processed_at).  Writing them through
# upsert_paper would bypass the R28 dual-write invariant logic in those setters,
# so upsert_paper rejects them with a redirect to the correct method.
_UPSERT_FLAG_COLUMNS: dict[str, str] = {
    "graph_indexed": "set_graph_indexed()",
    "chunks_indexed": "set_chunks_indexed()",
    "notes_synced": "set_notes_synced()",
    "ner_processed_at": "set_ner_processed_at()",
    "rel_processed_at": "set_rel_processed_at()",
}

# B3-F8: the hand-maintained column list that upsert_paper() writes.  Lifted to
# module scope so the drift guard (assert_upsert_columns_consistent) can compare
# it against the live papers schema without re-running upsert_paper.
_UPSERT_WRITABLE_COLUMNS: list[str] = [
    "doi", "title", "authors", "year", "journal", "zotero_key",
    "first_seen_date", "status", "source_type", "parent_id",
    "note_title", "note_path", "embeddings_indexed", "extraction_json",
    "extraction_provider", "extraction_model", "ocr_pages_json",
    "ocr_heavy", "keywords_json", "isbn", "last_updated",
]

# B3-F8: real papers columns that upsert_paper() intentionally does NOT write.
#   * The five _UPSERT_FLAG_COLUMNS are managed by dedicated setters whose R28
#     dual-write invariants would be bypassed by a COALESCE upsert.
#   * last_insight_run (G16) is a reserved column with no writer yet; it must
#     not be silently writable through upsert_paper either.
# The drift guard treats (written ∪ this set) as the full accounted-for column
# universe, so a NEW papers column added later — neither written nor listed
# here — trips a clear assertion telling the developer to decide where it lives.
_UPSERT_INTENTIONALLY_EXCLUDED: frozenset[str] = frozenset(
    set(_UPSERT_FLAG_COLUMNS) | {"last_insight_run"}
)


# FE2-1: SQL-safety whitelists for StateDB.list_papers().
# `sort` and `order` come from HTTP query params, so they MUST NOT be
# interpolated into SQL unchecked.  list_papers() maps a requested sort key to a
# trusted column name via this dict (default last_updated) and the order to a
# trusted keyword via _LIST_PAPERS_ORDER (default desc).  Only these constants —
# never the caller's raw string — ever reach the ORDER BY clause.
_LIST_PAPERS_SORT_COLUMNS: dict[str, str] = {
    "last_updated": "last_updated",
    "year": "year",
    "title": "title",
    "status": "status",
}
_LIST_PAPERS_ORDER: dict[str, str] = {"asc": "ASC", "desc": "DESC"}

# FE2-1: status_gap value → the column-based WHERE fragment that selects the
# gap.  low_confidence is deliberately absent here — it needs per-row
# extraction_json parsing and is applied in Python after the fetch.
_LIST_PAPERS_STATUS_GAP: dict[str, str] = {
    "missing_graph": "graph_indexed = 0",
    "missing_notes": "notes_synced = 0",
    "missing_embeddings": "embeddings_indexed = 0",
}


def upsert_writable_columns() -> list[str]:
    """Return a copy of the columns upsert_paper() writes (B3-F8 introspection)."""
    return list(_UPSERT_WRITABLE_COLUMNS)


def assert_upsert_columns_consistent(
    conn: sqlite3.Connection,
    written_cols: list[str] | None = None,
) -> None:
    """B3-F8 drift guard: verify upsert_paper's column list matches the schema.

    Two invariants, both AssertionErrors on violation:
      (1) every column upsert_paper writes must be a real ``papers`` column
          (catches a renamed/removed column), and
      (2) every real ``papers`` column must be accounted for — either written
          by upsert_paper or in ``_UPSERT_INTENTIONALLY_EXCLUDED`` — so a NEW
          schema column added later fails loudly until a human decides where it
          is written.

    ``written_cols`` defaults to the live ``_UPSERT_WRITABLE_COLUMNS``; tests
    pass a corrupted list to prove the guard bites.
    """
    written = list(_UPSERT_WRITABLE_COLUMNS if written_cols is None else written_cols)
    rows = conn.execute("PRAGMA table_info(papers)").fetchall()
    schema_cols = {r[1] for r in rows}

    # (0) a setter-managed flag column must never appear in the writable list.
    #     This direction is invisible to checks (1)/(2): a flag column is a real
    #     schema column AND is allowlisted, so it passes both — yet writing it via
    #     the COALESCE upsert would clobber setter-owned state (R28 invariant).
    leaked = sorted(set(written) & set(_UPSERT_FLAG_COLUMNS))
    assert not leaked, (
        f"setter-managed flag column(s) {leaked} appear in upsert_paper's "
        "writable list — they must stay setter-only (_UPSERT_FLAG_COLUMNS) to "
        "preserve the R28 dual-write invariant."
    )

    # (1) upsert must not write phantom columns.
    phantom = sorted(c for c in written if c not in schema_cols)
    assert not phantom, (
        f"upsert_paper writes column(s) {phantom} that do not exist in the "
        "papers table — the writable list has drifted from the schema "
        "(renamed/removed column?)."
    )

    # (2) every schema column must be written or explicitly excluded.
    accounted = set(written) | _UPSERT_INTENTIONALLY_EXCLUDED
    unaccounted = sorted(c for c in schema_cols if c not in accounted)
    assert not unaccounted, (
        f"papers column(s) {unaccounted} are neither written by upsert_paper "
        "nor in _UPSERT_INTENTIONALLY_EXCLUDED. Add each to "
        "_UPSERT_WRITABLE_COLUMNS (if upsert_paper should write it) or to "
        "_UPSERT_INTENTIONALLY_EXCLUDED (if a dedicated setter owns it)."
    )


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
                # Discovery history join: FK linking discovery_runs → run_log.run_id (uuid).
                # NULL for historical rows written before this column existed.
                ("discovery_runs", "run_log_id",
                 "ALTER TABLE discovery_runs ADD COLUMN run_log_id TEXT"),
                # Trigger marker: how this run was started — 'manual' (web "Run now"
                # / CLI) vs 'scheduled' (OS scheduler). SQLite permits adding a NOT
                # NULL column only with a non-NULL default; 'manual' backfills any
                # pre-existing rows.
                ("discovery_runs", "trigger",
                 "ALTER TABLE discovery_runs ADD COLUMN trigger TEXT NOT NULL DEFAULT 'manual'"),
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
            except Exception as exc:
                # P1: was a fully-silent `pass` — a failed M3 phase backfill
                # left existing rows with inconsistent simple/complex/fully
                # flags and logged NOTHING.  This is a data UPDATE (not DDL), so
                # escalating under --strict cannot brick an existing DB's
                # structure — route through strict_fallback (warn by default,
                # fatal under --strict).
                strict_fallback(
                    logger,
                    f"M3 phase-progress backfill failed: {exc}. Existing "
                    "brain_build_progress rows may have inconsistent "
                    "simple/complex/fully_complete flags.",
                    exc,
                )
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
                # P1: was warn-only — under --strict this must be fatal so CI
                # surfaces a cleanup that silently never completes.
                strict_fallback(
                    logger,
                    f"N7 stale-row cleanup failed (will retry on next startup): {exc}",
                    exc,
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
                # P1: was warn-only — under --strict this must be fatal so a
                # rename that silently never completes is caught in CI.
                strict_fallback(
                    logger,
                    "weekly→discovery run_type rename failed (will retry on "
                    f"next startup): {exc}",
                    exc,
                )
            # B3-F8: the drift guard (assert_upsert_columns_consistent) is NOT
            # called here on purpose.  Legacy DBs may have a minimal `papers`
            # table (e.g. only `doi`): _SCHEMA_SQL's CREATE TABLE IF NOT EXISTS
            # is a no-op on an existing table and the base columns are not part
            # of the additive_migrations list, so PRAGMA table_info(papers) can
            # legitimately lack base columns at init.  Asserting here would brick
            # startup on such a DB.  The guard is a TEST-time contract instead
            # (see TestUpsertColumnDriftGuard) — it catches developer-introduced
            # drift in CI without risking end-user startup.
    # -- Paper / review CRUD --
    def upsert_paper(self, data: dict[str, Any]) -> None:
        """Insert or replace a paper or review record.

        B4: validates every key in ``data`` against the set of upsert-able
        ``papers`` columns and raises ``ValueError`` on anything unknown.
        Previously unknown keys were SILENTLY dropped, so a typo
        (``"tite"`` for ``"title"``) or an attempt to write a flag column
        produced silent data loss with no error.  Flag columns
        (graph_indexed / chunks_indexed / notes_synced / ner_processed_at /
        rel_processed_at) are real columns but managed by dedicated setters,
        so they are rejected with a redirect rather than written here.
        """
        # B3-F8: single source of truth — the module-level writable list, also
        # checked against the live schema by assert_upsert_columns_consistent().
        cols = list(_UPSERT_WRITABLE_COLUMNS)

        # B4: fail loud on unknown keys.  Separate flag columns (redirect to
        # their setter) from genuine typos / non-existent columns (clear error).
        allowed = set(cols)
        flag_keys = sorted(k for k in data if k in _UPSERT_FLAG_COLUMNS)
        if flag_keys:
            redirects = ", ".join(
                f"{k} -> {_UPSERT_FLAG_COLUMNS[k]}" for k in flag_keys
            )
            raise ValueError(
                "upsert_paper() does not manage flag column(s) "
                f"{flag_keys}; use the dedicated setter(s) instead: {redirects}"
            )
        unknown = sorted(k for k in data if k not in allowed)
        if unknown:
            raise ValueError(
                f"upsert_paper() got unknown papers column(s) {unknown}; "
                f"valid columns are {sorted(allowed)}. Check for a typo or use "
                "the dedicated setter for flag columns."
            )

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

    def get_zotero_key(self, doi: str) -> str | None:
        """Return the Zotero item key for a paper by DOI, or None if unlinked.

        Read-only, parameterized. The caller is responsible for normalizing the
        DOI to the canonical form (see scripts/core/doi.py::normalize_doi and
        Audit Top Finding #4) before calling — this method does a literal match
        on the stored ``papers.doi`` primary key. An unknown DOI returns None
        (not an error), matching the convention used by the other read accessors.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT zotero_key FROM papers WHERE doi = ?",
                (doi,),
            ).fetchone()
        if row is None:
            return None
        return row["zotero_key"]

    def get_all_by_source_type(self, source_type: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE source_type = ?", (source_type,)
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_confidence(extraction_json: str | None) -> float | None:
        """Parse ``extraction_json`` → ``_overall_confidence`` (None on absent/bad).

        Never raises: a malformed blob, a missing key, or a non-numeric value
        all yield None so a single corrupt row cannot break the whole listing.
        """
        if not extraction_json:
            return None
        try:
            data = json.loads(extraction_json)
            conf = data.get("_overall_confidence")
            return float(conf) if conf is not None else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    @staticmethod
    def _row_authors(authors_json: str | None) -> list[str]:
        """Parse the JSON ``authors`` column → list[str] ([] on absent/bad)."""
        if not authors_json:
            return []
        try:
            parsed = json.loads(authors_json)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def list_papers(
        self,
        *,
        search: str | None = None,
        source_type: str | None = None,
        status_gap: str | None = None,
        theme: str | None = None,
        sort: str = "last_updated",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """FE2-1: search / filter / sort / paginate the processed papers corpus.

        A read-only lens over the ``papers`` table for the Corpus Explorer web
        surface. Returns ``(rows, total)`` where ``total`` is the count of rows
        matching the same filters (ignoring limit/offset) so the caller can
        paginate.

        SQL safety
        ----------
        Every user-supplied VALUE is passed as a ``?`` placeholder — search
        terms, source_type, theme, limit, offset never touch the SQL string.
        ``sort`` and ``order`` are SQL identifiers/keywords (cannot be
        parameterised), so they are mapped through the trusted
        ``_LIST_PAPERS_SORT_COLUMNS`` / ``_LIST_PAPERS_ORDER`` whitelists —
        anything unrecognised falls back to ``last_updated`` / ``desc``. No raw
        caller string ever reaches the ORDER BY clause.

        Filters
        -------
        - ``search``: case-insensitive substring on title OR doi.
        - ``source_type``: exact match (e.g. ``paper`` / ``review``).
        - ``status_gap``: ``missing_graph`` / ``missing_notes`` /
          ``missing_embeddings`` map to a column predicate in SQL;
          ``low_confidence`` is applied in Python after the fetch (see below).
        - ``theme``: active-cluster ``display_name`` match via an EXISTS
          subquery over ``cluster_assignments`` + ``clusters`` (no row
          multiplication, so ``total`` stays correct).

        Confidence and low_confidence
        ------------------------------
        ``confidence`` is parsed per-row from ``extraction_json`` →
        ``_overall_confidence`` (None when absent/bad). Because that lives
        inside a JSON blob, the ``low_confidence`` gap (confidence < 0.5) cannot
        be expressed as a column predicate. It is therefore applied as a
        post-fetch Python filter, and to keep pagination correct the full
        filtered set is materialised first: ``total`` is the post-filter count
        and ``limit``/``offset`` are sliced in Python. For all column-based
        filters ``total`` comes from a ``COUNT(*)`` with the same WHERE.
        """
        # --- Whitelist sort/order (NEVER interpolate the caller's raw value) ---
        sort_col = _LIST_PAPERS_SORT_COLUMNS.get(sort, "last_updated")
        order_kw = _LIST_PAPERS_ORDER.get((order or "").lower(), "DESC")

        # --- Build the shared WHERE clause from column-based filters ---
        where: list[str] = []
        params: list[Any] = []

        if search:
            term = f"%{search.lower()}%"
            where.append("(LOWER(title) LIKE ? OR LOWER(doi) LIKE ?)")
            params.extend([term, term])

        if source_type:
            where.append("source_type = ?")
            params.append(source_type)

        # Column-based status gaps map to a trusted predicate; low_confidence is
        # handled in Python below (its predicate lives inside extraction_json).
        if status_gap in _LIST_PAPERS_STATUS_GAP:
            where.append(_LIST_PAPERS_STATUS_GAP[status_gap])

        if theme:
            # EXISTS keeps one row per paper (a paper may have >1 assignment),
            # so COUNT(*) over the same WHERE stays exact.
            where.append(
                "EXISTS (SELECT 1 FROM cluster_assignments ca "
                "JOIN clusters c ON c.id = ca.cluster_id "
                "WHERE ca.doi = papers.doi AND c.archived = 0 "
                "AND c.display_name = ?)"
            )
            params.append(theme)

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        # sort_col/order_kw are whitelist outputs — safe to concatenate.
        # A stable secondary key on doi keeps pagination deterministic when the
        # primary sort column ties.
        order_sql = f" ORDER BY {sort_col} {order_kw}, doi {order_kw}"

        low_conf = status_gap == "low_confidence"

        with self._connect() as conn:
            if low_conf:
                # low_confidence needs per-row JSON parsing → fetch everything
                # matching the other filters, filter + paginate in Python.
                rows = conn.execute(
                    "SELECT * FROM papers" + where_sql + order_sql, params
                ).fetchall()
                dicts = [self._build_paper_row(r) for r in rows]
                filtered = [
                    d for d in dicts
                    if d["confidence"] is not None and d["confidence"] < 0.5
                ]
                total = len(filtered)
                page = filtered[offset:offset + limit] if limit else filtered[offset:]
                return page, total

            total_row = conn.execute(
                "SELECT COUNT(*) FROM papers" + where_sql, params
            ).fetchone()
            total = total_row[0] if total_row else 0

            rows = conn.execute(
                "SELECT * FROM papers" + where_sql + order_sql + " LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()

        return [self._build_paper_row(r) for r in rows], total

    def _build_paper_row(self, row: sqlite3.Row) -> dict:
        """FE2-1: shape one ``papers`` row into the Corpus list-view dict.

        Parses authors + confidence defensively and surfaces the active-cluster
        display_name as ``cluster``/``theme`` (None when unassigned).
        """
        data = dict(row)
        cluster = self.get_paper_cluster(data["doi"])
        theme_name = cluster["display_name"] if cluster else None
        return {
            "doi": data["doi"],
            "title": data.get("title"),
            "year": data.get("year"),
            "journal": data.get("journal"),
            "source_type": data.get("source_type"),
            "status": data.get("status"),
            "authors": self._row_authors(data.get("authors")),
            "confidence": self._row_confidence(data.get("extraction_json")),
            "embeddings_indexed": data.get("embeddings_indexed"),
            "graph_indexed": data.get("graph_indexed"),
            "chunks_indexed": data.get("chunks_indexed"),
            "notes_synced": data.get("notes_synced"),
            "note_path": data.get("note_path"),
            "zotero_key": data.get("zotero_key"),
            "last_updated": data.get("last_updated"),
            "cluster": theme_name,
            "theme": theme_name,
        }
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
            # B4: re-queue reset papers as 'pending' (the canonical "ready to be
            # processed" state — same value newly-ingested papers default to and
            # the only status get_pending() matches).  The old code set status=NULL,
            # which made every reset paper INVISIBLE to get_pending() so they were
            # never re-extracted — defeating the whole point of resetting.
            cur = conn.execute(
                "UPDATE papers SET extraction_json = NULL, status = 'pending', "
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

    # -- Bundle F: embedding provenance --
    def record_embedding_provenance(
        self,
        collection_name: str,
        provider: str,
        model: str,
        dim: int,
    ) -> None:
        """Bundle F: register a ChromaDB collection's embedding provenance.

        Uses INSERT OR REPLACE semantics so a second call for the same
        collection_name overwrites the previous record.  Sets is_current=1
        on the new/updated row; does NOT clear is_current on other rows —
        call set_current_embedding_collection() explicitly when switching.

        Parameters
        ----------
        collection_name:
            ChromaDB collection name (e.g. 'lit_monitor_v1').
        provider:
            'ollama' or 'litellm'.
        model:
            Model identifier string (e.g. 'mxbai-embed-large').
        dim:
            Embedding dimension (e.g. 1024).
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embedding_provenance "
                "(collection_name, provider, model, dim, created_at, is_current) "
                "VALUES (?, ?, ?, ?, datetime('now'), 1)",
                (collection_name, provider, model, dim),
            )

    def get_embedding_provenance(self, collection_name: str) -> dict | None:
        """Bundle F: read provenance for a collection; None if not registered.

        Returns a dict with keys: provider, model, dim, created_at, is_current.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT provider, model, dim, created_at, is_current "
                "FROM embedding_provenance WHERE collection_name = ?",
                (collection_name,),
            ).fetchone()
        if row is None:
            return None
        return {
            "provider": row[0],
            "model": row[1],
            "dim": row[2],
            "created_at": row[3],
            "is_current": bool(row[4]),
        }

    def set_current_embedding_collection(self, collection_name: str) -> None:
        """Bundle F: mark one collection as active; all others become non-current.

        Call this after building a new collection so the provenance table
        accurately reflects which collection is in active use.

        Parameters
        ----------
        collection_name:
            ChromaDB collection name to mark as the current (active) one.
        """
        with self._connect() as conn:
            conn.execute("UPDATE embedding_provenance SET is_current = 0")
            conn.execute(
                "UPDATE embedding_provenance SET is_current = 1 "
                "WHERE collection_name = ?",
                (collection_name,),
            )

    def list_embedding_provenance(self) -> list[dict]:
        """Bundle F: list all known embedding collections and their provenance.

        Returns rows in created_at DESC order (most recently created first).

        Returns
        -------
        list[dict]
            Each dict has keys: collection_name, provider, model, dim,
            created_at, is_current.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT collection_name, provider, model, dim, created_at, is_current "
                "FROM embedding_provenance ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "collection_name": r[0],
                "provider": r[1],
                "model": r[2],
                "dim": r[3],
                "created_at": r[4],
                "is_current": bool(r[5]),
            }
            for r in rows
        ]

    # -- Bundle G (v0.9): domain focus extraction --
    # Plural ↔ singular field_type mapping. Plural keys are the public/JSON
    # contract; singular values are the closed vocabulary stored in the table.
    _DFE_PLURAL_TO_SINGULAR: dict[str, str] = {
        "topics": "topic",
        "methods": "method",
        "materials": "material",
        "adjacent_fields": "adjacent_field",
        "exclusions": "exclusion",
    }
    _DFE_SINGULAR_TO_PLURAL: dict[str, str] = {
        "topic": "topics",
        "method": "methods",
        "material": "materials",
        "adjacent_field": "adjacent_fields",
        "exclusion": "exclusions",
    }

    def save_domain_extraction(self, extraction: dict[str, list[str]]) -> None:
        """Bundle G: persist a domain extraction with REPLACE semantics.

        Wipes ALL previous rows in domain_focus_extracted before inserting,
        so the table always reflects the most recent analysis — no merging,
        no append. Confirmed/rejected state from a previous extraction is
        intentionally discarded: a new analysis means the user is asking
        the LLM to re-think, and stale confirmations would taint the new
        result.

        Args:
            extraction: dict with plural keys (topics/methods/materials/
                adjacent_fields/exclusions), each mapped to a list of
                strings. Empty lists are fine.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM domain_focus_extracted")
            for plural, items in extraction.items():
                singular = self._DFE_PLURAL_TO_SINGULAR.get(plural)
                if singular is None:
                    # Unknown plural key — skip silently. Schema is the
                    # contract; bad keys from upstream are not our problem
                    # to fix at write time.
                    continue
                for item in items:
                    if not item or not str(item).strip():
                        continue
                    conn.execute(
                        "INSERT INTO domain_focus_extracted "
                        "(field_type, value) VALUES (?, ?)",
                        (singular, str(item).strip()),
                    )

    def list_domain_extraction(self) -> dict[str, list[dict]]:
        """Bundle G: read current extraction; group by field_type.

        Returns a dict with the SAME five plural keys as the input to
        save_domain_extraction. Each value is a list of row dicts:
        ``{id, value, user_confirmed, last_analyzed_at}``. Keys are
        always present even when empty (stable shape for HTTP consumers).
        """
        result: dict[str, list[dict]] = {
            plural: [] for plural in self._DFE_PLURAL_TO_SINGULAR
        }
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, field_type, value, user_confirmed, last_analyzed_at "
                "FROM domain_focus_extracted "
                "ORDER BY field_type, value"
            ).fetchall()
        for row in rows:
            plural = self._DFE_SINGULAR_TO_PLURAL.get(row[1])
            if plural is None:
                continue
            result[plural].append(
                {
                    "id": row[0],
                    "value": row[2],
                    "user_confirmed": bool(row[3]),
                    "last_analyzed_at": row[4],
                }
            )
        return result

    def clear_domain_extraction(self) -> int:
        """Bundle G: wipe domain_focus_extracted; return rows deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM domain_focus_extracted")
            return cur.rowcount

    def set_domain_extraction_confirmed(self, row_id: int, confirmed: bool) -> None:
        """Bundle G: toggle the user_confirmed flag on one extracted item."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE domain_focus_extracted "
                "SET user_confirmed = ? WHERE id = ?",
                (int(bool(confirmed)), int(row_id)),
            )

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

    def write_corpus_snapshot(
        self,
        *,
        papers: int,
        graph_nodes: int,
        themes: int,
        run_type: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Record one corpus-state snapshot (papers/graph_nodes/themes) for the
        stats-banner deltas. Called at run completion only."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO corpus_stat_snapshots "
                "(papers, graph_nodes, themes, run_type, run_id) VALUES (?, ?, ?, ?, ?)",
                (int(papers), int(graph_nodes), int(themes), run_type, run_id),
            )

    def get_recent_snapshots(self, limit: int = 2) -> list[dict]:
        """Return the most recent corpus snapshots, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, snapshot_at, papers, graph_nodes, themes, run_type, run_id "
                "FROM corpus_stat_snapshots ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- P1: structured discovery-run tracking --

    def start_discovery_run(
        self,
        run_params: dict,
        run_log_id: str | None = None,
        trigger: str = "manual",
    ) -> int:
        """P1: insert a discovery_runs row with status='running'; return new run_id.

        Args:
            run_params: Arbitrary dict of run parameters (topics, since_days,
                rag_mode, etc.) serialised as JSON for later auditing.
            run_log_id: The uuid of the sibling run_log row (from start_run), so
                discovery history can join the two tables. None for callers that
                don't create a run_log row (e.g. dry-run paths / legacy tests),
                which leaves the column NULL.
            trigger: How this run was started — 'manual' (web "Run now" / CLI) or
                'scheduled' (OS scheduler). Defaults to 'manual'.

        Returns:
            Integer primary-key of the newly inserted row.
        """
        import json as _json
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO discovery_runs (run_params_json, run_log_id, trigger) "
                "VALUES (?, ?, ?)",
                (_json.dumps(run_params), run_log_id, trigger),
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
            # AR-5 Fix 3 (Finding 10a): parameterize the LIMIT (mirror
            # `list_papers`) instead of f-string interpolation. `int()` coerces
            # any non-int caller value to a safe bound.
            sql += " LIMIT ?"
            params.append(int(limit))

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
            # AR-5 Fix 3 (Finding 10a): parameterize the LIMIT (mirror
            # `list_papers`) instead of f-string interpolation. `int()` coerces
            # any non-int caller value to a safe bound.
            sql += " LIMIT ?"
            params.append(int(limit))

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

    # ------------------------------------------------------------------ #
    # Bundle C: clustering helpers
    # ------------------------------------------------------------------ #
    # -- conn-based primitives (B3) --
    # These execute their SQL against a caller-supplied connection so that
    # replace_clusters() can run archive + every insert/upsert inside ONE
    # transaction.  The public methods below wrap them with their own
    # _connect() so existing callers/tests keep working unchanged.

    @staticmethod
    def _insert_cluster_conn(
        conn: sqlite3.Connection,
        display_name: str | None,
        n_papers: int,
        cohesion_score: float | None,
        centroid_blob: bytes,
    ) -> int:
        """INSERT a cluster row on ``conn``; return the new row id. No commit."""
        name = display_name or ""
        cursor = conn.execute(
            "INSERT INTO clusters (display_name, n_papers, cohesion_score, centroid_blob) "
            "VALUES (?, ?, ?, ?)",
            (name, n_papers, cohesion_score, centroid_blob),
        )
        return cursor.lastrowid

    @staticmethod
    def _upsert_cluster_by_id_conn(
        conn: sqlite3.Connection,
        cluster_id: int,
        display_name: str | None,
        n_papers: int,
        cohesion_score: float | None,
        centroid_blob: bytes,
    ) -> None:
        """UPSERT a cluster row by id on ``conn`` (reactivates archived). No commit."""
        name = display_name or ""
        conn.execute(
            "INSERT INTO clusters (id, display_name, n_papers, cohesion_score, "
            "centroid_blob, archived, computed_at) "
            "VALUES (?, ?, ?, ?, ?, 0, datetime('now')) "
            "ON CONFLICT(id) DO UPDATE SET "
            "  display_name=excluded.display_name, "
            "  n_papers=excluded.n_papers, "
            "  cohesion_score=excluded.cohesion_score, "
            "  centroid_blob=excluded.centroid_blob, "
            "  archived=0, "
            "  computed_at=datetime('now')",
            (cluster_id, name, n_papers, cohesion_score, centroid_blob),
        )

    @staticmethod
    def _update_cluster_display_name_conn(
        conn: sqlite3.Connection, cluster_id: int, name: str
    ) -> None:
        """UPDATE a cluster's display_name on ``conn``. No commit."""
        conn.execute(
            "UPDATE clusters SET display_name=? WHERE id=?",
            (name, cluster_id),
        )

    @staticmethod
    def _archive_clusters_conn(
        conn: sqlite3.Connection, cluster_ids: list[int]
    ) -> None:
        """Mark cluster_ids archived on ``conn``. No commit. No-op if empty."""
        if not cluster_ids:
            return
        placeholders = ",".join("?" * len(cluster_ids))
        conn.execute(
            f"UPDATE clusters SET archived=1 WHERE id IN ({placeholders})",
            cluster_ids,
        )

    def insert_cluster(
        self,
        display_name: str | None,
        n_papers: int,
        cohesion_score: float | None,
        centroid_blob: bytes,
    ) -> int:
        """Insert a new cluster row and return its AUTOINCREMENT id.

        Args:
            display_name: Human-readable theme name (None → persisted as empty string).
            n_papers: Number of papers assigned to this cluster.
            cohesion_score: Mean silhouette score for the cluster.
            centroid_blob: np.float32 bytes of the centroid vector.

        Returns:
            The new row's integer id.
        """
        with self._connect() as conn:
            return self._insert_cluster_conn(
                conn, display_name, n_papers, cohesion_score, centroid_blob
            )

    def upsert_cluster_by_id(
        self,
        cluster_id: int,
        display_name: str | None,
        n_papers: int,
        cohesion_score: float | None,
        centroid_blob: bytes,
    ) -> None:
        """Update an existing cluster row (reactivates archived rows).

        Used when recompute finds a new cluster near an existing one:
        the existing ID is preserved so user-applied tags don't drift.
        """
        with self._connect() as conn:
            self._upsert_cluster_by_id_conn(
                conn, cluster_id, display_name, n_papers, cohesion_score, centroid_blob
            )

    def update_cluster_display_name(self, cluster_id: int, name: str) -> None:
        """Update the display_name of an existing cluster (used for fallback labelling)."""
        with self._connect() as conn:
            self._update_cluster_display_name_conn(conn, cluster_id, name)

    def replace_clusters(
        self,
        archive_ids: list[int],
        new_clusters: list[NewClusterSpec],
    ) -> list[int]:
        """B3: atomically archive old clusters and install the new ones.

        Archive + every insert/upsert + display-name fallback run inside a
        SINGLE ``with self._connect()`` block, i.e. one SQLite transaction.
        ``_connect()`` commits on clean exit and rolls back on any exception,
        so a crash mid-persist can NEVER leave the library with zero active
        clusters: either the old rows stay (rollback) or the new rows fully
        land (commit).

        Args:
            archive_ids: Existing cluster ids to soft-archive (archived=1).
            new_clusters: Ordered specs to persist. inherited_id None → fresh
                INSERT; otherwise reactivate that id via upsert. display_name
                None → 'Cluster <id>' fallback applied with the now-known id.

        Returns:
            The assigned cluster ids, in the same order as ``new_clusters``.
        """
        assigned: list[int] = []
        with self._connect() as conn:
            # 1) Archive the old clusters first (still in this txn).
            self._archive_clusters_conn(conn, archive_ids)

            # 2) Insert/upsert every new cluster.
            for spec in new_clusters:
                if spec.inherited_id is not None:
                    self._upsert_cluster_by_id_conn(
                        conn,
                        spec.inherited_id,
                        spec.display_name,
                        spec.n_papers,
                        spec.cohesion_score,
                        spec.centroid_blob,
                    )
                    cid = spec.inherited_id
                else:
                    cid = self._insert_cluster_conn(
                        conn,
                        spec.display_name,
                        spec.n_papers,
                        spec.cohesion_score,
                        spec.centroid_blob,
                    )

                # 3) Display-name fallback — inside the same txn, using the
                #    now-known row id.
                if spec.display_name is None:
                    self._update_cluster_display_name_conn(
                        conn, cid, f"Cluster {cid}"
                    )

                assigned.append(cid)

        return assigned

    def get_cluster(self, cluster_id: int) -> dict | None:
        """Return a single cluster row as a dict, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM clusters WHERE id=?", (cluster_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_active_clusters(self) -> list[dict]:
        """Return all non-archived cluster rows."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM clusters WHERE archived=0 ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def archive_clusters(self, cluster_ids: list[int]) -> None:
        """Mark the given cluster IDs as archived (soft delete).

        Archived clusters are excluded from list_active_clusters() but kept
        in the DB for stable-ID mapping in the next recompute.
        """
        if not cluster_ids:
            return
        with self._connect() as conn:
            self._archive_clusters_conn(conn, cluster_ids)

    def upsert_cluster_assignment(
        self,
        doi: str,
        cluster_id: int,
        *,
        distance_to_centroid: float | None = None,
    ) -> None:
        """Write or update a cluster_assignments row.

        Idempotent: re-assigning the same doi to the same cluster updates
        distance and timestamp rather than creating a duplicate.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cluster_assignments (doi, cluster_id, distance_to_centroid, assigned_at) "
                "VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(doi, cluster_id) DO UPDATE SET "
                "  distance_to_centroid=excluded.distance_to_centroid, "
                "  assigned_at=datetime('now')",
                (doi, cluster_id, distance_to_centroid),
            )

    def replace_cluster_assignments(
        self,
        assignments: list[tuple[str, int, float | None]],
    ) -> int:
        """P3.5: atomically batch-upsert many cluster assignments.

        Every (doi, cluster_id, distance) upsert runs inside a SINGLE
        ``with self._connect()`` block, i.e. one SQLite transaction (the same
        precedent as ``replace_clusters``). ``_connect()`` commits on clean
        exit and rolls back on any exception, so a crash partway through the
        batch can NEVER leave the table half-updated: either every assignment
        in this call lands (commit) or none of them do (rollback).

        Upsert semantics match ``upsert_cluster_assignment`` exactly
        (idempotent on the ``(doi, cluster_id)`` conflict key), so this is a
        drop-in atomic replacement for a per-row write loop.

        Args:
            assignments: list of ``(doi, cluster_id, distance_to_centroid)``
                tuples. ``distance_to_centroid`` may be ``None``.

        Returns:
            Number of assignment rows written.
        """
        if not assignments:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO cluster_assignments "
                "(doi, cluster_id, distance_to_centroid, assigned_at) "
                "VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(doi, cluster_id) DO UPDATE SET "
                "  distance_to_centroid=excluded.distance_to_centroid, "
                "  assigned_at=datetime('now')",
                assignments,
            )
        return len(assignments)

    def get_cluster_assignments(self, cluster_id: int) -> list[dict]:
        """Return all cluster_assignments rows for the given cluster_id."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cluster_assignments WHERE cluster_id=?",
                (cluster_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_paper_cluster(self, doi: str) -> dict | None:
        """Return the active cluster assignment for a paper (nearest centroid), or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ca.*, c.display_name FROM cluster_assignments ca "
                "JOIN clusters c ON ca.cluster_id = c.id "
                "WHERE ca.doi=? AND c.archived=0 "
                "ORDER BY ca.distance_to_centroid ASC LIMIT 1",
                (doi,),
            ).fetchone()
        return dict(row) if row else None

    def get_cluster_last_surfaced(self) -> dict[int, str | None]:
        """Bundle K-b: ``{cluster_id: last_surfaced_iso | None}`` for active clusters.

        "Last surfaced" = the most recent ``discovery_runs.started_at`` among
        discovery runs that recorded *any* paper (``discovery_paper_results``)
        whose DOI maps to that cluster via ``cluster_assignments``. There is no
        per-result timestamp column, so the run's ``started_at`` is the surface
        time — the natural granularity for the ``quiet_weeks`` window.

        Every *active* cluster is represented: a cluster that has never surfaced
        a paper in any discovery run maps to ``None`` (→ under-engaged by the
        K-b rule). Read-only; safe to call when the tables are empty.
        """
        # Seed every active cluster at None so never-surfaced clusters appear.
        result: dict[int, str | None] = {
            int(c["id"]): None for c in self.list_active_clusters()
        }
        if not result:
            return result
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ca.cluster_id AS cluster_id, "
                "       MAX(dr.started_at) AS last_surfaced "
                "FROM discovery_paper_results dpr "
                "JOIN discovery_runs dr ON dpr.run_id = dr.id "
                "JOIN cluster_assignments ca ON ca.doi = dpr.doi "
                "JOIN clusters c ON c.id = ca.cluster_id "
                "WHERE dpr.doi IS NOT NULL AND dpr.doi <> '' "
                "  AND c.archived = 0 "
                "GROUP BY ca.cluster_id"
            ).fetchall()
        for r in rows:
            cid = int(r["cluster_id"])
            if cid in result:
                result[cid] = r["last_surfaced"]
        return result

    # ---------------------------------------------------------------------------
    # Bundle E (v0.9): trending-concept suggestion helpers
    # ---------------------------------------------------------------------------

    def persist_trending_suggestion(
        self,
        concept_text: str,
        concept_type: str,
        n_mentions_new: int,
        n_mentions_prev: int,
        growth_rate: float,
    ) -> int:
        """Insert a new trending-concept suggestion row; return its rowid.

        Always inserts a fresh row with user_action=pending so the full
        history of detections is preserved. Cooldown deduplication is handled
        upstream in find_trending_concepts() before calling this method.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO trending_concepts_suggested "
                "(concept_text, concept_type, n_mentions_new, n_mentions_prev, "
                " growth_rate, user_action) "
                "VALUES (?, ?, ?, ?, ?, 'pending')",
                (concept_text, concept_type, n_mentions_new, n_mentions_prev, growth_rate),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_pending_trending_suggestions(self) -> list[dict]:
        """Return all rows where user_action IS NULL or 'pending', newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trending_concepts_suggested "
                "WHERE user_action IS NULL OR user_action = 'pending' "
                "ORDER BY suggested_at DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_trending_suggestion_by_id(self, row_id: int) -> dict | None:
        """Return a single trending suggestion row by primary key, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trending_concepts_suggested WHERE id = ?",
                (row_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_trending_action(self, row_id: int, action: str) -> None:
        """Set user_action and action_at for a trending suggestion.

        action must be 'accepted' or 'dismissed'.
        """
        if action not in ("accepted", "dismissed"):
            raise ValueError(f"invalid trending action: {action!r}")
        with self._connect() as conn:
            conn.execute(
                "UPDATE trending_concepts_suggested "
                "SET user_action = ?, action_at = datetime('now') "
                "WHERE id = ?",
                (action, row_id),
            )

    def get_dismissed_trending_concepts(self, cooldown_days: int = 60) -> list[str]:
        """Return concept_text values dismissed within the cooldown window.

        Used by find_trending_concepts() to skip re-suggesting recently
        dismissed concepts. Returns lowercase concept_text strings.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT concept_text FROM trending_concepts_suggested "
                "WHERE user_action = 'dismissed' "
                "  AND action_at >= datetime('now', ? || ' days')",
                (f"-{cooldown_days}",),
            ).fetchall()
        return [r[0].lower() for r in rows]

    def list_all_trending_suggestions(self) -> list[dict]:
        """Return all trending suggestion rows ordered by suggested_at DESC.

        Used by the CLI `trending view` command.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trending_concepts_suggested ORDER BY suggested_at DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Bundle H (v0.9): feedback events --

    # Pre-computed Rocchio-style weight map per signal type.
    # 'rated' is handled separately; weight is scaled by (rating - 3) * 0.5.
    _FEEDBACK_WEIGHT_MAP: dict[str, float] = {
        "opened": 0.3,
        "saved": 1.0,
        "dismissed": -0.5,
        "rated": 0.5,        # base; multiplied by (rating - 3) * 0.5 at insert time
        "thumbs_up": 0.8,
        "thumbs_down": -0.8,
    }

    _VALID_SIGNAL_TYPES: frozenset[str] = frozenset(
        {"opened", "saved", "dismissed", "rated", "thumbs_up", "thumbs_down"}
    )

    def record_feedback_event(
        self,
        doi: str,
        signal_type: str,
        *,
        rating: int | None = None,
        source: str | None = None,
    ) -> None:
        """Bundle H: insert one feedback event row.

        Args:
            doi:         Paper DOI — stored verbatim; FK to papers(doi).
            signal_type: One of the closed vocabulary in _VALID_SIGNAL_TYPES.
            rating:      Integer 1-5. Only meaningful when signal_type='rated'.
            source:      Free-text origin tag (e.g. 'discovery', 'themes').

        Raises:
            ValueError: If signal_type is not in the closed vocabulary, or if
                        rating is provided for a non-'rated' signal.
        """
        if signal_type not in self._VALID_SIGNAL_TYPES:
            raise ValueError(
                f"invalid signal_type {signal_type!r}; "
                f"must be one of {sorted(self._VALID_SIGNAL_TYPES)}"
            )
        if signal_type == "rated":
            if rating is None or not (1 <= int(rating) <= 5):
                raise ValueError("rating (1-5) required when signal_type='rated'")
        elif rating is not None:
            raise ValueError("rating only valid when signal_type='rated'")

        base_weight = self._FEEDBACK_WEIGHT_MAP[signal_type]
        if signal_type == "rated" and rating is not None:
            # Rocchio-friendly: scale relative to the neutral midpoint (3).
            weight = base_weight * (int(rating) - 3) * 0.5
        else:
            weight = base_weight

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback_events (doi, signal_type, weight, rating, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (doi, signal_type, weight, rating, source),
            )

    def list_feedback_events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        since: str | None = None,
    ) -> list[dict]:
        """Bundle H: return recent feedback events, newest first.

        Args:
            limit:  Max rows to return (default 50).
            offset: Row offset for pagination (AR-7; default 0).
            since:  ISO date/datetime string; when provided, only rows with
                    created_at >= since are returned.

        Returns:
            List of row dicts with keys: id, doi, signal_type, weight,
            rating, source, created_at.
        """
        # AR-7: id DESC is a stable tiebreaker. created_at has only 1-second
        # resolution, so events recorded in the same second would otherwise
        # order non-deterministically across pages — breaking pagination.
        if since is not None:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM feedback_events "
                    "WHERE created_at >= ? "
                    "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                    (since, limit, offset),
                ).fetchall()
        else:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM feedback_events "
                    "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        return [dict(r) for r in rows]

    def count_feedback_events(self) -> int:
        """AR-7: total number of feedback events (for recent-events pagination)."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]

    def feedback_summary(self) -> dict[str, int]:
        """Bundle H: count feedback events per signal_type over all time.

        Returns:
            Dict mapping signal_type → count. All valid signal types are
            present even when count is 0.
        """
        result: dict[str, int] = {s: 0 for s in self._VALID_SIGNAL_TYPES}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT signal_type, COUNT(*) as n FROM feedback_events "
                "GROUP BY signal_type"
            ).fetchall()
        for row in rows:
            result[row[0]] = row[1]
        return result

    def feedback_summary_by_source(self) -> dict[str, int]:
        """FI-5: count feedback events per ``source`` over all time.

        Mirrors :meth:`feedback_summary` but groups by the free-text ``source``
        column instead of ``signal_type``. NULL sources are coalesced to the
        label ``"unknown"`` so they are counted rather than dropped.

        Sources are open-ended strings (e.g. ``"discovery"``, ``"themes"``),
        so — unlike :meth:`feedback_summary` — no keys are pre-seeded to 0;
        only sources actually present in the table appear in the result.

        Returns:
            Dict mapping source (or ``"unknown"`` for NULL) → count. Empty dict
            on any read failure (defensive, read-only).
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT COALESCE(source, 'unknown') AS src, COUNT(*) AS n "
                    "FROM feedback_events GROUP BY src"
                ).fetchall()
            return {row[0]: int(row[1]) for row in rows}
        except sqlite3.Error as exc:
            logger.warning("feedback_summary_by_source read failed: %s", exc)
            return {}

    #: Whitelist of allowed time-grouping SQL expressions for feedback_timeline.
    #: The granularity argument selects a fixed literal here — the user string is
    #: NEVER interpolated into SQL, only used as a dict key (unknown ⇒ "week").
    _FEEDBACK_PERIOD_EXPR: dict[str, str] = {
        "day": "DATE(created_at)",
        "week": "strftime('%Y-W%W', created_at)",
    }

    #: Whitelist of allowed second group-by dimensions for feedback_timeline.
    #: Maps a dimension token → ``(column_expr, output_alias)``. The dimension
    #: argument is used ONLY as a dict key — the user string is NEVER
    #: interpolated into SQL (unknown ⇒ "signal"). The alias becomes the row
    #: dict key, so ``"signal"`` keeps the backward-compatible ``signal_type``
    #: key while ``"source"`` exposes a ``source`` key (NULL → 'unknown').
    _FEEDBACK_DIMENSION_COL: dict[str, tuple[str, str]] = {
        "signal": ("signal_type", "signal_type"),
        "source": ("COALESCE(source, 'unknown')", "source"),
    }

    def feedback_timeline(
        self,
        granularity: str = "week",
        *,
        weeks: int | None = None,
        dimension: str = "signal",
    ) -> list[dict]:
        """FI-1/FI-5: feedback event counts grouped by time-period and a dimension.

        Read-only time-series for the Insights page. Defensive: any read failure
        degrades to an empty list rather than raising to the caller.

        Args:
            granularity: ``"day"`` or ``"week"`` (any other value falls back to
                         ``"week"``). Selected via a fixed whitelist map — the
                         raw string is never interpolated into SQL.
            weeks:       Optional lookback window. When set, only events with
                         ``created_at >= now - weeks*7 days`` are counted. The
                         cutoff is computed in Python and passed as a bound
                         parameter (never interpolated).
            dimension:   ``"signal"`` (default) or ``"source"`` — selects the
                         second group-by column. Any other value falls back to
                         ``"signal"``. Selected via a fixed whitelist map — the
                         raw string is never interpolated into SQL.

        Returns:
            List of ``{"period": str, <dim>: str, "count": int}`` dicts ordered
            by (period, dimension). The ``<dim>`` key is ``"signal_type"`` when
            ``dimension="signal"`` (backward-compatible) or ``"source"`` when
            ``dimension="source"``. Empty list on no data or error.
        """
        # Whitelist: the value chosen is a fixed literal, not the user string.
        period_expr = self._FEEDBACK_PERIOD_EXPR.get(
            granularity, self._FEEDBACK_PERIOD_EXPR["week"]
        )
        # Whitelist: dimension is a dict key only; unknown ⇒ "signal" (compat).
        dim_col, dim_alias = self._FEEDBACK_DIMENSION_COL.get(
            dimension, self._FEEDBACK_DIMENSION_COL["signal"]
        )

        params: list[object] = []
        where_clause = ""
        if weeks is not None:
            try:
                cutoff = datetime.now() - timedelta(weeks=int(weeks))
                where_clause = " WHERE created_at >= ?"
                params.append(cutoff.strftime("%Y-%m-%d %H:%M:%S"))
            except (ValueError, TypeError):
                # A bad ``weeks`` value degrades to "no lookback filter".
                where_clause = ""
                params = []

        sql = (
            f"SELECT {period_expr} AS period, {dim_col} AS {dim_alias}, "
            f"COUNT(*) AS count "
            f"FROM feedback_events{where_clause} "
            f"GROUP BY period, {dim_col} "
            f"ORDER BY period, {dim_alias}"
        )
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, tuple(params)).fetchall()
            return [
                {
                    "period": r["period"],
                    dim_alias: r[dim_alias],
                    "count": int(r["count"]),
                }
                for r in rows
            ]
        except sqlite3.Error as exc:
            logger.warning("feedback_timeline read failed: %s", exc)
            return []

    # -- P4 Part C: implicit Zotero-save support --

    #: ``source`` tag stamped on implicit-save feedback events so they are
    #: distinguishable from explicit web/CLI 'saved' signals.  Shared with
    #: ``scripts.pipelines._ingest`` so the writer and the idempotency reader
    #: can't drift.
    IMPLICIT_SAVE_SOURCE: str = "implicit_zotero_save"

    def was_surfaced_in_discovery(self, doi: str) -> bool:
        """P4 Part C: True iff DOI appeared in any prior discovery run.

        Read-only membership check against ``discovery_paper_results`` (keyed
        by ``doi``).  Used to distinguish a paper the user saved *after* it was
        recommended (→ implicit positive feedback) from pre-existing library
        baseline papers that were never surfaced (→ no feedback).

        An empty/blank DOI never matches: discovery may record candidates with
        an empty DOI string, but those can't be the paper being ingested.
        """
        if not doi or not doi.strip():
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM discovery_paper_results WHERE doi = ? LIMIT 1",
                (doi,),
            ).fetchone()
        return row is not None

    def has_implicit_save_feedback(self, doi: str) -> bool:
        """P4 Part C: True iff an implicit-save event already exists for DOI.

        Idempotency guard: re-running brain-build over the same paper must NOT
        re-log the implicit 'saved' signal.  Matches only rows with
        ``signal_type='saved'`` AND ``source=IMPLICIT_SAVE_SOURCE`` so an
        explicit web/CLI 'saved' does not suppress (or count as) an implicit one.
        """
        if not doi:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM feedback_events "
                "WHERE doi = ? AND signal_type = 'saved' AND source = ? "
                "LIMIT 1",
                (doi, self.IMPLICIT_SAVE_SOURCE),
            ).fetchone()
        return row is not None

    # -- Bundle J (v1.0 active-learning CORE): interest vectors --

    def store_interest_vector(
        self,
        scope: str,
        vector: np.ndarray,
        n_events: int,
    ) -> None:
        """Bundle J: atomically upsert one computed interest vector.

        The vector is stored using the SAME blob encoding as
        ``clusters.centroid_blob``: ``np.ndarray.astype(float32).tobytes()``.
        ``computed_at`` is refreshed to ``now`` on every write so a stale
        vector is never silently retained after a recompute.

        Args:
            scope:    Key for this vector — ``'global'`` for the library-wide
                      interest vector, or a per-cluster scope tag (Bundle K).
            vector:   1-D embedding-space direction. Cast to float32 on write.
            n_events: Number of feedback events that contributed to ``vector``;
                      read back by the rank-time soft-gate (Bundle J2).
        """
        import numpy as np

        blob = np.asarray(vector, dtype=np.float32).tobytes()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO interest_vectors (scope, vector, n_events, computed_at) "
                "VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(scope) DO UPDATE SET "
                "  vector=excluded.vector, "
                "  n_events=excluded.n_events, "
                "  computed_at=excluded.computed_at",
                (scope, blob, int(n_events)),
            )

    def get_interest_vector(
        self,
        scope: str = "global",
    ) -> tuple[np.ndarray, int] | None:
        """Bundle J: load a stored interest vector + its contributor count.

        Decodes the blob with the same ``np.frombuffer(..., float32).copy()``
        pattern used for cluster centroids (``.copy()`` so the array owns its
        buffer and stays writable).

        Args:
            scope: Vector key to load. Defaults to ``'global'``.

        Returns:
            ``(vector, n_events)`` if a row exists for ``scope``, else ``None``.
        """
        import numpy as np

        with self._connect() as conn:
            row = conn.execute(
                "SELECT vector, n_events FROM interest_vectors WHERE scope = ?",
                (scope,),
            ).fetchone()
        if row is None:
            return None
        blob = row["vector"]
        if blob is None:
            return None
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        return vec, int(row["n_events"])
