"""
KuzuDB schema DDL for lit-monitor Phase 1 knowledge graph.

``apply_schema(conn)`` runs all 10 CREATE … IF NOT EXISTS statements so the
function is idempotent: calling it on an already-initialised database is a
no-op.  The calling code (GraphDB.__init__) does not need to track whether
the schema has been applied before.

Node tables
-----------
  Paper   — one node per paper, keyed by DOI.
  Entity  — canonical named entity (gene, method, dataset, …), keyed by
            a canonical_id string.

Relationship tables (closed predicate set for v0.4.0)
------------------------------------------------------
Paper → Entity predicates (6):
  MENTIONS, DEPENDS_ON, PROPOSES, LIMITED_BY, INTRODUCES, RAISES_QUESTION

Paper → Paper predicates (2):
  CITES, COMPARES_TO

Every REL TABLE carries three optional provenance properties introduced in G14
(schema v2), populated by DB DEFAULTs when not explicitly set:
  confidence DOUBLE DEFAULT 1.0          — extraction confidence score
  extracted_at TIMESTAMP DEFAULT ...     — wall-clock time of extraction
  prompt_version STRING DEFAULT 'phase1.0' — extractor prompt tag for
                                             reproducibility / re-extraction

Phase 3 will add EXTENDS + CONTRADICTS — those are NOT defined here.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # kuzu types not importable at type-check time without the extra

logger = logging.getLogger(__name__)

# Bumped here (G14): adds confidence/extracted_at/prompt_version to all REL TABLEs.
SCHEMA_VERSION: int = 2

# ---------------------------------------------------------------------------
# Three provenance columns appended to every REL TABLE (G14).
# Using a shared fragment keeps the DDL consistent and easy to audit.
# NOTE: Kuzu 0.11.x does NOT support `COLUMN` keyword in ALTER TABLE ADD —
# use `ALTER TABLE <T> ADD <col> <type> DEFAULT <val>` (no COLUMN keyword).
# ---------------------------------------------------------------------------
_REL_PROVENANCE = (
    "confidence DOUBLE DEFAULT 1.0, "
    "extracted_at TIMESTAMP DEFAULT current_timestamp(), "
    "prompt_version STRING DEFAULT 'phase1.0'"
)

# ---------------------------------------------------------------------------
# DDL statements — executed in order by apply_schema().
# All use IF NOT EXISTS so re-runs are no-ops.
# ---------------------------------------------------------------------------
DDL_STATEMENTS: list[str] = [
    # --- Node tables ---
    (
        "CREATE NODE TABLE IF NOT EXISTS Paper("
        "doi STRING, title STRING, year INT64, journal STRING, "
        "PRIMARY KEY(doi))"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS Entity("
        "canonical_id STRING, type STRING, surface STRING, "
        "PRIMARY KEY(canonical_id))"
    ),
    # --- Relationship tables: Paper → Entity ---
    # MENTIONS had confidence DOUBLE in v1; the v2 column is now the shared
    # provenance column that replaces the v1 lone confidence field.
    (
        "CREATE REL TABLE IF NOT EXISTS MENTIONS("
        "FROM Paper TO Entity, "
        "source STRING, surface STRING, field STRING, "
        f"span_start INT64, span_end INT64, {_REL_PROVENANCE})"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS DEPENDS_ON("
        f"FROM Paper TO Entity, evidence STRING, {_REL_PROVENANCE})"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS PROPOSES("
        f"FROM Paper TO Entity, evidence STRING, {_REL_PROVENANCE})"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS LIMITED_BY("
        f"FROM Paper TO Entity, evidence STRING, {_REL_PROVENANCE})"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS INTRODUCES("
        f"FROM Paper TO Entity, evidence STRING, {_REL_PROVENANCE})"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS RAISES_QUESTION("
        f"FROM Paper TO Entity, evidence STRING, {_REL_PROVENANCE})"
    ),
    # --- Relationship tables: Paper → Paper ---
    (
        "CREATE REL TABLE IF NOT EXISTS CITES("
        f"FROM Paper TO Paper, resolution STRING, {_REL_PROVENANCE})"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS COMPARES_TO("
        f"FROM Paper TO Paper, evidence STRING, {_REL_PROVENANCE})"
    ),
]

# ---------------------------------------------------------------------------
# The v1 DDL (without provenance columns) is reproduced here verbatim so
# the migration test can build a v1-schema database without touching Git.
# This list is intentionally frozen at the G1 state; do NOT edit it.
# ---------------------------------------------------------------------------
_DDL_V1_STATEMENTS: list[str] = [
    "CREATE NODE TABLE IF NOT EXISTS Paper(doi STRING, title STRING, year INT64, journal STRING, PRIMARY KEY(doi))",
    "CREATE NODE TABLE IF NOT EXISTS Entity(canonical_id STRING, type STRING, surface STRING, PRIMARY KEY(canonical_id))",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Paper TO Entity, source STRING, surface STRING, field STRING, confidence DOUBLE, span_start INT64, span_end INT64)",
    "CREATE REL TABLE IF NOT EXISTS DEPENDS_ON(FROM Paper TO Entity, evidence STRING, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS PROPOSES(FROM Paper TO Entity, evidence STRING, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS LIMITED_BY(FROM Paper TO Entity, evidence STRING, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS INTRODUCES(FROM Paper TO Entity, evidence STRING, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS RAISES_QUESTION(FROM Paper TO Entity, evidence STRING, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS CITES(FROM Paper TO Paper, resolution STRING)",
    "CREATE REL TABLE IF NOT EXISTS COMPARES_TO(FROM Paper TO Paper, evidence STRING, confidence DOUBLE)",
]

# REL tables that gain the three new provenance columns in v1 → v2 migration.
_REL_TABLES_V1_TO_V2: list[str] = [
    "MENTIONS",
    "DEPENDS_ON",
    "PROPOSES",
    "LIMITED_BY",
    "INTRODUCES",
    "RAISES_QUESTION",
    "CITES",
    "COMPARES_TO",
]


def apply_schema(conn) -> None:  # type: ignore[type-arg]
    """Execute all DDL statements against an open KuzuDB Connection.

    Parameters
    ----------
    conn:
        A ``kuzu.Connection`` instance pointing at the target database.

    The function is idempotent — calling it more than once on the same
    database is safe because every statement uses ``IF NOT EXISTS``.
    """
    for ddl in DDL_STATEMENTS:
        conn.execute(ddl)
    logger.debug(
        "apply_schema: %d DDL statement(s) executed (idempotent).",
        len(DDL_STATEMENTS),
    )


def migrate_v1_to_v2(conn) -> None:  # type: ignore[type-arg]
    """Add provenance columns to all 8 REL TABLEs for schema v1 → v2.

    Three columns are added to every REL TABLE:
      - ``confidence DOUBLE DEFAULT 1.0``
      - ``extracted_at TIMESTAMP DEFAULT current_timestamp()``
      - ``prompt_version STRING DEFAULT 'phase1.0'``

    Implementation note (Kuzu 0.11.x)
    ----------------------------------
    KuzuDB 0.11.x supports ``ALTER TABLE <T> ADD <col> <type> DEFAULT <val>``
    but does **NOT** accept the ``COLUMN`` keyword (unlike standard SQL).
    The correct syntax is::

        ALTER TABLE MENTIONS ADD confidence DOUBLE DEFAULT 1.0

    Existing edges in the table receive the DEFAULT value immediately upon
    the ALTER — confirmed empirically on Kuzu 0.11.3.  No data-copy loop is
    required.

    Idempotency: if a column already exists, KuzuDB raises a RuntimeError with
    the text "already exists".  We catch those and log a warning so re-running
    the migration on an already-upgraded graph is safe.

    Parameters
    ----------
    conn:
        An open ``kuzu.Connection`` pointing at a v1-schema KuzuDB database.
    """
    new_cols = [
        ("confidence", "DOUBLE DEFAULT 1.0"),
        ("extracted_at", "TIMESTAMP DEFAULT current_timestamp()"),
        ("prompt_version", "STRING DEFAULT 'phase1.0'"),
    ]
    for table in _REL_TABLES_V1_TO_V2:
        for col_name, col_def in new_cols:
            ddl = f"ALTER TABLE {table} ADD {col_name} {col_def}"
            try:
                conn.execute(ddl)
                logger.debug("migrate_v1_to_v2: added %s.%s", table, col_name)
            except RuntimeError as exc:
                msg = str(exc).lower()
                # Kuzu 0.11.x raises:
                #   "MENTIONS table already has property confidence."
                # when the column already exists.  We normalise both "already
                # exists" (generic form) and "already has property" (Kuzu form)
                # so the migration is idempotent across engine minor versions.
                if "already has property" in msg or "already exists" in msg:
                    logger.warning(
                        "migrate_v1_to_v2: %s.%s already exists, skipping.",
                        table,
                        col_name,
                    )
                else:
                    raise
    logger.info("migrate_v1_to_v2: all REL TABLE provenance columns ensured.")


def apply_migrations(conn, current_version: int) -> int:  # type: ignore[type-arg]
    """Run any pending schema migrations and return the new schema version.

    Parameters
    ----------
    conn:
        An open ``kuzu.Connection`` to the target database.
    current_version:
        The schema version read from the sentinel file (or 1 if the file is
        absent — meaning the database was created by G1 and has never been
        migrated).

    Returns
    -------
    int
        The schema version after all applicable migrations have been applied.
        The caller (``GraphDB.__init__``) must persist this value.
    """
    if current_version < 2:
        logger.info(
            "apply_migrations: v%d → v2 (adding REL TABLE provenance columns).",
            current_version,
        )
        migrate_v1_to_v2(conn)
        current_version = 2

    # Future migrations:  if current_version < 3: migrate_v2_to_v3(conn) …

    return current_version
