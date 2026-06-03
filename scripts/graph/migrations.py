"""
KuzuDB schema DDL for lit-monitor Phase 1 knowledge graph.

``apply_schema(conn)`` runs all 12 CREATE … IF NOT EXISTS statements so the
function is idempotent: calling it on an already-initialised database is a
no-op.  The calling code (GraphDB.__init__) does not need to track whether
the schema has been applied before.

Node tables
-----------
  Paper   — one node per paper, keyed by DOI.
  Entity  — canonical named entity (gene, method, dataset, …), keyed by
            a canonical_id string.

Relationship tables (closed predicate set for v0.5.0)
------------------------------------------------------
Paper → Entity predicates (6):
  MENTIONS, DEPENDS_ON, PROPOSES, LIMITED_BY, INTRODUCES, RAISES_QUESTION

Paper → Paper predicates (4):
  CITES, COMPARES_TO, EXTENDS, CONTRADICTS

Every REL TABLE carries three optional provenance properties introduced in G14
(schema v2), populated by DB DEFAULTs when not explicitly set:
  confidence DOUBLE DEFAULT 1.0          — extraction confidence score
  extracted_at TIMESTAMP DEFAULT ...     — wall-clock INGEST time the edge was
                                           written (DEFAULT current_timestamp();
                                           add_paper never overrides it, so this
                                           is ingest/extraction time, NOT the
                                           paper's publication date). Trending
                                           detection (scripts/graph/trending.py)
                                           buckets MENTIONS edges by this column,
                                           so its "trending" signal is library-
                                           activity / ingest-recency, not field-
                                           level publication trends.
  prompt_version STRING DEFAULT 'phase1.0' — extractor prompt tag for
                                             reproducibility / re-extraction

Phase 3 R1 adds EXTENDS + CONTRADICTS (schema v3):
  EXTENDS     — paper A builds on / refines paper B's work
  CONTRADICTS — paper A disputes paper B's claim

Phase 4 (schema v4):
  CITES: renames 'resolution' column to 'evidence' so it matches all other
  typed Paper→Paper REL tables (EXTENDS, CONTRADICTS, COMPARES_TO).  The
  generic _upsert_typed_edge writes 'evidence'; CITES had diverged at G1.

Bundle D (schema v5):
  Paper: adds an ``authors`` STRING column (graph-in-ranker support).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# History: v1 (G1) → v2 (G14, provenance cols) → v3 (Phase 3 R1, new Paper→Paper RELs)
#          → v4 (Phase 4, CITES column parity fix) → v5 (Bundle D, Paper.authors column).
SCHEMA_VERSION: int = 5

# Default prompt version tag written to all REL TABLE edges by the DDL DEFAULT
# and used as the migration column default.  Centralised here so DDL and
# migration stay in sync when the tag is bumped.
_DEFAULT_PROMPT_VERSION: str = "phase1.0"

# ---------------------------------------------------------------------------
# Canonical typed-relationship predicate lists (shared with scripts/api/queries.py).
# Adding a new predicate here automatically surfaces it in the query layer.
# ---------------------------------------------------------------------------
# Paper → Entity predicates (5)
TYPED_PAPER_TO_ENTITY_PREDS: tuple[str, ...] = (
    "DEPENDS_ON",
    "PROPOSES",
    "LIMITED_BY",
    "INTRODUCES",
    "RAISES_QUESTION",
)
# Paper → Paper predicates (4)
TYPED_PAPER_TO_PAPER_PREDS: tuple[str, ...] = (
    "CITES",
    "COMPARES_TO",
    "EXTENDS",
    "CONTRADICTS",
)
# Combined list for callers that need all predicates.
TYPED_PREDICATES: tuple[str, ...] = TYPED_PAPER_TO_ENTITY_PREDS + TYPED_PAPER_TO_PAPER_PREDS

# ---------------------------------------------------------------------------
# Three provenance columns appended to every REL TABLE (G14).
# Using a shared fragment keeps the DDL consistent and easy to audit.
# NOTE: Kuzu 0.11.x does NOT support `COLUMN` keyword in ALTER TABLE ADD —
# use `ALTER TABLE <T> ADD <col> <type> DEFAULT <val>` (no COLUMN keyword).
# ---------------------------------------------------------------------------
_REL_PROVENANCE = (
    "confidence DOUBLE DEFAULT 1.0, "
    "extracted_at TIMESTAMP DEFAULT current_timestamp(), "
    f"prompt_version STRING DEFAULT '{_DEFAULT_PROMPT_VERSION}'"
)

# ---------------------------------------------------------------------------
# DDL statements — executed in order by apply_schema().
# All use IF NOT EXISTS so re-runs are no-ops.
# ---------------------------------------------------------------------------
DDL_STATEMENTS: list[str] = [
    # --- Node tables ---
    # v5: authors STRING column added (Bundle D graph-in-ranker).
    # Semicolon-separated author names, e.g. "Smith; Jones; Lee".
    # Empty string for papers where author data is unavailable.
    (
        "CREATE NODE TABLE IF NOT EXISTS Paper("
        "doi STRING, title STRING, year INT64, journal STRING, authors STRING, "
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
    # v4: uses 'evidence' (matches EXTENDS / CONTRADICTS / COMPARES_TO); the
    # original v1 DDL mistakenly used 'resolution'. Fresh installs get 'evidence'
    # directly; existing DBs get it via the v3→v4 migration (RENAME).
    (
        "CREATE REL TABLE IF NOT EXISTS CITES("
        f"FROM Paper TO Paper, evidence STRING, {_REL_PROVENANCE})"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS COMPARES_TO("
        f"FROM Paper TO Paper, evidence STRING, {_REL_PROVENANCE})"
    ),
    # --- Phase 3 R1: new Paper → Paper relationship tables ---
    # EXTENDS: paper A builds on / refines paper B's work.
    # CONTRADICTS: paper A disputes paper B's claim.
    # Both share the same G14 provenance block as COMPARES_TO.
    (
        "CREATE REL TABLE IF NOT EXISTS EXTENDS("
        f"FROM Paper TO Paper, evidence STRING, {_REL_PROVENANCE})"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS CONTRADICTS("
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
        ("prompt_version", f"STRING DEFAULT '{_DEFAULT_PROMPT_VERSION}'"),
    ]
    for table in _REL_TABLES_V1_TO_V2:
        for col_name, col_def in new_cols:
            ddl = f"ALTER TABLE {table} ADD {col_name} {col_def}"
            try:
                conn.execute(ddl)
                logger.debug("migrate_v1_to_v2: added %s.%s", table, col_name)
            except RuntimeError as exc:
                msg = str(exc).lower()
                # FRAGILITY NOTE: idempotency here relies on matching Kuzu's
                # error *message* text, which is not part of its API contract
                # and can change between minor versions. Kuzu 0.11.x raises:
                #   "MENTIONS table already has property confidence."
                # when the column already exists.  We normalise both "already
                # exists" (generic form) and "already has property" (Kuzu form)
                # so the migration is idempotent across engine minor versions.
                # The newer migrations (migrate_v3_to_v4 / migrate_v4_to_v5)
                # use CALL table_info(...) introspection instead — the robust
                # pattern.  This pre-v2 path predates that and is left as-is to
                # avoid rewriting a migration that only runs on legacy v1 DBs.
                if "already has property" in msg or "already exists" in msg:
                    logger.debug(
                        "migrate_v1_to_v2: %s.%s already exists, skipping.",
                        table,
                        col_name,
                    )
                else:
                    raise
    logger.info("migrate_v1_to_v2: all REL TABLE provenance columns ensured.")


# ---------------------------------------------------------------------------
# v2 → v3 migration: add EXTENDS + CONTRADICTS REL tables (Phase 3 R1)
# ---------------------------------------------------------------------------

# DDL for the two new Paper→Paper REL tables introduced in schema v3.
# Defined here (not inlined in migrate_v2_to_v3) so the test suite can
# reference the list independently.
_V2_TO_V3_REL_TABLES: list[tuple[str, str]] = [
    (
        "EXTENDS",
        (
            "CREATE REL TABLE IF NOT EXISTS EXTENDS("
            f"FROM Paper TO Paper, evidence STRING, {_REL_PROVENANCE})"
        ),
    ),
    (
        "CONTRADICTS",
        (
            "CREATE REL TABLE IF NOT EXISTS CONTRADICTS("
            f"FROM Paper TO Paper, evidence STRING, {_REL_PROVENANCE})"
        ),
    ),
]


def migrate_v2_to_v3(conn) -> int:  # type: ignore[type-arg]
    """Add EXTENDS + CONTRADICTS REL tables to a v2 graph (schema v2 → v3).

    Idempotent: if a table already exists (e.g. this migration has already
    been run), the ``IF NOT EXISTS`` guard in the DDL makes it a no-op.
    Any residual ``RuntimeError`` with "already exists" in the message is
    silently swallowed for belt-and-suspenders safety.

    Parameters
    ----------
    conn:
        An open ``kuzu.Connection`` pointing at a v2-schema KuzuDB database.

    Returns
    -------
    int
        The new schema version: ``3``.
    """
    for table_name, ddl in _V2_TO_V3_REL_TABLES:
        try:
            conn.execute(ddl)
            logger.debug("migrate_v2_to_v3: created REL TABLE %s", table_name)
        except RuntimeError as exc:
            msg = str(exc).lower()
            # Guard against engines that don't honour IF NOT EXISTS for REL tables.
            if "already exists" in msg or "already has" in msg:
                logger.debug(
                    "migrate_v2_to_v3: %s already exists (idempotent)", table_name
                )
            else:
                raise
    logger.info("migrate_v2_to_v3: EXTENDS and CONTRADICTS REL tables ensured.")
    return 3


def migrate_v3_to_v4(conn) -> int:  # type: ignore[type-arg]
    """Rename CITES.resolution → CITES.evidence (schema v3 → v4).

    CITES diverged from every other typed Paper→Paper REL table at G1 by
    using a ``resolution`` column instead of ``evidence``.  The generic
    ``_upsert_typed_edge`` writes ``evidence``, so calling
    ``add_paper(..., relationships=[("CITES", ...)])`` raised a Kuzu
    RuntimeError on v3 graphs.

    Implementation note (Kuzu 0.11.x)
    ----------------------------------
    Kuzu 0.11.x supports ``ALTER TABLE <T> RENAME <old> TO <new>``.
    Confirmed empirically: ADD, DROP, and RENAME all work on REL TABLEs
    in 0.11.x.  RENAME is the cleanest path — no data copy needed.

    Idempotency
    -----------
    If ``resolution`` is already gone (fresh install that got the fixed DDL,
    or migration re-run), the RENAME is skipped rather than raising.
    We inspect ``table_info('CITES')`` before acting.

    Parameters
    ----------
    conn:
        An open ``kuzu.Connection`` pointing at a v3-schema KuzuDB database.

    Returns
    -------
    int
        The new schema version: ``4``.
    """
    try:
        res = conn.execute("CALL table_info('CITES') RETURN *")
        cols: list[str] = []
        while res.has_next():
            row = res.get_next()
            # row layout: (property_id, name, type, default, direction)
            cols.append(row[1])

        if "resolution" in cols and "evidence" not in cols:
            # Rename the divergent column to match the rest.
            conn.execute("ALTER TABLE CITES RENAME resolution TO evidence")
            logger.info("migrate_v3_to_v4: CITES.resolution renamed to evidence.")
        elif "evidence" in cols:
            logger.debug(
                "migrate_v3_to_v4: CITES already has 'evidence' column, skipping (idempotent)."
            )
        else:
            # Neither column present — table was dropped/recreated elsewhere.
            # Not our problem to resolve; log and continue.
            logger.warning(
                "migrate_v3_to_v4: CITES has neither 'resolution' nor 'evidence'. "
                "Columns found: %s. Skipping rename.",
                cols,
            )
    except RuntimeError as exc:
        msg = str(exc).lower()
        # If CITES doesn't exist yet (fresh install still running apply_schema),
        # the table_info call will fail.  That's fine — apply_schema will create
        # CITES with 'evidence' from scratch.
        if "does not exist" in msg or "not exist" in msg or "cannot find" in msg:
            logger.debug(
                "migrate_v3_to_v4: CITES table not present yet (will be created by apply_schema), skipping."
            )
        else:
            raise

    return 4


def migrate_v4_to_v5(conn) -> int:  # type: ignore[type-arg]
    """Add Paper.authors STRING column to existing databases (schema v4 → v5).

    Bundle D (graph-in-ranker) uses author overlap as a ranking signal.
    The authors column stores a semicolon-separated list of author names.
    Empty string for papers where author data was not available at ingest time.

    Idempotency: if the column already exists (fresh install with v5 DDL),
    the ALTER TABLE is skipped rather than raising.

    Parameters
    ----------
    conn:
        An open ``kuzu.Connection`` pointing at a v4-schema KuzuDB database.

    Returns
    -------
    int
        The new schema version: ``5``.
    """
    try:
        res = conn.execute("CALL table_info('Paper') RETURN *")
        cols: list[str] = []
        while res.has_next():
            row = res.get_next()
            # row layout per Kuzu table_info: (property_id, name, type, ...)
            cols.append(str(row[1]))

        if "authors" not in cols:
            conn.execute("ALTER TABLE Paper ADD authors STRING DEFAULT ''")
            logger.info("migrate_v4_to_v5: Paper.authors column added.")
        else:
            logger.debug(
                "migrate_v4_to_v5: Paper.authors already exists (idempotent)."
            )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "already has property" in msg or "already exists" in msg:
            logger.debug("migrate_v4_to_v5: authors already present, skipping.")
        else:
            raise

    return 5


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

    if current_version < 3:
        logger.info(
            "apply_migrations: v%d → v3 (adding EXTENDS + CONTRADICTS REL tables).",
            current_version,
        )
        current_version = migrate_v2_to_v3(conn)

    if current_version < 4:
        logger.info(
            "apply_migrations: v%d → v4 (renaming CITES.resolution → evidence).",
            current_version,
        )
        current_version = migrate_v3_to_v4(conn)

    if current_version < 5:
        logger.info(
            "apply_migrations: v%d → v5 (adding Paper.authors column for Bundle D).",
            current_version,
        )
        current_version = migrate_v4_to_v5(conn)

    return current_version
