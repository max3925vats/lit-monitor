"""
KuzuDB schema DDL for lit-monitor Phase 1 knowledge graph (G1).

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

Phase 3 will add EXTENDS + CONTRADICTS — those are NOT defined here.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL statements — executed in order by apply_schema().
# All use IF NOT EXISTS so re-runs are no-ops.
# ---------------------------------------------------------------------------
_DDL_STATEMENTS: list[str] = [
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
    (
        "CREATE REL TABLE IF NOT EXISTS MENTIONS("
        "FROM Paper TO Entity, "
        "source STRING, surface STRING, field STRING, "
        "confidence DOUBLE, span_start INT64, span_end INT64)"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS DEPENDS_ON("
        "FROM Paper TO Entity, evidence STRING, confidence DOUBLE)"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS PROPOSES("
        "FROM Paper TO Entity, evidence STRING, confidence DOUBLE)"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS LIMITED_BY("
        "FROM Paper TO Entity, evidence STRING, confidence DOUBLE)"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS INTRODUCES("
        "FROM Paper TO Entity, evidence STRING, confidence DOUBLE)"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS RAISES_QUESTION("
        "FROM Paper TO Entity, evidence STRING, confidence DOUBLE)"
    ),
    # --- Relationship tables: Paper → Paper ---
    (
        "CREATE REL TABLE IF NOT EXISTS CITES("
        "FROM Paper TO Paper, resolution STRING)"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS COMPARES_TO("
        "FROM Paper TO Paper, evidence STRING, confidence DOUBLE)"
    ),
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
    for ddl in _DDL_STATEMENTS:
        conn.execute(ddl)
    logger.debug("apply_schema: %d DDL statement(s) executed (idempotent).", len(_DDL_STATEMENTS))
