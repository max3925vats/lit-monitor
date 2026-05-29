"""G5: mirror state.db citation_edges into KuzuDB CITES edges.

Runs (a) explicitly via the G10 backfill CLI (not yet built) and
(b) automatically at the end of ``obsidian build-citation-graph`` and
``obsidian rebuild-citations`` unless --no-graph is set.

Skips rows where either Paper node doesn't exist yet — G6's ingest path
or the user re-running mirror later will fill those in.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def safe_graph_db(persist_dir: str | None = None) -> Any | None:
    """Construct a GraphDB if the [graph] extra is installed; else log + return None.

    Used by CLI integration so commands don't hard-fail on installs without
    the ``[graph]`` optional extra.

    Args:
        persist_dir: path to pass to GraphDB.  Defaults to the production path
            ``~/.config/lit-monitor/graph.kuzu``.

    Returns:
        A ``GraphDB`` instance, or ``None`` if kuzu is not installed or the DB
        cannot be opened.
    """
    try:
        from scripts.graph import GraphDB
    except ImportError:
        logger.warning(
            "[graph] extra not installed; skipping graph mirror. "
            "Install with `uv sync --extra graph` to enable."
        )
        return None
    resolved_dir = persist_dir or "~/.config/lit-monitor/graph.kuzu"
    try:
        return GraphDB(persist_dir=resolved_dir)
    except Exception as exc:
        logger.warning("Could not open GraphDB at %s: %s", resolved_dir, exc)
        return None


def _paper_exists(conn: Any, doi: str) -> bool:
    """Return True if Kuzu has a Paper node with the given DOI."""
    res = conn.execute(
        "MATCH (p:Paper {doi: $doi}) RETURN count(p) AS n",
        {"doi": doi},
    )
    row = res.get_next()
    if not row:
        return False
    # get_next() returns a list; row[0] is the count value.
    return int(row[0]) > 0


def _cites_exists(conn: Any, src: str, tgt: str) -> bool:
    """Return True if a CITES edge from src → tgt already exists."""
    res = conn.execute(
        "MATCH (s:Paper {doi: $src})-[r:CITES]->(t:Paper {doi: $tgt}) "
        "RETURN count(r) AS n",
        {"src": src, "tgt": tgt},
    )
    row = res.get_next()
    if not row:
        return False
    return int(row[0]) > 0


def mirror_citations(
    graph_db: Any,
    state_db: Any,
    source_doi: str | None = None,
) -> int:
    """Mirror state.db citation edges into KuzuDB CITES edges.

    Walks ``state_db.get_citation_edges()`` (or a filtered subset when
    ``source_doi`` is given), then for each row with a non-null ``target_doi``
    creates a ``CITES`` edge in Kuzu — skipping rows where either endpoint
    Paper node is absent, and skipping edges that already exist (idempotent).

    Args:
        graph_db: ``GraphDB`` instance (from G1).  If ``None``, returns 0.
        state_db: ``StateDB`` instance with ``get_citation_edges()`` (E1).
                  If ``None``, returns 0.
        source_doi: optional filter — mirror only edges whose ``source_doi``
                    matches this value.  ``None`` mirrors all resolved edges.

    Returns:
        Number of ``CITES`` edges newly added (skips not counted).
    """
    if graph_db is None or state_db is None:
        logger.warning(
            "mirror_citations: graph_db or state_db is None; nothing to do."
        )
        return 0

    # Fetch edges — when source_doi is given, pass it through; otherwise we
    # need all edges.  state_db.get_citation_edges() requires a DOI argument,
    # so for the "all" case we use get_all_citation_edges() (added to StateDB
    # alongside this module).
    if source_doi is not None:
        edges = state_db.get_citation_edges(source_doi)
    else:
        edges = state_db.get_all_citation_edges()

    if not edges:
        return 0

    conn = graph_db._conn  # private access — consistent with G1/G14 test conventions
    added = 0
    skipped_no_paper = 0
    skipped_dup = 0

    for edge in edges:
        src = edge.get("source_doi")
        tgt = edge.get("target_doi")

        # Skip rows where either DOI is absent (unresolved citations).
        if not src or not tgt:
            continue

        # Skip if either Paper node doesn't exist in Kuzu yet.
        # G6's ingest path or a later mirror run will handle these.
        if not _paper_exists(conn, src) or not _paper_exists(conn, tgt):
            skipped_no_paper += 1
            continue

        # Idempotency: skip edges already present.
        if _cites_exists(conn, src, tgt):
            skipped_dup += 1
            continue

        # Use the DB resolution value, or fall back to a sensible default.
        resolution = edge.get("resolution") or "state_db_mirror"

        try:
            conn.execute(
                "MATCH (s:Paper {doi: $src}), (t:Paper {doi: $tgt}) "
                "CREATE (s)-[r:CITES {resolution: $res}]->(t)",
                {"src": src, "tgt": tgt, "res": resolution},
            )
            added += 1
        except Exception as exc:
            logger.warning(
                "mirror_citations: failed to create CITES %s -> %s: %s",
                src,
                tgt,
                exc,
            )

    if added or skipped_no_paper or skipped_dup:
        logger.info(
            "mirror_citations: added=%d skipped_no_paper=%d skipped_dup=%d",
            added,
            skipped_no_paper,
            skipped_dup,
        )

    return added
