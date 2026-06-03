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

# Production default kuzu location, used when neither an explicit persist_dir nor
# a configured retrieval.graph_db.persist_dir is available.
_DEFAULT_GRAPH_PATH = "~/.config/lit-monitor/graph.kuzu"


def _configured_graph_path() -> str | None:
    """Return ``retrieval.graph_db.persist_dir`` from config, or None if unset.

    Defensive throughout: config loading is optional here, and pre-graph configs
    may lack the ``retrieval.graph_db`` block entirely, so any failure simply
    yields None and the caller falls back to the production default.
    """
    try:
        from scripts.core.config import get_config

        graph_cfg = getattr(getattr(get_config(), "retrieval", None), "graph_db", None)
        if graph_cfg is None:
            return None
        # graph_db is a dot-access _Namespace when present (.get is safe); a
        # non-dict raw value won't have .get and yields None.
        persist_dir = graph_cfg.get("persist_dir") if hasattr(graph_cfg, "get") else None
        return persist_dir or None
    except Exception as exc:  # noqa: BLE001 — config optional; fall back to default
        logger.debug("safe_graph_db: could not read graph path from config: %s", exc)
        return None


def safe_graph_db(persist_dir: str | None = None) -> Any | None:
    """Construct a GraphDB if the [graph] extra is installed; else log + return None.

    Used by CLI integration so commands don't hard-fail on installs without
    the ``[graph]`` optional extra.

    Args:
        persist_dir: explicit path to pass to GraphDB. If omitted, the path is
            read from ``retrieval.graph_db.persist_dir`` in extraction.yaml, then
            falls back to the production default ``~/.config/lit-monitor/graph.kuzu``.

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
    resolved_dir = persist_dir or _configured_graph_path() or _DEFAULT_GRAPH_PATH
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

    # P3.4: wrap the whole CITES-creation loop in a single Kuzu write
    # transaction so a mid-loop failure leaves NO partial edges committed.
    # Previously each CREATE auto-committed individually, so a crash after N
    # edges left those N edges behind. Mirroring GraphDB.add_paper's pattern:
    # BEGIN, do the work, COMMIT; on any exception ROLLBACK and re-raise so the
    # caller sees the real failure and the graph is unchanged. A per-edge
    # CREATE failure now aborts the whole batch rather than being swallowed —
    # all-or-nothing is the point of the transaction.
    conn.execute("BEGIN TRANSACTION")
    try:
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

            # v4 schema: CITES uses 'evidence' (was 'resolution' before schema v4).
            # Prefer the 'evidence' key from the edge dict; fall back to 'resolution'
            # for backward-compat with any callers still passing the old key, then
            # default to a descriptive sentinel so the column is never NULL.
            evidence = (
                edge.get("evidence")
                or edge.get("resolution")
                or "state_db_mirror"
            )

            conn.execute(
                "MATCH (s:Paper {doi: $src}), (t:Paper {doi: $tgt}) "
                "CREATE (s)-[r:CITES {evidence: $ev}]->(t)",
                {"src": src, "tgt": tgt, "ev": evidence},
            )
            added += 1
        conn.execute("COMMIT")
    except Exception as exc:
        # Best-effort rollback — re-raise the ORIGINAL exception so the caller
        # sees the real cause even if the rollback itself fails.
        try:
            conn.execute("ROLLBACK")
        except Exception as rb_exc:  # pragma: no cover — defensive
            logger.error(
                "mirror_citations: ROLLBACK also failed: %s (original: %s)",
                rb_exc,
                exc,
            )
        logger.warning("mirror_citations rolled back: %s", exc)
        raise

    if added or skipped_no_paper or skipped_dup:
        logger.info(
            "mirror_citations: added=%d skipped_no_paper=%d skipped_dup=%d",
            added,
            skipped_no_paper,
            skipped_dup,
        )

    return added
