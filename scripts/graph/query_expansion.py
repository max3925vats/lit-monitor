"""Bundle E (v0.9): graph-driven query expansion — top-k co-occurring entity lookup.

For each topic query string, finds the top-k entities that co-occur in papers
that mention the topic's primary entity. Results are suggestion-only — never
auto-applied (per locked decision #4).

Defensive perimeter: suggest_expansions() never raises. All failures are
logged at WARNING and an empty list is returned.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 3


def suggest_expansions(
    topic_query: str,
    graph_db: Any,
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> list[str]:
    """Bundle E: find top-k entities that co-occur with the topic's primary entity.

    Steps:
    1. Normalize topic_query to a canonical entity by substring match on
       Entity.surface (first hit, case-insensitive).
    2. Find papers that MENTION that entity.
    3. Count all other entities mentioned by those same papers.
    4. Return top-k canonical_ids ordered by co-occurrence count descending.

    Args:
        topic_query: Topic name or query string to look up (matched against
                     Entity.surface via case-insensitive substring).
        graph_db:    GraphDB instance (must expose ._conn.execute).
        top_k:       Maximum number of expansion suggestions to return.
                     Defaults to 3 (per plan).

    Returns:
        List of canonical_id strings (at most top_k), ordered by co-count.
        Returns [] on any failure or when no entity is found.
    """
    try:
        # --- Step 1: Resolve topic_query → canonical_id ---
        entity_cid = _resolve_entity(topic_query, graph_db)
        if not entity_cid:
            logger.info(
                "E: no entity found in graph for topic query %r — skipping expansion",
                topic_query,
            )
            return []

        # --- Steps 2-4: Co-occurring entities ---
        return _find_co_occurring(entity_cid, graph_db, top_k=top_k)

    except Exception as exc:
        logger.warning(
            "E: expansion lookup failed for %r: %s",
            topic_query,
            exc,
        )
        return []


def _resolve_entity(topic_query: str, graph_db: Any) -> str | None:
    """Find the first Entity whose surface contains topic_query (case-insensitive).

    Returns the canonical_id string, or None if no match.
    """
    needle = topic_query.strip().lower()
    if not needle:
        return None

    # Kuzu's LOWER() function enables case-insensitive substring match.
    cypher = (
        "MATCH (e:Entity) "
        "WHERE lower(e.surface) CONTAINS $needle "
        "RETURN e.canonical_id "
        "LIMIT 1"
    )
    result = graph_db._conn.execute(cypher, {"needle": needle})
    if result.has_next():
        row = result.get_next()
        cid = row[0]
        return str(cid) if cid else None
    return None


def _find_co_occurring(
    entity_cid: str,
    graph_db: Any,
    *,
    top_k: int,
) -> list[str]:
    """Count entities co-mentioned with entity_cid; return top-k canonical_ids.

    A co-occurrence is counted once per paper (DISTINCT paper DOI) to avoid
    papers with many repeated mentions of the same entity dominating the signal.

    Args:
        entity_cid: Canonical ID of the seed entity.
        graph_db:   GraphDB instance.
        top_k:      Maximum results to return.

    Returns:
        List of canonical_id strings ordered by co-count descending.
    """
    cypher = (
        "MATCH (p:Paper)-[:MENTIONS]->(seed:Entity {canonical_id: $cid}) "
        "MATCH (p)-[:MENTIONS]->(other:Entity) "
        "WHERE other.canonical_id <> $cid "
        "RETURN other.canonical_id, count(DISTINCT p) AS co_count "
        "ORDER BY co_count DESC "
        "LIMIT $k"
    )
    result = graph_db._conn.execute(cypher, {"cid": entity_cid, "k": top_k})
    out: list[str] = []
    while result.has_next():
        row = result.get_next()
        cid = row[0]
        if cid:
            out.append(str(cid))
    return out
