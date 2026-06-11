"""Bundle K-b (v0.9): graph-driven exploration queries for under-engaged clusters.

Part B of the exploration budget. For each under-engaged cluster (see
``atrophy.find_under_engaged_clusters``) we:

1. Find the cluster's CENTRAL ENTITIES — the entities most frequently mentioned
   across the cluster's member papers in the KuzuDB graph (counted by DISTINCT
   member paper, so one chatty paper can't dominate).
2. Turn each central entity surface into a search query, optionally enriched by
   the Bundle E ``query_expansion.suggest_expansions`` co-occurrence lookup.
3. Return a bounded, de-duplicated list of exploration query strings.

Defensive perimeter: every public helper is non-fatal. A missing/empty graph,
a cluster with no member entities, or any Cypher failure yields an empty list
for that cluster rather than raising — exploration is enrichment, never a gate
on the discovery run.
"""

from __future__ import annotations

import logging
from typing import Any

from scripts.graph.query_expansion import suggest_expansions

logger = logging.getLogger(__name__)

#: Default number of central entities pulled per cluster.
DEFAULT_TOP_ENTITIES: int = 3

#: Default cap on exploration queries returned per cluster (central entities +
#: their expansions), so one cluster can't flood the budget.
DEFAULT_MAX_QUERIES_PER_CLUSTER: int = 5


def central_entities_for_dois(
    member_dois: list[str],
    graph_db: Any,
    *,
    top_k: int = DEFAULT_TOP_ENTITIES,
) -> list[str]:
    """Most-mentioned entity *surfaces* among ``member_dois`` in the graph.

    Counts DISTINCT member papers per entity so a single paper that mentions an
    entity many times contributes at most 1. Returns surface strings (the
    human-readable entity text), ordered by member-paper count descending,
    capped at ``top_k``.

    Non-fatal: returns ``[]`` on empty input, a ``None`` graph, or any Cypher
    error.
    """
    dois = [d for d in (member_dois or []) if d]
    if not dois or graph_db is None:
        return []
    try:
        cypher = (
            "MATCH (p:Paper)-[:MENTIONS]->(e:Entity) "
            "WHERE p.doi IN $dois "
            "RETURN e.surface AS surface, count(DISTINCT p.doi) AS n "
            "ORDER BY n DESC "
            "LIMIT $k"
        )
        result = graph_db._conn.execute(cypher, {"dois": dois, "k": int(top_k)})
        surfaces: list[str] = []
        while result.has_next():
            row = result.get_next()
            surface = row[0]
            if surface and str(surface).strip():
                surfaces.append(str(surface).strip())
        return surfaces
    except Exception as exc:  # noqa: BLE001 — exploration is enrichment, never a gate
        logger.warning(
            "K-b: central-entity lookup failed for %d member DOIs (non-fatal): %s",
            len(dois), exc,
        )
        return []


def build_exploration_queries_for_cluster(
    member_dois: list[str],
    graph_db: Any,
    *,
    top_entities: int = DEFAULT_TOP_ENTITIES,
    max_queries: int = DEFAULT_MAX_QUERIES_PER_CLUSTER,
    expand: bool = True,
) -> list[str]:
    """Bounded, de-duplicated exploration queries for ONE cluster.

    The cluster's central entity surfaces seed the query list; when ``expand``
    is True each central entity is also run through the Bundle E expansion
    lookup and any co-occurring entity ids are appended. The combined list is
    de-duplicated (case-insensitive) preserving order and capped at
    ``max_queries``.

    Returns ``[]`` when the cluster has no central entities (empty/absent graph,
    no member papers, or no mentions) — that cluster is simply skipped upstream.
    """
    centrals = central_entities_for_dois(member_dois, graph_db, top_k=top_entities)
    if not centrals:
        return []

    queries: list[str] = list(centrals)
    if expand and graph_db is not None:
        for surface in centrals:
            try:
                queries.extend(suggest_expansions(surface, graph_db))
            except Exception as exc:  # noqa: BLE001 — suggest_expansions is itself
                # defensive, but belt-and-suspenders so one bad surface can't abort.
                logger.warning(
                    "K-b: expansion failed for central entity %r (non-fatal): %s",
                    surface, exc,
                )

    # De-duplicate case-insensitively, preserving first-seen order, then cap.
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(q.strip())
    return deduped[: max(0, int(max_queries))]


def build_exploration_queries(
    cluster_member_dois: dict[int, list[str]],
    graph_db: Any,
    *,
    top_entities: int = DEFAULT_TOP_ENTITIES,
    max_queries_per_cluster: int = DEFAULT_MAX_QUERIES_PER_CLUSTER,
    global_cap: int | None = None,
    expand: bool = True,
) -> list[str]:
    """Exploration queries across several under-engaged clusters (de-duplicated).

    ``cluster_member_dois`` maps each under-engaged cluster id to its member
    DOIs. Each cluster contributes up to ``max_queries_per_cluster`` queries;
    the union is de-duplicated case-insensitively (preserving order) and, when
    ``global_cap`` is set, truncated to that many queries.

    Non-fatal end to end: an absent/empty graph or a cluster with no entities
    contributes nothing. Returns ``[]`` when no cluster yields a query.
    """
    if graph_db is None or not cluster_member_dois:
        return []

    all_queries: list[str] = []
    for cid, dois in cluster_member_dois.items():
        cluster_qs = build_exploration_queries_for_cluster(
            dois,
            graph_db,
            top_entities=top_entities,
            max_queries=max_queries_per_cluster,
            expand=expand,
        )
        if cluster_qs:
            logger.info(
                "K-b: cluster %s → %d exploration query(ies)", cid, len(cluster_qs)
            )
            all_queries.extend(cluster_qs)

    seen: set[str] = set()
    deduped: list[str] = []
    for q in all_queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(q.strip())

    if global_cap is not None and global_cap >= 0:
        deduped = deduped[:global_cap]
    return deduped
