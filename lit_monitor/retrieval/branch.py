"""G9: rag-mode dispatch for hybrid retrieval.

Provides a single entry point — ``retrieve_doi_candidates`` — that routes
to vector, graph, or hybrid (RRF-fused) retrieval based on ``rag_mode``.

Used by:
  - scripts/obsidian_tools/relink.py     (similar-paper expansion)
  - scripts/obsidian_tools/synthesize.py (topic-based retrieval)
  - scripts/pipelines/discovery.py       (candidate scoring)
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from scripts.core.doi import normalize_doi
from scripts.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

RagMode = Literal["vector", "graph", "hybrid"]


def retrieve_doi_candidates(
    rag_mode: str,
    *,
    seed_doi: str | None = None,
    query_text: str | None = None,
    entity_ids: list[str] | None = None,
    embeddings_db: Any | None = None,
    graph_db: Any | None = None,
    k: int = 20,
    exclude_id: str | None = None,
    rerank_with_query: str | None = None,
    reranker_config: Any | None = None,
) -> list[tuple[str, float]]:
    """Return one ranked (doi, score) list per rag_mode.

    Args:
        rag_mode:    One of "vector", "graph", "hybrid".
        seed_doi:    For graph mode: DOI of the source paper for
                     find_similar_papers(); ignored in pure vector mode.
        query_text:  For vector mode: text to embed-query against ChromaDB.
        entity_ids:  For graph mode: canonical entity IDs for
                     find_papers_by_entities() (used when seed_doi absent).
        embeddings_db:  EmbeddingsDB instance; may be None.
        graph_db:    GraphDB instance; may be None.
        k:           Maximum number of results.
        exclude_id:  G9 W1: forwarded to embeddings.find_similar_to_text so the
                     vector leg of hybrid skips the seed paper (matches v0.3.3).
        rerank_with_query:  G9 W1: forwarded to embeddings.find_similar_to_text
                     so the vector leg participates in cross-encoder reranking
                     (matches v0.3.3 N19 contract).
        reranker_config:  G9 W1: forwarded alongside ``rerank_with_query``.

    Returns:
        List of (doi, score) tuples ranked by relevance descending.
        Empty list when the relevant DB is None for the requested mode.

    Raises:
        ValueError: unknown rag_mode value.
    """
    if rag_mode == "vector":
        return _retrieve_vector(
            embeddings_db, query_text, k,
            exclude_id=exclude_id,
            rerank_with_query=rerank_with_query,
            reranker_config=reranker_config,
        )
    if rag_mode == "graph":
        return _retrieve_graph(graph_db, seed_doi, entity_ids, k)
    if rag_mode == "hybrid":
        vector_rank = _retrieve_vector(
            embeddings_db, query_text, k,
            exclude_id=exclude_id,
            rerank_with_query=rerank_with_query,
            reranker_config=reranker_config,
        )
        graph_rank = _retrieve_graph(graph_db, seed_doi, entity_ids, k)
        # RRF expects plain lists of doc IDs, not (doi, score) pairs.
        # AR-4: canonicalize DOIs from BOTH legs before fusion so a case/URL
        # mismatch between the vector and graph legs does not double-count the
        # same paper (RRF fuses by string equality).
        v_ids = [normalize_doi(d) for d, _ in vector_rank]
        g_ids = [normalize_doi(d) for d, _ in graph_rank]
        fused = reciprocal_rank_fusion([v_ids, g_ids])
        return fused[:k]
    raise ValueError(
        f"Unknown rag_mode: {rag_mode!r}. Must be one of 'vector', 'graph', 'hybrid'."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _retrieve_vector(
    embeddings_db: Any | None,
    query_text: str | None,
    k: int,
    *,
    exclude_id: str | None = None,
    rerank_with_query: str | None = None,
    reranker_config: Any | None = None,
) -> list[tuple[str, float]]:
    """Embed-query ChromaDB and return (doi, score) pairs.

    G9 W1: optional kwargs ``exclude_id``, ``rerank_with_query``, and
    ``reranker_config`` are forwarded to ``find_similar_to_text`` so the vector
    leg of hybrid behaves bit-for-bit like the v0.3.3 vector-only path.
    """
    if embeddings_db is None:
        logger.debug("vector retrieval skipped: embeddings_db is None")
        return []
    if not query_text:
        logger.debug("vector retrieval skipped: no query_text supplied")
        return []
    kwargs: dict[str, Any] = {"top_k": k}
    if exclude_id is not None:
        kwargs["exclude_id"] = exclude_id
    if rerank_with_query is not None:
        kwargs["rerank_with_query"] = rerank_with_query
    if reranker_config is not None:
        kwargs["reranker_config"] = reranker_config
    try:
        results = embeddings_db.find_similar_to_text(query_text, **kwargs)
    except Exception as exc:
        logger.warning("vector retrieval failed: %s", exc)
        return []
    # Normalise to (doi, score) — ChromaDB results carry "id" or "doi" key.
    out: list[tuple[str, float]] = []
    for r in results:
        doi_val: str = r.get("id") or r.get("doi") or ""
        score_val: float = float(r.get("score", 0.0))
        if doi_val:
            out.append((doi_val, score_val))
    return out


def _retrieve_graph(
    graph_db: Any | None,
    seed_doi: str | None,
    entity_ids: list[str] | None,
    k: int,
) -> list[tuple[str, float]]:
    """Query the knowledge graph and return (doi, score) pairs."""
    if graph_db is None:
        logger.debug("graph retrieval skipped: graph_db is None")
        return []
    try:
        if seed_doi:
            # G7: papers sharing the most entity MENTIONS with the source paper.
            return graph_db.find_similar_papers(seed_doi, k=k)
        if entity_ids:
            # G7: papers ranked by MENTIONS overlap with the entity set.
            return graph_db.find_papers_by_entities(entity_ids, k=k)
    except Exception as exc:
        logger.warning("graph retrieval failed: %s", exc)
    return []
