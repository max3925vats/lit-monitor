"""H1: shared query layer for HTTP + MCP.

Single source of truth for engine queries.  Both FastAPI HTTP routes
(scripts/server/routes/*.py) and MCP tools (scripts/mcp/tools.py) call into
this module.  Prevents drift between the two surfaces.

All functions return plain JSON-serializable Python (no Kuzu objects, no
non-string keys, no datetime that doesn't serialize via json.dumps).
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# DOI pattern: prefix 10.N+/ suffix (non-whitespace).
# We accept 1+ digits after '10.' so synthetic test DOIs like '10.0/x' pass.
# The validator's job is to reject non-DOI strings ("not-a-doi", URLs, etc.),
# not to enforce IANA-registry digit-count restrictions.
_DOI_RE = re.compile(r"^10\.\d+/\S+$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_doi(doi: str) -> None:
    """Raise ValueError if doi isn't a recognizable DOI shape.

    Args:
        doi: Candidate DOI string.

    Raises:
        ValueError: When doi is empty, not a string, or doesn't match the
            standard 10.XXXX/ prefix pattern.
    """
    if not isinstance(doi, str) or not _DOI_RE.match(doi.strip()):
        raise ValueError(f"invalid DOI: {doi!r}")


def _coerce_jsonable(value: Any) -> Any:
    """Recursively coerce Kuzu/datetime/exotic types to JSON-serializable Python.

    H1 invariant: every return from this module must survive json.dumps without
    raising.  Custom types from Kuzu (e.g. internal node references) and
    datetime objects get string-cast.

    Args:
        value: Any Python object.

    Returns:
        An equivalent structure containing only JSON-native types (str, int,
        float, bool, None, list, dict with str keys).
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(v) for v in value]
    # datetime / date / time — isoformat() is the standard serialization form.
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    # Fallback: stringify anything Kuzu might return that we don't recognize.
    return str(value)


# ---------------------------------------------------------------------------
# Typed relationship predicates tracked in GraphDB.
# Kept here so queries.py is self-contained; duplicates the constant in G6.
# ---------------------------------------------------------------------------
_TYPED_PREDS = (
    "CITES",
    "COMPARES_TO",
    "DEPENDS_ON",
    "PROPOSES",
    "LIMITED_BY",
    "INTRODUCES",
    "RAISES_QUESTION",
    "EXTENDS",
    "CONTRADICTS",
)


# ---------------------------------------------------------------------------
# Public API — six named functions
# ---------------------------------------------------------------------------


def get_paper_snapshot(doi: str, graph_db: Any) -> dict[str, Any]:
    """Return a full snapshot of one paper.

    Intended callers: HTTP /api/papers/{doi} (H2) and MCP get_paper_details
    (Phase 4b B2).

    Args:
        doi:      DOI of the target paper.
        graph_db: GraphDB instance.

    Returns:
        Dict with keys: metadata, entities_by_type, relationships_in,
        relationships_out.  All values JSON-serializable.
        Returns empty-shape dict (NOT raises) for an unknown DOI so the HTTP
        layer can decide between 200 (empty) and 404 without catching errors.

    Raises:
        ValueError: When doi fails DOI validation.
    """
    _validate_doi(doi)

    # --- Metadata via direct Cypher on Paper node properties ----------------
    metadata: dict[str, Any] = {}
    try:
        res = graph_db._conn.execute(
            "MATCH (p:Paper {doi: $d}) RETURN p.title, p.year, p.journal",
            {"d": doi},
        )
        if res.has_next():
            row = res.get_next()
            metadata = {
                "doi": doi,
                "title": str(row[0]) if row[0] is not None else "",
                "year": int(row[1]) if row[1] is not None else None,
                "journal": str(row[2]) if row[2] is not None else "",
            }
    except Exception as exc:
        logger.warning(
            "get_paper_snapshot: metadata query failed for %s: %s", doi, exc
        )

    # --- Entities grouped by type via MENTIONS edges ------------------------
    entities_by_type: dict[str, list[dict[str, Any]]] = {}
    try:
        res = graph_db._conn.execute(
            "MATCH (p:Paper {doi: $d})-[m:MENTIONS]->(e:Entity) "
            "RETURN e.canonical_id, e.type, e.surface, m.source, m.field, m.confidence",
            {"d": doi},
        )
        while res.has_next():
            row = res.get_next()
            type_ = str(row[1])
            entities_by_type.setdefault(type_, []).append(
                {
                    "canonical_id": str(row[0]),
                    "type": type_,
                    "surface": str(row[2]) if row[2] else "",
                    "source": str(row[3]) if row[3] else "",
                    "field": str(row[4]) if row[4] else None,
                    "confidence": float(row[5]) if row[5] is not None else None,
                }
            )
    except Exception as exc:
        logger.warning(
            "get_paper_snapshot: entities query failed for %s: %s", doi, exc
        )

    # --- Relationships in / out over all typed predicate tables -------------
    rels_in: list[dict[str, Any]] = []
    rels_out: list[dict[str, Any]] = []
    for pred in _TYPED_PREDS:
        # Outgoing edges: this Paper → any target
        try:
            res = graph_db._conn.execute(
                f"MATCH (p:Paper {{doi: $d}})-[r:{pred}]->(t) "
                "RETURN t.doi AS doi, t.canonical_id AS cid, "
                "r.evidence AS ev, r.confidence AS conf, r.prompt_version AS pv",
                {"d": doi},
            )
            while res.has_next():
                row = res.get_next()
                # Distinguish Paper targets (have .doi) from Entity targets (have .cid).
                target_doi = row[0]
                target_cid = row[1]
                if target_doi is not None:
                    target_kind = "Paper"
                    target_id = str(target_doi)
                else:
                    target_kind = "Entity"
                    target_id = str(target_cid or "")
                rels_out.append(
                    {
                        "predicate": pred,
                        "target_kind": target_kind,
                        "target_id": target_id,
                        "evidence": str(row[2]) if row[2] else "",
                        "confidence": float(row[3]) if row[3] is not None else None,
                        "prompt_version": str(row[4]) if row[4] else "",
                    }
                )
        except Exception:
            # A predicate table may be empty or absent on older schema versions.
            continue

        # Incoming edges: any Paper → this Paper (Paper–Paper predicates only)
        try:
            res = graph_db._conn.execute(
                f"MATCH (s:Paper)-[r:{pred}]->(p:Paper {{doi: $d}}) "
                "RETURN s.doi AS source_doi, r.evidence AS ev, "
                "r.confidence AS conf, r.prompt_version AS pv",
                {"d": doi},
            )
            while res.has_next():
                row = res.get_next()
                rels_in.append(
                    {
                        "predicate": pred,
                        "source_doi": str(row[0]) if row[0] else "",
                        "evidence": str(row[1]) if row[1] else "",
                        "confidence": float(row[2]) if row[2] is not None else None,
                        "prompt_version": str(row[3]) if row[3] else "",
                    }
                )
        except Exception:
            continue

    return _coerce_jsonable(
        {
            "metadata": metadata,
            "entities_by_type": entities_by_type,
            "relationships_in": rels_in,
            "relationships_out": rels_out,
        }
    )


def get_related_papers(
    doi: str,
    mode: str,
    k: int,
    cfg: Any = None,
    *,
    embeddings_db: Any = None,
    graph_db: Any = None,
    query_text: str | None = None,
) -> list[dict[str, Any]]:
    """Dispatch to retrieve_doi_candidates for the given retrieval mode.

    Intended callers: HTTP /api/papers/{doi}/related (H5) and MCP
    find_papers_by_query (Phase 4b B6).

    Args:
        doi:          Seed paper DOI.
        mode:         One of "vector", "graph", "hybrid".
        k:            Maximum number of results.
        cfg:          Optional config object forwarded downstream.
        embeddings_db: Optional EmbeddingsDB instance (required for vector/hybrid).
        graph_db:     Optional GraphDB instance (required for graph/hybrid).
        query_text:   Optional text for vector query leg.

    Returns:
        list of {"doi": str, "score": float} dicts, ranked descending.

    Raises:
        ValueError: When doi fails validation or mode is unrecognised.
    """
    _validate_doi(doi)
    if mode not in ("vector", "graph", "hybrid"):
        raise ValueError(f"mode must be vector/graph/hybrid, got {mode!r}")

    from scripts.retrieval.branch import retrieve_doi_candidates  # noqa: PLC0415

    results = retrieve_doi_candidates(
        mode,  # positional rag_mode
        seed_doi=doi,
        query_text=query_text,
        embeddings_db=embeddings_db,
        graph_db=graph_db,
        k=k,
    )
    # retrieve_doi_candidates returns list[tuple[str, float]]; coerce to list[dict].
    return _coerce_jsonable(
        [{"doi": d, "score": float(s)} for d, s in (results or [])]
    )


def get_entity_neighborhood(canonical_id: str, graph_db: Any) -> dict[str, Any]:
    """Papers mentioning this entity and its co-occurring entities.

    Intended callers: HTTP /api/entities/{id} (H7) and MCP entity_neighborhood
    (Phase 4b).

    Args:
        canonical_id: Normalized canonical entity identifier.
        graph_db:     GraphDB instance.

    Returns:
        Dict with keys: canonical_id, papers (list of {doi, title, year}),
        co_entities (list of {canonical_id, type, co_count} top-25 by co-count).
        All values JSON-serializable.

    Raises:
        ValueError: When canonical_id is empty or not a string.
    """
    if not isinstance(canonical_id, str) or not canonical_id.strip():
        raise ValueError(f"invalid canonical_id: {canonical_id!r}")

    papers: list[dict[str, Any]] = []
    try:
        res = graph_db._conn.execute(
            "MATCH (p:Paper)-[m:MENTIONS]->(e:Entity {canonical_id: $cid}) "
            "RETURN DISTINCT p.doi, p.title, p.year",
            {"cid": canonical_id},
        )
        while res.has_next():
            row = res.get_next()
            papers.append(
                {
                    "doi": str(row[0]),
                    "title": str(row[1]) if row[1] else "",
                    "year": int(row[2]) if row[2] is not None else None,
                }
            )
    except Exception as exc:
        logger.warning(
            "get_entity_neighborhood: papers query failed for %s: %s",
            canonical_id,
            exc,
        )

    co_entities: list[dict[str, Any]] = []
    try:
        res = graph_db._conn.execute(
            "MATCH (p:Paper)-[:MENTIONS]->(e:Entity {canonical_id: $cid}) "
            "MATCH (p)-[:MENTIONS]->(other:Entity) "
            "WHERE other.canonical_id <> $cid "
            "RETURN other.canonical_id, other.type, count(p) AS co_count "
            "ORDER BY co_count DESC LIMIT 25",
            {"cid": canonical_id},
        )
        while res.has_next():
            row = res.get_next()
            co_entities.append(
                {
                    "canonical_id": str(row[0]),
                    "type": str(row[1]),
                    "co_count": int(row[2]),
                }
            )
    except Exception as exc:
        logger.warning(
            "get_entity_neighborhood: co_entities query failed for %s: %s",
            canonical_id,
            exc,
        )

    return _coerce_jsonable(
        {
            "canonical_id": canonical_id,
            "papers": papers,
            "co_entities": co_entities,
        }
    )


def list_entities(
    entity_type: str,
    top_k: int,
    graph_db: Any,
) -> list[dict[str, Any]]:
    """List entities of a given type ranked by mention count.

    Intended callers: HTTP /api/entities?type=... (H8) and MCP list_entities.

    Args:
        entity_type: Entity type string (e.g. "method", "material").
        top_k:       Maximum number of results (must be >= 1).
        graph_db:    GraphDB instance.

    Returns:
        list of {canonical_id, type, surface, mention_count} dicts,
        ordered by mention_count descending.  All values JSON-serializable.

    Raises:
        ValueError: When entity_type is empty or top_k < 1.
    """
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise ValueError(f"invalid entity_type: {entity_type!r}")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be int >= 1, got {top_k!r}")

    out: list[dict[str, Any]] = []
    try:
        res = graph_db._conn.execute(
            "MATCH (e:Entity {type: $t})<-[m:MENTIONS]-(p:Paper) "
            "RETURN e.canonical_id, e.type, e.surface, count(p) AS mention_count "
            "ORDER BY mention_count DESC LIMIT $k",
            {"t": entity_type, "k": top_k},
        )
        while res.has_next():
            row = res.get_next()
            out.append(
                {
                    "canonical_id": str(row[0]),
                    "type": str(row[1]),
                    "surface": str(row[2]) if row[2] else "",
                    "mention_count": int(row[3]),
                }
            )
    except Exception as exc:
        logger.warning(
            "list_entities: query failed for type=%s: %s", entity_type, exc
        )

    return _coerce_jsonable(out)


def get_corpus_stats(graph_db: Any) -> dict[str, Any]:
    """Aggregate counts across the graph corpus.

    Intended callers: HTTP /api/stats (H3) and MCP get_corpus_stats.

    Args:
        graph_db: GraphDB instance.

    Returns:
        Dict with paper_count, entity_count, and edge_counts_by_predicate.
        All values JSON-serializable.
    """
    stats: dict[str, Any] = {}

    # Node counts for Paper and Entity tables.
    for label in ("Paper", "Entity"):
        try:
            res = graph_db._conn.execute(
                f"MATCH (n:{label}) RETURN count(n) AS c"
            )
            stats[f"{label.lower()}_count"] = (
                int(res.get_next()[0]) if res.has_next() else 0
            )
        except Exception as exc:
            logger.warning(
                "get_corpus_stats: count failed for %s: %s", label, exc
            )
            stats[f"{label.lower()}_count"] = 0

    # Per-predicate edge counts — missing / empty tables default to 0.
    pred_counts: dict[str, int] = {}
    for pred in ("MENTIONS",) + _TYPED_PREDS:
        try:
            res = graph_db._conn.execute(
                f"MATCH ()-[r:{pred}]->() RETURN count(r)"
            )
            pred_counts[pred] = int(res.get_next()[0]) if res.has_next() else 0
        except Exception:
            pred_counts[pred] = 0
    stats["edge_counts_by_predicate"] = pred_counts

    return _coerce_jsonable(stats)


def get_schema_text(graph_db: Any) -> str:
    """Markdown-formatted schema description for prompt inclusion.

    Used by Phase 4a ``ask`` endpoint to inject graph schema context into LLM
    prompts.  Defers to Phase 4a A1's schema_describer when it exists.  Falls
    back to a stub message before A1 ships — the fallback path becomes dead
    code once A1 lands.

    Args:
        graph_db: GraphDB instance passed through to describe_schema.

    Returns:
        Markdown string describing node types, relationship types, and
        properties.  Falls back to a placeholder stub if A1 hasn't shipped.
    """
    try:
        # Lazy import so H1 doesn't crash before Phase 4a A1 ships.
        from scripts.graph.schema_describer import (  # type: ignore[import-not-found]  # noqa: PLC0415
            describe_schema,
        )
    except (ImportError, ModuleNotFoundError):
        # A1 not yet built — return a recognisable placeholder.
        return "(schema describer not yet built — see Phase 4a A1)"

    try:
        return describe_schema(graph_db)
    except Exception as exc:
        logger.warning("get_schema_text: describe_schema failed: %s", exc)
        return f"(schema describer error: {exc})"
