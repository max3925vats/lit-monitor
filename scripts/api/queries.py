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

from scripts.graph.migrations import (
    TYPED_PAPER_TO_PAPER_PREDS as _PAPER_TO_PAPER_PREDS,
)
from scripts.graph.migrations import (
    TYPED_PREDICATES as _TYPED_PREDS,
)

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

    # Outgoing: all 9 predicates (Paper→Entity and Paper→Paper).
    for pred in _TYPED_PREDS:
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
        except Exception as exc:
            # A predicate table may be empty or absent on older schema versions.
            logger.warning(
                "get_paper_snapshot: predicate %s outgoing query failed: %s", pred, exc
            )
            continue

    # Incoming: only Paper→Paper predicates (5 structurally-impossible queries
    # avoided by scoping to _PAPER_TO_PAPER_PREDS instead of all 9).
    for pred in _PAPER_TO_PAPER_PREDS:
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
        except Exception as exc:
            logger.warning(
                "get_paper_snapshot: predicate %s incoming query failed: %s", pred, exc
            )
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
            "RETURN e.canonical_id, e.type, e.surface, count(DISTINCT p) AS mention_count "
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


# ---------------------------------------------------------------------------
# P5: discovery run read helpers (shared by HTTP routes + future MCP P9 + CLI
# P6 + export-md P7)
# ---------------------------------------------------------------------------


def get_discovery_runs(
    state_db: Any,
    *,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """P5: paginated list of discovery_runs rows.

    Returns a dict ``{runs: [...], total: N}`` where each run dict contains
    id, started_at, finished_at, status, total_found, total_ingested.

    Args:
        state_db: StateDB instance.
        limit:    Maximum number of rows to return (default 20).
        offset:   Row offset for pagination (default 0).

    Returns:
        Dict with keys ``runs`` (list of run dicts) and ``total`` (int).
    """
    cols = ["id", "started_at", "finished_at", "status", "total_found", "total_ingested"]
    with state_db._connect() as conn:
        rows = conn.execute(
            "SELECT " + ", ".join(cols) + " FROM discovery_runs "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        total: int = conn.execute("SELECT COUNT(*) FROM discovery_runs").fetchone()[0]
    return {"runs": [dict(zip(cols, r)) for r in rows], "total": total}


def get_discovery_run(state_db: Any, run_id: int) -> dict[str, Any] | None:
    """P5: single discovery_runs row (without papers). None if not found.

    Args:
        state_db: StateDB instance.
        run_id:   Primary key of the discovery_runs row.

    Returns:
        Dict with id, started_at, finished_at, status, total_found,
        total_ingested; or None when run_id does not exist.
    """
    cols = ["id", "started_at", "finished_at", "status", "total_found", "total_ingested"]
    with state_db._connect() as conn:
        row = conn.execute(
            "SELECT " + ", ".join(cols) + " FROM discovery_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    return dict(zip(cols, row)) if row else None


def get_discovery_run_papers(
    state_db: Any,
    run_id: int,
    *,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """P5: paper results for a discovery run, sorted by score DESC.

    Args:
        state_db: StateDB instance.
        run_id:   Primary key of the parent discovery_runs row.
        top_k:    Maximum number of papers to return (default 20).

    Returns:
        List of dicts with doi, title, score, rationale, ingested,
        ingested_at; ordered by score descending.  Empty list when
        run_id has no papers or does not exist.
    """
    cols = ["doi", "title", "score", "rationale", "ingested", "ingested_at"]
    with state_db._connect() as conn:
        rows = conn.execute(
            "SELECT " + ", ".join(cols) + " FROM discovery_paper_results "
            "WHERE run_id = ? ORDER BY score DESC LIMIT ?",
            (run_id, top_k),
        ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


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
    except (ImportError, ModuleNotFoundError, TypeError):
        # A1 not yet built — return a recognisable placeholder.
        # TypeError is caught because monkeypatching sys.modules[key] = None
        # causes the import machinery to raise TypeError, not ImportError.
        return "(schema describer not yet built — see Phase 4a A1)"

    try:
        return describe_schema(graph_db)
    except Exception as exc:
        logger.warning("get_schema_text: describe_schema failed: %s", exc)
        return f"(schema describer error: {exc})"


def get_papers_by_query(
    query: str,
    *,
    mode: str = "graph",
    k: int = 20,
    cfg: Any = None,  # noqa: ARG001 — reserved for future EmbeddingsDB lazy-init
    graph_db: Any = None,
    embeddings_db: Any = None,
) -> list[dict[str, Any]]:
    """H10: single source of truth for free-text paper retrieval.

    Called by POST /api/search AND MCP find_papers_by_query[_hybrid].
    Both surfaces share this one implementation to guarantee parity.

    Mode 'graph':  Resolve query as entity (Phase 2 alias chain), return
                   papers MENTIONS-linked to that entity, ranked by overlap.
    Mode 'hybrid': Graph leg + optional vector leg (ChromaDB), RRF-fused.
                   Degrades to graph-only when embeddings_db is None.
    Mode 'vector': Pure ChromaDB semantic search.  Returns [] when
                   embeddings_db is None (no exception).

    Args:
        query:         Free-text search string.  Empty / whitespace-only
                       returns [] without error.
        mode:          One of "graph", "hybrid", "vector".
        k:             Maximum number of results; must be in [1, 100].
        cfg:           Reserved for future use (lazy EmbeddingsDB init).
        graph_db:      GraphDB instance.  Lazily acquired via safe_graph_db()
                       when None and mode requires the graph leg.
        embeddings_db: Optional EmbeddingsDB for vector/hybrid modes.

    Returns:
        list of {doi, title, score} dicts, top-k by mode-appropriate score.
        May also include {year, journal} when graph_db enrichment succeeds.

    Raises:
        ValueError: When mode is unrecognised or k is outside [1, 100].
    """
    if mode not in ("graph", "hybrid", "vector"):
        raise ValueError(f"unknown mode: {mode!r}; expected graph/hybrid/vector")

    # k validation — Pydantic catches this at the HTTP layer, but callers via
    # MCP or direct Python import bypass Pydantic, so guard here too.
    if not isinstance(k, int) or k < 1 or k > 100:
        raise ValueError(f"k must be an integer in [1, 100], got {k!r}")

    # Empty / whitespace query → empty result, not an error.
    if not query or not query.strip():
        return []

    # Acquire graph_db if not injected (lazy — avoids I/O when not needed).
    if graph_db is None and mode in ("graph", "hybrid"):
        try:
            from scripts.graph.import_citations import safe_graph_db  # noqa: PLC0415
            graph_db = safe_graph_db()
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_papers_by_query: could not acquire graph_db: %s", exc)

    # --- Vector-only path -------------------------------------------------------
    if mode == "vector":
        if embeddings_db is None:
            logger.debug("get_papers_by_query: vector mode but no embeddings_db")
            return []
        try:
            results = embeddings_db.find_similar_to_text(query, top_k=k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_papers_by_query: vector query failed: %s", exc)
            return []
        out_vec: list[dict[str, Any]] = []
        for r in (results or []):
            doi = r.get("id") or r.get("doi", "")
            if not doi:
                continue
            md = r.get("metadata") or {}
            out_vec.append({
                "doi": doi,
                "title": md.get("title", ""),
                "score": float(r.get("score", 0.0)),
            })
        return _coerce_jsonable(out_vec[:k])

    # --- Graph leg (used by both "graph" and "hybrid") --------------------------
    graph_results: list[tuple[str, float]] = []
    if graph_db is not None:
        try:
            canonical_id = graph_db.resolve_query_entity(query, type_=None)
            if canonical_id is not None:
                graph_results = graph_db.find_papers_by_entities([canonical_id], k=k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_papers_by_query: graph leg failed: %s", exc)

    if mode == "graph":
        doi_scores = graph_results
    else:
        # --- Hybrid: RRF-fuse graph + vector ------------------------------------
        from scripts.retrieval.rrf import reciprocal_rank_fusion  # noqa: PLC0415

        vector_results: list[tuple[str, float]] = []
        if embeddings_db is not None:
            try:
                raw = embeddings_db.find_similar_to_text(query, top_k=k)
                vector_results = [
                    (r.get("id") or r.get("doi", ""), float(r.get("score", 0.0)))
                    for r in (raw or [])
                    if r.get("id") or r.get("doi")
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "get_papers_by_query: hybrid vector leg failed: %s", exc
                )

        g_ids = [d for d, _ in graph_results]
        v_ids = [d for d, _ in vector_results]
        # reciprocal_rank_fusion returns list[tuple[str, float]] sorted desc.
        fused = reciprocal_rank_fusion([g_ids, v_ids]) if (g_ids or v_ids) else []
        doi_scores = fused[:k]

    # --- Enrich with metadata --------------------------------------------------
    out: list[dict[str, Any]] = []
    if graph_db is not None:
        for doi, score in doi_scores[:k]:
            entry: dict[str, Any] = {"doi": doi, "score": float(score)}
            try:
                res = graph_db._conn.execute(
                    "MATCH (p:Paper {doi: $d}) RETURN p.title, p.year, p.journal",
                    {"d": doi},
                )
                if res.has_next():
                    row = res.get_next()
                    entry["title"] = str(row[0]) if row[0] else ""
                    entry["year"] = int(row[1]) if row[1] is not None else None
                    entry["journal"] = str(row[2]) if row[2] else ""
                else:
                    entry["title"] = ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "get_papers_by_query: metadata fetch failed for %s: %s", doi, exc
                )
                entry.setdefault("title", "")
            out.append(entry)
    else:
        # No graph_db — return bare doi + score
        out = [{"doi": d, "title": "", "score": float(s)} for d, s in doi_scores[:k]]

    return _coerce_jsonable(out)
