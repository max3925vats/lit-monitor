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
from datetime import UTC, datetime
from typing import Any

from lit_monitor.core.doi import normalize_doi
from lit_monitor.graph.migrations import (
    TYPED_PAPER_TO_PAPER_PREDS as _PAPER_TO_PAPER_PREDS,
)
from lit_monitor.graph.migrations import (
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


def _zotero_deeplink(zotero_key: str | None) -> str | None:
    """Build a Zotero deep-link for a paper's item key, or None when unlinked.

    v1 emits the personal (user) library form, matching the hardcoded link in
    scripts/server/templates/corpus/detail.html and the default
    config.zotero.library_type == "user". Group libraries use a different URL
    shape (zotero://select/groups/{group_id}/items/{key}); supporting them needs
    the library type + id threaded in.
    # TODO(zotero-group-library): branch on config.zotero.library_type/library_id.
    """
    if not zotero_key:
        return None
    return f"zotero://select/library/items/{zotero_key}"


# ---------------------------------------------------------------------------
# Public API — six named functions
# ---------------------------------------------------------------------------


def get_paper_snapshot(
    doi: str, graph_db: Any, state_db: Any = None
) -> dict[str, Any]:
    """Return a full snapshot of one paper.

    Intended callers: HTTP /api/papers/{doi} (H2) and MCP get_paper_details
    (Phase 4b B2).

    Args:
        doi:      DOI of the target paper.
        graph_db: GraphDB instance.
        state_db: Optional StateDB. When provided, metadata is enriched with
            zotero_key/zotero_deeplink. When omitted, those keys are None.

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

    # Enrich with Zotero linkage (lives in state.db, not the graph). Guarded by
    # `if metadata:` so the empty-metadata -> 404 contract is preserved (adding
    # keys to an empty dict would make it truthy). Keys are ALWAYS set when
    # metadata exists, so the cross-surface parity key-set stays stable.
    if metadata:
        zkey = None
        if state_db is not None:
            try:
                # Audit Top Finding #4: normalize to the canonical form the
                # ingest path used as the state.db key before the lookup.
                zkey = state_db.get_zotero_key(normalize_doi(doi))
            except Exception as exc:  # noqa: BLE001 — linkage is best-effort
                logger.warning(
                    "get_paper_snapshot: zotero_key lookup failed for %s: %s",
                    doi,
                    exc,
                )
        metadata["zotero_key"] = zkey
        metadata["zotero_deeplink"] = _zotero_deeplink(zkey)

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

    from lit_monitor.retrieval.branch import retrieve_doi_candidates  # noqa: PLC0415

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


def get_entity_counts_by_type(graph_db: Any) -> dict[str, int]:
    """Count Entity nodes grouped by their ``type`` property.

    Corpus-wide breakdown of how many entities exist per entity type
    (topic / method / material / author / journal / keyword).  Mirrors
    ``get_corpus_stats`` defensive style: a single grouped Cypher query
    wrapped in try/except so a query failure yields ``{}`` rather than
    propagating to the caller.

    Args:
        graph_db: GraphDB instance exposing ``._conn.execute(cypher)``.

    Returns:
        Mapping of entity-type string → count (int).  Empty dict on any
        query failure.  All keys are str and all values are int, so the
        result survives ``json.dumps``.
    """
    counts: dict[str, int] = {}
    try:
        res = graph_db._conn.execute(
            "MATCH (e:Entity) RETURN e.type, count(e)"
        )
        while res.has_next():
            row = res.get_next()
            # row[0] may be None for entities with no type set; skip those.
            if row[0] is None:
                continue
            counts[str(row[0])] = int(row[1])
    except Exception as exc:
        logger.warning("get_entity_counts_by_type: query failed: %s", exc)
        return {}

    return _coerce_jsonable(counts)


def get_entity_counts_by_source(graph_db: Any) -> dict[str, int]:
    """Count MENTIONS edges grouped by their ``source`` property.

    Corpus-wide breakdown of how many mention edges each extraction source
    contributed (``schema`` / ``biobert`` / ``llm_cloud``).  Mirrors
    ``get_corpus_stats`` defensive style: a single grouped Cypher query
    wrapped in try/except so a query failure yields ``{}`` rather than
    propagating to the caller.

    Args:
        graph_db: GraphDB instance exposing ``._conn.execute(cypher)``.

    Returns:
        Mapping of source string → count (int).  Empty dict on any query
        failure.  All keys are str and all values are int, so the result
        survives ``json.dumps``.
    """
    counts: dict[str, int] = {}
    try:
        res = graph_db._conn.execute(
            "MATCH ()-[r:MENTIONS]->() RETURN r.source, count(r)"
        )
        while res.has_next():
            row = res.get_next()
            # row[0] may be None for edges with no source set; skip those.
            if row[0] is None:
                continue
            counts[str(row[0])] = int(row[1])
    except Exception as exc:
        logger.warning("get_entity_counts_by_source: query failed: %s", exc)
        return {}

    return _coerce_jsonable(counts)


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
    cols = ["doi", "title", "score", "rationale", "ingested", "ingested_at",
            "score_breakdown_json"]
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
        from lit_monitor.graph.schema_describer import (  # type: ignore[import-not-found]  # noqa: PLC0415
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


def get_graph_signals_for_candidate(
    candidate_doi: str,
    graph_db: Any,
    library_dois: list[str],
) -> dict[str, Any]:
    """Bundle D: extract graph signals for a single candidate relative to the library.

    Returns a dict with raw counts and sample lists for all graph ranking signals.
    Always returns a complete dict (never None, never raises).  All counts default
    to 0 and all lists to [] when the graph is unavailable or the candidate is not
    present.

    Signal definitions
    ------------------
    n_shared_entities:
        Number of Entity nodes that both the candidate and at least one library
        paper mention via MENTIONS edges.
    shared_entity_canonical_ids:
        Canonical IDs of those shared entities (for explainability).
    n_cites_in_library:
        Count of library papers directly cited by the candidate (CITES edges
        from candidate → library).
    n_cited_by_library:
        Count of library papers that cite the candidate (CITES edges from
        library → candidate).
    n_extends_in_library:
        Count of library papers that the candidate EXTENDS.
    n_compares_to_library:
        Count of library papers that the candidate COMPARES_TO.
    n_shared_authors:
        Number of distinct author names (case-insensitive, stripped) that appear
        in both the candidate and at least one library paper.
    shared_authors_sample:
        Up to 3 shared author names (original casing from candidate) for the
        Bundle B score-decomposition UI.

    Args:
        candidate_doi: DOI of the candidate paper.
        graph_db:      GraphDB instance.
        library_dois:  List of DOIs that form the current library.

    Returns:
        Dict with all signal keys (see above).
    """
    result: dict[str, Any] = {
        "n_shared_entities": 0,
        "shared_entity_canonical_ids": [],
        "n_cites_in_library": 0,
        "n_cited_by_library": 0,
        "n_extends_in_library": 0,
        "n_compares_to_library": 0,
        "n_shared_authors": 0,
        "shared_authors_sample": [],
    }

    if graph_db is None or not library_dois:
        return result

    conn = graph_db._conn

    try:
        # ------------------------------------------------------------------
        # 1. Shared MENTIONS entities
        #    Find entities the candidate mentions that at least one library
        #    paper also mentions.
        # ------------------------------------------------------------------
        res = conn.execute(
            "MATCH (cand:Paper {doi: $cand_doi})-[:MENTIONS]->(e:Entity) "
            "WHERE EXISTS { "
            "  MATCH (lib:Paper)-[:MENTIONS]->(e) "
            "  WHERE lib.doi IN $library_dois "
            "} "
            "RETURN DISTINCT e.canonical_id",
            {"cand_doi": candidate_doi, "library_dois": library_dois},
        )
        shared_entity_ids: list[str] = []
        while res.has_next():
            row = res.get_next()
            if row[0] is not None:
                shared_entity_ids.append(str(row[0]))
        result["n_shared_entities"] = len(shared_entity_ids)
        result["shared_entity_canonical_ids"] = shared_entity_ids

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_graph_signals_for_candidate: entity query failed for %s: %s",
            candidate_doi, exc,
        )

    try:
        # ------------------------------------------------------------------
        # 2. Citations: candidate → library  (n_cites_in_library)
        # ------------------------------------------------------------------
        res = conn.execute(
            "MATCH (cand:Paper {doi: $cand_doi})-[:CITES]->(lib:Paper) "
            "WHERE lib.doi IN $library_dois "
            "RETURN count(lib)",
            {"cand_doi": candidate_doi, "library_dois": library_dois},
        )
        if res.has_next():
            result["n_cites_in_library"] = int(res.get_next()[0] or 0)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_graph_signals_for_candidate: cites_in_library query failed for %s: %s",
            candidate_doi, exc,
        )

    try:
        # ------------------------------------------------------------------
        # 3. Citations: library → candidate  (n_cited_by_library)
        # ------------------------------------------------------------------
        res = conn.execute(
            "MATCH (lib:Paper)-[:CITES]->(cand:Paper {doi: $cand_doi}) "
            "WHERE lib.doi IN $library_dois "
            "RETURN count(lib)",
            {"cand_doi": candidate_doi, "library_dois": library_dois},
        )
        if res.has_next():
            result["n_cited_by_library"] = int(res.get_next()[0] or 0)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_graph_signals_for_candidate: cited_by_library query failed for %s: %s",
            candidate_doi, exc,
        )

    try:
        # ------------------------------------------------------------------
        # 4. EXTENDS: candidate EXTENDS → library  (n_extends_in_library)
        # ------------------------------------------------------------------
        res = conn.execute(
            "MATCH (cand:Paper {doi: $cand_doi})-[:EXTENDS]->(lib:Paper) "
            "WHERE lib.doi IN $library_dois "
            "RETURN count(lib)",
            {"cand_doi": candidate_doi, "library_dois": library_dois},
        )
        if res.has_next():
            result["n_extends_in_library"] = int(res.get_next()[0] or 0)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_graph_signals_for_candidate: extends query failed for %s: %s",
            candidate_doi, exc,
        )

    try:
        # ------------------------------------------------------------------
        # 5. COMPARES_TO: candidate COMPARES_TO → library  (n_compares_to_library)
        # ------------------------------------------------------------------
        res = conn.execute(
            "MATCH (cand:Paper {doi: $cand_doi})-[:COMPARES_TO]->(lib:Paper) "
            "WHERE lib.doi IN $library_dois "
            "RETURN count(lib)",
            {"cand_doi": candidate_doi, "library_dois": library_dois},
        )
        if res.has_next():
            result["n_compares_to_library"] = int(res.get_next()[0] or 0)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_graph_signals_for_candidate: compares_to query failed for %s: %s",
            candidate_doi, exc,
        )

    try:
        # ------------------------------------------------------------------
        # 6. Shared authors
        #    Fetch candidate's authors and all library authors from Paper.authors
        #    (v5 schema column), compute case-insensitive intersection.
        # ------------------------------------------------------------------
        # Candidate authors
        cand_res = conn.execute(
            "MATCH (p:Paper {doi: $d}) RETURN p.authors",
            {"d": candidate_doi},
        )
        cand_authors_raw = ""
        if cand_res.has_next():
            row = cand_res.get_next()
            cand_authors_raw = str(row[0]) if row[0] else ""

        cand_names: set[str] = _parse_author_names(cand_authors_raw)

        if cand_names:
            # Library authors (one query for all library papers)
            lib_res = conn.execute(
                "MATCH (p:Paper) "
                "WHERE p.doi IN $library_dois "
                "RETURN p.authors",
                {"library_dois": library_dois},
            )
            lib_names_lower: set[str] = set()
            lib_names_display: dict[str, str] = {}  # lower → first display form
            while lib_res.has_next():
                row = lib_res.get_next()
                for name in _parse_author_names(str(row[0]) if row[0] else ""):
                    lower = name.lower()
                    lib_names_lower.add(lower)
                    lib_names_display.setdefault(lower, name)

            # Intersection: candidate names that appear in library (case-insensitive)
            shared: list[str] = [
                n for n in cand_names if n.lower() in lib_names_lower
            ]
            result["n_shared_authors"] = len(shared)
            # Return up to 3 names using the candidate's display form
            result["shared_authors_sample"] = shared[:3]

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_graph_signals_for_candidate: authors query failed for %s: %s",
            candidate_doi, exc,
        )

    return result


def _relative_time(iso_ts: str | None) -> str:
    """Human 'Nh ago' / 'Nm ago' / 'Nd ago' from a SQLite 'YYYY-MM-DD HH:MM:SS'
    UTC timestamp. Returns 'never' for None/unparseable."""
    if not iso_ts:
        return "never"
    try:
        dt = datetime.strptime(iso_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return "never"
    delta = datetime.now(UTC) - dt
    secs = max(0, int(delta.total_seconds()))
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def get_dashboard_stats(state_db: Any, graph_db: Any) -> dict[str, Any]:
    """Assemble the four-stat banner payload (pure read — never writes).

    Returns dict with keys papers/graph_nodes/themes (each {value:int,
    delta:int|None}) and last_run ({relative:str, status:str|None}). Every
    field degrades to a safe default; never raises.

    Args:
        state_db: StateDB instance.
        graph_db: Optional GraphDB instance (None for graph-absent degradation).

    Returns:
        Dict with keys: papers, graph_nodes, themes, last_run. All values
        JSON-serializable.
    """
    try:
        papers_val = int(state_db.count_with_extraction())
    except Exception as exc:
        logger.warning("dashboard_stats: papers count failed: %s", exc)
        papers_val = 0
    try:
        themes_val = len(state_db.list_active_clusters())
    except Exception as exc:
        logger.warning("dashboard_stats: themes count failed: %s", exc)
        themes_val = 0
    graph_val = 0
    if graph_db is not None:
        try:
            gs = get_corpus_stats(graph_db)
            graph_val = int(gs.get("paper_count", 0)) + int(gs.get("entity_count", 0))
        except Exception as exc:
            logger.warning("dashboard_stats: graph count failed: %s", exc)
            graph_val = 0

    try:
        snaps = state_db.get_recent_snapshots(limit=2)
    except Exception as exc:
        logger.warning("dashboard_stats: snapshots fetch failed: %s", exc)
        snaps = []

    def _delta(field: str) -> int | None:
        if len(snaps) < 2:
            return None
        return int(snaps[0].get(field, 0)) - int(snaps[1].get(field, 0))

    last_relative, last_status = "never", None
    try:
        runs = state_db.get_recent_runs(limit=1)
        if runs:
            r = runs[0]
            last_relative = _relative_time(r.get("finished_at") or r.get("started_at"))
            last_status = r.get("status")
    except Exception as exc:
        logger.warning("dashboard_stats: last run failed: %s", exc)

    try:
        return _coerce_jsonable({
            "papers": {"value": papers_val, "delta": _delta("papers")},
            "graph_nodes": {"value": graph_val, "delta": _delta("graph_nodes")},
            "themes": {"value": themes_val, "delta": _delta("themes")},
            "last_run": {"relative": last_relative, "status": last_status},
        })
    except Exception as exc:
        logger.warning("dashboard_stats: final assembly failed: %s", exc)
        return {
            "papers": {"value": papers_val, "delta": None},
            "graph_nodes": {"value": graph_val, "delta": None},
            "themes": {"value": themes_val, "delta": None},
            "last_run": {"relative": last_relative, "status": last_status},
        }


def _parse_author_names(authors_str: str) -> set[str]:
    """Parse a semicolon-separated authors string into a deduplicated set of names.

    Handles both '; ' and ';' separators.  Strips whitespace.  Returns an
    empty set for empty / whitespace-only input.

    Used by get_graph_signals_for_candidate for author overlap computation.
    """
    if not authors_str or not authors_str.strip():
        return set()
    names: set[str] = set()
    for part in authors_str.split(";"):
        name = part.strip()
        if name:
            names.add(name)
    return names


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
    # P3.6: track ownership so we close ONLY the instance we acquired here. A
    # caller-injected graph_db must outlive this call and is NOT closed by us.
    owns_graph_db = False
    if graph_db is None and mode in ("graph", "hybrid"):
        try:
            from lit_monitor.graph.import_citations import safe_graph_db  # noqa: PLC0415
            graph_db = safe_graph_db()
            if graph_db is not None:
                owns_graph_db = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_papers_by_query: could not acquire graph_db: %s", exc)

    # --- Vector-only path -------------------------------------------------------
    # (mode == "vector" never lazily acquires graph_db, so owns_graph_db is
    # False here and there is nothing to close on this early return.)
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

    # P3.6: from here on the graph leg may use graph_db. Wrap in try/finally so
    # a lazily-acquired GraphDB is ALWAYS closed on every exit path (return or
    # exception). A caller-injected graph_db (owns_graph_db False) is left open.
    try:
        # --- Graph leg (used by both "graph" and "hybrid") ----------------------
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
            # --- Hybrid: RRF-fuse graph + vector --------------------------------
            from lit_monitor.retrieval.rrf import reciprocal_rank_fusion  # noqa: PLC0415

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

        # --- Enrich with metadata ----------------------------------------------
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
    finally:
        # P3.6: close ONLY the GraphDB we acquired lazily; never a caller's.
        if owns_graph_db and graph_db is not None:
            try:
                graph_db.close()
            except Exception as exc:  # noqa: BLE001 — close must never mask the result
                logger.warning(
                    "get_papers_by_query: graph_db.close() failed: %s", exc
                )
