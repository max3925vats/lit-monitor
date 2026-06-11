"""B2: high-level MCP tool implementations.

Wraps GraphDB + H1 shared query layer (scripts/api/queries.py). All tools
validate inputs against the closed vocabularies (DOI regex, 10 predicates,
6 entity types) and return JSON-serialisable shapes.

Tools implemented here (B2):
- find_papers_by_entity      — alias-resolved entity lookup
- find_papers_by_relationship — predicate-typed paper retrieval
- get_paper_details          — full paper snapshot (delegates to H1)
- get_corpus_stats           — paper/entity counts (delegates to H1)
- list_entities_by_type      — entity listing by type (delegates to H1)
- get_schema                 — schema text (delegates to H1 / A1)
- find_papers_by_query       — graph free-text query
- find_papers_by_query_hybrid — RRF-fused graph + vector query

Also implemented (B3):
- run_cypher      — safety-reviewed Cypher execution

Also implemented (B6):
- semantic_search — ChromaDB vector path (paper + chunk granularity)

Also implemented (P9):
- get_recent_discovery_runs — most recent discovery runs (newest first)
- get_discovery_run_papers  — paper results for a specific discovery run

Total: 12 tools (cross-check scripts/mcp/graph_server.py::TOOL_NAMES).
"""
from __future__ import annotations

import logging
from typing import Any

from scripts.api.queries import (
    _coerce_jsonable,
    _validate_doi,
    get_paper_snapshot,
)
from scripts.api.queries import (
    get_corpus_stats as _q_corpus_stats,
)
from scripts.api.queries import (
    get_schema_text as _get_schema_text_impl,
)
from scripts.api.queries import (
    list_entities as _q_list_entities,
)
from scripts.graph.relationship_validator import VALID_PREDICATES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

# Canonical entity-type vocabulary (6 types, source: db.py line 588 + G2/G3).
_ENTITY_TYPES: frozenset[str] = frozenset(
    ("topic", "method", "material", "author", "journal", "keyword")
)

# _PREDICATES mirrors VALID_PREDICATES; imported directly so there is a single
# source of truth.  Exposed here for tests that patch against this module.
_PREDICATES: frozenset[str] = VALID_PREDICATES


# ---------------------------------------------------------------------------
# GraphDB accessor
# ---------------------------------------------------------------------------

def _get_graph_db() -> Any:
    """Return a GraphDB instance (lazy, process-global via safe_graph_db).

    Returns None if the [graph] extra is not installed or the DB is
    unavailable, which callers must handle gracefully.
    """
    from scripts.graph.import_citations import safe_graph_db  # noqa: PLC0415
    return safe_graph_db()


def _get_state_db() -> Any:
    """P9: lazy StateDB accessor — mirrors _get_graph_db's pattern.

    Returns a StateDB instance pointed at the configured state_db path.
    """
    from pathlib import Path  # noqa: PLC0415

    from scripts.core.config import get_config  # noqa: PLC0415
    from scripts.core.state_db import StateDB  # noqa: PLC0415

    cfg = get_config()
    return StateDB(Path(cfg.state_db.path).expanduser())


# ---------------------------------------------------------------------------
# Internal paper-shape helper
# ---------------------------------------------------------------------------

def _enrich_paper_rows(
    dois_scores: list[tuple[str, float]],
    graph_db: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    """Fetch title/year/journal for each (doi, score) pair and return shaped dicts.

    If the metadata query fails for any DOI we fall back gracefully to just
    the doi key so we never lose results entirely.
    """
    rows: list[dict[str, Any]] = []
    for doi, _score in dois_scores[:top_k]:
        entry: dict[str, Any] = {"doi": doi}
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("_enrich_paper_rows: metadata fetch failed for %s: %s", doi, exc)
        rows.append(entry)
    return _coerce_jsonable(rows)


# ---------------------------------------------------------------------------
# Tool 1: find_papers_by_entity
# ---------------------------------------------------------------------------

def find_papers_by_entity(entity: str, top_k: int = 20) -> list[dict[str, Any]]:
    """Find papers that mention the given entity.

    Resolves ``entity`` through the Phase 2 normalizer+alias chain on the
    live GraphDB vocabulary before querying.  Returns [] when the entity
    string cannot be resolved to any known canonical ID.

    Args:
        entity: Surface form or canonical ID of the entity to search for.
        top_k:  Maximum number of papers to return (default 20).

    Returns:
        list of {doi, title, year, journal} dicts, ranked by mention overlap.

    Raises:
        ValueError: When ``entity`` is empty or not a string.
    """
    if not isinstance(entity, str) or not entity.strip():
        raise ValueError(f"entity must be a non-empty string, got {entity!r}")

    db = _get_graph_db()
    if db is None:
        logger.warning("find_papers_by_entity: graph backend unavailable")
        return []

    # Phase 2 alias resolution: resolve query text → canonical_id before querying.
    canonical_id = db.resolve_query_entity(entity, type_=None)
    if canonical_id is None:
        logger.debug("find_papers_by_entity: no canonical match for %r", entity)
        return []

    doi_scores = db.find_papers_by_entities([canonical_id], k=top_k)
    return _enrich_paper_rows(doi_scores, db, top_k)


# ---------------------------------------------------------------------------
# Tool 2: find_papers_by_relationship
# ---------------------------------------------------------------------------

def find_papers_by_relationship(
    predicate: str,
    target: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Find papers that participate in a given typed relationship.

    Args:
        predicate: One of the 10 closed-vocabulary predicates (e.g. "PROPOSES",
                   "CITES", "MENTIONS").
        target:    Optional target entity canonical_id or paper DOI to filter by.
                   When omitted, all papers using this predicate are returned.
        top_k:     Maximum number of papers to return.

    Returns:
        list of {doi, title, year, journal} dicts.

    Raises:
        ValueError: When predicate is not in the closed vocabulary, or when
            top_k is not an int in [1, 100].
    """
    if predicate not in _PREDICATES:
        raise ValueError(
            f"unknown predicate: {predicate!r}; "
            f"expected one of {sorted(_PREDICATES)}"
        )
    if not isinstance(top_k, int) or not (1 <= top_k <= 100):
        raise ValueError("top_k must be int in [1, 100]")

    db = _get_graph_db()
    if db is None:
        logger.warning("find_papers_by_relationship: graph backend unavailable")
        return []

    try:
        if target:
            # Filter by target — match both Entity canonical_id and Paper doi
            # using UNION to cover Paper→Entity and Paper→Paper predicates.
            cypher = (
                f"MATCH (p:Paper)-[:{predicate}]->(e:Entity {{canonical_id: $tid}}) "
                "RETURN DISTINCT p.doi AS doi, p.title AS title, "
                "p.year AS year, p.journal AS journal "
                "UNION "
                f"MATCH (p:Paper)-[:{predicate}]->(t:Paper {{doi: $tid}}) "
                "RETURN DISTINCT p.doi AS doi, p.title AS title, "
                "p.year AS year, p.journal AS journal "
                "LIMIT $k"
            )
            res = db._conn.execute(cypher, {"tid": target, "k": int(top_k)})
        else:
            cypher = (
                f"MATCH (p:Paper)-[:{predicate}]->() "
                "RETURN DISTINCT p.doi AS doi, p.title AS title, "
                "p.year AS year, p.journal AS journal "
                "LIMIT $k"
            )
            res = db._conn.execute(cypher, {"k": int(top_k)})
    except Exception as exc:  # noqa: BLE001
        logger.warning("find_papers_by_relationship: query failed: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    while res.has_next():
        row = res.get_next()
        out.append(
            {
                "doi": str(row[0]) if row[0] else "",
                "title": str(row[1]) if row[1] else "",
                "year": int(row[2]) if row[2] is not None else None,
                "journal": str(row[3]) if row[3] else "",
            }
        )
    return _coerce_jsonable(out[:top_k])


# ---------------------------------------------------------------------------
# Tool 3: get_paper_details
# ---------------------------------------------------------------------------

def get_paper_details(doi: str) -> dict[str, Any]:
    """Return a full snapshot of one paper.

    Delegates to H1's ``get_paper_snapshot`` which returns metadata,
    entities grouped by type, and typed relationship edges in/out.

    Args:
        doi: DOI of the target paper (must match ``^10.[0-9]+/[^ ]+$``).

    Returns:
        Dict with keys: metadata, entities_by_type, relationships_in,
        relationships_out.

    Raises:
        ValueError: When doi fails DOI validation.
    """
    # _validate_doi raises ValueError on bad input
    _validate_doi(doi)

    db = _get_graph_db()
    if db is None:
        logger.warning("get_paper_details: graph backend unavailable")
        return {
            "metadata": {},
            "entities_by_type": {},
            "relationships_in": [],
            "relationships_out": [],
        }

    # Best-effort Zotero linkage: if config/state.db is unavailable (e.g. a
    # fresh checkout or CI with no config/paths.yaml), degrade to no state_db so
    # the snapshot still returns — zotero_key falls back to None.
    try:
        state_db = _get_state_db()
    except Exception:  # noqa: BLE001 — linkage is best-effort, never fatal
        state_db = None
    return get_paper_snapshot(doi, db, state_db)


# ---------------------------------------------------------------------------
# Tool 4: get_corpus_stats
# ---------------------------------------------------------------------------

def get_corpus_stats() -> dict[str, Any]:
    """Return aggregate counts for the graph corpus.

    Delegates to H1's ``get_corpus_stats``.

    Returns:
        Dict with paper_count, entity_count, and edge_counts_by_predicate.
    """
    db = _get_graph_db()
    if db is None:
        logger.warning("get_corpus_stats: graph backend unavailable")
        return {"paper_count": 0, "entity_count": 0, "edge_counts_by_predicate": {}}

    return _q_corpus_stats(db)


# ---------------------------------------------------------------------------
# Tool 5: list_entities_by_type
# ---------------------------------------------------------------------------

def list_entities_by_type(
    entity_type: str,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """List entities of a given type, ranked by mention count.

    The MCP-facing key is ``canonical`` (not ``canonical_id``) to keep the
    external surface clean — internally GraphDB uses ``canonical_id``.

    Args:
        entity_type: One of the 6 closed types:
                     topic, method, material, author, journal, keyword.
        top_k:       Maximum number of entities to return (default 50).

    Returns:
        list of {canonical, type, mention_count, surface_forms} dicts.

    Raises:
        ValueError: When entity_type is not in the closed vocabulary.
    """
    if entity_type not in _ENTITY_TYPES:
        raise ValueError(
            f"unknown entity_type: {entity_type!r}; "
            f"expected one of {sorted(_ENTITY_TYPES)}"
        )

    db = _get_graph_db()
    if db is None:
        logger.warning("list_entities_by_type: graph backend unavailable")
        return []

    raw = _q_list_entities(entity_type, top_k, db)

    # Remap canonical_id → canonical for the MCP-facing shape.
    # Also add surface_forms as a single-item list when surface is present.
    out: list[dict[str, Any]] = []
    for item in raw:
        mapped: dict[str, Any] = {
            "canonical": item.get("canonical_id", ""),
            "type": item.get("type", entity_type),
            "mention_count": item.get("mention_count", 0),
        }
        surface = item.get("surface", "")
        # surface_forms: list of known surface strings for this canonical entity.
        mapped["surface_forms"] = [surface] if surface else []
        out.append(mapped)
    return _coerce_jsonable(out)


# ---------------------------------------------------------------------------
# Tool 6: get_schema
# ---------------------------------------------------------------------------

def get_schema() -> str:
    """Return a Markdown-formatted schema description for prompt injection.

    Delegates to A1's ``describe_schema`` via H1's ``get_schema_text``.
    Falls back to a stub message before A1 ships.

    Returns:
        Markdown string describing node types, edge types, and properties.
    """
    db = _get_graph_db()
    if db is None:
        logger.warning("get_schema: graph backend unavailable")
        return "(graph backend unavailable)"

    return _get_schema_text_impl(db)


# ---------------------------------------------------------------------------
# Tool 7: find_papers_by_query (graph mode)
# ---------------------------------------------------------------------------

def find_papers_by_query(query: str, k: int = 20) -> list[dict[str, Any]]:
    """Find papers via free-text query using the graph entity alias chain.

    Resolves ``query`` to a canonical entity ID (via the Phase 2 normalizer),
    then returns papers ranked by MENTIONS overlap.  Returns [] if the query
    cannot be resolved.

    Calls ``scripts.api.queries.get_papers_by_query(query, mode="graph", k=k)``.
    If that function is not yet present (pre-H10), falls back to the inline
    entity-resolve + find_papers_by_entities path.

    Args:
        query: Free-text search string.
        k:     Maximum number of results (default 20).

    Returns:
        list of {doi, title, year, journal} dicts.

    Raises:
        ValueError: When query is empty/blank, or k is not an int in [1, 100].
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"query must be a non-empty string, got {query!r}")
    if not isinstance(k, int) or not (1 <= k <= 100):
        raise ValueError("k must be int in [1, 100]")

    db = _get_graph_db()
    if db is None:
        logger.warning("find_papers_by_query: graph backend unavailable")
        return []

    # Try H1's get_papers_by_query first (may not exist pre-H10).
    try:
        from scripts.api.queries import get_papers_by_query as _gpbq  # noqa: PLC0415
        return _gpbq(query, mode="graph", k=k, graph_db=db)
    except (ImportError, AttributeError):
        pass

    # Inline fallback: resolve entity + find_papers_by_entities.
    canonical_id = db.resolve_query_entity(query, type_=None)
    if canonical_id is None:
        return []
    doi_scores = db.find_papers_by_entities([canonical_id], k=k)
    return _enrich_paper_rows(doi_scores, db, k)


# ---------------------------------------------------------------------------
# Tool 8: find_papers_by_query_hybrid
# ---------------------------------------------------------------------------

def find_papers_by_query_hybrid(query: str, k: int = 20) -> list[dict[str, Any]]:
    """Find papers via free-text query using RRF-fused graph + vector retrieval.

    Graph leg: entity alias-resolve → find_papers_by_entities.
    Vector leg: ChromaDB semantic search (requires EmbeddingsDB).  When the
    vector backend is unavailable, degrades gracefully to graph-only results.

    Calls ``scripts.api.queries.get_papers_by_query(query, mode="hybrid", k=k)``.
    If that function is not yet present (pre-H10), falls back to the inline
    graph-only path (see comment below).

    Args:
        query: Free-text search string.
        k:     Maximum number of results (default 20).

    Returns:
        list of {doi, title, year, journal} dicts, RRF-fused when both legs
        are available, otherwise graph-only.

    Raises:
        ValueError: When query is empty/blank, or k is not an int in [1, 100].
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"query must be a non-empty string, got {query!r}")
    if not isinstance(k, int) or not (1 <= k <= 100):
        raise ValueError("k must be int in [1, 100]")

    db = _get_graph_db()
    if db is None:
        logger.warning("find_papers_by_query_hybrid: graph backend unavailable")
        return []

    # Try H1's get_papers_by_query first (may not exist pre-H10).
    try:
        from scripts.api.queries import get_papers_by_query as _gpbq  # noqa: PLC0415
        return _gpbq(query, mode="hybrid", k=k, graph_db=db)
    except (ImportError, AttributeError):
        pass

    # Inline fallback: graph leg only.
    # NOTE: hybrid truly needs EmbeddingsDB for the vector leg; that path is
    # wired by H10.  Until then, this returns graph results with an informative
    # log so downstream engineers know what is missing.
    logger.info(
        "find_papers_by_query_hybrid: get_papers_by_query not yet available "
        "(pre-H10); falling back to graph-only results"
    )
    canonical_id = db.resolve_query_entity(query, type_=None)
    if canonical_id is None:
        return []

    # Graph leg using RRF — single ranking, so RRF is identity; still use it
    # so the code path is exercised and H10 can slot the vector leg in.
    from scripts.retrieval.rrf import reciprocal_rank_fusion  # noqa: PLC0415

    doi_scores = db.find_papers_by_entities([canonical_id], k=k)
    doi_list = [d for d, _ in doi_scores]
    fused = reciprocal_rank_fusion([doi_list])[:k]
    fused_scores = [(doi, score) for doi, score in fused]

    return _enrich_paper_rows(fused_scores, db, k)


# ---------------------------------------------------------------------------
# Tool 9: run_cypher (B3) — read-only Cypher escape hatch
# ---------------------------------------------------------------------------


def run_cypher(query: str, limit: int = 100) -> list[dict[str, Any]]:
    """Execute a read-only Cypher query against the live GraphDB.

    This is a power-user escape hatch for the MCP agent to author its own
    Cypher queries.  Every query passes through the safety guard
    (:func:`scripts.mcp.cypher_guard.guard`) before reaching Kuzu:

    - Comments are stripped first (``//`` and ``/* */``).
    - Any mutation keyword (CREATE, MERGE, DELETE, SET, DROP, ALTER,
      REMOVE, LOAD CSV) raises :class:`CypherSafetyError`.
    - A ``LIMIT {limit}`` clause is appended when the query has none.

    Result rows are coerced to plain ``dict[str, Any]`` using H1's
    :func:`scripts.api.queries._coerce_jsonable` so callers always get
    JSON-serializable output.

    Args:
        query: Raw Cypher query string.  Must not contain mutation keywords.
        limit: Maximum rows to return (default 100).  Forwarded to the guard
               as ``hard_limit``; ignored when the query already has LIMIT.

    Returns:
        list of row dicts, one per result row.  Column names are the keys.

    Raises:
        CypherSafetyError: (a :class:`ValueError` subclass) when *query*
            contains a forbidden keyword.  MCP's ``ValueError`` handler
            converts this to ``TextContent("Error: ...")``.
        RuntimeError: When the graph backend is unavailable.
    """
    # Late import keeps the module importable even when cypher_guard isn't
    # installed yet (e.g. in certain unit-test stubs).
    from scripts.mcp.cypher_guard import guard  # noqa: PLC0415

    # guard() raises CypherSafetyError (ValueError subclass) if unsafe.
    safe_query = guard(query, hard_limit=limit)

    db = _get_graph_db()
    if db is None:
        raise RuntimeError("Graph backend unavailable")

    result = db._conn.execute(safe_query)

    col_names = result.get_column_names()
    rows: list[dict[str, Any]] = []
    while result.has_next():
        raw = result.get_next()
        # _coerce_jsonable flattens Kuzu node objects to their properties dict
        # and handles datetime/exotic types → strings.
        row = {col_names[i]: _coerce_jsonable(raw[i]) for i in range(len(col_names))}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# B6: semantic_search — ChromaDB vector retrieval
# ---------------------------------------------------------------------------

# Closed vocabulary for granularity parameter.
_ALLOWED_GRANULARITY: frozenset[str] = frozenset(("paper", "chunk"))

# Module-level EmbeddingsDB cache — lazy-loaded on first query.
# Reset to None by close_embeddings_db() on SIGTERM.
_EMBEDDINGS_DB: Any = None

# One-shot warning flag: logs "empty index" at most once per server lifetime.
_EMPTY_WARNED: bool = False


def _resolve_persist_dir() -> str:
    """Derive the ChromaDB persist dir from config.

    Pure indirection so tests can monkeypatch this without touching disk.
    Mirrors the canonical derivation in scripts/cli.py::_make_embeddings_db:
    ``persist_dir = str(Path(config.state_db.path).parent / "chroma")``
    """
    from pathlib import Path  # noqa: PLC0415

    from scripts.core.config import get_config  # noqa: PLC0415

    cfg = get_config()
    return str(Path(cfg.state_db.path).parent / "chroma")


def _get_embeddings_db() -> Any:
    """Lazy-load and cache an EmbeddingsDB instance.

    Returns the cached instance on subsequent calls.
    Returns None (with a one-shot WARNING) when the persist dir is missing —
    callers treat None as "no embeddings index" and return [].

    Lazy import keeps graph_server.py startup cheap (B1 requirement).
    """
    global _EMBEDDINGS_DB, _EMPTY_WARNED  # noqa: PLW0603

    if _EMBEDDINGS_DB is not None:
        return _EMBEDDINGS_DB

    import os  # noqa: PLC0415

    persist_dir = _resolve_persist_dir()
    if not os.path.isdir(persist_dir):
        if not _EMPTY_WARNED:
            logger.warning("semantic_search: empty embeddings index at %s", persist_dir)
            _EMPTY_WARNED = True
        return None

    # Lazy import — EmbeddingsDB triggers chromadb import which is expensive.
    from scripts.core.config import get_config  # noqa: PLC0415
    from scripts.output.embeddings import EmbeddingsDB  # noqa: PLC0415

    cfg = get_config()
    ollama_host = getattr(cfg.embeddings, "ollama_host", "http://localhost:11434")
    # Config field is `embeddings.model`; EmbeddingsDB constructor param is `embed_model`.
    embed_model = getattr(cfg.embeddings, "model", "mxbai-embed-large")

    _EMBEDDINGS_DB = EmbeddingsDB(
        persist_dir=persist_dir,
        ollama_host=ollama_host,
        embed_model=embed_model,
    )
    return _EMBEDDINGS_DB


def close_embeddings_db() -> None:
    """Drop the module-level EmbeddingsDB handle.

    Called by graph_server SIGTERM handler before exit.  ChromaDB's
    PersistentClient cleans up its own resources; we just release the handle
    so the object can be GC-ed.
    """
    global _EMBEDDINGS_DB  # noqa: PLW0603
    _EMBEDDINGS_DB = None


def semantic_search(
    query: str,
    top_k: int = 10,
    granularity: str = "paper",
) -> list[dict[str, Any]]:
    """Vector retrieval over the production ChromaDB persist dir.

    Args:
        query:       Free-text query string (embedded on the fly by EmbeddingsDB).
        top_k:       Number of results to return.  Must be an int in [1, 100].
        granularity: ``"paper"`` — paper-level hits with {doi, title, year, score}.
                     ``"chunk"`` — chunk-level passages with {doi, chunk_id,
                     snippet (≤500 chars), score}.

    Returns:
        List of result dicts ordered by similarity score descending.
        Returns [] (without raising) when the persist dir is missing or the
        collection is empty.

    Raises:
        ValueError: When ``granularity`` is not ``"paper"`` or ``"chunk"``.
        ValueError: When ``top_k`` is not an int in [1, 100].
    """
    if granularity not in _ALLOWED_GRANULARITY:
        raise ValueError("granularity must be 'paper' or 'chunk'")
    if not isinstance(top_k, int) or not (1 <= top_k <= 100):
        raise ValueError("top_k must be int in [1, 100]")

    db = _get_embeddings_db()
    if db is None:
        return []

    try:
        if granularity == "paper":
            raw = db.find_similar_to_text(query, top_k=top_k)
        else:
            raw = db.find_similar_chunks(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_search query failed: %s", exc)
        return []

    return _coerce_hits(raw, granularity)


def _coerce_hits(raw: list[dict[str, Any]], granularity: str) -> list[dict[str, Any]]:
    """Coerce EmbeddingsDB hits to JSON-serializable dicts.

    Both find_similar_to_text and find_similar_chunks return:
        [{id, score, document, metadata}, ...]
    sorted by similarity descending.

    For ``granularity="paper"``:
        - id   → doi (paper DOI, used as ChromaDB document ID)
        - metadata → {title, year, ...}  (title/year may be absent on old indices)
    For ``granularity="chunk"``:
        - id   → chunk_id  (format: "<doi>#<chunk_index>")
        - metadata → {doi, section_heading, chunk_index}
        - document → snippet (truncated to 500 chars)

    score is cast to plain float() to strip numpy scalar wrappers.
    """
    out: list[dict[str, Any]] = []
    for row in raw:
        md = row.get("metadata") or {}
        if granularity == "paper":
            out.append(
                {
                    "doi": row.get("id") or md.get("doi", ""),
                    "title": md.get("title", ""),
                    "year": md.get("year"),
                    "score": float(row.get("score", 0.0)),
                }
            )
        else:
            # chunk id format: "<doi>#<chunk_index>" — split on first '#'
            raw_id = row.get("id", "")
            doi = md.get("doi") or (raw_id.split("#")[0] if "#" in raw_id else raw_id)
            snippet = (row.get("document") or "")[:500]
            out.append(
                {
                    "doi": doi,
                    "chunk_id": raw_id,
                    "snippet": snippet,
                    "score": float(row.get("score", 0.0)),
                }
            )
    return out


# ---------------------------------------------------------------------------
# P9: Discovery run tools
# ---------------------------------------------------------------------------


def get_recent_discovery_runs(limit: int = 5) -> list[dict]:
    """P9: most recent discovery runs (newest first).

    Wraps queries.get_discovery_runs. Returns a list of run dicts, each
    containing: id, started_at, finished_at, status, total_found,
    total_ingested.

    Args:
        limit: Maximum number of runs to return (must be 1–100, default 5).

    Returns:
        List of run dicts ordered by id DESC.

    Raises:
        ValueError: When limit is out of bounds.
    """
    if not isinstance(limit, int) or limit < 1 or limit > 100:
        raise ValueError("limit must be int in [1, 100]")
    from scripts.api.queries import get_discovery_runs  # noqa: PLC0415

    result = get_discovery_runs(_get_state_db(), limit=limit, offset=0)
    return result.get("runs", [])


def get_discovery_run_papers(run_id: int, top_k: int = 20) -> list[dict]:
    """P9: paper results for a specific discovery run, sorted by score DESC.

    Wraps queries.get_discovery_run_papers. Returns a list of paper dicts,
    each containing: doi, title, score, rationale, ingested, ingested_at.

    Args:
        run_id: Primary key of the discovery_runs row (must be >= 1).
        top_k:  Maximum number of papers to return (must be 1–100, default 20).

    Returns:
        List of paper dicts ordered by score descending.

    Raises:
        ValueError: When run_id or top_k are out of bounds.
    """
    if not isinstance(run_id, int) or run_id < 1:
        raise ValueError("run_id must be positive int")
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        raise ValueError("top_k must be int in [1, 100]")
    from scripts.api.queries import get_discovery_run_papers as _qpapers  # noqa: PLC0415

    return _qpapers(_get_state_db(), run_id, top_k=top_k)
