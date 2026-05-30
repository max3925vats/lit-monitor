"""Shared R28 invariant: embed + chunk + (G6) graph write + mark-phases in the right order.

Both discovery and brain-build pipelines must defer phase-progress marking
until embeddings succeed, so a degraded ChromaDB doesn't leave papers
permanently stuck in post-extraction state. This helper centralises that
contract. Each caller still owns its own pipeline-specific concerns
(rate limiting, batch tracking, source-type routing, etc.).

G6 extension — graph write inside the R28 invariant:
  - ``graph_db`` kwarg threads through from the pipeline runner; ``None``
    when the ``[graph]`` extra is not installed.
  - When ``graph_db`` is non-None AND the vector ``add_paper`` succeeded,
    we call ``graph_db.add_paper(...)`` with the pre-computed entity /
    relationship tuples from G3 / G4.
  - Graph add success -> ``state_db.set_graph_indexed(doi, 1)``.
  - Graph add failure -> WARN log, ``graph_indexed`` stays 0, but PHASE
    MARKING STILL FIRES.  This is the R28 enrichment semantic: the graph
    is not a correctness gate for the vector-only ingest path.

N4 extension — NER-augmented entities:
  - ``build_ner_mention_edges`` replaces ``build_graph_tuples`` at callsites
    that want BioBERT + cloud-LLM enrichment in addition to schema entities.
  - Returns ``list[MentionEdge]`` (N3 dataclass); ``graph_db.add_paper``
    accepts these via N4 duck-typing in ``db._upsert_entity_and_mentions``.
  - ``build_graph_tuples`` is preserved unchanged for backward compatibility
    and for callers that explicitly want schema-only entities.
"""
from __future__ import annotations

import logging
from typing import Any

_module_logger = logging.getLogger(__name__)

# R4: lazy module-level binding so monkeypatch can override the symbol.
# The real import is deferred to keep this module importable without the
# [graph] extra installed.  If the package is absent, the name resolves to
# None and maybe_extract_llm_relationships will catch it gracefully.
try:
    from scripts.graph.relationship_llm import extract_llm_relationships  # noqa: E402
except Exception:  # ImportError or missing optional dep
    extract_llm_relationships = None  # type: ignore[assignment]


def build_graph_tuples(
    extraction: dict[str, Any],
    paper_metadata: dict[str, Any],
    source_doi: str,
    state_db: Any,
) -> tuple[list[Any], list[Any]]:
    """G6 callsite helper: build (entities, relationships) for one paper.

    Wraps G3 ``extract_entities`` + G4 ``extract_relationships`` so both the
    brain-build and discovery pipelines can compute the graph payload with
    a single call.  Returns ``([], [])`` and logs WARN if anything raises —
    the graph payload is enrichment, never a correctness gate.

    Imports are lazy so this module stays importable on installs without
    the ``[graph]`` extra (the imports themselves do not require kuzu,
    only ``GraphDB`` instantiation does).
    """
    try:
        from scripts.graph.aliases import load_aliases
        from scripts.graph.entity_extractor import extract_entities
        from scripts.graph.normalizer import EntityNormalizer
        from scripts.graph.relationship_extractor import extract_relationships
    except Exception as exc:  # pragma: no cover — defensive
        _module_logger.warning("G6 build_graph_tuples: import failed: %s", exc)
        return [], []

    try:
        aliases = load_aliases()
        normalizer = EntityNormalizer(aliases=aliases)
        entities = extract_entities(extraction, paper_metadata, normalizer)
        # Build (surface_lower, type) -> canonical_id lookup for G4.
        entity_lookup: dict[tuple[str, str], str] = {}
        for ent in entities:
            entity_lookup[(ent.surface.lower().strip(), ent.type)] = ent.canonical_id
        relationships = extract_relationships(
            extraction_json=extraction,
            source_doi=source_doi,
            entity_lookup=entity_lookup,
            state_db=state_db,
        )
        return entities, relationships
    except Exception as exc:
        _module_logger.warning(
            "G6 build_graph_tuples failed for %s (non-fatal, graph_indexed stays 0): %s",
            source_doi, exc,
        )
        return [], []


def build_ner_mention_edges(
    extraction: dict[str, Any],
    paper_metadata: dict[str, Any],
    source_doi: str,
    state_db: Any,  # noqa: ARG001 — kept for call-site symmetry with build_graph_tuples
    fulltext: str | None = None,
    use_biobert: bool = True,
    use_cloud_llm: bool = True,
) -> list[Any]:
    """N4 callsite helper: build NER-augmented MentionEdge list for one paper.

    Orchestrates schema (N3) + BioBERT (N1) + cloud-Ollama (N2) → merge_mentions.
    Wraps ``ner_pipeline.build_mention_edges`` with the same defensive perimeter
    as ``build_graph_tuples``: any failure returns ``[]`` and logs WARN — the
    graph payload is enrichment, never a correctness gate for vector ingest.

    Returns a ``list[MentionEdge]`` accepted by ``graph_db.add_paper``
    via N4 duck-typing.  An empty list is a valid return value.

    The ``state_db`` parameter is accepted for API symmetry with
    ``build_graph_tuples`` so callers can swap the two without restructuring;
    it is not used internally.
    """
    try:
        from scripts.graph.aliases import load_aliases  # noqa: PLC0415
        from scripts.graph.ner_pipeline import build_mention_edges  # noqa: PLC0415
        from scripts.graph.normalizer import EntityNormalizer  # noqa: PLC0415
    except Exception as exc:
        _module_logger.warning(
            "N4 build_ner_mention_edges: import failed for %s: %s", source_doi, exc,
        )
        return []

    try:
        aliases = load_aliases()
        normalizer = EntityNormalizer(aliases=aliases)
        return build_mention_edges(
            paper_id=source_doi,
            text=fulltext or "",
            extraction_json=extraction,
            paper_metadata=paper_metadata,
            normalizer=normalizer,
            use_biobert=use_biobert,
            use_cloud_llm=use_cloud_llm,
        )
    except Exception as exc:
        _module_logger.warning(
            "N4 build_ner_mention_edges failed for %s (non-fatal, graph_indexed stays 0): %s",
            source_doi, exc,
        )
        return []


def index_embeddings_and_mark_phases(
    *,
    doi: str,
    zotero_key: str,              # used for mark_brain_build_phase(zotero_key, ...)
    fulltext: str,
    paper_metadata: dict[str, Any],
    chunks: list,
    state_db,
    embeddings_db,
    phases_to_mark: tuple[str, ...],
    logger: logging.Logger,
    graph_db: Any | None = None,                              # G6
    graph_entities: list[Any] | None = None,                  # G6 — EntityTuple list
    graph_relationships: list[Any] | None = None,             # G6 — RelationshipTuple list
    graph_paper_metadata: dict[str, Any] | None = None,       # G6 — Paper node fields
    prompt_version: str = "phase1.0",                         # G6
) -> tuple[bool, str | None]:
    """Index a paper, then mark phase progress IFF embed_add_paper succeeded.

    R28 contract:
      - embeddings_db.add_paper() failure -> phases NOT marked, returns (False, err).
        Graph write is NOT attempted on embed failure.
      - embeddings_db.add_chunks() failure -> logged as WARN; phases STILL marked.
      - graph_db.add_paper() failure (G6) -> logged as WARN; phases STILL marked;
        graph_indexed stays 0 (the R28 dual-write invariant).
      - All-OK -> phases marked, returns (True, None).
        If graph_db was provided AND graph add_paper succeeded,
        state_db.set_graph_indexed(doi, 1) is called.

    Parameters
    ----------
    graph_db:
        GraphDB instance from ``safe_graph_db()``, or ``None`` when the
        ``[graph]`` extra isn't installed.  Vector-only behaviour is
        identical to v0.3.x when ``None``.
    graph_entities:
        Pre-computed ``EntityTuple`` list from G3 ``extract_entities``.
        Defaults to empty list.
    graph_relationships:
        Pre-computed ``RelationshipTuple`` list from G4 ``extract_relationships``.
        Defaults to empty list.
    prompt_version:
        Provenance tag written onto every graph edge; G14 column default
        matches this value.
    """
    try:
        embeddings_db.add_paper(doi, fulltext, paper_metadata)
    except Exception as exc:
        logger.warning("Embed add_paper failed for %s: %s", doi, exc)
        return False, f"add_paper_failed: {exc}"

    try:
        embeddings_db.add_chunks(doi, chunks)
    except Exception as exc:
        # Non-fatal: chunks are an enrichment, not a correctness gate.
        # Matches existing behaviour in discovery.py and brain_build.py.
        logger.warning("Embed add_chunks failed for %s: %s", doi, exc)

    # G6: graph write — non-fatal enrichment.  Same precedence as chunks:
    # the vector add_paper must succeed first; graph_db is allowed to be
    # absent or to fail without blocking the vector ingest.
    if graph_db is not None:
        try:
            # If caller didn't supply a separate graph_paper_metadata, fall
            # back to the (scalar-only) embeddings metadata — this keeps the
            # API forgiving for tests and for any future callsite that
            # doesn't split the two.
            _graph_meta = (
                graph_paper_metadata if graph_paper_metadata is not None
                else paper_metadata
            )
            graph_db.add_paper(
                doi=doi,
                entities=graph_entities or [],
                relationships=graph_relationships or [],
                paper_metadata=_graph_meta,
                prompt_version=prompt_version,
            )
            state_db.set_graph_indexed(doi, 1)
            logger.debug("G6: graph_indexed=1 for %s", doi)
        except Exception as exc:
            # The R28 invariant: graph failure must NOT block phase marking
            # and must NOT flip graph_indexed.  The user keeps a usable
            # vector index; a subsequent re-ingest or backfill will retry
            # the graph write.
            logger.warning(
                "G6: graph add_paper failed for %s; graph_indexed stays 0: %s",
                doi, exc,
            )

    for phase in phases_to_mark:
        state_db.mark_brain_build_phase(zotero_key, phase)
    return True, None


def maybe_extract_llm_relationships(
    paper_doi: str,
    fulltext: str | None,
    extraction_json: dict[str, Any] | None,
    cfg: Any,
) -> list[Any]:  # list[RelationshipTuple]
    """R4: conditionally run R2 LLM relationship extraction during ingest.

    Returns a list of RelationshipTuple when the flag is enabled and
    extraction succeeds.  Returns an empty list when:
      - ``graph.relationships.llm_enabled`` is false or missing from cfg
      - ``fulltext`` is empty or None
      - ``extract_llm_relationships`` raises for any reason (R28 invariant:
        LLM failure must NEVER abort ingest or block phase marking)

    Never raises.  Matches the N4 ``build_ner_mention_edges`` enrichment
    pattern: single call per paper, guarded, returns [] on any failure.
    """
    # --- flag check ---
    try:
        enabled = bool(cfg.graph.relationships.llm_enabled)
    except (AttributeError, TypeError):
        # Config section missing or malformed → treat as disabled (default off).
        enabled = False
    if not enabled:
        return []

    # --- guard: nothing to send to the LLM ---
    if not fulltext:
        return []

    # --- extraction call (R28: any failure → WARN + empty, not a raise) ---
    try:
        if extract_llm_relationships is None:
            raise ImportError(
                "scripts.graph.relationship_llm not available "
                "(install the [graph] extra)"
            )
        return extract_llm_relationships(
            fulltext=fulltext,
            paper_doi=paper_doi,
            extraction_json=extraction_json or {},
        )
    except Exception as exc:
        _module_logger.warning(
            "R4: LLM relationship extraction failed for %s (non-fatal, "
            "graph_indexed unaffected): %s",
            paper_doi,
            exc,
        )
        return []
