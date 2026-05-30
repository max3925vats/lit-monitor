"""G10: backfill + rebuild operator commands for the KuzuDB graph.

Functions
---------
backfill_papers     -- incremental; only papers where state.db.graph_indexed=0.
rebuild_all         -- drop + recreate the full graph, reset all graph_indexed=0.
rebuild_aliases_only -- re-normalize existing Entity nodes (no LLM, no re-extract).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from scripts.graph.aliases import load_aliases
from scripts.graph.entity_extractor import extract_entities
from scripts.graph.migrations import apply_schema
from scripts.graph.ner_pipeline import build_mention_edges
from scripts.graph.normalizer import EntityNormalizer
from scripts.graph.relationship_extractor import extract_relationships

logger = logging.getLogger(__name__)


def _build_entity_lookup(entities: list[Any]) -> dict[tuple[str, str], str]:
    """G6-style lookup table for the relationship extractor."""
    return {(e.surface.lower(), e.type): e.canonical_id for e in entities}


def backfill_papers(
    state_db: Any,
    graph_db: Any,
    *,
    filter_doi: str | None = None,
    since: datetime | None = None,
    progress_callback: Any = None,
) -> int:
    """Walk state.db papers and write each to the knowledge graph.

    Only processes papers where ``graph_indexed = 0``.  After a successful
    write, sets ``graph_indexed = 1`` via :meth:`StateDB.set_graph_indexed`.

    Args:
        state_db:          Open :class:`StateDB` instance.
        graph_db:          Open :class:`GraphDB` instance.
        filter_doi:        When set, process only this single DOI.
        since:             When set, restrict to papers whose ``last_updated``
                           column is >= this datetime (ISO comparison in SQLite).
        progress_callback: Optional callable invoked after each paper as
                           ``callback(doi, done_count, total_count)``.

    Returns:
        Count of papers successfully processed.
    """
    sql = "SELECT * FROM papers WHERE graph_indexed = 0"
    params: list[Any] = []

    if filter_doi:
        sql += " AND doi = ?"
        params.append(filter_doi)
    if since is not None:
        # Papers with NULL last_updated have never been processed — treat
        # them as "always eligible" even with a --since filter.
        sql += " AND (last_updated IS NULL OR last_updated >= ?)"
        params.append(since.isoformat())

    with state_db._connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        logger.info("backfill_papers: nothing to do (0 rows match)")
        return 0

    total = len(rows)
    normalizer = EntityNormalizer(aliases=load_aliases())
    processed = 0

    for row in rows:
        paper = dict(row)
        doi = paper.get("doi")
        if not doi:
            continue

        try:
            # Parse extraction_json — may arrive as a JSON string or already a dict.
            extraction: dict[str, Any] = {}
            raw = paper.get("extraction_json")
            if isinstance(raw, str):
                try:
                    extraction = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    extraction = {}
            elif isinstance(raw, dict):
                extraction = raw

            paper_metadata: dict[str, Any] = {
                "title": paper.get("title") or "",
                "year": paper.get("year") or 0,
                "journal": paper.get("journal") or "",
                "authors": paper.get("authors") or "",
                "keywords_json": paper.get("keywords_json") or "",
            }

            entities = extract_entities(extraction, paper_metadata, normalizer)
            entity_lookup = _build_entity_lookup(entities)
            relationships = extract_relationships(
                extraction,
                source_doi=doi,
                entity_lookup=entity_lookup,
                state_db=state_db,
            )
            graph_db.add_paper(
                doi=doi,
                entities=entities,
                relationships=relationships,
                paper_metadata=paper_metadata,
                prompt_version="phase1.0",
            )
            state_db.set_graph_indexed(doi, 1)
            processed += 1
            logger.info(
                "backfill: indexed %s (%d entities, %d relationships)",
                doi, len(entities), len(relationships),
            )
            if progress_callback is not None:
                progress_callback(doi, processed, total)

        except Exception as exc:  # noqa: BLE001 — log + continue; don't abort the run
            logger.warning("backfill: failed to index %s: %s", doi, exc)

    return processed


def backfill_ner(
    state_db: Any,
    graph_db: Any,
    *,
    with_llm: bool = False,
    limit: int | None = None,
    since: datetime | None = None,
    progress_callback: Any = None,
) -> dict[str, int]:
    """N5: run the NER-augmented entity pipeline (N1+N2+N3+N4) over existing papers.

    Only processes papers where ``ner_processed_at IS NULL`` (idempotent).
    After a successful per-paper run, stamps ``ner_processed_at`` via
    :meth:`StateDB.set_ner_processed_at`.

    Args:
        state_db:          Open :class:`StateDB` instance.
        graph_db:          Open :class:`GraphDB` instance.
        with_llm:          When True, enables cloud-Ollama long-tail NER (N2).
        limit:             Cap on the number of candidate papers.
        since:             Restrict candidates to papers with
                           ``last_updated >= since``.
        progress_callback: Optional callable ``(doi, done_count, total_count)``.

    Returns:
        Summary dict with keys ``papers_processed``, ``edges_added``,
        ``failures``.
    """
    candidates = state_db.get_papers_for_ner_backfill(
        since=since, limit=limit, only_unprocessed=True,
    )
    if not candidates:
        logger.info("backfill_ner: no candidates — nothing to do")
        return {"papers_processed": 0, "edges_added": 0, "failures": 0}

    normalizer = EntityNormalizer(aliases=load_aliases())
    summary: dict[str, int] = {"papers_processed": 0, "edges_added": 0, "failures": 0}
    total = len(candidates)

    for idx, paper in enumerate(candidates):
        doi = paper.get("doi")
        if not doi:
            continue

        try:
            # Parse extraction_json — may be a JSON string or already a dict.
            extraction: dict[str, Any] = {}
            raw = paper.get("extraction_json")
            if isinstance(raw, str):
                try:
                    extraction = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    extraction = {}
            elif isinstance(raw, dict):
                extraction = raw

            paper_metadata: dict[str, Any] = {
                "title": paper.get("title") or "",
                "year": paper.get("year") or 0,
                "journal": paper.get("journal") or "",
                "authors": paper.get("authors") or "",
                "keywords_json": paper.get("keywords_json") or "",
            }

            # Fulltext may not exist for all papers; pass empty string rather than None.
            text: str = paper.get("fulltext") or ""

            edges = build_mention_edges(
                paper_id=doi,
                text=text,
                extraction_json=extraction,
                paper_metadata=paper_metadata,
                normalizer=normalizer,
                use_biobert=True,
                use_cloud_llm=with_llm,
            )

            # MERGE into graph via G6 add_paper semantics (idempotent).
            # N5 does NOT touch typed-relationship edges — entity/MENTIONS only.
            graph_db.add_paper(
                doi=doi,
                entities=edges,
                relationships=[],
                paper_metadata=paper_metadata,
                prompt_version="phase2.0",
            )

            # Stamp so re-runs skip this paper.
            state_db.set_ner_processed_at(doi, datetime.now().isoformat())

            summary["papers_processed"] += 1
            summary["edges_added"] += len(edges)

            if (idx + 1) % 25 == 0:
                logger.info(
                    "backfill_ner: processed %d/%d papers", idx + 1, total,
                )
            if progress_callback is not None:
                progress_callback(doi, idx + 1, total)

        except Exception as exc:  # noqa: BLE001 — log + skip; don't abort the run
            logger.warning("backfill_ner: failed for %s: %s", doi, exc)
            summary["failures"] += 1

    return summary


def rebuild_all(state_db: Any, graph_db: Any) -> int:
    """Drop all graph data, reset graph_indexed flags, and re-run full backfill.

    Drops every Kuzu node and edge table in dependency order, re-applies the
    schema DDL, then resets every ``papers.graph_indexed`` row to 0 so that a
    fresh :func:`backfill_papers` re-processes the entire corpus.

    Args:
        state_db: Open :class:`StateDB` instance.
        graph_db: Open :class:`GraphDB` instance.

    Returns:
        Count of papers re-indexed after the rebuild.
    """
    conn = graph_db._conn

    # Drop in reverse dependency order: edges before nodes.
    edge_tables = [
        "MENTIONS", "CITES", "COMPARES_TO", "DEPENDS_ON",
        "PROPOSES", "LIMITED_BY", "INTRODUCES", "RAISES_QUESTION",
    ]
    node_tables = ["Paper", "Entity"]

    for name in edge_tables + node_tables:
        try:
            conn.execute(f"DROP TABLE {name}")
            logger.debug("rebuild_all: dropped table %s", name)
        except Exception as exc:
            # Table may not exist on a fresh install — that's fine.
            logger.debug("rebuild_all: DROP TABLE %s skipped: %s", name, exc)

    # Re-create schema.
    apply_schema(conn)

    # Reset all graph_indexed flags so backfill starts clean.
    with state_db._connect() as state_conn:
        state_conn.execute("UPDATE papers SET graph_indexed = 0")
        state_conn.commit()

    # Invalidate any cached query results.
    if hasattr(graph_db, "invalidate_query_cache"):
        graph_db.invalidate_query_cache()

    return backfill_papers(state_db, graph_db, filter_doi=None, since=None)


def rebuild_aliases_only(graph_db: Any) -> int:
    """Re-normalize all existing Entity nodes against the current alias YAML.

    For each Entity node, re-runs the normalizer with the freshly loaded alias
    dictionary.  When the new canonical ID differs from the stored one, updates
    the node in-place.

    Note: collapsing two distinct nodes that now share the same canonical ID
    is deferred to v0.8+ (requires MERGE semantics not yet supported in Kuzu
    0.11.x).  This function only renames individual nodes.

    Args:
        graph_db: Open :class:`GraphDB` instance.

    Returns:
        Count of Entity nodes whose ``canonical_id`` was updated.
    """
    conn = graph_db._conn
    normalizer = EntityNormalizer(aliases=load_aliases())

    # Read all existing entity nodes.
    res = conn.execute("MATCH (e:Entity) RETURN e.canonical_id, e.type, e.surface")
    entities_to_check: list[tuple[str, str, str]] = []
    while res.has_next():
        row = res.get_next()
        entities_to_check.append((str(row[0]), str(row[1]), str(row[2])))

    renormalized = 0
    for old_canonical, type_, surface in entities_to_check:
        new_canonical, _via = normalizer.normalize(surface, type_=type_)
        if new_canonical == old_canonical:
            continue
        try:
            conn.execute(
                "MATCH (e:Entity {canonical_id: $old}) SET e.canonical_id = $new",
                {"old": old_canonical, "new": new_canonical},
            )
            renormalized += 1
            logger.info(
                "rebuild_aliases: %s -> %s (surface=%r)", old_canonical, new_canonical, surface
            )
        except Exception as exc:
            logger.warning(
                "rebuild_aliases: failed to rename %s: %s", old_canonical, exc
            )

    if hasattr(graph_db, "invalidate_query_cache"):
        graph_db.invalidate_query_cache()

    return renormalized
