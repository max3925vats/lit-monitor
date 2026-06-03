"""
Obsidian re-extract — re-run LLM extraction for one or both phases or specific
field(s), then update the state DB and rerender the note.

Use cases:
  - Re-run simple phase:  re_extract(doi, phases=["simple"], ...)
  - Re-run complex phase: re_extract(doi, phases=["complex"], ...)
  - Refresh a specific field: re_extract(doi, fields=["actionable_insights"], ...)
  - Bulk re-extraction: re_extract_all_failed_phase("simple", ...)
  - Graph-context injection (N7): re_extract(doi, ..., rag_mode="graph")

Single-item failures log and continue in bulk mode.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.core.strict_mode import strict_fallback
from scripts.llm.extractor import (
    ALL_PAPER_FIELDS,
    compute_confidence_score,
    extract_fields,
    extract_paper,
)
from scripts.llm.prompt_safety import sanitize_for_prompt
from scripts.obsidian_tools.rerender import rerender_note

logger = logging.getLogger(__name__)
def re_extract(
    doi: str,
    config,
    state_db,
    llm,
    phases: list[str] | None = None,
    fields: list[str] | None = None,
    rerender: bool = True,
    zotero_client=None,
    rag_mode: str = "vector",
    graph_k: int = 8,
) -> dict[str, Any]:
    """Re-run LLM extraction for a single paper or review.

    Parameters
    ----------
    doi:
        The DOI of the item to re-extract.
    phases:
        Which phases to re-run — "simple", "complex", or both.
        When None and fields is also None, defaults to both phases.
    fields:
        If set, re-extract only these specific fields via a focused prompt.
        Cannot be combined with ``phases``.
    rerender:
        If True, rerender the Obsidian note after extraction.
    rag_mode:
        Retrieval mode for context injection. "vector" (default) uses the
        existing extraction path unchanged. "graph" (N7) fetches related
        entities and adjacent paper titles from KuzuDB and injects them as
        {graph_context} into the re_extract_with_graph prompt. Falls back to
        "vector" with a WARNING if [graph] extra is missing or Kuzu is
        unreachable.
    graph_k:
        Number of related papers/entities to fetch for graph-mode context.
        Default 8.

    Returns
    -------
    dict: updated extraction JSON
    """
    record = state_db.get_paper(doi)
    if record is None:
        raise ValueError(f"No record found for doi: {doi!r}")
    source_type = record.get("source_type", "paper")
    existing = json.loads(record.get("extraction_json") or "{}")

    if fields and source_type == "paper":
        unknown = [f for f in fields if f not in ALL_PAPER_FIELDS]
        if unknown:
            raise ValueError(
                f"Unknown field(s) for paper re-extraction: {unknown!r}. "
                f"Valid fields: {ALL_PAPER_FIELDS}."
            )

    fulltext = _load_fulltext(record, zotero_client=zotero_client)
    ocr_heavy = bool(record.get("ocr_heavy", 0))

    if source_type not in ("paper", "review"):
        raise ValueError(
            f"Unknown source_type: {source_type!r}. "
            "Re-extraction supports 'paper' and 'review' only."
        )

    # N7: graph-mode — try to open GraphDB and inject corpus context.
    # Falls back to vector silently on failure so the command never hard-fails.
    effective_rag = rag_mode
    if rag_mode == "graph":
        graph_db = _open_graph_db_safe()
        if graph_db is None:
            logger.warning(
                "N7: --rag-mode graph requested but graph DB unavailable; "
                "falling back to vector mode for %s",
                doi,
            )
            effective_rag = "vector"

    if effective_rag == "graph":
        # Build graph-context string and route through the graph-aware prompt.
        graph_context = _fetch_graph_context(graph_db, doi, k=graph_k)
        merged = _re_extract_with_graph(
            doi=doi,
            record=record,
            source_type=source_type,
            existing=existing,
            fulltext=fulltext,
            config=config,
            state_db=state_db,
            llm=llm,
            rerender=rerender,
            graph_context=graph_context,
        )
        return merged

    # --- vector path (original behaviour, unchanged) ---

    # E3: targeted field extraction — focused prompt, no full phase run.
    if fields is not None and phases is None:
        targeted = extract_fields(fulltext, fields, llm, ocr_heavy=ocr_heavy,
                                  content_type=source_type)
        merged = dict(existing)
        merged.update(targeted)
        merged["_overall_confidence"] = compute_confidence_score(merged)
        state_db.update_extraction_json(doi, merged)
        if rerender and record.get("note_path"):
            try:
                rerender_note(doi, config, state_db)
            except Exception as exc:
                # P4.4: escalate in --strict; non-strict logs and continues.
                strict_fallback(
                    logger, f"Rerender failed after field re-extract for {doi}: {exc}", exc
                )
        logger.info("Field re-extract %s: %s", doi, fields)
        return merged

    # Phase-based re-extraction (default).
    if phases is None:
        phases = ["simple", "complex"]

    new_extraction = extract_paper(
        fulltext, llm,
        content_type=source_type,
        ocr_heavy=ocr_heavy,
        phases=tuple(phases),
        existing_extraction=existing,
    )
    if fields:
        # Both --phase and --field given: keep only the requested fields.
        merged = dict(existing)
        for f in fields:
            if f in new_extraction:
                merged[f] = new_extraction[f]
            conf_key = f"{f}_confidence"
            if conf_key in new_extraction:
                merged[conf_key] = new_extraction[conf_key]
        merged["_overall_confidence"] = compute_confidence_score(merged)
    else:
        merged = new_extraction

    state_db.update_extraction_json(doi, merged)
    if rerender and record.get("note_path"):
        try:
            rerender_note(doi, config, state_db)
        except Exception as exc:
            # P4.4: escalate in --strict; non-strict logs and continues.
            strict_fallback(logger, f"Rerender failed after re-extract for {doi}: {exc}", exc)
    logger.info("Re-extracted %s: %s", source_type, doi)
    return merged
def re_extract_all_failed_phase(
    phase: str,
    config,
    state_db,
    llm,
    source_type: str = "paper",
    zotero_client=None,
) -> dict[str, int]:
    """Re-run a specific phase for all items that have a _<phase>_error key.

    M3 replacement for re_extract_all_failed().  Scans extraction_json for
    ``_simple_error`` or ``_complex_error`` markers and re-runs that phase.

    Returns
    -------
    dict with keys: re_extracted, failed
    """
    if phase not in ("simple", "complex"):
        raise ValueError(f"phase must be 'simple' or 'complex', got {phase!r}")
    records = state_db.get_all_by_source_type(source_type)
    stats = {"re_extracted": 0, "failed": 0}
    error_key = f"_{phase}_error"
    for record in records:
        doi = record.get("doi", "")
        raw = record.get("extraction_json", "")
        if not doi or not raw:
            continue
        try:
            extraction = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if error_key not in extraction:
            continue
        try:
            re_extract(
                doi=doi, config=config, state_db=state_db, llm=llm,
                phases=[phase], zotero_client=zotero_client,
            )
            stats["re_extracted"] += 1
        except Exception as exc:
            logger.error("Phase re-extract failed for %s: %s", doi, exc)
            # P4.4: escalate in --strict; non-strict records the failure in
            # stats["failed"] and continues to the next paper.
            strict_fallback(logger, f"Phase re-extract failed for {doi}: {exc}", exc)
            stats["failed"] += 1
    logger.info(
        "Re-extract all failed (phase=%s, %s): %d ok, %d failed",
        phase, source_type, stats["re_extracted"], stats["failed"],
    )
    return stats




def _open_graph_db_safe() -> Any | None:
    """N7: open GraphDB via safe_graph_db helper; return None on any failure.

    Lazily imports so the module remains usable without the [graph] extra.
    """
    try:
        from scripts.graph.import_citations import safe_graph_db
        return safe_graph_db()
    except Exception as exc:
        logger.warning("N7: could not open GraphDB: %s", exc)
        return None


def _fetch_graph_context(graph_db: Any, doi: str, k: int = 8) -> str:
    """N7: build a graph context block for the LLM prompt.

    Returns a human-readable string with:
      - Top-K related papers (via G7 find_similar_papers + titles)
      - Entities mentioned by this paper (via MENTIONS edges)

    Returns an empty string if graph_db is None or the neighbourhood is empty.
    Falls back gracefully on any Kuzu error, logging a WARNING.
    """
    if graph_db is None:
        return ""
    try:
        related: list[tuple[str, float]] = graph_db.find_similar_papers(doi, k=k)

        lines: list[str] = []

        if not related:
            lines.append("(no related papers found in the corpus)")
        else:
            lines.append(
                f"Top {len(related)} most-related papers (by shared MENTIONS overlap):"
            )
            for related_doi, score in related:
                # Fetch the title for this neighbour paper.
                res = graph_db._conn.execute(
                    "MATCH (p:Paper {doi: $d}) RETURN p.title",
                    {"d": related_doi},
                )
                if res.has_next():
                    title = str(res.get_next()[0])
                else:
                    title = "(no title)"
                lines.append(
                    f"  - {related_doi} (overlap={int(score)}): {title}"
                )

        # Entities mentioned by this paper.
        ent_res = graph_db._conn.execute(
            "MATCH (p:Paper {doi: $d})-[:MENTIONS]->(e:Entity) "
            "RETURN e.canonical_id, e.type LIMIT $k",
            {"d": doi, "k": k},
        )
        ent_rows: list[tuple[str, str]] = []
        while ent_res.has_next():
            row = ent_res.get_next()
            ent_rows.append((str(row[0]), str(row[1])))

        if ent_rows:
            lines.append("")
            lines.append("Entities mentioned by this paper:")
            for canonical, type_ in ent_rows:
                lines.append(f"  - {canonical} [{type_}]")

        return "\n".join(lines)

    except Exception as exc:
        # P4.4: escalate in --strict; non-strict logs and returns "" so graph
        # re-extraction can still proceed with an empty context block.
        strict_fallback(logger, f"N7: _fetch_graph_context failed for {doi}: {exc}", exc)
        return ""


def _re_extract_with_graph(
    doi: str,
    record: dict[str, Any],
    source_type: str,
    existing: dict[str, Any],
    fulltext: str,
    config: Any,
    state_db: Any,
    llm: Any,
    rerender: bool,
    graph_context: str,
) -> dict[str, Any]:
    """N7: single-pass LLM re-extraction using the graph-context prompt.

    Renders the re_extract_with_graph prompt with the paper's title, a text
    excerpt, and the pre-built graph_context block, then calls llm.complete()
    directly (same pattern used in other non-extractor LLM sites in this repo).
    Merges the result into existing and persists to state_db.
    """
    from scripts.llm.prompt_registry import load_prompt

    prompt = load_prompt("re_extract_with_graph")

    # Use title + abstract fields from the DB record if available; otherwise
    # fall back to the full text (already sanitized by _load_fulltext).
    extraction = existing or {}
    title = record.get("title") or extraction.get("title") or "(no title)"
    abstract = record.get("abstract") or fulltext

    user_msg = prompt.render_user(
        title=title,
        abstract=abstract,
        graph_context=graph_context or "(graph context unavailable)",
    )

    raw_response: str = llm.complete(
        system=prompt.system,
        user=user_msg,
        max_tokens=prompt.max_tokens,
    )

    # Parse the JSON response — tolerate leading/trailing whitespace and
    # markdown code-fences the model sometimes emits.
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        # Remove closing fence if present.
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        new_fields: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "N7: LLM response for %s was not valid JSON; keeping existing extraction",
            doi,
        )
        return existing

    merged = dict(existing)
    merged.update(new_fields)
    merged["_overall_confidence"] = compute_confidence_score(merged)

    state_db.update_extraction_json(doi, merged)
    if rerender and record.get("note_path"):
        try:
            rerender_note(doi, config, state_db)
        except Exception as exc:
            # P4.4: escalate in --strict; non-strict logs and continues.
            strict_fallback(
                logger, f"N7: Rerender failed after graph re-extract for {doi}: {exc}", exc
            )
    logger.info("N7: Graph re-extracted %s: %s", source_type, doi)
    return merged


def _load_fulltext(record: dict[str, Any], zotero_client=None) -> str:
    """Load full-text for re-extraction.

    Priority:
      1. Markdown attachment via ZoteroClient (M1 — primary path)
      2. Read the rendered Obsidian note
      3. Reconstruct from extraction_json summary fields (last resort)
    """
    zotero_key = record.get("zotero_key", "")
    # Priority 1: markdown attachment (M1 path)
    if zotero_key and zotero_client is not None:
        try:
            md = zotero_client.get_markdown_attachment(zotero_key)
            if md:
                return sanitize_for_prompt(md)
        except Exception as exc:
            logger.debug("Markdown attachment lookup failed for %s: %s", zotero_key, exc)
    # Fallback: read the rendered Obsidian note
    note_path = record.get("note_path", "")
    if note_path and Path(note_path).exists():
        return sanitize_for_prompt(Path(note_path).read_text(encoding="utf-8"))
    # Last resort: reconstruct from extraction JSON
    extraction = json.loads(record.get("extraction_json") or "{}")
    parts = []
    for field in ["core_finding", "methods_summary", "results_summary"]:
        if extraction.get(field):
            parts.append(str(extraction[field]))
    return sanitize_for_prompt("\n\n".join(parts) if parts else "")
