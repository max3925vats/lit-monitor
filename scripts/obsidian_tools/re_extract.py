"""
Obsidian re-extract — re-run LLM extraction for one or both phases or specific
field(s), then update the state DB and rerender the note.

Use cases:
  - Re-run simple phase:  re_extract(doi, phases=["simple"], ...)
  - Re-run complex phase: re_extract(doi, phases=["complex"], ...)
  - Refresh a specific field: re_extract(doi, fields=["actionable_insights"], ...)
  - Bulk re-extraction: re_extract_all_failed_phase("simple", ...)

Single-item failures log and continue in bulk mode.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
                logger.warning("Rerender failed after field re-extract for %s: %s", doi, exc)
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
            logger.warning("Rerender failed after re-extract for %s: %s", doi, exc)
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
            stats["failed"] += 1
    logger.info(
        "Re-extract all failed (phase=%s, %s): %d ok, %d failed",
        phase, source_type, stats["re_extracted"], stats["failed"],
    )
    return stats




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
