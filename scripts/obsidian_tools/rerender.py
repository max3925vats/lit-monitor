"""
Obsidian rerender — regenerate a note from the extraction JSON stored in
the state DB, preserving all persist zone content.
Use this after changing templates or schema labels without re-running the LLM.
rerender_note(doi, config, state_db):
    - Loads extraction_json from state DB
    - Re-renders the appropriate template (paper / book / chapter)
    - Calls update_note_preserve_persist_zones() so user edits survive
rerender_all(source_type, config, state_db):
    - Calls rerender_note() for every item of the given source_type
    - Single failures log and continue
"""
from __future__ import annotations

import json
import logging

from scripts.core.strict_mode import strict_fallback
from scripts.output.obsidian_writer import write_paper_note

logger = logging.getLogger(__name__)
def rerender_note(doi: str, config, state_db) -> str:
    """
    Re-render and write the Obsidian note for a single item.
    Returns the note path (str). Raises ValueError if item not found in
    state DB or extraction_json is missing.
    """
    record = state_db.get_paper(doi)
    if record is None:
        raise ValueError(f"No record found for doi: {doi!r}")
    raw_json = record.get("extraction_json")
    if not raw_json:
        raise ValueError(f"No extraction_json for doi: {doi!r}")
    extraction = json.loads(raw_json)
    source_type = record.get("source_type", "paper")
    existing_note_path = record.get("note_path") or None
    if source_type in ("paper", "review"):
        from scripts.vocabulary.normalizer import assign_themes
        authors = json.loads(record.get("authors") or "[]")
        keywords = json.loads(record.get("keywords_json") or "[]")
        themes = assign_themes(keywords)
        paper_record = {
            "doi": doi,
            "title": record.get("title", ""),
            "authors": authors,
            "year": record.get("year", 0),
            "journal": record.get("journal", ""),
            "zotero_key": record.get("zotero_key", ""),
            "first_seen_date": record.get("first_seen_date", ""),
            "keywords": keywords,
            "themes": themes,
            "tracked_author": False,
            "fulltext_analyzed": True,
        }
        note_path = write_paper_note(
            paper_record, extraction, config,
            note_path_override=existing_note_path,
        )
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}")
    logger.info("Rerendered %s note: %s → %s", source_type, doi, note_path)
    return note_path
def rerender_all(
    source_type: str,
    config,
    state_db,
) -> dict[str, int]:
    """
    Re-render all notes of the given source_type.
    source_type: 'paper' | 'review'
    Returns

    -------
    dict with keys: rerendered, failed
    """
    records = state_db.get_all_by_source_type(source_type)
    stats = {"rerendered": 0, "failed": 0}
    for record in records:
        doi = record.get("doi", "")
        if not doi:
            continue
        try:
            rerender_note(doi, config, state_db)
            stats["rerendered"] += 1
        except Exception as exc:
            strict_fallback(
                logger,
                f"Rerender failed for {doi}: {exc}",
                exc,
            )
            stats["failed"] += 1
    logger.info(
        "Rerender all %s: %d ok, %d failed",
        source_type, stats["rerendered"], stats["failed"],
    )
    return stats
