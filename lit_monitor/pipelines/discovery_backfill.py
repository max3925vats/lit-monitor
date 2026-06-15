"""Backfill discovery_paper_results.ingested for recommendations that were
ingested before the attribution wiring shipped. Forward-only, re-runnable."""
from __future__ import annotations

import logging
from typing import Any

from lit_monitor.core.doi import normalize_doi

logger = logging.getLogger(__name__)


def backfill_ingested(state_db: Any) -> int:
    """For every DOI ever shown in a digest that is now in the library
    (status != 'discovered'), flip its discovery_paper_results rows to
    ingested. Uses the paper's last_updated as the timestamp when available
    (SQLite datetime('now') format: YYYY-MM-DD HH:MM:SS, which passes the
    ISO regex in mark_discovery_recommendations_ingested). Falls back to
    datetime('now') when last_updated is NULL. Returns the count of distinct
    DOIs whose rows were flipped.

    Args:
        state_db: StateDB instance connected to the application state database.

    Returns:
        Number of distinct DOIs that had at least one discovery_paper_results
        row updated (i.e. rows changed from ingested=0 to ingested=1).
    """
    # Build the set of DOIs already in the library (normalised for comparison)
    library: set[str] = {normalize_doi(d) for d in state_db.library_dois()}
    flipped = 0

    for doi in state_db.surfaced_dois():
        if normalize_doi(doi) not in library:
            continue

        # Use the paper's own last_updated as the backfill timestamp so the
        # ingested_at column approximates the real ingest time.  SQLite stores
        # datetime('now') as 'YYYY-MM-DD HH:MM:SS', which passes the ISO regex
        # checked inside mark_discovery_recommendations_ingested.  If the field
        # is NULL (e.g. a paper row written by an older schema version), pass
        # when=None and let the method stamp datetime('now').
        paper = state_db.get_paper(doi) or {}
        when: str | None = paper.get("last_updated")  # None → method uses now()

        n = state_db.mark_discovery_recommendations_ingested(doi, when=when)
        if n:
            flipped += 1

    logger.info("backfill-ingested: flipped %d recommendation DOI(s)", flipped)
    return flipped
