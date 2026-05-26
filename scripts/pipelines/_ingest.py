"""Shared R28 invariant: embed + chunk + mark-phases in the right order.

Both discovery and brain-build pipelines must defer phase-progress marking
until embeddings succeed, so a degraded ChromaDB doesn't leave papers
permanently stuck in post-extraction state. This helper centralises that
contract. Each caller still owns its own pipeline-specific concerns
(rate limiting, batch tracking, source-type routing, etc.).
"""
from __future__ import annotations

import logging
from typing import Any


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
) -> tuple[bool, str | None]:
    """Index a paper, then mark phase progress IFF add_paper succeeded.

    R28 contract:
      - add_paper() failure -> phases NOT marked, returns (False, err).
      - add_chunks() failure -> logged as WARN; phases STILL marked
        (preserves existing non-fatal behaviour in both call sites).
      - All-OK -> phases marked, returns (True, None).
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

    for phase in phases_to_mark:
        state_db.mark_brain_build_phase(zotero_key, phase)
    return True, None
