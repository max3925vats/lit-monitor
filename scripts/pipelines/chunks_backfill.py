"""CB1: chunks backfill — close the R28 chunks retry gap.

Walks state.db for papers WHERE chunks_indexed=0 AND fully_complete=1,
re-chunks each via the stored fulltext_path, and (re-)indexes into
ChromaDB. NO LLM calls — uses the existing chunk_markdown helper and
EmbeddingsDB.add_chunks (which is delete-then-add, so idempotent).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.core.state_db import StateDB

logger = logging.getLogger(__name__)


def backfill_chunks(
    state_db: StateDB,
    embeddings_db: Any,
    *,
    doi: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """CB1: re-chunk + re-index papers whose chunks_indexed=0.

    Selects papers with ``chunks_indexed=0 AND fully_complete=1`` from
    state.db, reads their stored ``fulltext_path``, runs ``chunk_markdown``,
    calls ``embeddings_db.add_chunks``, then sets ``chunks_indexed=1`` on
    success. Per-paper exceptions are logged as WARNING and skipped
    (``failed`` incremented) so one bad paper never blocks the rest.

    Parameters
    ----------
    state_db:
        Open StateDB instance.
    embeddings_db:
        Open EmbeddingsDB instance.
    doi:
        When provided, restricts the backfill to this single DOI.
        The paper is still only processed if its ``chunks_indexed=0``.
    since:
        When provided, further restricts to papers with
        ``last_updated >= since`` (ISO date string, e.g. "2024-01-01").
    limit:
        When provided, caps the number of papers processed in this run.

    Returns
    -------
    dict[str, int]
        ``{processed, succeeded, failed}`` counts.
    """
    from scripts.core.chunker import chunk_markdown

    # embeddings_indexed=1 means add_paper succeeded (the vector index is current).
    # That is the right gate: we only re-chunk papers whose main embedding
    # already landed so the chunk query (granularity="chunk") is coherent with
    # the paper-level query.
    query = (
        "SELECT doi, extraction_json FROM papers "
        "WHERE chunks_indexed = 0 AND embeddings_indexed = 1"
    )
    params: list[Any] = []

    if doi:
        query += " AND doi = ?"
        params.append(doi)
    if since:
        query += " AND last_updated >= ?"
        params.append(since)

    # Stable order so --limit is deterministic.
    query += " ORDER BY doi"

    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    with state_db._connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()

    processed = 0
    succeeded = 0
    failed = 0

    for row in rows:
        paper_doi = row[0]
        extraction_raw = row[1]
        processed += 1
        try:
            extraction = json.loads(extraction_raw) if extraction_raw else {}
            fulltext_path = extraction.get("fulltext_path")
            if not fulltext_path or not Path(fulltext_path).is_file():
                raise FileNotFoundError(
                    f"fulltext_path missing or not on disk: {fulltext_path!r}"
                )
            text = Path(fulltext_path).read_text(encoding="utf-8")
            chunks = chunk_markdown(text, paper_doi)
            embeddings_db.add_chunks(paper_doi, chunks)
            # Only flip the flag after a confirmed successful add_chunks.
            state_db.set_chunks_indexed(paper_doi, 1)
            succeeded += 1
            logger.info(
                "CB1: re-indexed chunks for %s (%d chunks)", paper_doi, len(chunks)
            )
        except Exception as exc:
            failed += 1
            logger.warning("CB1: chunks backfill failed for %s: %s", paper_doi, exc)

    return {"processed": processed, "succeeded": succeeded, "failed": failed}
