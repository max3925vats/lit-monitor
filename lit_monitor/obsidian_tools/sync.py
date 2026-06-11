"""P10: bulk Obsidian note sync.

When ``discovery.notes.auto_write_per_paper=false``, the inline write is
skipped in brain_build / discovery; notes accumulate as ``notes_synced=0``
in state.db.  This module processes the pending queue in one controlled
pass, writing each note via the same rerender path used by
``lit-monitor obsidian rerender``.

Typical usage
-------------
    from lit_monitor.obsidian_tools.sync import sync_notes
    result = sync_notes(state_db, cfg, limit=50)
    # result = {"processed": N, "succeeded": M, "failed": K}

Per-paper failures are logged at WARNING and do not abort the run;
``notes_synced`` stays 0 so the paper is retried on the next invocation.
"""
from __future__ import annotations

import logging
from typing import Any

from lit_monitor.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)


def _rerender_one(doi: str, state_db: Any, cfg: Any) -> None:
    """Thin wrapper around rerender_note — monkeypatch this in tests.

    Signature matches rerender.rerender_note: (doi, config, state_db).
    """
    from lit_monitor.obsidian_tools.rerender import rerender_note
    rerender_note(doi, cfg, state_db)


def sync_notes(
    state_db: Any,
    cfg: Any,
    *,
    doi: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """P10: bulk-sync pending Obsidian notes from state.db to the vault.

    Walks the queue of papers with ``embeddings_indexed=1`` and
    ``notes_synced=0``, calling :func:`_rerender_one` for each.  On
    success the ``notes_synced`` flag is flipped to 1; on failure the flag
    stays 0 so the paper is retried on the next run.

    Parameters
    ----------
    state_db:
        StateDB instance.
    cfg:
        Config instance (passed through to rerender_note).
    doi:
        If given, process only this specific DOI (bypasses the pending
        queue — useful for manual one-off syncs regardless of the current
        flag value).
    since:
        ISO date string (YYYY-MM-DD or datetime).  If given, restrict to
        papers with ``last_updated >= since`` and ``notes_synced=0``.
    limit:
        Maximum number of papers to process in this run.  Applied after
        the ``doi``/``since`` filter.

    Returns
    -------
    dict with keys ``processed``, ``succeeded``, ``failed``.
    """
    # --- Build the work list ---
    if doi:
        # Single-DOI path: always process regardless of notes_synced state.
        # Useful for manual re-syncs or testing.
        dois: list[str] = [doi]
    elif since:
        with state_db._connect() as conn:
            rows = conn.execute(
                "SELECT doi FROM papers "
                "WHERE embeddings_indexed = 1 AND notes_synced = 0 "
                "AND last_updated >= ? "
                "ORDER BY last_updated DESC",
                (since,),
            ).fetchall()
        dois = [r[0] for r in rows if r[0]]
    else:
        # Default bulk path: get_notes_pending already applies the
        # embeddings_indexed + notes_synced gate.
        dois = state_db.get_notes_pending(limit=limit)

    # Apply limit after the since-filter (get_notes_pending handles its own limit).
    if limit is not None and since and len(dois) > limit:
        dois = dois[:limit]

    processed = 0
    succeeded = 0
    failed = 0

    for d in dois:
        processed += 1
        try:
            _rerender_one(d, state_db, cfg)
            state_db.set_notes_synced(d, 1)
            succeeded += 1
            logger.info("P10 sync: rendered note for %s", d)
        except Exception as exc:
            failed += 1
            # P4.4: per-paper best-effort. Escalate in --strict; non-strict logs,
            # leaves notes_synced=0 (retried next run), and continues.
            strict_fallback(logger, f"P10 sync: failed for {d}: {exc}", exc)
            # notes_synced stays 0; paper will be retried on next run.

    logger.info(
        "P10 sync complete: processed=%d succeeded=%d failed=%d",
        processed,
        succeeded,
        failed,
    )
    return {"processed": processed, "succeeded": succeeded, "failed": failed}
