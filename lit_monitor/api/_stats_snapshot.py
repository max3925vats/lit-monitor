"""Best-effort corpus snapshot at run completion (stats-banner deltas).

Isolated from the pipeline modules so the hot-path call site is a single
guarded line. NEVER raises — a snapshot failure must not affect a run.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def record_corpus_snapshot(
    state_db: Any, *, run_type: str | None = None, run_id: str | None = None
) -> None:
    """Compute current corpus counts and write one snapshot row. Best-effort."""
    try:
        papers = int(state_db.count_with_extraction())
    except Exception:
        papers = 0
    try:
        themes = len(state_db.list_active_clusters())
    except Exception:
        themes = 0
    graph_nodes = 0
    try:
        from lit_monitor.api.queries import get_corpus_stats
        from lit_monitor.graph import safe_graph_db

        db = safe_graph_db()
        if db is not None:
            try:
                gs = get_corpus_stats(db)
                graph_nodes = int(gs.get("paper_count", 0)) + int(gs.get("entity_count", 0))
            finally:
                try:
                    db.close()
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("record_corpus_snapshot: graph count skipped: %s", exc)

    try:
        state_db.write_corpus_snapshot(
            papers=papers, graph_nodes=graph_nodes, themes=themes,
            run_type=run_type, run_id=run_id,
        )
    except Exception as exc:
        logger.warning("record_corpus_snapshot: write failed (non-fatal): %s", exc)
