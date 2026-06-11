"""FG-2: graph-health check surfaced through the web UI's health badge.

``check_graph()`` reports whether the KuzuDB knowledge graph is built and how
much of the corpus has been mirrored into it (``papers.graph_indexed=1``). It is
wired as the 5th section of :func:`scripts.setup.health_check.run_health_check`.

Design notes:
    - The ``[graph]`` extra is optional. ``safe_graph_db`` already guards the
      KuzuDB import (returns ``None`` when the extra is absent or the DB can't be
      opened), so a graph-absent install yields a yellow *warn* row, never a
      crash.
    - A partially-indexed graph is *healthy* (``ok``): the user may simply not
      have backfilled everything yet. Only an unreachable graph or a load error
      is a ``warn``; nothing here returns ``fail``.
"""
from __future__ import annotations

import logging

# Re-export at module scope so tests can monkeypatch
# ``scripts.setup.check_graph.safe_graph_db``. This import is cheap and safe even
# without the [graph] extra: ``safe_graph_db`` itself lazily imports KuzuDB and
# returns None when it is missing (see scripts/graph/import_citations.py).
from lit_monitor.graph import safe_graph_db
from lit_monitor.setup.check_configured import CheckResult

logger = logging.getLogger(__name__)


def _indexed_total() -> tuple[int, int]:
    """Return ``(graph_indexed_count, total_papers)`` from the state DB.

    Mirrors the count idiom in ``scripts/cli.py`` (``SELECT COUNT(*) FROM papers
    WHERE graph_indexed = 1`` and the unfiltered total). Opens a fresh
    :class:`StateDB` via the same config accessor the CLI uses.

    Defensive: any failure (missing config, unbuilt DB, schema mismatch) returns
    ``(0, 0)`` so the health aggregator never crashes.
    """
    try:
        from lit_monitor.core.config import get_config
        from lit_monitor.core.state_db import StateDB

        state_db = StateDB(get_config().state_db.path)
        with state_db._connect() as conn:
            indexed = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE graph_indexed = 1"
            ).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        return int(indexed), int(total)
    except Exception as exc:  # noqa: BLE001 — health check must never raise
        logger.debug("check_graph._indexed_total failed: %s", exc)
        return (0, 0)


def check_graph() -> CheckResult:
    """Probe the knowledge graph and report a single badge-ready status.

    Returns:
        A :class:`CheckResult`:
            - graph absent / [graph] extra missing → ``warn`` (ok=False).
            - graph present but no papers yet → ``warn``.
            - graph present but nothing backfilled → ``warn``.
            - graph present and ≥1 paper indexed → ``ok`` (partial counts as
              healthy). Never ``fail``.
    """
    graph = safe_graph_db()
    if graph is None:
        return CheckResult(
            False,
            "Knowledge graph not built (or [graph] extra not installed)",
            severity="warn",
        )
    # Release the handle promptly — we only needed the availability gate.
    try:
        graph.close()
    except Exception:  # noqa: BLE001 — close is best-effort
        pass

    indexed, total = _indexed_total()
    if total == 0:
        return CheckResult(
            False, "Knowledge graph reachable but no papers yet", severity="warn"
        )
    if indexed == 0:
        return CheckResult(
            False,
            f"Knowledge graph not yet backfilled (0/{total} papers indexed)",
            severity="warn",
        )
    return CheckResult(
        True, f"{indexed}/{total} papers indexed in graph", severity="ok"
    )


__all__ = ["check_graph", "CheckResult"]
