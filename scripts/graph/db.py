"""
GraphDB — thin wrapper around a KuzuDB embedded database.

Mirrors the shape of ``scripts.output.embeddings.EmbeddingsDB``:
  - ``__init__(self, persist_dir: str)``
  - lazy import of kuzu (only at instantiation, not at module load)
  - idempotent init via DDL ``IF NOT EXISTS`` guards in ``apply_schema``

The kuzu import is deferred so that ``import scripts.graph`` succeeds even
when kuzu is not installed.  Only calling ``GraphDB(...)`` will raise an
``ImportError`` in that case — non-graph users see no disruption.
"""
from __future__ import annotations

import logging
from pathlib import Path

from scripts.graph.migrations import apply_schema

logger = logging.getLogger(__name__)


class GraphDB:
    """Embedded KuzuDB knowledge graph for lit-monitor.

    Parameters
    ----------
    persist_dir:
        Path to the KuzuDB directory.  Created (including parents) if absent.
        Use ``~/.config/lit-monitor/graph.kuzu`` in production.

    Raises
    ------
    ImportError
        When ``kuzu`` is not importable (i.e. the ``[graph]`` optional extra
        has not been installed).  Install with ``uv sync --extra graph``.
    """

    def __init__(self, persist_dir: str) -> None:
        # Expand ~ and resolve so the path is always absolute.
        # ``persist_dir`` is the path passed to kuzu.Database().  KuzuDB stores
        # its data at this path as a file (not a directory).  We ensure only
        # the PARENT directory exists — kuzu.Database() creates the file itself.
        self._persist_dir = Path(persist_dir).expanduser()
        self._persist_dir.parent.mkdir(parents=True, exist_ok=True)

        # Lazy kuzu import — deferred so non-graph users never hit ImportError
        # at module-load time.  Only instantiation requires kuzu to be present.
        try:
            import kuzu  # noqa: PLC0415  (intentional lazy import)
        except ImportError as exc:
            raise ImportError(
                "kuzu is required for GraphDB but is not installed. "
                "Install with: uv sync --extra graph"
            ) from exc

        logger.debug("Opening KuzuDB at %s", self._persist_dir)
        self._db = kuzu.Database(str(self._persist_dir))
        self._conn = kuzu.Connection(self._db)

        # Apply the Phase-1 schema DDL — idempotent on re-open.
        apply_schema(self._conn)
        logger.debug("GraphDB ready: %s", self._persist_dir)
