"""
GraphDB — thin wrapper around a KuzuDB embedded database.

Mirrors the shape of ``scripts.output.embeddings.EmbeddingsDB``:
  - ``__init__(self, persist_dir: str)``
  - lazy import of kuzu (only at instantiation, not at module load)
  - idempotent init via DDL ``IF NOT EXISTS`` guards in ``apply_schema``

Schema versioning (G14)
-----------------------
The current SCHEMA_VERSION is stored as plain text in a sentinel file whose
name is derived from the KuzuDB path:

    <persist_dir>.schema_version

For example: ``~/.config/lit-monitor/graph.kuzu.schema_version``

On ``__init__``:
1. ``apply_schema`` runs the full DDL (IF NOT EXISTS — idempotent on
   already-initialised databases).
2. The sentinel file is read; if absent, version is assumed to be 1
   (the G1 baseline — database existed before G14 shipped).
3. ``apply_migrations`` is called with the persisted version; any pending
   migrations run and the new version is returned.
4. The sentinel file is written back with the current SCHEMA_VERSION.

The kuzu import is deferred so that ``import scripts.graph`` succeeds even
when kuzu is not installed.  Only calling ``GraphDB(...)`` will raise an
``ImportError`` in that case — non-graph users see no disruption.
"""
from __future__ import annotations

import logging
from pathlib import Path

from scripts.graph.migrations import apply_migrations, apply_schema

logger = logging.getLogger(__name__)


class GraphDB:
    """Embedded KuzuDB knowledge graph for lit-monitor.

    Parameters
    ----------
    persist_dir:
        Path to the KuzuDB file.  Parent directory is created (including
        parents) if absent.  Use ``~/.config/lit-monitor/graph.kuzu`` in
        production.

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

        # Sentinel file that persists the schema version across process restarts.
        # Stored alongside the DB file: <persist_dir>.schema_version
        self._version_file = Path(str(self._persist_dir) + ".schema_version")

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

        # Step 1: Apply the full schema DDL — idempotent on re-open.
        apply_schema(self._conn)

        # Step 2: Read persisted schema version (absent → G1 baseline = 1).
        persisted_version = self._read_schema_version()

        # Step 3: Run any pending migrations.
        new_version = apply_migrations(self._conn, current_version=persisted_version)

        # Step 4: Persist the updated version so future opens skip migrations.
        if new_version != persisted_version:
            self._write_schema_version(new_version)
            logger.info(
                "GraphDB migrated: schema v%d → v%d at %s",
                persisted_version,
                new_version,
                self._persist_dir,
            )

        logger.debug(
            "GraphDB ready (schema v%d): %s", new_version, self._persist_dir
        )

    # ------------------------------------------------------------------
    # Schema version helpers
    # ------------------------------------------------------------------

    def _read_schema_version(self) -> int:
        """Return the persisted schema version, or 1 if the sentinel is absent.

        Version 1 is the G1 baseline: a database created before G14 shipped
        and therefore lacking the three provenance columns.
        """
        if not self._version_file.exists():
            # Pre-G14 database — treat as v1 so migrations run on first open.
            return 1
        try:
            return int(self._version_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError) as exc:
            logger.warning(
                "Could not read schema version from %s (%s); defaulting to 1.",
                self._version_file,
                exc,
            )
            return 1

    def _write_schema_version(self, version: int) -> None:
        """Write the schema version integer to the sentinel file."""
        self._version_file.write_text(str(version), encoding="utf-8")

    def close(self) -> None:
        """Release the KuzuDB connection and database handle deterministically.

        kuzu.Connection / kuzu.Database are released when dereferenced; this method
        drops our references explicitly so callers can free resources without
        relying on CPython refcounting. Safe to call multiple times.
        """
        self._conn = None  # type: ignore[assignment]
        self._db = None    # type: ignore[assignment]
        logger.debug("GraphDB closed: %s", self._persist_dir)

    def __enter__(self) -> GraphDB:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
