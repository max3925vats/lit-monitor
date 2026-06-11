"""Shared server-side runtime holder with lazy client construction.

Mirrors the per-call factory pattern used by ``scripts/cli.py`` so the
FastAPI app can boot even when configuration or credentials are missing.
The actual client instances are created on first attribute access and
cached for the lifetime of the process.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_secrets() -> dict:
    """Best-effort load of ~/.config/lit-monitor/config.toml.

    Mirrors ``scripts.cli._load_secrets``: returns ``{}`` when the file
    is missing or unreadable rather than raising. This keeps the server
    importable in fresh-clone development environments.
    """

    secrets_path = Path.home() / ".config" / "lit-monitor" / "config.toml"
    if not secrets_path.exists():
        return {}
    try:
        import tomllib  # Python 3.11+

        with secrets_path.open("rb") as fh:
            return tomllib.load(fh)
    except Exception:
        # Intentionally swallow: server should still boot.
        return {}


@dataclass
class ProcessSlot:
    """A registered subprocess + its bookkeeping.

    ``output`` is a bounded list (last ~1000 lines) so memory doesn't grow
    unbounded for long-running pipelines. The full record stays in the
    JSONL log; this in-memory buffer is just for the dashboard's "Live
    progress" panel when reloading the page mid-run.
    """

    process: asyncio.subprocess.Process | None = None
    started_at: str | None = None  # ISO-format
    # Set by /api/discovery/start so the controls fragment can label the run
    # as "dry-run" vs "running". Other slots ignore this field.
    dry_run: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    output: list[str] = field(default_factory=list)
    output_cap: int = 1000
    # Free-form per-slot metadata (e.g. state.db mtime snapshot for dev_dryrun).
    metadata: dict = field(default_factory=dict)

    def append_line(self, line: str) -> None:
        """Append a line, truncating from the start if cap exceeded."""
        self.output.append(line)
        if len(self.output) > self.output_cap:
            # Drop ~10% from the front so we don't reallocate every line.
            drop = max(1, self.output_cap // 10)
            del self.output[:drop]

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def stop(self, *, timeout: float = 30.0) -> bool:
        """SIGTERM the process; SIGKILL after timeout. Returns True if stopped."""
        async with self.lock:
            p = self.process
            if p is None or p.returncode is not None:
                return False
            p.terminate()
            try:
                await asyncio.wait_for(p.wait(), timeout=timeout)
            except TimeoutError:
                logger.warning(
                    "subprocess pid=%s didn't exit on SIGTERM after %ss; SIGKILL",
                    p.pid, timeout,
                )
                p.kill()
                await p.wait()
            self.process = None
            self.started_at = None
            self.dry_run = False
            return True


@dataclass
class ServerRuntime:
    """Lazily-constructed holder for shared per-process clients.

    Each property defers import + construction until first access so the
    server can start before config or credentials are present. Failures
    propagate to the caller (route handlers should handle them and return
    a sensible response) instead of crashing the app at startup.
    """

    _config: Any | None = field(default=None, repr=False)
    _state_db: Any | None = field(default=None, repr=False)
    _embeddings_db: Any | None = field(default=None, repr=False)
    _zotero_client: Any | None = field(default=None, repr=False)
    _secrets: dict | None = field(default=None, repr=False)
    _graph_db: Any | None = field(default=None, repr=False)
    # graph_db is allowed to legitimately resolve to None (the [graph] extra
    # may be absent or the DB may not open). A plain ``is None`` cache guard
    # would re-attempt construction on every access, so we track resolution
    # with a separate sentinel flag to cache the None result too.
    _graph_db_resolved: bool = field(default=False, repr=False)

    # Process registry — one slot per long-running pipeline. Start endpoints
    # refuse to spawn when the matching slot is already running.
    processes: dict[str, ProcessSlot] = field(
        default_factory=lambda: {
            "brain_build": ProcessSlot(),
            "discovery": ProcessSlot(),
            "vocabulary": ProcessSlot(),
            "dev_dryrun": ProcessSlot(),
            "dev_graph_backfill": ProcessSlot(),
        }
    )

    @property
    def secrets(self) -> dict:
        if self._secrets is None:
            self._secrets = _load_secrets()
        return self._secrets

    @property
    def config(self) -> Any:
        if self._config is None:
            # Local import: avoids paying the cost at module load time.
            from lit_monitor.core.config import get_config

            self._config = get_config()
        return self._config

    @property
    def state_db(self) -> Any:
        if self._state_db is None:
            from lit_monitor.core.state_db import StateDB

            self._state_db = StateDB(self.config.state_db.path)
        return self._state_db

    @property
    def embeddings_db(self) -> Any:
        if self._embeddings_db is None:
            from lit_monitor.output.embeddings import EmbeddingsDB

            cfg = self.config
            persist_dir = str(Path(cfg.state_db.path).parent / "chroma")
            ollama_host = getattr(cfg.embeddings, "ollama_host", None)
            if ollama_host is None:
                ollama_host = getattr(
                    cfg.brain_build, "ollama_host", "http://localhost:11434"
                )
            embed_model = getattr(cfg.embeddings, "model", "mxbai-embed-large")
            self._embeddings_db = EmbeddingsDB(
                persist_dir=persist_dir,
                ollama_host=ollama_host,
                embed_model=embed_model,
            )
        return self._embeddings_db

    @property
    def zotero_client(self) -> Any:
        if self._zotero_client is None:
            from lit_monitor.core.zotero_client import ZoteroClient

            cfg = self.config
            zot_secrets = self.secrets.get("zotero", {})
            self._zotero_client = ZoteroClient(
                library_id=str(zot_secrets.get("library_id", cfg.zotero.library_id)),
                api_key=str(zot_secrets.get("api_key", "")),
                library_type=cfg.zotero.library_type,
                local_storage_path=cfg.zotero.local_storage_path,
            )
        return self._zotero_client

    @property
    def graph_db(self) -> Any | None:
        """Lazily construct the Kuzu-backed GraphDB, or None when unavailable.

        Mirrors the other lazy properties but tolerates a legitimate ``None``
        result: ``safe_graph_db`` returns ``None`` when the ``[graph]`` extra
        is not installed or the DB cannot be opened. We cache that ``None`` via
        ``_graph_db_resolved`` so a missing graph install isn't re-probed on
        every request. As with the other lazy props, picking up a later graph
        install requires a runtime reset — acceptable and consistent.
        """
        if not self._graph_db_resolved:
            from lit_monitor.graph.import_citations import safe_graph_db

            # Argless call uses the default production path
            # (~/.config/lit-monitor/graph.kuzu), matching every other
            # server route (papers/entities/query/ingest) so the topics
            # endpoint reads the same graph.
            self._graph_db = safe_graph_db()
            self._graph_db_resolved = True
        return self._graph_db


# Module-level singleton: one runtime per server process.
_RUNTIME: ServerRuntime | None = None


def get_runtime() -> ServerRuntime:
    """Return the per-process ``ServerRuntime`` singleton."""

    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = ServerRuntime()
    return _RUNTIME


def reset_runtime() -> None:
    """Drop the cached runtime singleton.

    Intended for tests that need a fresh ServerRuntime per case. Production
    code never calls this — the server runs with a single runtime per process.
    """
    global _RUNTIME
    _RUNTIME = None


__all__ = ["ServerRuntime", "ProcessSlot", "get_runtime", "reset_runtime"]
