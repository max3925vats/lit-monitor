"""Shared server-side runtime holder with lazy client construction.

Mirrors the per-call factory pattern used by ``scripts/cli.py`` so the
FastAPI app can boot even when configuration or credentials are missing.
The actual client instances are created on first attribute access and
cached for the lifetime of the process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

    @property
    def secrets(self) -> dict:
        if self._secrets is None:
            self._secrets = _load_secrets()
        return self._secrets

    @property
    def config(self) -> Any:
        if self._config is None:
            # Local import: avoids paying the cost at module load time.
            from scripts.core.config import get_config

            self._config = get_config()
        return self._config

    @property
    def state_db(self) -> Any:
        if self._state_db is None:
            from scripts.core.state_db import StateDB

            self._state_db = StateDB(self.config.state_db.path)
        return self._state_db

    @property
    def embeddings_db(self) -> Any:
        if self._embeddings_db is None:
            from scripts.output.embeddings import EmbeddingsDB

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
            from scripts.core.zotero_client import ZoteroClient

            cfg = self.config
            zot_secrets = self.secrets.get("zotero", {})
            self._zotero_client = ZoteroClient(
                library_id=str(zot_secrets.get("library_id", cfg.zotero.library_id)),
                api_key=str(zot_secrets.get("api_key", "")),
                library_type=cfg.zotero.library_type,
                local_storage_path=cfg.zotero.local_storage_path,
            )
        return self._zotero_client


# Module-level singleton: one runtime per server process.
_RUNTIME: ServerRuntime | None = None


def get_runtime() -> ServerRuntime:
    """Return the per-process ``ServerRuntime`` singleton."""

    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = ServerRuntime()
    return _RUNTIME


__all__ = ["ServerRuntime", "get_runtime"]
