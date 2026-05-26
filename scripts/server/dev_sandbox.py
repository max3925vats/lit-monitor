"""Disposable sandbox for the /dev test page.

The sandbox isolates ingestion runs from production state:
  - separate sqlite DB:     ~/.config/lit-monitor/state_dev.db
  - separate ChromaDB collections (same persist dir, different names):
        dev-papers   /  dev-chunks
  - separate vault subfolder:  <vault_path>/Literature/_Dev/

Why this shape?

* ``EmbeddingsDB`` hardcodes its production collection names (``lit_monitor_v1``
  and ``lit_monitor_chunks_v1``), so the sandbox creates its sibling
  collections directly through ``chromadb.PersistentClient`` rather than
  instantiating a parallel ``EmbeddingsDB``. That keeps the production class
  unchanged.
* ``StateDB`` already takes the DB path as its first constructor arg, so the
  sandbox just hands it a different file.
* Vault writes go into a clearly-marked ``_Dev`` subfolder so a user can
  always tell which markdown files came from the test surface.

Clear via ``clear_sandbox(confirm=True)`` — destructive, requires explicit flag.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.core.config import get_config
from scripts.core.state_db import StateDB

logger = logging.getLogger(__name__)

# Module-level constants. Tests monkeypatch SANDBOX_STATE_DB_PATH to redirect
# the sandbox sqlite file into a tmp_path so they never touch the real one.
SANDBOX_STATE_DB_PATH: Path = Path("~/.config/lit-monitor/state_dev.db").expanduser()
SANDBOX_VAULT_SUBFOLDER: str = "Literature/_Dev"
# ChromaDB rejects names that start with an underscore (must start/end with
# [a-zA-Z0-9]). Use the ``dev-`` prefix instead so the sandbox collections
# can actually be created at runtime — the leading underscore from earlier
# drafts was silently swallowed by sandbox_status()'s try/except, masking
# the failure.
SANDBOX_PAPERS_COLLECTION: str = "dev-papers"
SANDBOX_CHUNKS_COLLECTION: str = "dev-chunks"


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------
def _chroma_persist_dir() -> str:
    """Resolve the ChromaDB persist directory using the same convention as
    cli._make_embeddings_db / runtime: sibling of the state DB file."""
    cfg = get_config()
    return str(Path(cfg.state_db.path).expanduser().parent / "chroma")


def sandbox_state_db() -> StateDB:
    """Return a StateDB instance pointed at the sandbox sqlite file."""
    # StateDB.__init__(self, db_path) — see scripts/core/state_db.py.
    return StateDB(SANDBOX_STATE_DB_PATH)


def sandbox_vault_subfolder() -> Path:
    """Resolve the sandbox vault output dir, creating it on demand."""
    cfg = get_config()
    p = Path(cfg.obsidian.vault_path).expanduser() / SANDBOX_VAULT_SUBFOLDER
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# ChromaDB collection helpers (sandbox-only)
# ---------------------------------------------------------------------------
def _sandbox_chroma_client():
    """Return a chromadb PersistentClient at the shared persist dir.

    Imported lazily so the module load cost stays cheap and so tests that
    monkeypatch chromadb don't have to patch at module-import time.
    """
    import chromadb  # local import — keeps boot fast and side-effects minimal

    return chromadb.PersistentClient(path=_chroma_persist_dir())


def sandbox_embeddings_db():
    """Return an ``EmbeddingsDB`` bound to the sandbox ``dev-*`` collections.

    Reuses the production class so chunk/paper add+embed logic stays in one
    place; the new ``papers_collection`` / ``chunks_collection`` kwargs route
    its writes to the ``dev-*`` sandbox collections instead of
    ``lit_monitor_v1`` / ``lit_monitor_chunks_v1``.
    """
    from scripts.output.embeddings import EmbeddingsDB

    cfg = get_config()
    ollama_host = getattr(cfg.embeddings, "ollama_host", None) or getattr(
        cfg.brain_build, "ollama_host", "http://localhost:11434"
    )
    embed_model = getattr(cfg.embeddings, "model", "mxbai-embed-large")
    return EmbeddingsDB(
        persist_dir=_chroma_persist_dir(),
        ollama_host=ollama_host,
        embed_model=embed_model,
        papers_collection=SANDBOX_PAPERS_COLLECTION,
        chunks_collection=SANDBOX_CHUNKS_COLLECTION,
    )


def sandbox_collections() -> tuple[Any, Any]:
    """Return (papers_collection, chunks_collection) for the sandbox.

    Thin wrapper for callers that need raw ChromaDB collection access (e.g.
    ``sandbox_status()`` counts). Ingestion paths should use
    ``sandbox_embeddings_db()`` instead so they share the production embed
    logic.
    """
    client = _sandbox_chroma_client()
    papers = client.get_or_create_collection(
        name=SANDBOX_PAPERS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    chunks = client.get_or_create_collection(
        name=SANDBOX_CHUNKS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return papers, chunks


# ---------------------------------------------------------------------------
# Status panel
# ---------------------------------------------------------------------------
def sandbox_status() -> dict[str, Any]:
    """Return live counts for the dev-page status panel.

    Each sub-section is wrapped in its own try/except — a missing vault path
    must not knock out the chroma counts and vice versa.
    """
    state_rows = 0
    papers_chroma = 0
    chunks_chroma = 0
    vault_files = 0
    last_modified: str | None = None

    if SANDBOX_STATE_DB_PATH.exists():
        try:
            db = sandbox_state_db()
            with db._connect() as conn:  # noqa: SLF001 — intentional, status-only peek
                cur = conn.execute("SELECT COUNT(*) FROM papers")
                state_rows = cur.fetchone()[0]
        except Exception as exc:
            logger.warning("Sandbox state row count failed: %s", exc)
        try:
            last_modified = datetime.fromtimestamp(
                SANDBOX_STATE_DB_PATH.stat().st_mtime
            ).isoformat()
        except Exception:
            pass

    try:
        papers_col, chunks_col = sandbox_collections()
        papers_chroma = papers_col.count()
        chunks_chroma = chunks_col.count()
    except Exception as exc:
        logger.warning("Sandbox chroma count failed: %s", exc)

    try:
        vault_dir = sandbox_vault_subfolder()
        vault_files = sum(1 for _ in vault_dir.rglob("*.md"))
    except Exception as exc:
        logger.warning("Sandbox vault file count failed: %s", exc)

    return {
        "state_db_rows": state_rows,
        "papers_collection_count": papers_chroma,
        "chunks_collection_count": chunks_chroma,
        "vault_file_count": vault_files,
        "last_modified": last_modified,
    }


# ---------------------------------------------------------------------------
# Destructive: clear the sandbox
# ---------------------------------------------------------------------------
def clear_sandbox(*, confirm: bool = False) -> dict[str, str]:
    """Wipe all sandbox state. Destructive — requires confirm=True.

    Returns a dict summarising what was cleared. Individual failures are
    captured in the ``actions`` log rather than raising, so a partial wipe
    (e.g. vault gone but chroma still around) is visible to the caller.
    """
    if not confirm:
        return {"status": "skipped", "reason": "confirm=False"}

    actions: list[str] = []
    failed = False

    # 1) State DB file (+ SQLite WAL/SHM siblings if present)
    if SANDBOX_STATE_DB_PATH.exists():
        try:
            SANDBOX_STATE_DB_PATH.unlink()
            for suffix in ("-wal", "-shm"):
                sib = SANDBOX_STATE_DB_PATH.with_name(
                    SANDBOX_STATE_DB_PATH.name + suffix
                )
                sib.unlink(missing_ok=True)
            actions.append(f"unlinked {SANDBOX_STATE_DB_PATH}")
        except Exception as exc:
            actions.append(f"FAILED unlink state_dev.db: {exc}")
            failed = True

    # 2) ChromaDB sandbox collections — delete; if absent, ignore silently.
    try:
        client = _sandbox_chroma_client()
        for col in (SANDBOX_PAPERS_COLLECTION, SANDBOX_CHUNKS_COLLECTION):
            try:
                client.delete_collection(col)
                actions.append(f"deleted chroma collection {col}")
            except Exception:
                # Collection was never created — that's fine.
                pass
    except Exception as exc:
        actions.append(f"FAILED chroma cleanup: {exc}")
        failed = True

    # 3) Vault subfolder
    try:
        # Resolve directly (without auto-create) so we don't recreate then nuke.
        cfg = get_config()
        vault_dir = Path(cfg.obsidian.vault_path).expanduser() / SANDBOX_VAULT_SUBFOLDER
        if vault_dir.exists():
            shutil.rmtree(vault_dir, ignore_errors=True)
            actions.append(f"removed {vault_dir}")
    except Exception as exc:
        actions.append(f"FAILED vault rmtree: {exc}")
        failed = True

    return {
        "status": "partial" if failed else "cleared",
        "actions": ";".join(actions),
    }


__all__ = [
    "SANDBOX_STATE_DB_PATH",
    "SANDBOX_VAULT_SUBFOLDER",
    "SANDBOX_PAPERS_COLLECTION",
    "SANDBOX_CHUNKS_COLLECTION",
    "sandbox_state_db",
    "sandbox_vault_subfolder",
    "sandbox_collections",
    "sandbox_embeddings_db",
    "sandbox_status",
    "clear_sandbox",
]
