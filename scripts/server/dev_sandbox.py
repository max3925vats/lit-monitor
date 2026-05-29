"""Disposable sandbox for the /dev test page.

The sandbox isolates ingestion runs from production state:
  - separate sqlite DB:        ~/.config/lit-monitor/state_dev.db
  - separate ChromaDB persist: ~/.config/lit-monitor/chroma_dev/
        (collections: dev-papers / dev-chunks)
  - separate vault subfolder:  <vault_path>/Literature/_Dev/

Why this shape?

* ``chromadb.PersistentClient`` enforces a single client instance per persist
  path per process. The runtime already holds a production client at
  ``~/.config/lit-monitor/chroma``; opening a second one there with different
  collection-name settings raises
  ``ValueError: An instance of Chroma already exists ... with different settings``.
  Giving the sandbox its own ``chroma_dev/`` dir sidesteps the singleton entirely.
* ``EmbeddingsDB`` hardcodes its production collection names (``lit_monitor_v1``
  and ``lit_monitor_chunks_v1``); we still override them with ``dev-*`` names so
  the sandbox is visibly distinct even when inspected raw.
* ``StateDB`` already takes the DB path as its first constructor arg, so the
  sandbox just hands it a different file.
* Vault writes go into a clearly-marked ``_Dev`` subfolder so a user can
  always tell which markdown files came from the test surface.

Clear via ``clear_sandbox(confirm=True)`` — destructive, requires explicit flag.
"""
from __future__ import annotations

import functools
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.core.config import get_config
from scripts.core.state_db import StateDB

logger = logging.getLogger(__name__)

# Module-level constants. Tests monkeypatch SANDBOX_STATE_DB_PATH and
# SANDBOX_CHROMA_DIR to redirect the sandbox sqlite file + chroma persist dir
# into a tmp_path so they never touch the real ones.
SANDBOX_STATE_DB_PATH: Path = Path("~/.config/lit-monitor/state_dev.db").expanduser()
# Separate persist dir from production chroma at ~/.config/lit-monitor/chroma —
# see module docstring for the singleton-collision rationale.
SANDBOX_CHROMA_DIR: Path = Path("~/.config/lit-monitor/chroma_dev").expanduser()
SANDBOX_VAULT_SUBFOLDER: str = "Literature/_Dev"
# G12: sandbox KuzuDB path — parallel to the other sandbox stores.
# KuzuDB stores its data as a FILE at this path (not a directory), so the type
# is str to match GraphDB(persist_dir: str). Production sits at graph.kuzu;
# the sandbox uses graph_dev.kuzu so clear_sandbox never touches production data.
# Tests monkeypatch SANDBOX_GRAPH_DIR to redirect into tmp_path.
SANDBOX_GRAPH_DIR: str = str(Path("~/.config/lit-monitor/graph_dev.kuzu").expanduser())
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
    """Resolve the sandbox ChromaDB persist directory.

    Returns the module-level ``SANDBOX_CHROMA_DIR`` (a separate path from the
    production ``~/.config/lit-monitor/chroma``) so chromadb's single-client
    invariant per path never collides with the production runtime client.
    Tests monkeypatch ``SANDBOX_CHROMA_DIR`` directly.
    """
    return str(SANDBOX_CHROMA_DIR)


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
@functools.lru_cache(maxsize=1)
def sandbox_embeddings_db():
    """Singleton ``EmbeddingsDB`` bound to the sandbox ``dev-*`` collections.

    Cached as a module-level singleton because chromadb's ``PersistentClient``
    rejects a second client on the same persist path when its ``Settings``
    object isn't identical to the first one. ``sandbox_status()`` (Panel 7,
    polled every 5s) and ``_run_sandbox_ingest()`` (Mode A/B/C) both construct
    an ``EmbeddingsDB`` on ``SANDBOX_CHROMA_DIR``; without caching, the second
    construction crashes with::

        ValueError: An instance of Chroma already exists for <path> with
        different settings

    The cache is invalidated by ``clear_sandbox(confirm=True)`` after the
    persist dir is removed, so the next access constructs a fresh client
    against the recreated (empty) dir.

    Reuses the production ``EmbeddingsDB`` class so chunk/paper add+embed
    logic stays in one place; the ``papers_collection`` /
    ``chunks_collection`` kwargs route writes to the ``dev-*`` sandbox
    collections instead of ``lit_monitor_v1`` / ``lit_monitor_chunks_v1``.

    Note: ``sandbox_status()`` re-reads counts via ``col.count()`` on each
    call, which queries chromadb live — so the cached client doesn't cause
    stale counts. Only the client *handle* is cached.
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

    Routes through ``sandbox_embeddings_db()`` so every chromadb client at
    SANDBOX_CHROMA_DIR is the same one — otherwise chromadb's "instance
    already exists with different settings" guard fires when raw-count callers
    (sandbox_status / Panel 7) and EmbeddingsDB-using callers (Mode A/B/C
    ingest) both run in the same process.
    """
    emb = sandbox_embeddings_db()
    # EmbeddingsDB stores the papers collection at `_collection` and the
    # chunks collection at `_chunks_collection` (scripts/output/embeddings.py:64-76).
    papers = getattr(emb, "_collection", None)
    chunks = getattr(emb, "_chunks_collection", None)
    if papers is None or chunks is None:
        # Defensive: if EmbeddingsDB's internal attribute names ever change,
        # surface a clear error rather than crashing deeper in chromadb.
        raise AttributeError(
            "EmbeddingsDB does not expose _collection/_chunks_collection attributes; "
            "sandbox_collections() needs an update to match the new shape."
        )
    return papers, chunks


# ---------------------------------------------------------------------------
# KuzuDB graph helper (G12)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def sandbox_graph_db():
    """Return a cached GraphDB bound to the sandbox graph path.

    Mirrors ``sandbox_embeddings_db()`` — module-level lru_cache so both
    ``sandbox_status()`` (polled every 5 s) and ``_run_sandbox_ingest()``
    see the same client handle. The cache is invalidated by
    ``clear_sandbox(confirm=True)`` after the DB file is removed.

    KuzuDB stores its data as a single FILE at ``SANDBOX_GRAPH_DIR`` (not a
    directory), so ``GraphDB(persist_dir=SANDBOX_GRAPH_DIR)`` is the correct
    constructor form — matching the production pattern in
    ``scripts/graph/import_citations.py::safe_graph_db``.
    """
    from scripts.graph import GraphDB  # lazy — kuzu optional extra

    return GraphDB(persist_dir=SANDBOX_GRAPH_DIR)


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

    # Only probe chroma if the sandbox persist dir exists — first-run flow has
    # no dir yet, and calling PersistentClient on a missing path would
    # silently create one, defeating the "zero counts when empty" contract.
    if SANDBOX_CHROMA_DIR.exists():
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

    # G12: graph node count — wrap in its own try/except so a missing kuzu
    # install (or an empty sandbox) degrades gracefully to zero without
    # masking the other status fields.
    graph_node_count = 0
    try:
        graph = sandbox_graph_db()
        res = graph._conn.execute("MATCH (n) RETURN count(n) AS c")  # noqa: SLF001
        if res.has_next():
            row = res.get_next()
            graph_node_count = int(row[0])
    except Exception as exc:  # noqa: BLE001 — defensive, kuzu may not be installed
        logger.debug("sandbox_status: graph node count failed: %s", exc)

    return {
        "state_db_rows": state_rows,
        "papers_collection_count": papers_chroma,
        "chunks_collection_count": chunks_chroma,
        "vault_file_count": vault_files,
        "last_modified": last_modified,
        "graph_node_count": graph_node_count,
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

    # 2) Sandbox chroma persist dir — rmtree the whole thing. We use a fully
    # separate dir from production (see SANDBOX_CHROMA_DIR), so wiping is safe.
    # rmtree is preferred over delete_collection because chromadb's singleton
    # may already be holding open this path; rmtree forces a clean slate on
    # next access.
    if SANDBOX_CHROMA_DIR.exists():
        try:
            shutil.rmtree(SANDBOX_CHROMA_DIR, ignore_errors=False)
            actions.append(f"removed chroma persist dir {SANDBOX_CHROMA_DIR}")
        except Exception as exc:
            actions.append(f"FAILED chroma rmtree: {exc}")
            failed = True

    # Drop the cached EmbeddingsDB so the next caller constructs a fresh client
    # against the freshly-empty (or recreated) sandbox dir. Without this the
    # next sandbox_embeddings_db() call would hand back a client still pointing
    # at the rmtree'd path and writes would fail.
    sandbox_embeddings_db.cache_clear()

    # ALSO drop chromadb's internal singleton-client cache. chromadb's
    # PersistentClient is keyed by path inside its own process-global cache;
    # rmtree'ing the persist dir doesn't evict the cached client, so the next
    # `PersistentClient(path=SANDBOX_CHROMA_DIR)` returns the stale handle
    # whose underlying sqlite file has been deleted. Subsequent writes then
    # fail with SQLITE_READONLY_DBMOVED (code 1032). Clearing chromadb's
    # system cache forces the next constructor call to allocate a fresh
    # client against the recreated directory.
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
        actions.append("cleared chromadb system cache")
    except Exception as exc:  # noqa: BLE001 — defensive; varies by chromadb version
        actions.append(f"chromadb system-cache reset skipped: {exc}")
        # Not flagged as failed — chromadb may not be installed in test env,
        # and on fresh dirs (no prior client) the reset is a no-op anyway.

    # 3) Sandbox graph DB (G12) — KuzuDB stores as a FILE, but we handle both
    # a file and a directory defensively (mirrors the dual-shape handling in
    # perform_state_reset). Also remove the .schema_version sentinel if present.
    graph_path = Path(SANDBOX_GRAPH_DIR)
    try:
        if graph_path.exists():
            if graph_path.is_dir():
                shutil.rmtree(graph_path, ignore_errors=False)
            else:
                graph_path.unlink()
            actions.append(f"removed graph sandbox {graph_path}")
        sentinel = Path(str(SANDBOX_GRAPH_DIR) + ".schema_version")
        if sentinel.exists():
            sentinel.unlink()
            actions.append(f"removed graph sentinel {sentinel}")
    except Exception as exc:
        actions.append(f"FAILED graph sandbox remove: {exc}")
        failed = True
    # Drop the cached GraphDB so the next caller gets a fresh instance against
    # the re-created (empty) sandbox dir — same pattern as sandbox_embeddings_db.
    sandbox_graph_db.cache_clear()

    # 4) Vault subfolder
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
    "SANDBOX_CHROMA_DIR",
    "SANDBOX_GRAPH_DIR",
    "SANDBOX_VAULT_SUBFOLDER",
    "SANDBOX_PAPERS_COLLECTION",
    "SANDBOX_CHUNKS_COLLECTION",
    "sandbox_state_db",
    "sandbox_vault_subfolder",
    "sandbox_collections",
    "sandbox_embeddings_db",
    "sandbox_graph_db",
    "sandbox_status",
    "clear_sandbox",
]
