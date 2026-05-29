"""Pure logic for the ``lit-monitor reset {state|vault|all}`` CLI command.

Splits enumeration from deletion so the Click wrapper can:
  1. Build the target list with sizes for the confirmation prompt.
  2. Refuse to touch anything unless the user types the exact confirmation
     phrase.
  3. Delete each target, returning a per-target result so the CLI can render
     ✓ deleted / - skipped lines.

What ``reset state`` removes:
  - ``~/.config/lit-monitor/state.db`` (and ``state.db-wal`` / ``state.db-shm``)
  - ``~/.config/lit-monitor/chroma/`` (entire ChromaDB persist dir)
  - ``config/concepts_draft.yaml`` (auto-generated)
  - ``config/concepts_draft_refinement_raw.txt`` (auto-generated)
  - ``config/topics_suggested.yaml`` (auto-generated)

What ``reset vault`` removes:
  - ``<vault>/<papers_folder>/`` markdown notes (covers regular papers AND
    review-type Zotero items — both are written to ``papers_folder`` by
    ``obsidian_writer.write_paper_note``)
  - ``<vault>/<digests_folder>/`` markdown notes
  - ``<vault>/<connections_folder>/`` markdown notes (synthesis output)

What is NEVER touched:
  - ``~/.config/lit-monitor/config.toml`` — credentials. User must delete
    manually if they really want it gone.
  - ``~/.config/lit-monitor/chroma_dev/`` and ``state_dev.db`` — dev sandbox.
  - ``~/.config/lit-monitor/logs/`` — kept for post-reset debugging.
  - Any ``.example.yaml`` files.
  - Tracked YAML configs (paths.yaml, topics.yaml, concepts.yaml, etc.).
  - ``<vault>/<books_folder>/`` — books/bookSections are routed to the
    ``skip`` pipeline (see ``config/item_routing.yaml``); any notes there are
    user-managed content, not lit-monitor-generated.
  - ``<vault>/Literature/_Dev/`` — sandbox subfolder managed elsewhere.
  - Theme pages at the vault root (may carry user-added context).

ChromaDB gotcha: ``chromadb.PersistentClient`` keeps a process-global cache of
``System`` objects keyed by persist path. Removing the persist directory does
*not* evict that cache, so the next ``EmbeddingsDB()`` in the same process
hands back a stale handle whose sqlite file no longer exists, producing
``SQLITE_READONLY_DBMOVED`` on write. ``perform_state_reset`` calls
``SharedSystemClient.clear_system_cache()`` after rmtree to force a fresh
client on next access — same pattern as ``dev_sandbox.clear_sandbox``.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ResetTarget:
    """A single path the reset command would consider deleting.

    ``size_bytes`` and ``file_count`` are pre-computed for the confirmation
    prompt so the user can see the impact before typing the confirmation
    phrase. Absent targets have ``exists=False`` and zeroed metrics, and are
    still kept in the list so the prompt shows them as ``(not present)``
    rather than silently omitting them — this is a deliberate UX choice so
    re-running the command after a successful reset still surfaces the full
    set of locations it inspects.
    """

    label: str
    path: Path
    size_bytes: int  # 0 if absent
    file_count: int  # 0 if file or absent; recursive for directory
    exists: bool


@dataclass
class ResetResult:
    """Per-target outcome reported by ``perform_*_reset`` for CLI rendering."""

    label: str
    path: Path
    deleted: bool
    skipped_reason: str | None = None  # e.g. "not present"


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------
def _format_size_bytes(num_bytes: int) -> str:
    """Render a byte count as a short human-readable string.

    Matches the diagnose-style output (KB / MB / GB) used elsewhere in the
    CLI surface. Values under 1 KB render in bytes so empty/near-empty
    targets don't all collapse to ``0 KB``.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    for unit in ("KB", "MB", "GB", "TB"):
        num_bytes_f = num_bytes / 1024.0
        if num_bytes_f < 1024 or unit == "TB":
            return f"{num_bytes_f:.1f} {unit}"
        num_bytes = int(num_bytes_f)
    # Unreachable, but keep mypy happy.
    return f"{num_bytes} B"


def _probe_path(path: Path) -> tuple[bool, int, int]:
    """Return (exists, size_bytes, file_count) for a file or directory path.

    For directories: recursive walk, summing file sizes and counting files
    only (not subdirs). For files: file_count is 0 (a single file isn't
    really a "count" — the prompt reads the path itself for context).
    Missing or unreadable entries return zeroes so probing never raises.
    """
    try:
        if not path.exists():
            return False, 0, 0
        if path.is_file():
            return True, path.stat().st_size, 0
        # Directory: recursive walk.
        total_size = 0
        file_count = 0
        for sub in path.rglob("*"):
            try:
                if sub.is_file():
                    total_size += sub.stat().st_size
                    file_count += 1
            except OSError:
                # Broken symlink / permission error — skip but keep counting.
                continue
        return True, total_size, file_count
    except OSError as exc:
        logger.warning("Path probe failed for %s: %s", path, exc)
        return False, 0, 0


def _make_target(label: str, path: Path) -> ResetTarget:
    """Build a ResetTarget by probing the filesystem."""
    exists, size, count = _probe_path(path)
    return ResetTarget(
        label=label,
        path=path,
        size_bytes=size,
        file_count=count,
        exists=exists,
    )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _state_db_path(config: Any) -> Path:
    """Resolve the state DB path from config (already expanded by Config)."""
    return Path(config.state_db.path)


def _chroma_persist_dir(config: Any) -> Path:
    """Resolve the production ChromaDB persist dir.

    Mirrors ``_make_embeddings_db()`` in scripts/cli.py — chroma lives
    alongside the state DB so a user override of state_db.path automatically
    co-locates the persist dir. Do NOT hard-code ~/.config/lit-monitor/chroma.
    """
    return _state_db_path(config).parent / "chroma"


def _project_config_dir() -> Path:
    """Resolve the in-repo ``config/`` directory (auto-generated files live here)."""
    # Module is at scripts/setup/reset.py; project root is two parents up.
    return Path(__file__).resolve().parent.parent.parent / "config"


# ---------------------------------------------------------------------------
# Target enumeration
# ---------------------------------------------------------------------------
def state_targets(config: Any) -> list[ResetTarget]:
    """Enumerate every path ``reset state`` would delete.

    Order matters for the UI — list durable runtime state first (state DB,
    chroma), then auto-generated config drafts. The sqlite WAL/SHM siblings
    are not listed separately to keep the prompt readable; they are removed
    automatically alongside the main file in ``perform_state_reset``.
    """
    state_db = _state_db_path(config)
    chroma = _chroma_persist_dir(config)
    cfg_dir = _project_config_dir()
    return [
        _make_target("state DB", state_db),
        _make_target("ChromaDB persist dir", chroma),
        _make_target("concepts_draft.yaml", cfg_dir / "concepts_draft.yaml"),
        _make_target(
            "concepts_draft_refinement_raw.txt",
            cfg_dir / "concepts_draft_refinement_raw.txt",
        ),
        _make_target("topics_suggested.yaml", cfg_dir / "topics_suggested.yaml"),
    ]


def vault_targets(config: Any) -> list[ResetTarget]:
    """Enumerate every path ``reset vault`` would delete.

    Returns one target per lit-monitor-generated content folder. We read
    each folder name from ``config.obsidian.*_folder`` so user overrides in
    paths.yaml are honoured. The dev sandbox subfolder (``Literature/_Dev``)
    and the vault root theme pages are deliberately excluded — see module
    docstring.

    Reviews are co-located with regular papers: ``obsidian_writer`` writes
    both ``source_type='paper'`` and ``source_type='review'`` notes into
    ``papers_folder`` (see ``scripts/output/obsidian_writer.py``), so the
    Papers folder entry covers both. There is no separate Reviews target.

    ``books_folder`` is intentionally omitted: books/bookSections are
    routed to ``pipeline: "skip"`` in ``config/item_routing.yaml`` and are
    not written by any production pipeline. Any markdown in that folder is
    user-curated and outside lit-monitor's scope.

    Synthesis notes live in ``connections_folder`` (default
    ``Literature/Connections``) — see ``scripts/obsidian_tools/synthesize.py``.
    """
    vault_root = Path(config.obsidian.vault_path).expanduser()
    return [
        _make_target(
            "Papers folder",
            vault_root / getattr(config.obsidian, "papers_folder", "Literature/Papers"),
        ),
        _make_target(
            "Digests folder",
            vault_root / getattr(config.obsidian, "digests_folder", "Literature/Digests"),
        ),
        _make_target(
            "Synthesis folder",
            vault_root
            / getattr(config.obsidian, "connections_folder", "Literature/Connections"),
        ),
    ]


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
def _delete_file(path: Path) -> None:
    """Remove a single file. Missing → no-op via missing_ok."""
    path.unlink(missing_ok=True)


def _delete_directory(path: Path) -> None:
    """Remove a directory tree. Missing → no-op."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=False)


def _delete_state_db(path: Path) -> bool:
    """Remove state.db plus its SQLite WAL/SHM siblings if present.

    Returns True iff at least one file was actually unlinked. The WAL/SHM
    pair only exists while the DB is open with WAL journaling; treat both
    as best-effort siblings of the main file.
    """
    deleted_any = False
    if path.exists():
        path.unlink()
        deleted_any = True
    for suffix in ("-wal", "-shm"):
        sib = path.with_name(path.name + suffix)
        if sib.exists():
            sib.unlink()
            deleted_any = True
    return deleted_any


def _clear_chromadb_cache() -> None:
    """Evict chromadb's per-path System cache so a new client opens cleanly.

    chromadb caches ``PersistentClient`` system objects keyed by persist
    path. After ``rmtree``ing the persist dir, the next constructor would
    otherwise return a stale handle whose sqlite file has been deleted —
    writes then fail with SQLITE_READONLY_DBMOVED (code 1032). Same pattern
    as ``dev_sandbox.clear_sandbox``. Best-effort: chromadb may not be
    installed (test env) or may not expose the cache helper on older
    versions; both are no-ops here.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception as exc:  # noqa: BLE001 — defensive, varies by chromadb version
        logger.debug("chromadb system-cache reset skipped: %s", exc)


def _delete_directory_contents(path: Path, pattern: str = "*.md") -> int:
    """Delete every file matching ``pattern`` under ``path`` recursively.

    Returns the count of deleted files. The directory itself is preserved
    (Obsidian users may have folder-level metadata or other tooling that
    expects the folder to exist). Empty subdirectories left behind are also
    preserved.
    """
    if not path.exists():
        return 0
    count = 0
    for f in path.rglob(pattern):
        if f.is_file():
            try:
                f.unlink()
                count += 1
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", f, exc)
    return count


def perform_state_reset(targets: list[ResetTarget]) -> list[ResetResult]:
    """Delete every state target. Missing paths are silently skipped.

    The deletion strategy varies by target label:
      - "state DB"            → unlink file + WAL/SHM siblings
      - "ChromaDB persist dir" → rmtree, then clear chromadb's system cache
      - everything else        → unlink as a regular file

    After the chroma rmtree we ALWAYS call ``_clear_chromadb_cache()`` —
    even if the persist dir didn't exist — because a prior in-process
    EmbeddingsDB may have populated chromadb's cache against the now-missing
    path. The cost is a single dict.clear on the chromadb side.
    """
    results: list[ResetResult] = []
    chroma_seen = False

    for tgt in targets:
        if not tgt.exists:
            results.append(
                ResetResult(
                    label=tgt.label,
                    path=tgt.path,
                    deleted=False,
                    skipped_reason="not present",
                )
            )
            # Even when the chroma dir is absent, evict the cache in case a
            # previous in-process client cached the path.
            if tgt.label == "ChromaDB persist dir":
                chroma_seen = True
            continue

        try:
            if tgt.label == "state DB":
                _delete_state_db(tgt.path)
            elif tgt.label == "ChromaDB persist dir":
                _delete_directory(tgt.path)
                chroma_seen = True
            else:
                _delete_file(tgt.path)
            results.append(
                ResetResult(label=tgt.label, path=tgt.path, deleted=True)
            )
        except OSError as exc:
            logger.error("Failed to delete %s (%s): %s", tgt.label, tgt.path, exc)
            raise

    # Always clear chromadb's system cache when chroma was in the target set,
    # regardless of whether the persist dir existed — see _clear_chromadb_cache
    # docstring for the stale-handle rationale.
    if chroma_seen:
        _clear_chromadb_cache()

    return results


def perform_vault_reset(targets: list[ResetTarget]) -> list[ResetResult]:
    """Delete every vault target's markdown content.

    Each target is a folder; we recursively remove ``*.md`` files inside it
    but leave the folder itself in place so downstream pipelines that
    auto-create + write into the folder don't have to special-case its
    absence. The dev sandbox subfolder and theme pages at the vault root
    are excluded at the target-enumeration layer, not here.
    """
    results: list[ResetResult] = []
    for tgt in targets:
        if not tgt.exists:
            results.append(
                ResetResult(
                    label=tgt.label,
                    path=tgt.path,
                    deleted=False,
                    skipped_reason="not present",
                )
            )
            continue
        deleted_count = _delete_directory_contents(tgt.path, pattern="*.md")
        results.append(
            ResetResult(
                label=tgt.label,
                path=tgt.path,
                deleted=deleted_count > 0,
                skipped_reason=(
                    None if deleted_count > 0 else "no .md files present"
                ),
            )
        )
    return results


__all__ = [
    "ResetTarget",
    "ResetResult",
    "state_targets",
    "vault_targets",
    "perform_state_reset",
    "perform_vault_reset",
    "_format_size_bytes",
]
