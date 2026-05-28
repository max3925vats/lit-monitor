"""Shared path utilities used across the codebase.

Kept under :mod:`scripts.core` (not under any domain package) so that lightweight
consumers — e.g. the web server's config loader — can resolve config files
without dragging in heavier LLM-domain imports.

Public API:
  resolve_path(path) -> Path
"""
from __future__ import annotations

from pathlib import Path


def resolve_path(path: Path) -> Path:
    """Resolve a config path relative to cwd, then ancestor directories.

    Falls back to a sibling ``*.example.yaml`` if the real file is missing.
    This keeps the package working on a fresh clone (CI, new contributors)
    before ``install.sh`` has seeded the personal configs.

    Args:
        path: Relative config path (e.g. ``Path("config/prompts/rationale.yaml")``).

    Returns:
        An existing absolute (or cwd-relative) path — either the requested file,
        or its ``.example.yaml`` fallback if the real file is absent.

    Raises:
        FileNotFoundError: If neither the requested file nor its example
            fallback can be located in cwd or any ancestor of this module.
    """
    candidates = [path]
    if path.suffix == ".yaml" and not path.name.endswith(".example.yaml"):
        candidates.append(path.with_suffix(".example.yaml"))

    if path.exists():
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in candidates:
            full = parent / cand
            if full.exists():
                return full
    raise FileNotFoundError(
        f"Cannot find {path} — searched cwd and ancestors of {here}"
    )
