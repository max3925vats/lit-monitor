"""Filesystem browser endpoints for the wizard's vault picker.

The web UI is bound to localhost, but as defence-in-depth these endpoints
also refuse any request for a path outside the current user's ``$HOME``
directory. ``~`` is expanded, symlinks are resolved, and ``..`` traversal
is collapsed before the check, so a symlink pointing outside ``$HOME``
will be rejected after resolution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/fs", tags=["fs"])


def _normalize(raw: str) -> Path:
    """Expand ``~``, resolve symlinks/``..``, and verify the path stays inside ``$HOME``.

    We deliberately use ``resolve(strict=False)`` because callers may ask
    about non-existent paths (the ``/exists`` endpoint must work for them).
    Raises ``HTTPException(403)`` if the resolved path escapes ``$HOME``.
    """

    home = Path.home().resolve()
    candidate = Path(raw).expanduser().resolve(strict=False)
    # Allow exactly $HOME and any descendant; refuse everything else.
    if candidate != home and home not in candidate.parents:
        raise HTTPException(status_code=403, detail="path outside HOME")
    return candidate


@router.get("/ls")
def ls(
    path: str = Query(..., description="Absolute or ~-relative directory path."),
) -> dict:
    """List the entries of a directory inside ``$HOME``.

    Directories sort lexicographically first, then files/symlinks/other.
    Hidden entries (names beginning with ``.``) are included because users
    legitimately need to navigate into ``~/.config``, ``~/Library``, etc.
    """

    target = _normalize(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"not found: {target}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {target}")

    entries = []
    for child in target.iterdir():
        # is_symlink() must be checked before is_dir/is_file because those
        # follow symlinks and would hide the symlink-ness from callers.
        try:
            if child.is_symlink():
                etype: Literal["dir", "file", "symlink", "other"] = "symlink"
            elif child.is_dir():
                etype = "dir"
            elif child.is_file():
                etype = "file"
            else:
                etype = "other"
        except OSError:
            # Race: child was removed between iterdir() and the stat call.
            etype = "other"
        entries.append(
            {"name": child.name, "type": etype, "full_path": str(child)}
        )

    # Dirs first (lex), everything else after (lex). Case-insensitive sort
    # so "Applications" and "applications" interleave naturally.
    entries.sort(key=lambda e: (0 if e["type"] == "dir" else 1, e["name"].lower()))

    home = Path.home().resolve()
    # Don't expose the path above $HOME — at the HOME root, parent is null.
    parent: str | None = str(target.parent) if target != home else None
    return {"path": str(target), "parent": parent, "entries": entries}


@router.get("/exists")
def exists(
    path: str = Query(..., description="Absolute or ~-relative path."),
) -> dict:
    """Report existence, directory-ness, and writability of a path."""

    target = _normalize(path)
    if not target.exists():
        return {"exists": False, "is_dir": False, "is_writable": False}
    return {
        "exists": True,
        "is_dir": target.is_dir(),
        "is_writable": os.access(target, os.W_OK),
    }
