"""Shared atomic-write helper for crash-safe file writes.

Writes content to a sibling temp file in the target's parent directory, fsyncs
it, then ``os.replace()`` it into place.  Because the temp file lives next to
the target it stays on the same filesystem, which is required for ``os.replace``
to be atomic on POSIX.  On any failure the temp file is unlinked before the
exception propagates so we do not leak ``.tmp`` droppings.

This is the canonical helper — :mod:`scripts.server.config_io` predates it and
keeps its own private copy for now; new call sites should use this one.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> Path:
    """Atomically write ``content`` to ``path``.

    Parameters
    ----------
    path:
        Destination file path. Parent directory must already exist — the caller
        is responsible for ``mkdir(parents=True, exist_ok=True)`` if needed.
    content:
        Text to write.
    encoding:
        File encoding (default ``"utf-8"``).

    Returns
    -------
    Path
        The final destination path (same as ``path``).

    Raises
    ------
    OSError
        Any I/O error from the write or rename is re-raised after best-effort
        cleanup of the temp file.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup so a failed write does not leak a .tmp sibling.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
