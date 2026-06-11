"""First-run seeding of the user config dir from packaged example configs.

A pip-installed wheel ships its example configs as package data under
``lit_monitor._data.config_examples``. On first run we copy each
``<name>.example.yaml`` to ``<target>/<name>.yaml`` so the user has an editable
starting point in ``~/.config/lit-monitor/``. Seeding is idempotent and never
clobbers a file the user already has.
"""
from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path

logger = logging.getLogger(__name__)

_EXAMPLES_PACKAGE = "lit_monitor._data.config_examples"


def seed_user_config(target: Path) -> list[Path]:
    """Copy packaged ``*.example.yaml`` files into ``target`` as ``*.yaml``.

    For every ``<name>.example.yaml`` shipped under
    :data:`_EXAMPLES_PACKAGE`, write ``<target>/<name>.yaml`` — but only when
    that destination does not already exist (never clobber a user's edits).

    Args:
        target: The config directory to seed (created if absent), typically
            ``~/.config/lit-monitor``.

    Returns:
        The list of destination paths that were newly written this call.
        Empty when every file already existed (idempotent re-run).
    """
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    examples_root = files(_EXAMPLES_PACKAGE)
    for resource in examples_root.iterdir():
        name = resource.name
        # Only seed the top-level *.example.yaml templates; skip the nested
        # examples/ domain directories and anything non-example.
        if not (resource.is_file() and name.endswith(".example.yaml")):
            continue
        dest_name = name.replace(".example.yaml", ".yaml")
        dest = target / dest_name
        if dest.exists():
            # Never clobber an existing user file.
            continue
        dest.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(dest)
        logger.info("Seeded %s", dest)

    return written
