"""G2: alias loader for the entity normalizer.

Resolves the real ``entity_aliases.yaml`` through the three-tier config
resolver (``config_dir()``) so a pip-installed wheel finds the user's file in
``~/.config/lit-monitor/``. Falls back to the *packaged* example shipped under
``lit_monitor._data.config_examples`` (so it works with no repo present).
Returns a per-type dict keyed by lower-cased surface form for O(1) lookup.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from lit_monitor.core.config import config_dir

logger = logging.getLogger(__name__)


def _load_packaged_example() -> dict[str, Any] | None:
    """Read the packaged ``entity_aliases.example.yaml`` via importlib.resources.

    Returns the parsed YAML dict, or ``None`` when the packaged example is not
    available (e.g. running before the example has been relocated into the
    package, in which case the repo ``config/`` copy is tried as a fallback).
    """
    try:
        from importlib.resources import files

        example = files("lit_monitor._data.config_examples") / "entity_aliases.example.yaml"
        if example.is_file():
            return yaml.safe_load(example.read_text(encoding="utf-8")) or {}
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    # Dev fallback: the repo copy that has not yet been relocated into the package.
    repo_example = config_dir() / "entity_aliases.example.yaml"
    if repo_example.exists():
        return yaml.safe_load(repo_example.read_text(encoding="utf-8")) or {}
    return None


def load_aliases(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load alias dictionary keyed by entity type.

    Args:
        path: Explicit YAML file to load.  When *None*, tries the real
              ``entity_aliases.yaml`` in the resolved config dir first, then
              falls back to the packaged ``entity_aliases.example.yaml``.

    Returns:
        ``{type_name: {lower(surface): canonical}}``.
        Empty dict (+ WARNING logged) when no usable file is found.
    """
    raw: dict[str, Any]
    if path is None:
        # Prefer the real (user-edited) file in the resolved config dir.
        real = config_dir() / "entity_aliases.yaml"
        if real.exists():
            path = real
        else:
            # Fall back to the packaged example (no real file present).
            example_raw = _load_packaged_example()
            if example_raw is None:
                logger.warning("No entity_aliases.yaml found; returning empty aliases.")
                return {}
            raw = example_raw
            return _index_aliases(raw, "entity_aliases.example.yaml (packaged)")

    if not Path(path).exists():
        logger.warning("Entity aliases file not found: %s", path)
        return {}

    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _index_aliases(raw, str(path))


def _index_aliases(raw: dict[str, Any], source: str) -> dict[str, dict[str, str]]:
    """Build the ``{type: {lower(surface): canonical}}`` index from raw YAML."""
    out: dict[str, dict[str, str]] = {}
    out: dict[str, dict[str, str]] = {}
    for type_, mapping in raw.items():
        if not isinstance(mapping, dict):
            # Malformed YAML row (e.g. a type mapped to a scalar/list instead of
            # a surface→canonical dict). Warn so the bad entry is observable
            # rather than silently dropped, then skip it.
            logger.warning(
                "Skipping malformed alias entry %r in %s: expected a mapping, got %s.",
                type_, source, type(mapping).__name__,
            )
            continue
        # Lowercase the surface-form keys so lookup is always case-insensitive.
        out[type_] = {str(k).lower(): str(v) for k, v in mapping.items()}
    return out
