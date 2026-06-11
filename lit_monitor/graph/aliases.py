"""G2: alias loader for the entity normalizer.

Reads config/entity_aliases.yaml (preferred) or falls back to
config/entity_aliases.example.yaml. Returns a per-type dict keyed by
lower-cased surface form so the normalizer can do O(1) lookup.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Paths are resolved relative to the project root (one level up from scripts/).
_HERE = Path(__file__).parent  # scripts/graph/
_PROJECT_ROOT = _HERE.parent.parent  # lit-monitor/

_DEFAULT_REAL = _PROJECT_ROOT / "config" / "entity_aliases.yaml"
_DEFAULT_EXAMPLE = _PROJECT_ROOT / "config" / "entity_aliases.example.yaml"


def load_aliases(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load alias dictionary keyed by entity type.

    Args:
        path: Explicit YAML file to load.  When *None*, tries
              ``config/entity_aliases.yaml`` first, then
              ``config/entity_aliases.example.yaml``.

    Returns:
        ``{type_name: {lower(surface): canonical}}``.
        Empty dict (+ WARNING logged) when no usable file is found.
    """
    if path is None:
        # Prefer the real (user-edited) file; fall back to the shipped example.
        for candidate in (_DEFAULT_REAL, _DEFAULT_EXAMPLE):
            if candidate.exists():
                path = candidate
                break
        else:
            logger.warning("No entity_aliases.yaml found; returning empty aliases.")
            return {}

    if not Path(path).exists():
        logger.warning("Entity aliases file not found: %s", path)
        return {}

    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    out: dict[str, dict[str, str]] = {}
    for type_, mapping in raw.items():
        if not isinstance(mapping, dict):
            # Malformed YAML row (e.g. a type mapped to a scalar/list instead of
            # a surface→canonical dict). Warn so the bad entry is observable
            # rather than silently dropped, then skip it.
            logger.warning(
                "Skipping malformed alias entry %r in %s: expected a mapping, got %s.",
                type_, path, type(mapping).__name__,
            )
            continue
        # Lowercase the surface-form keys so lookup is always case-insensitive.
        out[type_] = {str(k).lower(): str(v) for k, v in mapping.items()}
    return out
