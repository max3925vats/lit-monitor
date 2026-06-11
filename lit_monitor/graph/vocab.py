"""G15: closed-vocabulary loader for predicates and entity types.

PredicateVocab and EntityTypeVocab share a ``_BaseVocab`` implementation and
``_Entry`` dataclass. Both support:

- ``aliases``: collapse multiple surface forms to a canonical name at extract
  time, so future predicate renames (e.g. EXTENDS ↔ BUILDS_ON) don't require
  an edge rebuild.
- ``deprecated``: the name still resolves to its canonical, but a one-shot
  WARNING is logged per deprecated name per process lifetime (module-level
  ``_WARNED`` set). This lets v0.8+ stage a rename across releases without
  silently dropping edges.

Downstream consumers:
- G3 (entity_extractor.py) will call ``EntityTypeVocab.resolve``
- G4 (relationship_validator.py) will call ``PredicateVocab.resolve``

NOTE (intentionally unwired): this is a foundation module for the v0.8+
closed-vocabulary feature and is deliberately NOT yet wired into the active
pipeline — no production code imports it today. It is kept on purpose (user
decision) as the staging point for the planned predicate/entity-type
resolution work; it is NOT dead code awaiting removal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import yaml

logger = logging.getLogger(__name__)

# Module-level one-shot warning cache.  Shared across PredicateVocab and
# EntityTypeVocab — one WARNING per deprecated name per process lifetime is
# enough.  Tests that exercise deprecation must call ``_WARNED.clear()``
# before each case to ensure deterministic isolation.
_WARNED: set[str] = set()


@dataclass(frozen=True)
class _Entry:
    """A single vocab entry: canonical name plus optional metadata."""

    canonical: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    deprecated: bool = False
    description: str = ""


class _BaseVocab:
    """Shared YAML loader and surface-form resolver for closed vocabularies.

    Subclasses set ``_KIND`` to differentiate log messages (e.g. "predicate"
    vs "entity type").
    """

    _KIND: ClassVar[str] = "vocab"

    def __init__(self, entries: list[_Entry]) -> None:
        self._entries = entries
        # surface form → canonical name (includes canonical → canonical)
        self._lookup: dict[str, str] = {}
        # canonical name → entry (for deprecated check)
        self._by_canonical: dict[str, _Entry] = {}

        for e in entries:
            self._lookup[e.canonical] = e.canonical
            self._by_canonical[e.canonical] = e
            for alias in e.aliases:
                # Aliases silently overwrite each other only if duplicate —
                # not expected in well-formed YAML, but we prefer last-write
                # over crashing.
                self._lookup[alias] = e.canonical

    @classmethod
    def load(cls, path: Path | str) -> _BaseVocab:
        """Load vocab from a YAML file.

        Args:
            path: Path to a YAML file whose root is a mapping of
                  ``CANONICAL_NAME -> {aliases: [...], deprecated: bool, ...}``.

        Returns:
            A new vocab instance backed by the parsed entries.

        Raises:
            ValueError: If the YAML root is not a mapping.
        """
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"{path}: root must be a YAML mapping, got {type(raw).__name__}"
            )
        entries: list[_Entry] = []
        for canonical, body in raw.items():
            if not isinstance(body, dict):
                # Tolerate bare scalars but warn — keeps load() non-fatal.
                logger.warning(
                    "%s: entry %r is not a mapping (got %s); skipping.",
                    path.name,
                    canonical,
                    type(body).__name__,
                )
                continue
            entries.append(
                _Entry(
                    canonical=str(canonical),
                    aliases=tuple(str(a) for a in (body.get("aliases") or [])),
                    deprecated=bool(body.get("deprecated", False)),
                    description=str(body.get("description", "")),
                )
            )
        return cls(entries)

    def resolve(self, name: str) -> str | None:
        """Resolve a surface form to its canonical name.

        Args:
            name: A predicate or entity-type string, which may be a canonical
                  name, an alias, or an unknown string.

        Returns:
            The canonical name if ``name`` is known (directly or via alias);
            ``None`` if unknown.

        Side-effect:
            Logs a one-shot WARNING if the resolved canonical is marked
            ``deprecated``. The warning is suppressed for all subsequent calls
            to any vocab instance in the same process (keyed on ``name``).
        """
        canonical = self._lookup.get(name)
        if canonical is None:
            return None

        entry = self._by_canonical.get(canonical)
        if entry is not None and entry.deprecated and name not in _WARNED:
            logger.warning(
                "%s %r is deprecated; resolved to canonical %r. "
                "Update upstream extractors to emit %r directly.",
                self._KIND,
                name,
                canonical,
                canonical,
            )
            _WARNED.add(name)

        return canonical

    def canonicals(self) -> list[str]:
        """Return all canonical names declared in this vocabulary."""
        return list(self._by_canonical.keys())


class PredicateVocab(_BaseVocab):
    """Closed vocabulary for relationship predicates (MENTIONS, CITES, …)."""

    _KIND = "predicate"


class EntityTypeVocab(_BaseVocab):
    """Closed vocabulary for entity types (topic, method, material, …)."""

    _KIND = "entity type"
