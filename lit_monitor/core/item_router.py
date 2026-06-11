"""
Item routing — maps Zotero item types to pipeline, extraction schema, and source_type.

Routing decisions for every Zotero item type live in config/item_routing.yaml
rather than scattered if/elif chains in brain_build.py and discovery.py.

Public API
----------
ItemRoute        — Pydantic model for one item-type routing entry.
load_item_routes() — Load and cache routing table from YAML.
get_route(item_type) — Look up routing for a specific item type; raises on unknown types.
detect_review(s2_types, title, abstract, journal) — True when a paper is a review (H4).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# H4 — review detection constants
# ---------------------------------------------------------------------------
# Phrases that unambiguously identify a review article in title or abstract.
_REVIEW_KEYWORDS: frozenset[str] = frozenset({
    "systematic review",
    "meta-analysis",
    "scoping review",
    "narrative review",
    "literature review",
})
# Journal name substrings that strongly suggest a review journal.
_REVIEW_JOURNAL_SUBSTRINGS: frozenset[str] = frozenset({
    "reviews",
    "annual review",
})

def _resolve_item_routing_path() -> Path:
    """Locate item_routing.yaml in a CWD/install-independent way.

    The old ``Path(__file__).parent.parent.parent / "config"`` default pointed
    at site-packages in a pip-installed wheel, so the file was never found.
    Resolution order:

    1. ``<config_dir()>/item_routing.yaml`` — dev (repo ./config) or a user
       override under ~/.config/lit-monitor.
    2. The packaged ``lit_monitor/_data/item_routing.yaml`` (ships in the wheel).
    """
    from lit_monitor.core.config import config_dir

    user_path = config_dir() / "item_routing.yaml"
    if user_path.exists():
        return user_path
    from importlib.resources import files

    return Path(str(files("lit_monitor._data") / "item_routing.yaml"))

Pipeline = Literal["brain_build", "skip"]
# After R-10 (textbook-build removed), only "paper" is a live schema.
# "review" is assigned dynamically by detect_review() at runtime, not in YAML.
# Skipped item types (book, thesis, bookSection, report) carry default_schema: "paper"
# as a placeholder — unused because they short-circuit on pipeline == "skip".
Schema = Literal["paper"]


class ItemRoute(BaseModel):
    """Routing configuration for one Zotero item type.

    Attributes
    ----------
    pipeline:
        Which pipeline processes this item type.
        ``"skip"`` means the item is logged to the run report and not extracted.
    default_schema:
        Extraction schema to use before detect_review() runs (H4).
        ``"review"`` is never the default — it is assigned dynamically at runtime.
    source_type:
        Value written to ``papers.source_type`` in the state DB.
    """
    pipeline: Pipeline
    default_schema: Schema
    source_type: str

    @field_validator("source_type")
    @classmethod
    def source_type_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source_type must be a non-empty string")
        return v


@lru_cache(maxsize=1)
def load_item_routes(config_path: Path | None = None) -> dict[str, ItemRoute]:
    """Load item routing table from YAML and return as {item_type: ItemRoute}.

    Results are cached after first load — identical to the extraction_schema.py
    singleton pattern.  Pass a different ``config_path`` in tests to inject a
    temporary YAML without modifying the production file.  When ``None``, the
    path is resolved via :func:`_resolve_item_routing_path` (wheel-safe).
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for item routing") from exc

    if config_path is None:
        config_path = _resolve_item_routing_path()

    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    routes_raw: dict = raw.get("routes", {})
    if not routes_raw:
        raise ValueError(f"item_routing.yaml at {config_path} has no 'routes' key")

    routes: dict[str, ItemRoute] = {}
    for item_type, entry in routes_raw.items():
        try:
            routes[item_type] = ItemRoute(**entry)
        except Exception as exc:
            raise ValueError(
                f"Invalid item_routing.yaml entry for {item_type!r}: {exc}"
            ) from exc

    logger.debug("Loaded item routes for %d item types", len(routes))
    return routes


def get_route(item_type: str, config_path: Path | None = None) -> ItemRoute:
    """Look up routing for a specific Zotero item type.

    Raises
    ------
    KeyError
        When ``item_type`` is not in the routing table.  This is intentional —
        unknown item types should surface as an explicit error rather than
        silently falling back to a wrong pipeline.
    """
    routes = load_item_routes(config_path)
    if item_type not in routes:
        raise KeyError(
            f"Unknown Zotero item type: {item_type!r}. "
            f"Known types: {sorted(routes.keys())}. "
            "Add an entry to config/item_routing.yaml to route this type."
        )
    return routes[item_type]


def detect_review(
    s2_types: list[str] | None,
    title: str,
    abstract: str,
    journal: str,
) -> bool:
    """Detect whether a paper is a review article.

    Priority order (conservative — returns False when uncertain):
    1. S2 publicationTypes contains "Review" (authoritative classification).
    2. Title or abstract contains a review keyword phrase.
    3. Journal name contains a review-journal substring.

    Parameters
    ----------
    s2_types:
        List of Semantic Scholar publicationTypes strings, or None if S2 enrichment
        was not available (treated as empty).
    title:
        Paper title (case-insensitive matching applied).
    abstract:
        Paper abstract (case-insensitive matching applied).
    journal:
        Publication venue name (case-insensitive matching applied).

    Returns
    -------
    bool
        True when the paper is confidently identified as a review/meta-analysis.
        False when uncertain — conservative by design to avoid misrouting.
    """
    # Priority 1: S2 authoritative classification
    if s2_types:
        for t in s2_types:
            if t.lower() == "review":
                logger.debug("detect_review: S2 classification 'Review' detected")
                return True

    # Priority 2: keyword in title or abstract
    combined = (title + " " + abstract).lower()
    for kw in _REVIEW_KEYWORDS:
        if kw in combined:
            logger.debug("detect_review: keyword %r found in title/abstract", kw)
            return True

    # Priority 3: journal name substring
    journal_lower = journal.lower()
    for substr in _REVIEW_JOURNAL_SUBSTRINGS:
        if substr in journal_lower:
            logger.debug("detect_review: journal substring %r matched in %r", substr, journal)
            return True

    return False
