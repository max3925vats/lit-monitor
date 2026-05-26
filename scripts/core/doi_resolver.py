"""
D2 — CrossRef DOI resolution via habanero.

Attempts to find a DOI for Zotero items that have no DOI field by querying
the CrossRef API using the paper title and author names.  Called from
brain_build and discovery ingestion *before* skipping a no-DOI item.

Only returns a DOI when the top CrossRef result exceeds ``min_score``
(default 70.0).  This is deliberately conservative — false positives (wrong
paper matched) are worse than false negatives (item stays skipped).

CrossRef polite pool
--------------------
Providing a contact email routes requests through CrossRef's polite pool,
which has higher rate limits and better reliability.  Set the environment
variable ``LIT_MONITOR_MAILTO`` to your email, or pass ``mailto=`` explicitly
to ``resolve_doi()``.  Without a mailto the API still works but may be
rate-limited more aggressively.
"""
from __future__ import annotations

import logging
import os

from habanero import Crossref

from scripts.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)

# Read from environment so tests and CI don't need to hard-code credentials.
_DEFAULT_MAILTO: str | None = os.environ.get("LIT_MONITOR_MAILTO")


def resolve_doi(
    title: str,
    authors: list[str],
    year: str | int | None = None,
    *,
    min_score: float = 70.0,
    mailto: str | None = _DEFAULT_MAILTO,
) -> str | None:
    """Attempt to resolve a DOI from title + authors via the CrossRef API.

    Parameters
    ----------
    title : str
        Full paper title.  Empty string always returns None immediately.
    authors : list[str]
        Author display names in "LastName, FirstName" format.
        Up to 3 last names are used as the ``query.author`` filter.
    year : str | int | None
        Publication year — currently logged only (CrossRef scoring already
        weights recency; filtering by year risks excluding valid matches).
    min_score : float
        Minimum CrossRef relevance score for the top result to be accepted.
        70.0 is a conservative threshold suitable for full-title queries.
        Lower it cautiously — CrossRef scores are not normalised across queries.
    mailto : str | None
        Contact email for the CrossRef polite pool.  Reads
        ``LIT_MONITOR_MAILTO`` env var by default.

    Returns
    -------
    str | None
        Resolved DOI string (lower-case, as returned by CrossRef), or None
        when resolution fails, score is too low, or title is empty.
    """
    if not title or not title.strip():
        return None

    if year is not None:
        logger.debug("resolve_doi: title=%r year=%s", title[:60], year)

    cr = Crossref(mailto=mailto) if mailto else Crossref()

    # CrossRef query.author accepts comma-separated last names; use up to 3
    # to keep the query specific without over-constraining it.
    author_str: str | None = None
    if authors:
        last_names = [a.split(",")[0].strip() for a in authors[:3] if a]
        author_str = ", ".join(last_names) if last_names else None

    try:
        result = cr.works(
            query_bibliographic=title.strip(),
            query_author=author_str,
            limit=3,
        )
    except Exception as exc:
        strict_fallback(
            logger,
            f"CrossRef API error resolving DOI for {title[:80]!r}: {exc}",
            exc,
        )
        return None

    try:
        items: list[dict] = result.get("message", {}).get("items", [])
    except (AttributeError, TypeError) as exc:
        strict_fallback(
            logger,
            f"CrossRef returned unexpected response for {title[:80]!r}: {exc}",
            exc,
        )
        return None

    if not items:
        logger.debug("CrossRef: no results for %r", title[:80])
        return None

    top = items[0]
    score = float(top.get("score", 0.0))
    doi = (top.get("DOI") or "").strip()

    if score < min_score:
        logger.debug(
            "CrossRef: score %.1f below threshold %.1f for %r — skipping",
            score,
            min_score,
            title[:80],
        )
        return None

    if not doi:
        logger.debug("CrossRef: top result has no DOI for %r", title[:80])
        return None

    logger.info(
        "CrossRef resolved DOI (score %.1f): %s  for %r",
        score,
        doi,
        title[:80],
    )
    return doi
