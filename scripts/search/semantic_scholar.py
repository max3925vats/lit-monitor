"""
D3 — Semantic Scholar integration.

Two entry points:

(a) ``enrich_paper(doi)`` — metadata enrichment for a known DOI.
    Fetches citation counts, influential citation count, S2 paper ID, and
    open-access PDF URL.  Returns a dict of ``s2_*`` keys to be merged into
    the paper's ``extraction_json``.

(b) ``search_semantic_scholar(query, since_days, limit)`` — discovery search.
    Returns paper dicts in pipeline format (same schema as findpapers results)
    so they can be merged with findpapers results and passed through
    ``filter_known_dois → rank_papers`` unchanged.

API key
-------
Set ``S2_API_KEY`` in the shell environment for higher rate limits (currently
100 requests/5-minute window without a key; 1 request/second with a key).
Without a key the API still works but is more aggressively rate-limited.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import date, timedelta
from typing import Any

from scripts.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)

_DEFAULT_S2_API_KEY: str | None = os.environ.get("S2_API_KEY")

try:
    from semanticscholar import SemanticScholar as _SemanticScholar
    _S2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SemanticScholar = None  # type: ignore[assignment]
    _S2_AVAILABLE = False

# Minimal field list for enrichment — avoids fetching massive citation/reference lists.
# publicationTypes added for H3: values like ["JournalArticle"], ["Review"], ["Conference"]
# are used by H4 detect_review() to decide schema before extraction starts.
_ENRICH_FIELDS = [
    "paperId",
    "citationCount",
    "influentialCitationCount",
    "openAccessPdf",
    "publicationTypes",
]

# Fields for discovery search results.
_SEARCH_FIELDS = [
    "title",
    "authors",
    "year",
    "externalIds",
    "publicationVenue",
    "abstract",
    "fieldsOfStudy",
]


def enrich_paper(
    doi: str,
    *,
    api_key: str | None = _DEFAULT_S2_API_KEY,
) -> dict[str, Any]:
    """Fetch Semantic Scholar metadata for a known DOI.

    Parameters
    ----------
    doi : str
        DOI string (lower-case, no https:// prefix).
    api_key : str | None
        Semantic Scholar API key.  Reads ``S2_API_KEY`` env var by default.

    Returns
    -------
    dict
        Zero or more of:  ``s2_paper_id``, ``s2_citation_count``,
        ``s2_influential_citations``, ``s2_open_access_pdf`` (URL string).
        Returns ``{}`` on any error (logged as WARNING, never raised).
    """
    if not _S2_AVAILABLE:
        logger.debug("semanticscholar package not available — skipping enrichment")
        return {}
    if not doi or not doi.strip():
        return {}

    sch = _SemanticScholar(api_key=api_key)
    try:
        paper = sch.get_paper(f"DOI:{doi.strip()}", fields=_ENRICH_FIELDS)
    except Exception as exc:
        strict_fallback(
            logger,
            f"S2 enrichment failed for DOI {doi}: {exc}. "
            "s2_citation_count and related fields will be missing from this paper.",
            exc,
        )
        return {}

    if paper is None:
        logger.debug("S2: no result found for DOI %s", doi)
        return {}

    result: dict[str, Any] = {}
    if paper.paperId:
        result["s2_paper_id"] = paper.paperId
    if paper.citationCount is not None:
        result["s2_citation_count"] = int(paper.citationCount)
    if paper.influentialCitationCount is not None:
        result["s2_influential_citations"] = int(paper.influentialCitationCount)
    if paper.openAccessPdf and isinstance(paper.openAccessPdf, dict):
        oa_url = paper.openAccessPdf.get("url") or ""
        result["s2_open_access_pdf"] = oa_url
    # H3: publicationTypes — list of strings like ["JournalArticle"], ["Review"].
    # Used by H4 detect_review() before extraction to decide schema routing.
    pub_types = getattr(paper, "publicationTypes", None)
    if pub_types is not None:
        result["s2_publication_types"] = [str(t) for t in pub_types] if pub_types else []

    logger.info(
        "S2 enriched %s: %d citations, %d influential, paperId=%s",
        doi,
        result.get("s2_citation_count", 0),
        result.get("s2_influential_citations", 0),
        result.get("s2_paper_id", "—"),
    )
    return result


def search_semantic_scholar(
    query: str,
    since_days: int = 14,
    limit: int = 100,
    *,
    api_key: str | None = _DEFAULT_S2_API_KEY,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar by keyword, returning pipeline-format paper dicts.

    Parameters
    ----------
    query : str
        Search query.  Findpapers ``[bracket]`` notation is stripped
        automatically so the same topic strings from ``topics.yaml`` work here.
    since_days : int
        Only return papers published in the last N days.
    limit : int
        Maximum results to return (Semantic Scholar API cap per request).
    api_key : str | None
        Semantic Scholar API key.  Reads ``S2_API_KEY`` env var by default.

    Returns
    -------
    list[dict]
        Paper dicts with keys: ``doi``, ``title``, ``authors``, ``year``,
        ``journal``, ``abstract``, ``keywords``, ``source_databases``,
        ``tracked_author``.  Compatible with findpapers result format so
        results merge directly into the discovery pipeline.
    """
    if not _S2_AVAILABLE:
        logger.debug("semanticscholar package not available — skipping search")
        return []
    if not query or not query.strip():
        return []

    # S2 does relevance ranking, not Boolean search — strip findpapers syntax.
    clean_query = _strip_findpapers_format(query)

    since_date = date.today() - timedelta(days=since_days)
    # S2 publication_date_or_year: "YYYY-MM-DD:" means "from this date onward".
    pub_date_filter = f"{since_date}:"

    sch = _SemanticScholar(api_key=api_key)
    try:
        results = sch.search_paper(
            clean_query,
            fields=_SEARCH_FIELDS,
            publication_date_or_year=pub_date_filter,
            limit=limit,
        )
    except Exception as exc:
        logger.warning("S2 search failed for %r: %s", clean_query[:80], exc)
        return []

    papers: list[dict[str, Any]] = []
    for p in results:
        try:
            doi = ((p.externalIds or {}).get("DOI") or "").lower().strip()
            year: int | None = getattr(p, "year", None)
            journal = ""
            pub_venue = getattr(p, "publicationVenue", None)
            if pub_venue is not None:
                journal = getattr(pub_venue, "name", "") or ""
            authors = [
                a.name for a in (p.authors or [])
                if hasattr(a, "name") and a.name
            ]
            fos: list[str] = list(getattr(p, "fieldsOfStudy", []) or [])
            papers.append({
                "doi": doi,
                "title": p.title or "",
                "authors": authors,
                "year": year,
                "journal": journal,
                "abstract": getattr(p, "abstract", "") or "",
                "keywords": fos,
                "source_databases": ["semantic_scholar"],
                "tracked_author": False,
            })
        except Exception as exc:
            logger.warning("Failed to convert S2 search result: %s", exc)

    logger.info(
        "S2 search %r (since %s): %d results",
        clean_query[:60], since_date, len(papers),
    )
    return papers


def _strip_findpapers_format(query: str) -> str:
    """Convert a findpapers [bracket] query to plain text for Semantic Scholar.

    ``[ultrafiltration] AND [monoclonal antibody]``
    → ``ultrafiltration monoclonal antibody``

    Plain queries without brackets are returned with only whitespace collapsed.
    Boolean operators (AND, OR, AND NOT) are removed because S2 search does
    relevance ranking rather than Boolean matching.
    """
    # Remove square brackets, keeping the term inside
    plain = re.sub(r"\[([^\]]+)\]", r"\1", query)
    # Remove Boolean operators — S2 ranks by relevance, not Boolean logic
    plain = re.sub(r"\bAND NOT\b|\bAND\b|\bOR\b", " ", plain, flags=re.IGNORECASE)
    # Collapse whitespace and strip parens
    plain = re.sub(r"[()]", " ", plain)
    return re.sub(r"\s+", " ", plain).strip()
