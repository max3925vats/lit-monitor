"""
E1 — Citation graph resolver.

Resolves ``key_citations`` ref_ids from pass-4 extraction to cited-paper DOIs
via the Semantic Scholar references API, then writes ``citation_edges`` rows
to the state DB.

Resolution strategies (in priority order):
  1. Numeric  — ``"[N]"`` → N-th entry in S2 reference list (1-indexed).
  2. Author-year fuzzy — ``"Smith et al. (2020)"`` matched against S2 ref
     titles + first-author name + year via rapidfuzz token_sort_ratio.
  3. Unresolved — stored with ``target_doi=NULL`` for transparency.

Rate limits: every S2 call wraps with exponential backoff (default max 4
retries, starting at 1 s).  A rate-limit response is detected by the
presence of "429", "rate", or "too many" in the exception message.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz

from scripts.search._constants import S2_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from scripts.core.state_db import StateDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level availability guard (mirrors semantic_scholar.py)
# ---------------------------------------------------------------------------
_DEFAULT_S2_API_KEY: str | None = os.environ.get("S2_API_KEY")

try:
    from semanticscholar import SemanticScholar as _SemanticScholar
    _S2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SemanticScholar = None  # type: ignore[assignment]
    _S2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUMERIC_RE = re.compile(r"^\[(\d+)\]$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_FIRST_AUTHOR_RE = re.compile(r"^\W*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'.\-]+)")
_BASE_BACKOFF: float = 1.0      # seconds; doubles on each retry
_AUTHOR_SCORE_CUTOFF: int = 72  # fuzz.ratio threshold for last-name matching


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CitationEdge:
    ref_id: str
    target_doi: str | None
    target_s2_id: str | None
    context: str
    resolution: str  # 'numeric_index' | 'author_year_fuzzy' | 'unresolved'


@dataclass
class CitationGraphResult:
    source_doi: str
    edges: list[CitationEdge] = field(default_factory=list)
    n_resolved: int = 0
    n_unresolved: int = 0
    s2_references_count: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_citation_graph(
    doi: str,
    state_db: StateDB,
    *,
    api_key: str | None = _DEFAULT_S2_API_KEY,
    max_retries: int = 3,
) -> CitationGraphResult:
    """Resolve ``key_citations`` for a paper and write ``citation_edges`` rows.

    Parameters
    ----------
    doi:
        DOI of the citing paper.  Pass-4 extraction must have run first.
    state_db:
        Open StateDB instance used both to read ``key_citations`` and to write
        resolved edges.
    api_key:
        Semantic Scholar API key.  Reads ``S2_API_KEY`` env var by default.
    max_retries:
        Maximum number of retry attempts on S2 rate-limit responses.

    Returns
    -------
    CitationGraphResult
        Summary of resolved and unresolved edges.  Edges are also persisted to
        ``citation_edges`` in the state DB before this returns.
    """
    result = CitationGraphResult(source_doi=doi)

    # Step 1 — load key_citations from pass-4 extraction
    extraction = state_db.get_extraction_json(doi)
    if not extraction:
        logger.warning("build_citation_graph: no extraction_json found for %s", doi)
        return result

    key_citations = extraction.get("key_citations") or []
    if not key_citations:
        logger.info("build_citation_graph: no key_citations in extraction for %s", doi)
        return result

    # Step 2 — fetch S2 reference list for the citing paper
    s2_refs = _fetch_s2_references(doi, api_key=api_key, max_retries=max_retries)
    result.s2_references_count = len(s2_refs)

    # Step 3 — resolve each key_citation and persist
    for kc in key_citations:
        if not isinstance(kc, dict):
            continue
        ref_id = str(kc.get("ref_id") or "").strip()
        context = str(kc.get("context") or "").strip()
        if not ref_id:
            continue

        edge = _resolve_ref_id(ref_id, context, s2_refs)
        result.edges.append(edge)

        if edge.resolution != "unresolved":
            result.n_resolved += 1
        else:
            result.n_unresolved += 1

        state_db.upsert_citation_edge(
            source_doi=doi,
            ref_id=ref_id,
            target_doi=edge.target_doi,
            target_s2_id=edge.target_s2_id,
            context=context,
            resolution=edge.resolution,
        )

    logger.info(
        "Citation graph %s: %d key_citations → %d resolved, %d unresolved "
        "(%d S2 references fetched)",
        doi,
        len(result.edges),
        result.n_resolved,
        result.n_unresolved,
        result.s2_references_count,
    )
    return result


# ---------------------------------------------------------------------------
# S2 fetch with exponential backoff
# ---------------------------------------------------------------------------
def _fetch_s2_references(
    doi: str,
    *,
    api_key: str | None = None,
    max_retries: int = 3,
) -> list[Any]:
    """Fetch the ordered reference list for a paper via S2.

    Retries on rate-limit responses with exponential backoff: with the default
    ``max_retries=3`` a persistently rate-limited DOI makes 1 initial attempt
    plus 3 backoff retries (4 HTTP calls total), sleeping 1s, 2s, 4s between
    them. Returns ``[]`` on permanent failures (logged as WARNING).
    """
    if not _S2_AVAILABLE:
        logger.debug("semanticscholar not available — citation graph disabled")
        return []

    sch = _SemanticScholar(api_key=api_key, timeout=S2_TIMEOUT_SECONDS)
    for attempt in range(max_retries + 1):
        try:
            paper = sch.get_paper(f"DOI:{doi.strip()}", fields=["references"])
            if paper is None:
                logger.debug("S2: no record found for DOI %s", doi)
                return []
            refs = list(getattr(paper, "references", None) or [])
            logger.debug("S2: fetched %d references for %s", len(refs), doi)
            return refs
        except Exception as exc:
            # The semanticscholar client maps an HTTP 429 to a plain
            # ConnectionRefusedError (see semanticscholar/ApiRequester.py),
            # so that type is the closest thing to a structured rate-limit
            # signal it exposes. Match on it first, then fall back to a
            # message substring scan as a defensive net for client versions
            # / proxies that surface 429s through a different exception.
            msg = str(exc).lower()
            is_rate_limit = (
                isinstance(exc, ConnectionRefusedError)
                or "429" in msg
                or "rate" in msg
                or "too many" in msg
            )
            if is_rate_limit and attempt < max_retries:
                wait = _BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "S2 rate limit for %s (attempt %d/%d); retrying in %.1fs",
                    doi, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                logger.warning("S2 references fetch failed for %s: %s", doi, exc)
                return []
    return []  # exhausted retries


# ---------------------------------------------------------------------------
# Resolution strategies
# ---------------------------------------------------------------------------
def _resolve_ref_id(
    ref_id: str,
    context: str,
    s2_refs: list[Any],
) -> CitationEdge:
    """Map a verbatim ref_id to one S2 reference entry using two strategies."""

    # Strategy 1: numeric "[N]" → 1-indexed position in S2 reference list
    m = _NUMERIC_RE.match(ref_id)
    if m:
        idx = int(m.group(1)) - 1  # convert 1-based to 0-based
        if 0 <= idx < len(s2_refs):
            ref = s2_refs[idx]
            return CitationEdge(
                ref_id=ref_id,
                target_doi=_doi_from_ref(ref),
                target_s2_id=_s2id_from_ref(ref),
                context=context,
                resolution="numeric_index",
            )

    # Strategy 2: author-year fuzzy match
    matched = _fuzzy_match_author_year(ref_id, s2_refs)
    if matched is not None:
        return CitationEdge(
            ref_id=ref_id,
            target_doi=_doi_from_ref(matched),
            target_s2_id=_s2id_from_ref(matched),
            context=context,
            resolution="author_year_fuzzy",
        )

    return CitationEdge(
        ref_id=ref_id,
        target_doi=None,
        target_s2_id=None,
        context=context,
        resolution="unresolved",
    )


def _fuzzy_match_author_year(
    ref_id: str,
    s2_refs: list[Any],
    cutoff: int = _AUTHOR_SCORE_CUTOFF,
) -> Any | None:
    """Fuzzy-match a ref_id like 'Smith et al. (2020)' against S2 references.

    Two-phase strategy:
    1. Hard filter on year: if the ref_id contains a 4-digit year (``\b19XX\b``
       or ``\b20XX\b``), only consider S2 references whose year field matches.
       This prevents "Smith 1999" from matching "Smith 2015 Some Paper".
    2. Fuzzy author match: use rapidfuzz ratio on the first-author last name
       extracted from the ref_id vs the S2 reference's first-author last name.
    """
    year = _extract_year(ref_id)
    author = _extract_first_author(ref_id)
    if not author:
        return None

    best_score = 0
    best_ref = None
    for ref in s2_refs:
        # Phase 1: hard year filter (skip refs with wrong year)
        if year is not None:
            ref_year = getattr(ref, "year", None)
            if ref_year != year:
                continue
        # Phase 2: fuzzy last-name comparison
        ref_last = _ref_last_name(ref)
        score = fuzz.ratio(author.lower(), ref_last.lower())
        if score > best_score:
            best_score = score
            best_ref = ref
    if best_ref is not None and best_score >= cutoff:
        return best_ref
    return None


# ---------------------------------------------------------------------------
# Reference field helpers
# ---------------------------------------------------------------------------
def _doi_from_ref(ref: Any) -> str | None:
    ext_ids = getattr(ref, "externalIds", None) or {}
    doi = (ext_ids.get("DOI") or "").lower().strip()
    return doi if doi else None


def _s2id_from_ref(ref: Any) -> str | None:
    return getattr(ref, "paperId", None) or None


def _ref_last_name(ref: Any) -> str:
    """Extract the first-author last name from an S2 reference object."""
    authors = getattr(ref, "authors", []) or []
    if not authors:
        return ""
    first_name = str(getattr(authors[0], "name", "") or "")
    if "," in first_name:
        # "Lastname, Firstname" format
        return first_name.split(",")[0].strip()
    # "Firstname Lastname" format
    parts = first_name.split()
    return parts[-1] if parts else ""


def _extract_year(ref_id: str) -> int | None:
    """Extract a 4-digit publication year (1900-2099) from a ref_id string."""
    m = _YEAR_RE.search(ref_id)
    return int(m.group()) if m else None


def _extract_first_author(ref_id: str) -> str:
    """Extract the lead author's surname from a ref_id like 'Smith et al. (2020)'."""
    m = _FIRST_AUTHOR_RE.search(ref_id)
    return m.group(1) if m else ""


def _ref_search_string(ref: Any) -> str:
    """Build 'LastName Year Title' from an S2 reference object.

    Used only by tests that inspect the search string directly;
    production matching uses the two-phase year+author strategy.
    """
    title = str(getattr(ref, "title", "") or "")
    year = str(getattr(ref, "year", "") or "")
    return f"{_ref_last_name(ref)} {year} {title}".strip()
