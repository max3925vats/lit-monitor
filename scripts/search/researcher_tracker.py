"""
Researcher tracker — author-name-based searches for tracked researchers.
Key constraints from findpapers 0.6.x audit:
- findpapers has NO Author-ID (ORCID / Scopus AU-ID) support
- Only standard author name queries via AU field
- Name collisions are possible for common names
- Supplementary: direct OpenAlex author endpoint for ORCID-based
  high-precision lookup (optional, only if orcid configured)

Expected researchers.yaml format:
  researchers:
    - name: "Author One"
      orcid: "0000-0001-2345-6789"  # optional
    - name: "Author Two"
"""
from __future__ import annotations

import datetime
import logging
import os
import tempfile
from typing import Any

try:
    import findpapers as _findpapers
    from findpapers.utils.persistence_util import load as _fp_load
except ImportError:  # pragma: no cover
    _findpapers = None  # type: ignore[assignment]
    _fp_load = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
_OPENALEX_AUTHOR_URL = "https://api.openalex.org/works"
_DEFAULT_DATABASES = ["pubmed", "arxiv", "scopus"]


def run_researcher_searches(
    config,
    databases: list[str] | None = None,
    since_days: int = 14,
    limit_per_author: int = 50,
) -> list[dict[str, Any]]:
    """
    Run author-name searches for all tracked researchers.
    For researchers with an ORCID configured, also runs a supplementary
    OpenAlex ORCID lookup for higher precision.

    Parameters
    ----------
    config:
        Config object with .researchers (list of {name, orcid?}).
    databases:
        Override database list (default: DEFAULT_DATABASES).
    since_days:
        Search window in days from today.
    limit_per_author:
        Max results per author per database.

    Returns
    -------
    list[dict]
        Deduplicated paper dicts with tracked_author=True.
    """
    from scripts.search.search_runner import _convert_findpapers_results, _load_api_secrets

    researchers = _get_researchers(config)
    if not researchers:
        logger.info("No researchers configured — skipping researcher searches")
        return []

    if _findpapers is None:  # pragma: no cover
        raise ImportError("findpapers is not installed — run: pip install findpapers")

    if databases is None:
        databases = _DEFAULT_DATABASES

    since = datetime.date.today() - datetime.timedelta(days=since_days)
    secrets = _load_api_secrets()
    scopus_key = secrets.get("scopus_api_key")
    ieee_key = secrets.get("ieee_api_key")

    all_papers: dict[str, dict[str, Any]] = {}

    for researcher in researchers:
        name = researcher.get("name", "").strip()
        orcid = researcher.get("orcid", "").strip()
        if not name:
            continue

        # findpapers 0.6.x requires [term] format; AU: field qualifier is not supported.
        # Searching by full name as a term finds papers where the name appears in
        # title/abstract/keywords. This is a best-effort approach since findpapers
        # has no author-ID or author-field support.
        au_query = f'[{name}]'
        logger.info("Searching for researcher: %r (query: %r)", name, au_query)
        tmp = tempfile.mktemp(suffix=".json")
        try:
            _findpapers.search(
                outputpath=tmp,
                query=au_query,
                since=since,
                databases=databases,
                limit_per_database=limit_per_author,
                scopus_api_token=scopus_key,
                ieee_api_token=ieee_key,
            )
            search_result = _fp_load(tmp)
            papers = _convert_findpapers_results(search_result.papers, tracked_author=True)
            for paper in papers:
                doi = paper.get("doi", "").strip().lower()
                if doi and doi not in all_papers:
                    all_papers[doi] = paper
            logger.info("Researcher %r: %d results", name, len(papers))
        except Exception as exc:
            logger.error("Search failed for researcher %r: %s", name, exc)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        # Supplementary: OpenAlex ORCID lookup (if orcid provided)
        if orcid:
            try:
                openalex_papers = _openalex_author_lookup(orcid, since, config)
                for paper in openalex_papers:
                    doi = paper.get("doi", "").strip().lower()
                    if doi and doi not in all_papers:
                        all_papers[doi] = paper
                logger.info(
                    "OpenAlex ORCID lookup for %r: %d additional results",
                    name, len(openalex_papers),
                )
            except Exception as exc:
                logger.warning("OpenAlex lookup failed for %r (ORCID %s): %s", name, orcid, exc)

    result = list(all_papers.values())
    logger.info("Total unique results from researcher searches: %d", len(result))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_researchers(config) -> list[dict[str, str]]:
    """Extract researcher list from config.researchers."""
    researchers = getattr(config, "researchers", None)
    if not researchers:
        return []
    if isinstance(researchers, list):
        result = []
        for r in researchers:
            if isinstance(r, str):
                result.append({"name": r})
            elif isinstance(r, dict):
                result.append(r)
            elif hasattr(r, "_data"):
                result.append(r._data if isinstance(r._data, dict) else {"name": str(r)})
            elif hasattr(r, "name"):
                result.append({"name": r.name, "orcid": getattr(r, "orcid", "")})
        return result
    return []


def _openalex_author_lookup(
    orcid: str,
    since: datetime.date,
    config,
) -> list[dict[str, Any]]:
    """
    Query OpenAlex works API filtered by author ORCID.
    Returns a list of paper dicts (tracked_author=True).
    """
    from scripts.core.http_client import get_json
    orcid_url = orcid if orcid.startswith("http") else f"https://orcid.org/{orcid}"
    from_date = since.isoformat()
    params_str = (
        f"?filter=author.orcid:{orcid_url},from_publication_date:{from_date}"
        f"&per-page=50&select=id,doi,title,authorships,publication_year,"
        f"primary_location,keywords,abstract_inverted_index"
    )
    from scripts.search.search_runner import _load_api_secrets
    email = (
        os.environ.get("LIT_MONITOR_MAILTO")
        or _load_api_secrets().get("email", "")
        or "lit-monitor@example.com"
    )
    url = _OPENALEX_AUTHOR_URL + params_str + f"&mailto={email}"
    data = get_json(url)
    results = data.get("results", [])
    return [_convert_openalex_work(work) for work in results]


def _convert_openalex_work(work: dict) -> dict[str, Any]:
    """Convert an OpenAlex work dict to a standard paper dict."""
    doi_raw = work.get("doi", "") or ""
    doi = doi_raw.replace("https://doi.org/", "").replace("http://doi.org/", "").lower().strip()
    authorships = work.get("authorships", [])
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in authorships
        if a.get("author", {}).get("display_name")
    ]
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    journal = source.get("display_name", "") or ""
    keywords = [k.get("keyword", "") for k in work.get("keywords", []) if k.get("keyword")]
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    return {
        "doi": doi,
        "title": work.get("title", "") or "",
        "authors": authors,
        "year": work.get("publication_year"),
        "journal": journal,
        "abstract": abstract,
        "keywords": keywords,
        "source_databases": ["openalex"],
        "tracked_author": True,
    }


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    try:
        positions: list[tuple[int, str]] = []
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions.append((pos, word))
        positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in positions)
    except Exception:
        return ""
