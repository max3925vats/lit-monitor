"""
Search runner — wraps findpapers (0.6.x functional API) for multi-database search.
findpapers.search() writes results to a temp file; we load and convert them.

Supported databases in findpapers 0.6.x: pubmed, arxiv, scopus (key required),
ieee (key required), acm, biorxiv, medrxiv.
WoS, CrossRef, and OpenAlex are NOT supported by findpapers 0.6.x.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import logging
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any

# findpapers is vendored (lit_monitor/_vendor/findpapers) and always available —
# it ships with the package, so this is a plain import, not an optional dep.
from lit_monitor._vendor import findpapers as _findpapers
from lit_monitor._vendor.findpapers.utils.persistence_util import load as _fp_load

try:
    from lit_monitor.search.semantic_scholar import (
        _S2_AVAILABLE,
        search_semantic_scholar,
    )
except ImportError:  # pragma: no cover
    _S2_AVAILABLE = False  # type: ignore[assignment]
    search_semantic_scholar = None  # type: ignore[assignment]

from lit_monitor.core.doi import normalize_doi
from lit_monitor.core.strict_mode import strict_fallback
from lit_monitor.search._constants import FINDPAPERS_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Databases available in findpapers 0.6.x (lowercase, as accepted by databases= param)
DEFAULT_DATABASES = ["pubmed", "arxiv", "scopus"]
_SECRETS_PATH = Path.home() / ".config" / "lit-monitor" / "config.toml"


def _load_api_secrets() -> dict[str, str | None]:
    """Read API keys and email from config.toml. Returns empty dict if absent."""
    if not _SECRETS_PATH.exists():
        return {}
    try:
        with _SECRETS_PATH.open("rb") as fh:
            data = tomllib.load(fh)
        return {
            "scopus_api_key": data.get("scopus", {}).get("api_key") or None,
            "ieee_api_key": data.get("ieee", {}).get("api_key") or None,
            "email": data.get("pubmed", {}).get("email") or None,
        }
    except Exception as exc:
        strict_fallback(
            logger,
            f"Could not load API keys from config.toml: {exc}. "
            "Searches requiring Scopus/IEEE keys will be skipped.",
            exc,
        )
        return {}


def run_searches(
    config,
    databases: list[str] | None = None,
    since_days: int = 14,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """
    Run keyword searches for all topics in config.topics.

    Parameters
    ----------
    config:
        Config object with .topics (list of topic strings).
    databases:
        Override database list. Defaults to DEFAULT_DATABASES.
        Must be lowercase identifiers from findpapers 0.6.x:
        pubmed, arxiv, scopus, ieee, acm, biorxiv, medrxiv.
    since_days:
        Search window in days from today.
    limit:
        Maximum results per database per query.

    Returns
    -------
    list[dict]
        Deduplicated list of paper dicts. Each dict has:
        doi, title, authors (list), year, journal, abstract, keywords (list),
        source_databases (list), tracked_author (False).
    """
    if databases is None:
        databases = DEFAULT_DATABASES

    topics = _get_topics(config)
    if not topics:
        logger.warning("No topics configured — skipping topic searches")
        return []

    since = datetime.date.today() - datetime.timedelta(days=since_days)
    secrets = _load_api_secrets()
    scopus_key = secrets.get("scopus_api_key")
    ieee_key = secrets.get("ieee_api_key")
    if not scopus_key and "scopus" in databases:
        logger.warning("No Scopus API key in config.toml — Scopus database will be skipped")
    if not ieee_key and "ieee" in databases:
        logger.warning("No IEEE API key in config.toml — IEEE database will be skipped")

    all_papers: dict[str, dict[str, Any]] = {}

    for topic in topics:
        logger.info("Searching topic: %r", topic)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as _tmp_fh:
            tmp = _tmp_fh.name
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _findpapers.search,
                    outputpath=tmp,
                    query=_format_query(topic),
                    since=since,
                    databases=databases,
                    limit_per_database=limit,
                    scopus_api_token=scopus_key,
                    ieee_api_token=ieee_key,
                )
                try:
                    future.result(timeout=FINDPAPERS_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    # Note: future.cancel() returns False here because the
                    # thread is already running; findpapers will keep going
                    # until it finishes on its own. We just stop waiting.
                    logger.warning(
                        "findpapers search for topic %r timed out after %ds; skipping",
                        topic, FINDPAPERS_TIMEOUT_SECONDS,
                    )
                    continue
            search_result = _fp_load(tmp)
            papers = _convert_findpapers_results(search_result.papers, tracked_author=False)
            for paper in papers:
                # AR-4: canonical key collapses URL-wrap + case so the same
                # paper from different sources dedups to one entry.
                doi = normalize_doi(paper.get("doi"))
                if doi and doi not in all_papers:
                    all_papers[doi] = paper
                elif not doi:
                    title_key = f"__notitle_{paper.get('title', '')[:60]}"
                    if title_key not in all_papers:
                        all_papers[title_key] = paper
            logger.info(
                "Topic %r: %d results (%d unique so far)", topic, len(papers), len(all_papers)
            )
        except Exception as exc:
            strict_fallback(
                logger,
                f"Search failed for topic {topic!r}: {exc}. "
                "Results for this topic are missing from the digest.",
                exc,
            )
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    # D3: Semantic Scholar supplementary search — best-effort, runs after findpapers
    # so that DOI deduplication covers both sources.  Errors never abort the run.
    if _S2_AVAILABLE and search_semantic_scholar is not None:
        for topic in topics:
            try:
                logger.info("S2 search for topic: %r", topic[:60])
                s2_results = search_semantic_scholar(topic, since_days=since_days)
                added = 0
                for paper in s2_results:
                    # AR-4: same canonical key as the findpapers loop so an S2
                    # result with a URL-wrapped DOI dedups against findpapers.
                    doi = normalize_doi(paper.get("doi"))
                    if doi and doi not in all_papers:
                        all_papers[doi] = paper
                        added += 1
                    elif not doi:
                        # No-DOI S2 results cannot be deduplicated — skip.
                        pass
                logger.info("S2 topic %r: %d new results added", topic[:60], added)
            except Exception as exc:
                logger.warning("S2 search failed for topic %r: %s", topic[:60], exc)

    result = list(all_papers.values())
    logger.info("Total unique results from topic searches: %d", len(result))
    return result


def filter_known_dois(
    papers: list[dict[str, Any]],
    state_db,
) -> list[dict[str, Any]]:
    """
    Remove papers whose DOI is already in the state database.
    Papers without a DOI are always included (cannot be deduplicated).
    """
    # AR-4 + lifecycle Part B: suppress against the LIBRARY (papers actually
    # ingested) UNION papers already SHOWN in a digest — NOT bare 'discovered'
    # candidates that were merely retrieved-and-ranked, so an unshown candidate
    # stays eligible to surface later under improved ranking.
    known = {normalize_doi(d) for d in state_db.library_dois()} \
        | {normalize_doi(d) for d in state_db.surfaced_dois()}
    new_papers = []
    for paper in papers:
        doi = normalize_doi(paper.get("doi"))
        if not doi or doi not in known:
            new_papers.append(paper)
    logger.info(
        "filter_known_dois: %d in / %d known → %d new",
        len(papers), len(known), len(new_papers),
    )
    return new_papers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_query(topic: str) -> str:
    """Convert a human-written Boolean query to findpapers [term] bracket format.

    findpapers 0.6.x requires every search term wrapped in square brackets:
        [ultrafiltration] AND [monoclonal antibody]
    Multi-word phrases stay as one term: [monoclonal antibody] (not split further).

    Any pre-existing '[' or ']' in the input is stripped first, so the canonical
    formatting path runs every time. This guarantees validation regardless of
    whether the caller already wrapped terms in brackets (Audit_31 H2).
    """
    # Strip any user-supplied brackets so the canonical path re-adds them
    # exactly where findpapers expects.
    topic = topic.replace('[', '').replace(']', '')

    if not re.search(r'\b(?:AND NOT|AND|OR)\b', topic, re.IGNORECASE):
        return f'[{topic.strip()}]'

    # Split on Boolean operators and parentheses (capturing groups keep delimiters)
    tokens = re.split(r'(\s+AND NOT\s+|\s+AND\s+|\s+OR\s+|[()])', topic, flags=re.IGNORECASE)
    result = []
    for token in tokens:
        stripped = token.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper in ('AND NOT', 'AND', 'OR'):
            result.append(f' {upper} ')
        elif stripped in ('(', ')'):
            result.append(stripped)
        else:
            result.append(f'[{stripped}]')
    return ''.join(result)


def _get_topics(config) -> list[str]:
    """Extract topic query strings from config.topics.

    topics.yaml uses the schema:
        searches:
          - name: "..."
            query: "..."
            databases: [...]
    config.topics is the list returned by `raw.get("searches", [])`.
    Each element may be a dict with a "query" key, a plain string, or a
    namespace object — handle all three.
    """
    topics = getattr(config, "topics", None)
    if not topics:
        return []
    if isinstance(topics, list):
        result = []
        for t in topics:
            if isinstance(t, dict):
                q = t.get("query") or t.get("name")
                if q:
                    result.append(str(q))
            elif t:
                result.append(str(t))
        return result
    if hasattr(topics, "queries"):
        return list(topics.queries) if topics.queries else []
    if hasattr(topics, "_data"):
        data = topics._data
        if isinstance(data, list):
            result = []
            for t in data:
                if isinstance(t, dict):
                    q = t.get("query") or t.get("name")
                    if q:
                        result.append(str(q))
                elif t:
                    result.append(str(t))
            return result
        if isinstance(data, dict):
            return [str(q) for q in data.get("queries", []) if q]
    return []


def _convert_findpapers_results(papers, tracked_author: bool) -> list[dict[str, Any]]:
    """
    Convert a list of findpapers Paper objects to flat paper dicts.

    findpapers Paper attributes used here:
        title (str), abstract (str), authors (list[str]),
        publication (Publication | None), publication_date (date | None),
        doi (str | None), keywords (set | None), databases (set | None).
    """
    result: list[dict[str, Any]] = []
    for paper in papers:
        try:
            # AR-4: store the canonical DOI. The system already stores bare
            # lowercased DOIs; normalize_doi() additionally strips URL-wrapping,
            # so the stored value stays consistent and the dedup key in
            # run_searches() matches what lands in state.db.
            doi = normalize_doi(paper.doi)
            year: int | None = None
            if paper.publication_date:
                try:
                    year = paper.publication_date.year
                except Exception:
                    pass
            journal = ""
            if paper.publication:
                journal = getattr(paper.publication, "title", "") or ""
            result.append({
                "doi": doi,
                "title": paper.title or "",
                "authors": list(paper.authors or []),
                "year": year,
                "journal": journal,
                "abstract": paper.abstract or "",
                "keywords": list(paper.keywords or []),
                "source_databases": list(paper.databases or []),
                "tracked_author": tracked_author,
            })
        except Exception as exc:
            logger.warning("Failed to convert paper result: %s", exc)
    return result
