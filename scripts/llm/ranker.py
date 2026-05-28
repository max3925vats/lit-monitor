"""
LLM-based relevance ranker for discovery results.
Takes candidate papers (from findpapers search), embeds each against the
ChromaDB collection, sorts by cosine similarity, then asks the LLM to
write a brief relevance rationale for the top-K results.

System and user prompts come from config/prompts/rationale.yaml.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    # Base class for chromadb client/connection errors. Imported lazily-safe
    # at module level so we can `except ChromaError` below without surprising
    # ImportError at call-time.
    from chromadb.errors import ChromaError
except Exception:  # pragma: no cover - chromadb missing in some envs
    ChromaError = ()  # type: ignore[assignment,misc]

from scripts.llm.prompt_registry import load_prompt
from scripts.llm.prompt_safety import sanitize_for_prompt

logger = logging.getLogger(__name__)


def rank_papers(
    candidates: list[dict[str, Any]],
    embeddings_db,
    llm,
    top_k: int = 20,
    domain_context: str = "",
) -> list[dict[str, Any]]:
    """
    Rank candidate papers by similarity to the existing knowledge base,
    then add LLM rationale for the top-K.
    Parameters
    ----------
    candidates:
        List of paper dicts. Each must have at minimum:
          doi, title. Optionally: abstract, authors, year.
    embeddings_db:
        EmbeddingsDB instance (used for similarity lookup).
    llm:
        LLMClient instance (used for rationale generation).
    top_k:
        Number of top results to send to the LLM for rationale.
    domain_context:
        Optional extra context prepended to the LLM system prompt.
    Returns
    -------
    list[dict]
        Candidates sorted by similarity score (descending), each augmented
        with ``similarity_score`` and (for top-K) ``llm_rationale``.
    """
    if not candidates:
        return []
    # 1. Embed each candidate and find similarity score
    scored: list[dict[str, Any]] = []
    for paper in candidates:
        doi = paper.get("doi", "")
        embed_text = _paper_query_text(paper)
        try:
            results = embeddings_db.find_similar_to_text(
                embed_text, top_k=1, exclude_id=doi
            )
            score = results[0]["score"] if results else 0.0
        except ChromaError:
            # Real DB / connection problem — do NOT silently degrade to 0.0,
            # otherwise every paper would be ranked as "no match" and the
            # caller would never learn the backend is broken.
            raise
        except Exception as exc:
            # Per-paper failures (e.g. bad embed text) are non-fatal: log a
            # warning with traceback and continue with score 0.0.
            logger.warning(
                "Similarity lookup failed for %s: %s", doi, exc, exc_info=True
            )
            score = 0.0
        scored.append({**paper, "similarity_score": score})
    # 2. Sort descending by score
    scored.sort(key=lambda p: p.get("similarity_score", 0.0), reverse=True)
    # 3. LLM rationale for top-K
    top = scored[:top_k]
    rationales = _get_rationales(top, llm, domain_context)
    for paper in scored:
        doi = paper.get("doi", "")
        paper["llm_rationale"] = rationales.get(doi, "")
    return scored


def _get_rationales(
    papers: list[dict[str, Any]],
    llm,
    domain_context: str,
) -> dict[str, str]:
    """
    Ask the LLM for one-sentence rationales for a list of papers.
    Returns {doi: rationale_str}. On any failure returns empty dict
    (rationale fields will be empty strings).
    """
    if not papers:
        return {}
    prompt = load_prompt("rationale")
    paper_summaries = []
    for p in papers:
        doi = sanitize_for_prompt(p.get("doi", "unknown"))
        title = sanitize_for_prompt(p.get("title", ""))
        abstract = sanitize_for_prompt((p.get("abstract") or "")[:300])
        paper_summaries.append(prompt.render_paper_card(
            doi=doi, title=title, abstract=abstract,
        ))

    papers_text = prompt.paper_card_separator.join(paper_summaries)
    user_prompt = prompt.render_user(papers=papers_text)

    system = prompt.system
    if domain_context:
        system = domain_context + "\n\n" + system

    try:
        raw = llm.complete(system, user_prompt, max_tokens=prompt.max_tokens)
        from scripts.llm.llm_client import parse_llm_json
        return parse_llm_json(raw)
    except Exception as exc:
        logger.warning("LLM rationale generation failed: %s", exc)
        return {}


def _paper_query_text(paper: dict[str, Any]) -> str:
    """Build a query string for embedding lookup."""
    parts = []
    if paper.get("title"):
        parts.append(paper["title"])
    if paper.get("abstract"):
        parts.append(paper["abstract"][:500])
    return " ".join(parts)
