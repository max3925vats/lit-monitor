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

import numpy as np

try:
    # Base class for chromadb client/connection errors. Imported lazily-safe
    # at module level so we can `except ChromaError` below without surprising
    # ImportError at call-time.
    from chromadb.errors import ChromaError
except ImportError:  # chromadb optional in some test envs
    # Empty tuple = "match no exception" when used with `except`.
    # Do NOT change to `Exception` — that would re-mask backend outages.
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
    *,
    domain_context_emb: np.ndarray | None = None,
    domain_context_weight: float = 0.0,
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
    domain_context_emb:
        Bundle A: optional pre-computed embedding of the domain_context paragraph.
        When None or domain_context_weight == 0.0: behavior is byte-for-byte
        identical to v0.8.0 (regression-test-locked).
    domain_context_weight:
        Bundle A: additive weight for the domain_context cosine score.
        0.0 (default) = no contribution → v0.8.0 behavior preserved.
    Returns
    -------
    list[dict]
        Candidates sorted by similarity score (descending), each augmented
        with ``similarity_score`` and (for top-K) ``llm_rationale``.
        When domain_context_emb is provided and weight > 0, each paper also
        gets ``_domain_score`` (the raw cosine value before weighting).
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

    # 2. Bundle A: optional domain_context additive score.
    #    Runs only when both the embedding AND a non-zero weight are provided.
    #    When either is absent/zero: no change to scores → v0.8.0 behavior.
    if domain_context_emb is not None and domain_context_weight > 0.0:
        _domain_norm = np.linalg.norm(domain_context_emb)
        for paper in scored:
            cand_emb = paper.get("_embedding")
            if cand_emb is None:
                # Candidate has no stored embedding — skip silently (no score change).
                continue
            cand_arr = np.asarray(cand_emb, dtype=np.float32)
            _cand_norm = float(np.linalg.norm(cand_arr))
            if _cand_norm < 1e-9 or _domain_norm < 1e-9:
                continue
            domain_score = float(np.dot(cand_arr, domain_context_emb) / (_cand_norm * _domain_norm))
            # Store raw domain cosine for Bundle B's score decomposition / explainability.
            paper["_domain_score"] = domain_score
            paper["similarity_score"] = paper["similarity_score"] + domain_context_weight * domain_score

    # 3. Sort descending by score
    scored.sort(key=lambda p: p.get("similarity_score", 0.0), reverse=True)
    # 4. LLM rationale for top-K
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
