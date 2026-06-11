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

from scripts.core.strict_mode import strict_fallback
from scripts.learning.rocchio import soft_gate
from scripts.llm.prompt_registry import load_prompt
from scripts.llm.prompt_safety import sanitize_for_prompt

logger = logging.getLogger(__name__)

# Graph-signal normalization caps. Each raw count is divided by its cap and
# clamped to [0, 1] before weighting. Conservative v0.9 baselines. These are
# imported by scripts/pipelines/discovery.py so the two ranking paths stay in
# lock-step — do not re-hardcode the divisors there.
_GRAPH_ENTITY_NORM_CAP: float = 5.0
_GRAPH_CITATION_NORM_CAP: float = 3.0
_GRAPH_AUTHORS_NORM_CAP: float = 3.0


def rank_papers(
    candidates: list[dict[str, Any]],
    embeddings_db,
    llm,
    top_k: int = 20,
    domain_context: str = "",
    *,
    domain_context_emb: np.ndarray | None = None,
    domain_context_weight: float = 0.0,
    cluster_centroids=None,         # list[Cluster] | None — Bundle C
    cluster_centroid_weight: float = 0.0,  # Bundle C
    graph_signals: dict[str, dict] | None = None,   # Bundle D: doi → signals dict
    graph_weights: dict[str, float] | None = None,  # Bundle D: entity_overlap/citation/shared_authors
    interest_vec: np.ndarray | None = None,          # Bundle J2: Rocchio interest direction
    interest_vec_n_events: int = 0,                  # Bundle J2: contributor count (soft-gate input)
    interest_weight: float = 0.0,                    # Bundle J2: ranking.weights.feedback
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
    cluster_centroids:
        Bundle C: list of Cluster objects (with centroid_vec and display_name).
        When None or cluster_centroid_weight == 0.0: no cluster signal added.
    cluster_centroid_weight:
        Bundle C: additive weight for max-cosine-to-nearest-centroid signal.
        0.0 (default) = no contribution → backward compatible.
    graph_signals:
        Bundle D: mapping from candidate DOI to a dict of graph signals as
        returned by ``get_graph_signals_for_candidate()``.  When None or
        all graph weights are 0.0, no graph contribution is added → backward
        compatible with v0.8/A/B/C behavior.
    graph_weights:
        Bundle D: dict with keys ``entity_overlap``, ``citation``,
        ``shared_authors``.  All default to 0.0 when absent.  Uses weighted
        sum (NOT RRF) per the v0.9 locked design decision.
    interest_vec:
        Bundle J2: the stored Rocchio interest direction (a unit-ish vector in
        embedding space).  When None: no interest contribution → backward
        compatible (byte-for-byte regression preserved, like the domain leg).
    interest_vec_n_events:
        Bundle J2: number of feedback events that produced ``interest_vec``.
        Fed into ``soft_gate``: below the cold-start floor the gate returns 0.0
        and the whole leg is inert even with a vector and a non-zero weight.
    interest_weight:
        Bundle J2: additive weight for the interest-vector cosine score, read
        from ``ranking.weights.feedback``.  0.0 (default) = no contribution.

    Returns
    -------
    list[dict]
        Candidates sorted by similarity score (descending), each augmented
        with ``similarity_score`` and (for top-K) ``llm_rationale``.
        Each paper gets ``score_breakdown`` containing ``vector``,
        ``domain_context``, ``cluster_centroid``, ``graph_entity_overlap``,
        ``graph_citation``, and ``graph_shared_authors`` keys.
        When a cluster match is found, ``cluster_matched`` is set to the
        cluster's display_name (used by Bundle B's web UI).
        When graph signals are present, ``graph_shared_authors_sample`` and
        ``graph_shared_entities`` are set per-paper for the UI annotation panel.
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
        _domain_norm = float(np.linalg.norm(domain_context_emb))
        if _domain_norm < 1e-9 or not np.isfinite(_domain_norm):
            # Degenerate domain-context embedding: either all-zero (norm≈0) or
            # non-finite (NaN/inf norm). The whole domain leg is a no-op, so skip
            # it entirely — do NOT fall into the per-candidate loop. Critically,
            # a NaN norm would slip past a bare `_domain_norm < 1e-9` check
            # (`nan < 1e-9` is False) and corrupt every finite candidate's score
            # via `np.dot(...) / (... * nan)`; guarding the whole leg here is the
            # only place that closes that path. Surface it once so a
            # misconfigured/empty/NaN domain context is observable, not invisible.
            logger.warning(
                "Domain-context embedding is degenerate (norm≈0 or non-finite); "
                "skipping domain-context scoring for this run."
            )
        else:
            # AR-6 (log hygiene): a corpus with many bad stored vectors used to
            # flood the logs with one WARNING per degenerate candidate. Tally the
            # skips and emit ONE summary WARNING after the loop instead — the skip
            # behavior (which candidates are excluded, the scores) is unchanged.
            _degenerate_skips = 0
            for paper in scored:
                cand_emb = paper.get("_embedding")
                if cand_emb is None:
                    # Candidate has no stored embedding — skip silently (no score change).
                    continue
                cand_arr = np.asarray(cand_emb, dtype=np.float32)
                _cand_norm = float(np.linalg.norm(cand_arr))
                if _cand_norm < 1e-9 or not np.isfinite(_cand_norm):
                    # Degenerate candidate embedding: all-zero (norm≈0) or
                    # non-finite (NaN/inf — a NaN norm slips past the zero-norm
                    # check since `nan < 1e-9` is False and would corrupt the
                    # score via np.dot). Numerically skipped exactly as before;
                    # just counted here and reported once below instead of per-row.
                    _degenerate_skips += 1
                    continue
                domain_score = float(np.dot(cand_arr, domain_context_emb) / (_cand_norm * _domain_norm))
                # Store raw domain cosine for Bundle B's score decomposition / explainability.
                paper["_domain_score"] = domain_score
                paper["similarity_score"] = paper["similarity_score"] + domain_context_weight * domain_score

            if _degenerate_skips > 0:
                # One summary line instead of one-per-candidate (AR-6 log hygiene).
                logger.warning(
                    "ranker: skipped %d candidate(s) with degenerate embeddings",
                    _degenerate_skips,
                )

    # 2b. Bundle C: optional cluster-centroid additive score.
    #     Runs only when both cluster_centroids AND non-zero weight are provided.
    #     When either is absent/zero: no change to scores → backward compat.
    if cluster_centroids and cluster_centroid_weight > 0.0:
        _centroid_matrix = np.array(
            [c.centroid_vec for c in cluster_centroids], dtype=np.float32
        )  # (K, dim)
        _cen_norms = np.linalg.norm(_centroid_matrix, axis=1) + 1e-9  # (K,)

        if not np.isfinite(_centroid_matrix).all():
            # A stored centroid contains non-finite values (NaN/inf). The matmul
            # would propagate NaN into every candidate's cosines and np.argmax
            # would still pick a garbage cluster, silently poisoning cluster_score
            # (symmetric to the candidate-side guard below). Skip the whole
            # cluster leg, surfaced once, rather than corrupt the ranking.
            logger.warning(
                "One or more cluster centroids contain non-finite values; "
                "skipping cluster-centroid scoring for this run."
            )
        else:
            for paper in scored:
                cand_emb = paper.get("_embedding")
                if cand_emb is None:
                    continue
                cand_arr = np.asarray(cand_emb, dtype=np.float32)
                cand_norm = float(np.linalg.norm(cand_arr))
                if cand_norm < 1e-9 or not np.isfinite(cand_norm):
                    # Degenerate candidate embedding: all-zero (norm≈0) or non-finite
                    # (NaN/inf — a NaN norm slips past the zero-norm check since
                    # `nan < 1e-9` is False and would corrupt cluster_score via the
                    # dot product). Skip it from cluster scoring, consistent with the
                    # domain-context block above.
                    logger.warning(
                        "Candidate %s has a degenerate embedding (norm≈0 or "
                        "non-finite); skipping cluster-centroid score for it.",
                        paper.get("doi", "?"),
                    )
                    continue
                # Cosine similarities to all centroids; pick the max
                dots = _centroid_matrix @ cand_arr          # (K,)
                cosines = dots / (_cen_norms * cand_norm)   # (K,)
                best_idx = int(np.argmax(cosines))
                cluster_score = float(cosines[best_idx])
                paper["_cluster_score"] = cluster_score
                paper["_cluster_matched_idx"] = best_idx
                paper["similarity_score"] = (
                    paper["similarity_score"] + cluster_centroid_weight * cluster_score
                )
                # Annotate matched cluster name for Bundle B's web UI
                paper["cluster_matched"] = cluster_centroids[best_idx].display_name or ""

    # 2b-J. Bundle J2: optional interest-vector (Rocchio) additive score.
    #     Runs only when an interest_vec is provided AND interest_weight > 0.0
    #     AND the cold-start soft-gate is open (soft_gate(n_events) > 0.0). The
    #     gate is 0.0 below COLD_START_FLOOR feedback events, so the loop stays
    #     fully inert until enough data accumulates — even with a weight and a
    #     vector. When any condition is unmet: no change to scores → byte-for-byte
    #     v0.8/A/B/C/D behavior preserved.
    _interest_gate = soft_gate(interest_vec_n_events)
    if interest_vec is not None and interest_weight > 0.0 and _interest_gate > 0.0:
        _interest_arr = np.asarray(interest_vec, dtype=np.float32).ravel()
        _interest_norm = float(np.linalg.norm(_interest_arr))
        if _interest_norm < 1e-9 or not np.isfinite(_interest_norm):
            # Degenerate interest vector: all-zero (norm≈0) or non-finite
            # (NaN/inf). Symmetric to the domain-context leg's whole-leg guard —
            # a NaN norm slips past a bare `_interest_norm < 1e-9` check
            # (`nan < 1e-9` is False) and would corrupt every finite candidate's
            # score via np.dot(...) / (... * nan). Skip the WHOLE leg, surfaced
            # once, rather than corrupt the ranking.
            logger.warning(
                "Interest vector is degenerate (norm≈0 or non-finite); "
                "skipping interest-vector scoring for this run."
            )
        else:
            for paper in scored:
                cand_emb = paper.get("_embedding")
                if cand_emb is None:
                    # Candidate has no stored embedding — skip silently (no score change).
                    continue
                cand_arr = np.asarray(cand_emb, dtype=np.float32)
                _cand_norm = float(np.linalg.norm(cand_arr))
                if _cand_norm < 1e-9 or not np.isfinite(_cand_norm):
                    # Degenerate candidate embedding: all-zero (norm≈0) or
                    # non-finite (NaN/inf — a NaN norm slips past the zero-norm
                    # check since `nan < 1e-9` is False and would corrupt the
                    # score via np.dot). Numerically skipped, but log it so a bad
                    # stored vector is observable, consistent with the domain leg.
                    logger.warning(
                        "Candidate %s has a degenerate embedding (norm≈0 or "
                        "non-finite); skipping interest-vector score for it.",
                        paper.get("doi", "?"),
                    )
                    continue
                interest_score = float(
                    np.dot(cand_arr, _interest_arr) / (_cand_norm * _interest_norm)
                )
                # Store raw interest cosine for Bundle B's score decomposition.
                paper["_interest_score"] = interest_score
                paper["similarity_score"] = (
                    paper["similarity_score"]
                    + interest_weight * _interest_gate * interest_score
                )

    # 2c. Bundle D: optional graph-signal additive scores.
    #     Runs only when graph_signals dict AND graph_weights are provided AND
    #     at least one weight is non-zero.  Weighted sum (NOT RRF).
    #     All weights default 0.0 → byte-for-byte regression preserved.
    _gw_entity = 0.0
    _gw_citation = 0.0
    _gw_authors = 0.0
    _graph_active = False
    if graph_signals is not None and graph_weights is not None:
        _gw_entity = float(graph_weights.get("entity_overlap", 0.0))
        _gw_citation = float(graph_weights.get("citation", 0.0))
        _gw_authors = float(graph_weights.get("shared_authors", 0.0))
        _graph_active = (_gw_entity > 0.0 or _gw_citation > 0.0 or _gw_authors > 0.0)

    if _graph_active:
        for paper in scored:
            doi = paper.get("doi") or paper.get("id", "")
            sig = graph_signals.get(doi, {}) if graph_signals else {}  # type: ignore[union-attr]

            # Normalize each raw count to [0, 1] before weighting.
            entity_norm = min(
                float(sig.get("n_shared_entities", 0)) / _GRAPH_ENTITY_NORM_CAP, 1.0
            )
            citation_total = (
                float(sig.get("n_cites_in_library", 0))
                + float(sig.get("n_cited_by_library", 0))
                + float(sig.get("n_extends_in_library", 0))
                + float(sig.get("n_compares_to_library", 0))
            )
            citation_norm = min(citation_total / _GRAPH_CITATION_NORM_CAP, 1.0)
            authors_norm = min(
                float(sig.get("n_shared_authors", 0)) / _GRAPH_AUTHORS_NORM_CAP, 1.0
            )

            # Store raw normalised values for score_breakdown assembly below.
            paper["_graph_entity_norm"] = entity_norm
            paper["_graph_citation_norm"] = citation_norm
            paper["_graph_authors_norm"] = authors_norm

            # Additive contribution to composite score (weighted sum).
            paper["similarity_score"] = (
                paper.get("similarity_score", 0.0)
                + _gw_entity * entity_norm
                + _gw_citation * citation_norm
                + _gw_authors * authors_norm
            )

            # Annotations for Bundle B's "Why this paper?" panel.
            paper["graph_shared_authors_sample"] = sig.get("shared_authors_sample", [])
            paper["graph_shared_entities"] = sig.get("shared_entity_canonical_ids", [])

    # 3. Bundle B: attach per-signal score_breakdown to every paper.
    #    Defaults preserve v0.8 behavior — breakdown is additive metadata only.
    for paper in scored:
        # domain_context weighted contribution (0.0 when not used)
        domain_raw = paper.get("_domain_score", 0.0)
        domain_contribution = round(float(domain_raw * domain_context_weight), 3)
        # Bundle C: cluster_centroid contribution
        cluster_raw = paper.get("_cluster_score", 0.0)
        cluster_contribution = round(float(cluster_raw * cluster_centroid_weight), 3)
        # Bundle J2: interest-vector (feedback) contribution. The gate is folded
        # into the contribution exactly as it was added to similarity_score above,
        # so vector_only stays the pure base score. 0.0 when the leg was inert.
        interest_raw = paper.get("_interest_score", 0.0)
        interest_contribution = round(
            float(interest_raw * interest_weight * _interest_gate), 3
        )
        # Bundle D: graph signal contributions
        graph_entity_contribution = round(
            float(paper.get("_graph_entity_norm", 0.0)) * _gw_entity, 3
        )
        graph_citation_contribution = round(
            float(paper.get("_graph_citation_norm", 0.0)) * _gw_citation, 3
        )
        graph_authors_contribution = round(
            float(paper.get("_graph_authors_norm", 0.0)) * _gw_authors, 3
        )
        # vector = base score excluding all additive signals
        vector_only = round(
            float(paper.get("similarity_score", 0.0))
            - domain_contribution
            - cluster_contribution
            - interest_contribution
            - graph_entity_contribution
            - graph_citation_contribution
            - graph_authors_contribution,
            3,
        )
        paper["score_breakdown"] = {
            "vector": vector_only,
            "domain_context": domain_contribution,
            "cluster_centroid": cluster_contribution,
            "feedback": interest_contribution,
            "graph_entity_overlap": graph_entity_contribution,
            "graph_citation": graph_citation_contribution,
            "graph_shared_authors": graph_authors_contribution,
        }

    # 4. Sort descending by score
    scored.sort(key=lambda p: p.get("similarity_score", 0.0), reverse=True)
    # 5. LLM rationale for top-K
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
        # C1: domain_context is a free-text user config paragraph; sanitize it
        # before prepending so a role marker there cannot break out of the
        # system prompt. Short metadata → default fence-stripping.
        system = sanitize_for_prompt(domain_context) + "\n\n" + system

    try:
        raw = llm.complete(system, user_prompt, max_tokens=prompt.max_tokens)
        from scripts.llm.llm_client import parse_llm_json
        return parse_llm_json(raw)
    except Exception as exc:
        # P4.2: escalate in --strict (raises) so a rationale-LLM failure is not
        # silently returned as empty. Non-strict UX is preserved: log at WARNING
        # and return {} (rationale fields become empty strings downstream).
        strict_fallback(logger, f"LLM rationale generation failed: {exc}", exc)
        return {}


def _paper_query_text(paper: dict[str, Any]) -> str:
    """Build a query string for embedding lookup."""
    parts = []
    if paper.get("title"):
        parts.append(paper["title"])
    if paper.get("abstract"):
        parts.append(paper["abstract"][:500])
    return " ".join(parts)
