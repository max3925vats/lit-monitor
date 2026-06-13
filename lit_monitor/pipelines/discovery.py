"""
Discovery pipeline.
Discovery + ingestion in two blocks:
DISCOVERY BLOCK:
  1. run_searches() + run_researcher_searches() from topics/researchers config
  2. Filter DOIs already in state DB
  3. Embed each new candidate (against live ChromaDB)
  4. Sort by cosine similarity to existing knowledge base
  5. LLM rationale for top-K
  6. Write Literature/Digests/Discovery_{date}.md
  7. Store new DOIs as status='discovered'
INGESTION BLOCK:
  8. Poll Zotero for items added since the last recorded library version
     (stored as kv_store key 'zotero_library_version' in state DB).
     First run: stores baseline version and skips processing.
  9. For each new item with markdown attachment:
       - get_markdown_attachment → extract_paper (simple + complex phases)
       - write_paper_note → add_paper to ChromaDB → relink (new note only)
  10. Persist updated Zotero library version to state DB
dry_run=True: discovery block runs (search + rank + digest); NO state DB writes,
  NO notes written, NO `last_run_date` update.
screen_all=True: Send all new results to LLM rationale (not just top-K).
"""
from __future__ import annotations

import json
import logging
import math
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import requests

from lit_monitor.api.queries import get_graph_signals_for_candidate
from lit_monitor.core.doi import normalize_doi
from lit_monitor.core.doi_resolver import resolve_doi
from lit_monitor.core.item_router import detect_review
from lit_monitor.core.strict_mode import strict_fallback
from lit_monitor.graph import safe_graph_db
from lit_monitor.llm.extractor import extract_paper
from lit_monitor.llm.llm_client import RateLimitError
from lit_monitor.llm.ranker import (
    _GRAPH_AUTHORS_NORM_CAP,
    _GRAPH_CITATION_NORM_CAP,
    _GRAPH_ENTITY_NORM_CAP,
    rank_papers,
)
from lit_monitor.obsidian_tools.relink import relink_note
from lit_monitor.output.obsidian_writer import write_paper_note
from lit_monitor.pipelines._ingest import (
    build_graph_tuples,
    build_ner_mention_edges,
    index_embeddings_and_mark_phases,
    maybe_extract_llm_relationships,
)
from lit_monitor.pipelines.brain_build import (
    RateLimitExhausted,  # P5.5: reuse the P4.5 domain exception (catchable in API/web)
    _check_item_quality,  # A2: metadata quality gate
    _extract_keywords,
    _llm_model_str,  # F15: handles single client and dict[int, LLMClient]
    _llm_provider_str,  # F15: handles single client and dict[int, LLMClient]
    _now,
    _paper_embed_text,
    _parse_year,
)
from lit_monitor.search.researcher_tracker import run_researcher_searches
from lit_monitor.search.search_runner import DEFAULT_DATABASES, filter_known_dois, run_searches
from lit_monitor.search.semantic_scholar import enrich_paper

logger = logging.getLogger(__name__)

# Historical default for the notification deep-link target; used when no
# [server] block is configured or the [server] extra isn't installed.
_DEFAULT_APP_URL = "http://localhost:8765"


def _resolve_app_url() -> str:
    """Build the web-app base URL for notification deep links from the persisted
    ``[server]`` config block, falling back to the historical default when the
    block is absent or the optional ``[server]`` extra (which reads it) is not
    installed. Isolated in its own try/except so a failure here never suppresses
    the notification itself.
    """
    try:
        from lit_monitor.server.config_io import load_server_config

        srv = load_server_config()
        host = srv.get("host") or "localhost"
        port = srv.get("port") or 8765
        return f"http://{host}:{port}"
    except Exception as exc:  # noqa: BLE001 — [server] extra optional; keep default
        logger.debug("discovery: falling back to default app_url (%s): %s", _DEFAULT_APP_URL, exc)
        return _DEFAULT_APP_URL


@dataclass
class _RunSummary:
    new_papers_found: int = 0
    papers_ingested: int = 0
    papers_failed: int = 0
    digest_path: str = ""
    errors: list[str] = field(default_factory=list)
def run_discovery(
    config,
    state_db,
    zotero_client,
    embeddings_db,
    llm,
    dry_run: bool = False,
    screen_all: bool = False,
    top_k: int = 20,
    sim_threshold: float = 0.3,
    rag_mode: str | None = None,
) -> _RunSummary:
    """
    Run the discovery + ingestion pipeline.
    Parameters
    ----------
    config:
        Config object.
    state_db:
        StateDB instance.
    zotero_client:
        ZoteroClient instance.
    embeddings_db:
        EmbeddingsDB instance.
    llm:
        LLMClient instance.
    dry_run:
        If True, run discovery only; no writes to state DB or vault.
    screen_all:
        If True, generate LLM rationale for all results (not just top-K).
    top_k:
        Number of top results for LLM rationale (when screen_all is False).
    sim_threshold:
        Minimum similarity score for a result to be highlighted.
    rag_mode:
        One of "vector", "graph", "hybrid". Default reads from
        config.retrieval.default_mode; falls back to "vector".
        When "graph" or "hybrid", graph proximity is used for paper ranking.
    Returns
    -------
    _RunSummary
    """
    # G9: resolve rag_mode once for the whole run.
    _cfg_mode = (
        getattr(getattr(config, "retrieval", None), "default_mode", "vector")
        if config is not None else "vector"
    )
    _rag_mode = rag_mode or _cfg_mode or "vector"
    summary = _RunSummary()
    run_id = str(uuid.uuid4())
    # P1: track structured discovery-run row id (None in dry_run mode).
    _disc_run_id: int | None = None
    if not dry_run:
        state_db.start_run(run_id, "discovery")
        # P1: open the structured discovery_runs row; finish in finally below.
        _disc_run_id = state_db.start_discovery_run(
            {
                "rag_mode": _rag_mode,
                "top_k": top_k,
                "screen_all": screen_all,
                "sim_threshold": sim_threshold,
            },
            run_log_id=run_id,  # link discovery_runs → run_log for unified history
        )

    # A4: compute search window from last successful run date.
    # Cap at 90 days to avoid overwhelming databases after a long gap.
    _MAX_SINCE_DAYS = 90
    last_run_date_str = state_db.get_kv("last_run_date")
    if last_run_date_str:
        try:
            last_run = datetime.fromisoformat(last_run_date_str).date()
            since_days = min((date.today() - last_run).days + 1, _MAX_SINCE_DAYS)
            logger.warning("Search window: %d days (last run: %s)", since_days, last_run_date_str)
        except ValueError:
            since_days = 14
            logger.warning("Could not parse last_run_date %r — defaulting to 14 days", last_run_date_str)
    else:
        since_days = 14
        logger.warning("No last_run_date stored — defaulting search window to 14 days")

    # ----------------------------------------------------------------
    # Discovery block
    # ----------------------------------------------------------------
    try:
        try:
            raw_papers, search_errors = _run_discovery(config, since_days=since_days)
            summary.errors.extend(search_errors)
            # Bundle K-b: SUPPLEMENT topic results with graph-expanded exploration
            # searches for under-engaged clusters. Fully gated + non-fatal: with no
            # clusters / no graph / budget 0 this is a no-op (zero extra searches),
            # so a fresh user sees no behaviour change. Exploration candidates are
            # appended (never replace topic results) and capped to ~budget_pct of
            # the topic pool; they de-dup + rank through the same downstream path.
            _explore_papers = _run_exploration_searches(
                config, state_db, raw_papers, since_days=since_days
            )
            if _explore_papers:
                raw_papers = raw_papers + _explore_papers
            # A4: advance date after successful discovery so subsequent runs compute
            # the correct search window.
            # H1: dry_run is supposed to be DB-read-only (per module docstring) — gate
            # the write so dry runs do not silently mutate kv_store.
            if not dry_run:
                state_db.set_kv("last_run_date", str(date.today()))
        except Exception as exc:
            logger.error("Discovery searches failed: %s", exc, exc_info=True)
            summary.errors.append(f"discovery: {exc}")
            raw_papers = []
        new_papers = filter_known_dois(raw_papers, state_db)
        summary.new_papers_found = len(new_papers)
        logger.warning("Discovery: %d new results after dedup", len(new_papers))
        # Bundle D: fetch graph signals once per candidate when any graph weight > 0.
        _graph_signals, _graph_weights = _fetch_graph_signals_if_needed(
            new_papers, config, state_db
        )
        # Bundle J2: load the global interest vector + feedback weight once.
        # Non-fatal / inert when the weight is 0.0 or no vector is stored yet;
        # the cold-start soft-gate is applied inside rank_papers via n_events.
        _interest_vec, _interest_n_events, _interest_weight = (
            _load_interest_signal_if_needed(config, state_db)
        )

        # G9: rank by similarity — vector path is unchanged; graph/hybrid path uses
        # the branch helper to score candidates by knowledge-graph proximity.
        if _rag_mode == "vector":
            ranked = rank_papers(
                new_papers,
                embeddings_db,
                llm,
                top_k=len(new_papers) if screen_all else top_k,
                graph_signals=_graph_signals,
                graph_weights=_graph_weights,
                interest_vec=_interest_vec,
                interest_vec_n_events=_interest_n_events,
                interest_weight=_interest_weight,
            )
        else:
            # graph or hybrid: score each new paper against the knowledge graph,
            # then add LLM rationale for the top results.
            ranked = _rank_papers_graph(
                new_papers,
                embeddings_db=embeddings_db,
                llm=llm,
                top_k=len(new_papers) if screen_all else top_k,
                rag_mode=_rag_mode,
                graph_signals=_graph_signals,
                graph_weights=_graph_weights,
                interest_vec=_interest_vec,
                interest_vec_n_events=_interest_n_events,
                interest_weight=_interest_weight,
            )
        # P1: record each ranked candidate in discovery_paper_results (non-dry only).
        if not dry_run and _disc_run_id is not None:
            for paper in ranked:
                state_db.add_discovery_paper(
                    run_id=_disc_run_id,
                    doi=paper.get("doi", ""),
                    title=paper.get("title", ""),
                    score=float(paper.get("similarity_score", 0.0)),
                    rationale=paper.get("llm_rationale", ""),
                    # ingested is determined later; set False here — updated by ingestion
                    # tracking in _run_ingestion or subsequent passes (P2+).
                    ingested=False,
                    # Bundle B: persist per-signal breakdown for the HTTP endpoint + web UI.
                    score_breakdown=paper.get("score_breakdown"),
                )
        # P10b: gate the digest write on discovery.digest.auto_write (default True).
        if _resolve_digest_auto_write(config):
            digest_path = _write_digest(
                ranked, config, sim_threshold, dry_run=dry_run,
            )
            summary.digest_path = digest_path
        else:
            logger.info(
                "P10b: digest auto-write disabled; use "
                "'lit-monitor discovery export-md --run latest' to render."
            )
        # Persist discovered papers
        if not dry_run:
            for paper in new_papers:
                doi = paper.get("doi", "")
                if doi:
                    # B4: similarity_score / llm_rationale are NOT papers columns
                    # (they were silently dropped here); the ranked score+rationale
                    # are persisted in discovery_paper_results via add_discovery_paper
                    # above, so they are not passed to upsert_paper.
                    state_db.upsert_paper({
                        "doi": doi,
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "year": paper.get("year", 0),
                        "status": "discovered",
                        "source_type": "paper",
                        "first_seen_date": str(date.today()),
                        "last_updated": _now(),
                    })
        # ----------------------------------------------------------------
        # Ingestion block
        # ----------------------------------------------------------------
        if not dry_run:
            _run_ingestion(
                config=config,
                state_db=state_db,
                zotero_client=zotero_client,
                embeddings_db=embeddings_db,
                llm=llm,
                summary=summary,
                run_id=run_id,
            )
            state_db.finish_run(
                run_id,
                status="complete",
                processed=summary.papers_ingested,
                failed=summary.papers_failed,
                errors=summary.errors,
            )
    finally:
        # P1: always close the structured discovery_runs row, even on exception.
        if not dry_run and _disc_run_id is not None:
            _disc_status = "success" if not summary.errors else "error"
            state_db.finish_discovery_run(
                _disc_run_id,
                status=_disc_status,
                total_found=summary.new_papers_found,
                total_ingested=summary.papers_ingested,
            )
            from lit_monitor.api._stats_snapshot import record_corpus_snapshot
            record_corpus_snapshot(state_db, run_type="discovery", run_id=str(_disc_run_id))
        # P2: fire OS notification after the run row is committed.
        # Wrapped in try/except so any failure here is non-fatal.
        try:
            from lit_monitor.notify.os_notification import notify_discovery_complete
            # Defensive getattr chain: handles configs that predate the
            # discovery.notify block — falls through with enabled=False.
            notify_cfg = getattr(getattr(config, "discovery", None), "notify", None)
            notify_discovery_complete(
                run_id=_disc_run_id if _disc_run_id is not None else 0,
                paper_count=summary.papers_ingested,
                app_url=_resolve_app_url(),
                enabled=getattr(notify_cfg, "enabled", False) if notify_cfg else False,
                on_zero_results=(
                    getattr(notify_cfg, "on_zero_results", True) if notify_cfg else True
                ),
            )
        except Exception as exc:
            logger.warning("P2: notify dispatch failed (non-fatal): %s", exc)
    return summary

# ---------------------------------------------------------------------------
# G9: graph/hybrid ranking helper
# ---------------------------------------------------------------------------

def _rank_papers_graph(
    candidates: list[dict],
    *,
    embeddings_db,
    llm,
    top_k: int,
    rag_mode: str,
    graph_signals: dict[str, dict] | None = None,   # Bundle D
    graph_weights: dict[str, float] | None = None,  # Bundle D
    interest_vec: np.ndarray | None = None,          # Bundle J2
    interest_vec_n_events: int = 0,                  # Bundle J2
    interest_weight: float = 0.0,                    # Bundle J2
) -> list[dict]:
    """G9: rank candidates using graph or hybrid retrieval for the discovery loop.

    For each candidate paper, we use retrieve_doi_candidates to ask the graph
    (or vector+graph fused via RRF) how many papers in the existing knowledge
    base are adjacent. The count / score becomes the similarity_score, and the
    LLM rationale is generated for the top-K as in the vector path.

    Falls back to the standard rank_papers() if graph_db is unavailable.

    Bundle D: graph_signals and graph_weights are forwarded to the underlying
    rank_papers() fallback path and also applied to the graph-scored papers
    via a synthetic rank_papers() call at the end.

    Bundle J2: the interest-vector kwargs are forwarded to the rank_papers()
    fallback path (graph_db unavailable). The graph-active path scores by graph
    neighbour counts, not candidate embeddings, so the interest leg does not
    apply there — consistent with the domain/cluster legs, which the graph path
    also does not run.
    """
    from lit_monitor.retrieval.branch import retrieve_doi_candidates
    graph_db = safe_graph_db()
    try:
        if graph_db is None:
            logger.warning(
                "G9: graph_db unavailable for rag_mode=%r — falling back to vector ranking",
                rag_mode,
            )
            return rank_papers(
                candidates, embeddings_db, llm, top_k=top_k,
                graph_signals=graph_signals, graph_weights=graph_weights,
                interest_vec=interest_vec,
                interest_vec_n_events=interest_vec_n_events,
                interest_weight=interest_weight,
            )

        scored: list[dict] = []
        for paper in candidates:
            doi = paper.get("doi", "")
            embed_text = (paper.get("title", "") + " " + paper.get("abstract", "")).strip()
            # Retrieve and score by number of graph neighbours.
            pairs = retrieve_doi_candidates(
                rag_mode,
                seed_doi=doi if doi else None,
                query_text=embed_text or None,
                embeddings_db=embeddings_db,
                graph_db=graph_db,
                k=1,  # we only need the top score as the relevance signal
            )
            score = pairs[0][1] if pairs else 0.0
            scored.append({**paper, "similarity_score": score})

        # Bundle D: apply graph signal additive weights to graph-ranked papers.
        # We reuse rank_papers's graph-signal logic by delegating to _apply_graph_signals.
        if graph_signals is not None and graph_weights is not None:
            _apply_graph_signals_to_scored(scored, graph_signals, graph_weights)

        # Attach score_breakdown to every paper (matches rank_papers behavior).
        _attach_score_breakdown(scored, graph_weights=graph_weights)

        # Sort descending by score.
        scored.sort(key=lambda p: p.get("similarity_score", 0.0), reverse=True)

        # Add LLM rationale for top-K (mirrors rank_papers behaviour).
        # C2 fix: _get_rationales requires `domain_context: str` — the previous
        # call omitted it and TypeError was silently swallowed by the broad
        # except below, leaving every graph/hybrid paper with rationale="".
        _top = scored[:top_k]
        try:
            from lit_monitor.llm.ranker import _get_rationales  # type: ignore[attr-defined]
            rationales = _get_rationales(_top, llm, domain_context="")
            for p in _top:
                p["llm_rationale"] = rationales.get(p.get("doi", ""), "")
        except Exception as exc:
            logger.warning("LLM rationale generation failed in graph ranking: %s", exc)
            for p in _top:
                p.setdefault("llm_rationale", "")

        return scored
    finally:
        if graph_db is not None:
            try:
                graph_db.close()
            except Exception:  # pragma: no cover
                pass


def _fetch_graph_signals_if_needed(
    candidates: list[dict[str, Any]],
    config: Any,
    state_db: Any,
) -> tuple[dict[str, dict] | None, dict[str, float] | None]:
    """Bundle D: fetch graph signals for all candidates when any graph weight > 0.

    Returns (graph_signals, graph_weights) where graph_signals maps each
    candidate DOI to its signals dict.  Returns (None, None) when weights are
    all zero or graph_db is unavailable — ensures zero overhead for users who
    haven't opted in.

    Parameters
    ----------
    candidates:
        List of candidate paper dicts from the discovery search.
    config:
        Config object (used to read ranking.weights.*).
    state_db:
        StateDB instance (used to enumerate library DOIs).
    """
    # Read weights from config; default all to 0.0.
    weights_ns = getattr(getattr(config, "ranking", None), "weights", None)
    w_entity = float(getattr(weights_ns, "graph_entity_overlap", 0.0))
    w_citation = float(getattr(weights_ns, "graph_citation", 0.0))
    w_authors = float(getattr(weights_ns, "graph_shared_authors", 0.0))

    if w_entity == 0.0 and w_citation == 0.0 and w_authors == 0.0:
        # All weights zero — skip graph I/O entirely.
        return None, None

    graph_db = safe_graph_db()
    if graph_db is None:
        logger.debug("_fetch_graph_signals_if_needed: graph_db unavailable, skipping.")
        return None, None

    try:
        # Fetch the current library DOIs (papers table, all known DOIs).
        library_dois = list(state_db.known_dois())

        graph_signals: dict[str, dict] = {}
        for paper in candidates:
            doi = paper.get("doi") or paper.get("id", "")
            if doi:
                graph_signals[doi] = get_graph_signals_for_candidate(
                    doi, graph_db, library_dois
                )

        graph_weights: dict[str, float] = {
            "entity_overlap": w_entity,
            "citation": w_citation,
            "shared_authors": w_authors,
        }
        return graph_signals, graph_weights

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_fetch_graph_signals_if_needed: failed (non-fatal), skipping graph signals: %s",
            exc,
        )
        return None, None
    finally:
        try:
            graph_db.close()
        except Exception:  # noqa: BLE001
            pass


def _load_interest_signal_if_needed(
    config: Any,
    state_db: Any,
) -> tuple[np.ndarray | None, int, float]:
    """Bundle J2: load the global Rocchio interest vector + its weight.

    Returns ``(interest_vec, n_events, weight)`` to forward into ``rank_papers``.
    The leg is INERT (returns ``(None, 0, 0.0)``) whenever:
      - ``ranking.weights.feedback`` is 0.0 (default — opt-in only), OR
      - no interest vector has been stored yet (table empty / never recomputed).
    Loading is non-fatal: any error degrades to no-op so a missing/corrupt
    interest row can never break a discovery run. The cold-start soft-gate is
    applied later inside ``rank_papers`` via ``n_events``.
    """
    weights_ns = getattr(getattr(config, "ranking", None), "weights", None)
    weight = float(getattr(weights_ns, "feedback", 0.0))
    if weight <= 0.0:
        # Opt-in only: skip the DB read entirely when the weight is off.
        return None, 0, 0.0
    try:
        stored = state_db.get_interest_vector("global")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_load_interest_signal_if_needed: failed to load interest vector "
            "(non-fatal), skipping feedback signal: %s",
            exc,
        )
        return None, 0, 0.0
    if stored is None:
        # No vector recomputed yet — leg stays off (weight has nothing to apply).
        return None, 0, weight
    vec, n_events = stored
    return vec, int(n_events), weight


def _apply_graph_signals_to_scored(
    scored: list[dict],
    graph_signals: dict[str, dict],
    graph_weights: dict[str, float],
) -> None:
    """Bundle D: in-place additive graph score application for _rank_papers_graph path.

    Mirrors the logic in rank_papers() so the graph/hybrid code path produces
    consistent score_breakdown shapes.

    Parameters
    ----------
    scored:
        List of paper dicts with similarity_score already set.
    graph_signals:
        Mapping from candidate DOI to signals dict.
    graph_weights:
        Mapping with keys entity_overlap, citation, shared_authors.
    """
    w_entity = float(graph_weights.get("entity_overlap", 0.0))
    w_citation = float(graph_weights.get("citation", 0.0))
    w_authors = float(graph_weights.get("shared_authors", 0.0))

    for paper in scored:
        doi = paper.get("doi") or paper.get("id", "")
        sig = graph_signals.get(doi, {})

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

        paper["_graph_entity_norm"] = entity_norm
        paper["_graph_citation_norm"] = citation_norm
        paper["_graph_authors_norm"] = authors_norm

        paper["similarity_score"] = (
            paper.get("similarity_score", 0.0)
            + w_entity * entity_norm
            + w_citation * citation_norm
            + w_authors * authors_norm
        )
        paper["graph_shared_authors_sample"] = sig.get("shared_authors_sample", [])
        paper["graph_shared_entities"] = sig.get("shared_entity_canonical_ids", [])


def _attach_score_breakdown(
    scored: list[dict],
    *,
    graph_weights: dict[str, float] | None = None,
) -> None:
    """Bundle D: attach score_breakdown to each paper after graph-path scoring.

    Produces the same shape as rank_papers()'s Bundle B block so the HTTP
    endpoint and UI see a consistent structure regardless of the ranking path.

    Parameters
    ----------
    scored:
        List of paper dicts already augmented by _apply_graph_signals_to_scored.
    graph_weights:
        Optional dict with entity_overlap/citation/shared_authors weights.
    """
    gw = graph_weights or {}
    w_entity = float(gw.get("entity_overlap", 0.0))
    w_citation = float(gw.get("citation", 0.0))
    w_authors = float(gw.get("shared_authors", 0.0))

    for paper in scored:
        graph_entity = round(float(paper.get("_graph_entity_norm", 0.0)) * w_entity, 3)
        graph_citation = round(float(paper.get("_graph_citation_norm", 0.0)) * w_citation, 3)
        graph_auth = round(float(paper.get("_graph_authors_norm", 0.0)) * w_authors, 3)
        vector_only = round(
            float(paper.get("similarity_score", 0.0))
            - graph_entity - graph_citation - graph_auth,
            3,
        )
        paper["score_breakdown"] = {
            "vector": vector_only,
            "domain_context": 0.0,
            "cluster_centroid": 0.0,
            "graph_entity_overlap": graph_entity,
            "graph_citation": graph_citation,
            "graph_shared_authors": graph_auth,
        }


# ---------------------------------------------------------------------------
# Bundle A: soft pre-rank domain filter helpers
# ---------------------------------------------------------------------------

def _apply_soft_domain_filter(
    candidates: list[dict[str, Any]],
    domain_emb: np.ndarray,
    threshold: float = 0.35,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bundle A: split candidates into in-domain and off-domain pools.

    Computes cosine(candidate._embedding, domain_emb) for each candidate.
    Candidates above/at threshold → in_domain; below → off_domain.
    Candidates with no ``_embedding`` key default to in_domain (fail-safe —
    we can't evaluate them, so we keep them rather than silently discard).

    The ``_domain_cosine`` key is written onto each candidate so downstream
    assembly can sort the off-domain pool by domain relevance.

    Parameters
    ----------
    candidates:
        List of paper dicts.
    domain_emb:
        Pre-computed embedding of the domain_context paragraph (numpy array).
    threshold:
        Cosine similarity cutoff.  Candidates with cosine >= threshold go to
        in_domain; candidates with cosine < threshold go to off_domain.

    Returns
    -------
    (in_domain, off_domain)
        Two lists of paper dicts (mutually exclusive, together exhaustive).
    """
    in_domain: list[dict[str, Any]] = []
    off_domain: list[dict[str, Any]] = []
    _domain_norm = float(np.linalg.norm(domain_emb))

    for cand in candidates:
        emb = cand.get("_embedding")
        if emb is None:
            # Can't evaluate cosine → conservative default: keep in in_domain.
            in_domain.append(cand)
            continue
        arr = np.asarray(emb, dtype=np.float32)
        _cand_norm = float(np.linalg.norm(arr))
        if _cand_norm < 1e-9 or _domain_norm < 1e-9:
            # Zero-vector edge case → can't compute cosine; treat as in-domain.
            in_domain.append(cand)
            continue
        cosine = float(np.dot(arr, domain_emb) / (_cand_norm * _domain_norm))
        cand["_domain_cosine"] = cosine
        if cosine >= threshold:
            in_domain.append(cand)
        else:
            off_domain.append(cand)

    return in_domain, off_domain


def assemble_with_soft_floor(
    in_domain_ranked: list[dict[str, Any]],
    off_domain_ranked: list[dict[str, Any]],
    digest_size: int,
    min_off_domain_pct: float = 0.05,
) -> list[dict[str, Any]]:
    """Bundle A: merge in-domain and off-domain pools with a guaranteed off-domain floor.

    Guarantees that at least ``⌈digest_size × min_off_domain_pct⌉`` slots in the
    final list come from the off-domain pool — preserving serendipitous discoveries
    that might be filtered out purely on cosine grounds.

    Parameters
    ----------
    in_domain_ranked:
        Sorted (by score, desc) in-domain candidates.
    off_domain_ranked:
        Sorted (by score, desc) off-domain candidates.
    digest_size:
        Target number of results (typically ``config.discovery_top_k``).
    min_off_domain_pct:
        Fraction of digest_size reserved for off-domain candidates.
        Default 0.05 = 5%.

    Returns
    -------
    list[dict]
        Final assembled list of at most ``digest_size`` papers.
        Off-domain candidates appear at the end to preserve in-domain ordering.
    """
    min_off_slots = max(1, math.ceil(digest_size * min_off_domain_pct))

    # Take off-domain first (top by score) up to the reserved quota.
    off_take = off_domain_ranked[:min_off_slots]
    # Fill the rest from in-domain.
    remaining_slots = digest_size - len(off_take)
    in_take = in_domain_ranked[:remaining_slots]

    return in_take + off_take


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def _run_discovery(
    config, since_days: int = 14
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run topic + researcher searches and merge results.

    Parameters
    ----------
    since_days:
        Search window in days from today. Computed from the stored
        'last_run_date' kv_store entry so that skipped weeks are
        automatically caught up on the next run.
    """
    papers: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        papers.extend(run_searches(config, since_days=since_days))
    except Exception as exc:
        logger.warning("Topic searches failed: %s", exc)
        errors.append(f"topic_search: {exc}")
    try:
        papers.extend(run_researcher_searches(config, since_days=since_days))
    except Exception as exc:
        logger.warning("Researcher searches failed: %s", exc)
        errors.append(f"researcher_search: {exc}")
    # Deduplicate by DOI across both search types
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for p in papers:
        doi = p.get("doi", "")
        if doi and doi not in seen:
            seen.add(doi)
            deduped.append(p)
        elif not doi:
            deduped.append(p)  # keep no-DOI results (filtered later)
    return deduped, errors


# ---------------------------------------------------------------------------
# Bundle K-b: exploration budget — probe under-engaged clusters
# ---------------------------------------------------------------------------
#
# WHAT A "SLOT" IS HERE (DOCUMENTED — see the pooled question in the bundle
# report). The discovery search step has no fixed per-topic "slot" array: each
# topic search streams results into one DOI-keyed pool that is then ranked and
# truncated to ``top_k`` for the digest. So "reserve 20% of slots" is mapped
# onto the *candidate pool*: exploration-sourced candidates are capped at
# ``ceil(exploration_budget_pct * len(topic_results))`` and APPENDED to the
# topic pool (they never displace a topic result). They then flow through the
# same dedup → filter_known_dois → rank → top_k path as everything else, so the
# 20% is a soft reservation of pool share, realised downstream by ranking.
# ---------------------------------------------------------------------------


def _resolve_exploration_budget_pct(config: Any) -> float:
    """K-b: read ``feedback.exploration_budget_pct`` (default 0.20), clamped to
    ``[0, 1]``. Returns 0.0 (exploration OFF) on any read failure so a malformed
    config can never silently widen the budget."""
    try:
        fb = getattr(config, "feedback", None)
        val = float(getattr(fb, "exploration_budget_pct", 0.20)) if fb else 0.20
    except (TypeError, ValueError, AttributeError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return max(0.0, min(1.0, val))


def _run_exploration_searches(
    config: Any,
    state_db: Any,
    topic_results: list[dict[str, Any]],
    *,
    since_days: int,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """K-b: build + run graph-expanded exploration searches for under-engaged
    clusters, returning a capped, supplementary list of candidate paper dicts.

    GATING (all must hold, else returns ``[]`` with ZERO extra searches — a
    fresh user with no clusters/graph sees no behaviour change):
      * ``exploration_budget_pct > 0``;
      * at least one active cluster exists;
      * the KuzuDB graph is available (``safe_graph_db()`` is not None);
      * ``find_under_engaged_clusters`` returns at least one cluster;
      * those clusters yield at least one exploration query from the graph.

    Cap: exploration candidates are truncated to
    ``ceil(pct * len(topic_results))`` (at least 1 when any exploration query
    runs and the topic pool is non-empty). NO fake-fill: if the searches return
    nothing relevant, we return ``[]`` and discovery proceeds normally.

    Wholly non-fatal: any failure logs at WARNING and returns ``[]`` so a
    discovery run is never broken by exploration.
    """
    try:
        pct = _resolve_exploration_budget_pct(config)
        if pct <= 0.0:
            return []

        # K-b review #1: the exploration budget is sized as a fraction of the
        # topic pool — an empty pool means cap=0, so short-circuit BEFORE issuing
        # any cluster/graph/search work to avoid issuing-then-discarding external
        # exploration searches.
        if not topic_results:
            return []

        clusters = state_db.list_active_clusters()
        if not clusters:
            # Fresh user / no clustering yet → no exploration, no extra searches.
            return []

        graph_db = safe_graph_db()
        if graph_db is None:
            logger.debug("K-b: graph unavailable — skipping exploration.")
            return []

        try:
            from lit_monitor.clustering.atrophy import (
                compute_cluster_feedback_weights_from_db,
                find_under_engaged_clusters,
            )
            from lit_monitor.clustering.exploration import build_exploration_queries

            ref_now = now or datetime.now()
            floor = float(
                getattr(getattr(config, "feedback", None), "minimum_cluster_floor", 0.1)
            )
            quiet_weeks = int(
                getattr(getattr(config, "feedback", None), "cluster_quiet_weeks", 4)
            )

            last_surfaced = state_db.get_cluster_last_surfaced()
            feedback_weights = compute_cluster_feedback_weights_from_db(
                state_db, now=ref_now, floor=floor
            )
            under_engaged = find_under_engaged_clusters(
                last_surfaced,
                feedback_weights,
                now=ref_now,
                quiet_weeks=quiet_weeks,
                floor=floor,
            )
            if not under_engaged:
                logger.info("K-b: no under-engaged clusters — skipping exploration.")
                return []

            # Member DOIs per under-engaged cluster (for central-entity lookup).
            cluster_member_dois: dict[int, list[str]] = {}
            for cid in under_engaged:
                rows = state_db.get_cluster_assignments(cid)
                cluster_member_dois[cid] = [
                    r.get("doi") for r in rows if r.get("doi")
                ]

            queries = build_exploration_queries(cluster_member_dois, graph_db)
            if not queries:
                logger.info(
                    "K-b: %d under-engaged cluster(s) but no graph-expanded "
                    "queries — skipping exploration.",
                    len(under_engaged),
                )
                return []
        finally:
            try:
                graph_db.close()
            except Exception:  # noqa: BLE001 — defensive close
                pass

        # Run the exploration queries through the SAME search path as topics by
        # handing run_searches a config shim whose .topics is the query list.
        # run_searches reads only `.topics` off config (API keys come from
        # config.toml), so the shim is sufficient and reuses findpapers + S2.
        explore_cfg = SimpleNamespace(topics=list(queries))
        explore_results = run_searches(explore_cfg, since_days=since_days)
        # AR-5 Fix 1 (Finding 5 + 4): drop exploration candidates already present
        # in the topic pool BEFORE the budget cap, so duplicate hits never consume
        # budget slots and the configured exploration rate is actually honoured.
        # Compare with the canonical `normalize_doi` (not a bare `.lower()`) so
        # URL-wrapped / `doi:`-prefixed variants of the same DOI dedup too.
        _topic_dois = {
            nd for p in topic_results
            if (nd := normalize_doi(p.get("doi")))
        }
        explore_results = [
            p for p in explore_results
            if normalize_doi(p.get("doi")) not in _topic_dois
        ]
        if not explore_results:
            # NO fake-fill: exploration found nothing new → proceed normally.
            logger.info("K-b: exploration searches returned 0 new candidates.")
            return []

        # Cap exploration candidates to ~pct of the topic-search effort.
        cap = max(1, math.ceil(pct * len(topic_results))) if topic_results else 0
        if cap <= 0:
            return []
        capped = explore_results[:cap]
        for paper in capped:
            paper["_exploration"] = True  # provenance tag (non-load-bearing)
        logger.warning(
            "K-b: exploration reserved %d/%d slot(s) (budget %.0f%%) from %d "
            "under-engaged cluster(s); %d exploration candidate(s) supplement "
            "%d topic candidate(s).",
            cap, len(topic_results), pct * 100, len(under_engaged),
            len(capped), len(topic_results),
        )
        return capped
    except Exception as exc:  # noqa: BLE001 — exploration must never break discovery
        logger.warning("K-b: exploration step failed (non-fatal): %s", exc)
        return []


def _resolve_digest_auto_write(cfg: Any) -> bool:
    """P10b: read discovery.digest.auto_write from config; default True.

    Defensive: returns True on any AttributeError or missing key so the
    default (write the digest) is preserved for all pre-P10b configs.
    """
    try:
        digest_ns = getattr(getattr(cfg, "discovery", None), "digest", None)
        if digest_ns is None:
            return True
        val = getattr(digest_ns, "auto_write", True)
        return bool(val) if val is not None else True
    except Exception:  # noqa: BLE001 — defensive; must never suppress digest write
        return True


def _write_digest(
    ranked: list[dict[str, Any]],
    config,
    sim_threshold: float,
    dry_run: bool = False,
    n_databases: int | None = None,
) -> str:
    """Render and write the discovery digest note. Returns the note path.

    Delegates rendering to ``scripts.output.digest_renderer.render_digest``
    (single renderer, DR1).  n_databases defaults to
    len(DEFAULT_DATABASES) + 1 (for S2 supplementary search).
    """
    from lit_monitor.output.digest_renderer import render_digest  # DR1: single renderer

    run_date = str(date.today())
    vault_path = Path(config.obsidian.vault_path)
    digests_folder = vault_path / config.obsidian.digests_folder
    if not dry_run:
        digests_folder.mkdir(parents=True, exist_ok=True)

    _n_db = n_databases if n_databases is not None else len(DEFAULT_DATABASES) + 1
    content = render_digest(
        {},
        ranked,
        sim_threshold=sim_threshold,
        n_databases=_n_db,
        _today=run_date,
    )

    digest_path = digests_folder / f"Discovery_{run_date}.md"
    if not dry_run:
        digest_path.write_text(content, encoding="utf-8")
        logger.warning("Wrote discovery digest: %s", digest_path)
    return str(digest_path)


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------
def _run_ingestion(
    config,
    state_db,
    zotero_client,
    embeddings_db,
    llm,
    summary: _RunSummary,
    run_id: str,
) -> None:
    """
    Poll Zotero for new items and ingest papers with PDFs.

    Version-based polling: reads 'zotero_library_version' from the state DB
    kv_store. On first invocation (no stored version) the current library
    version is saved as a baseline and the function returns without processing
    — the full library is already handled by brain-build. Subsequent calls
    fetch only items added after the stored version and update the stored
    version on completion.
    """
    from lit_monitor.core.zotero_client import ZoteroClient
    # Version-based Zotero polling — compare against last-seen library version.
    # On the first run there is no baseline; store the current version and skip
    # processing (the full library is already handled by brain-build).
    last_version_str = state_db.get_kv("zotero_library_version")
    if last_version_str is None:
        current_version = zotero_client.get_current_version()
        state_db.set_kv("zotero_library_version", str(current_version))
        logger.warning("First discovery run: baseline Zotero version %d stored", current_version)
        return
    last_version = int(last_version_str)
    try:
        new_items = zotero_client.get_items_since(last_version)
    except Exception as exc:
        logger.warning("Zotero poll failed: %s", exc)
        return
    logger.warning("Zotero poll: %d new items since version %d", len(new_items), last_version)
    consecutive_429 = 0  # V-9: abort after 3 consecutive rate-limit hits
    # M4: (doi, discovered_topics) pairs — mirrors brain_build.py:run_brain_build
    # (parallel collection pattern; same name reused intentionally).
    _topics_batch: list[tuple[str, list[str]]] = []
    # G6: open the GraphDB once for this ingestion pass.  None when the
    # [graph] extra is absent — vector-only behaviour is unchanged.
    graph_db = safe_graph_db()
    try:
        for item in new_items:
            data = item.get("data", {})
            doi = (data.get("DOI") or "").strip() or None
            zotero_key = item.get("key", "")
            title = data.get("title", "")
            if not doi:
                # D2: try CrossRef before giving up — ZoteroClient is already
                # imported at the top of this function (line: from lit_monitor.core…).
                _authors = ZoteroClient.extract_authors(data)
                doi = resolve_doi(
                    title, _authors, _parse_year(data.get("date", ""))
                ) or None
                if not doi:
                    logger.debug("New Zotero item %s has no DOI — skipping", zotero_key)
                    continue
                logger.debug(
                    "CrossRef resolved DOI for new item %s (%s): %s",
                    zotero_key, title[:60], doi,
                )

            # A2: metadata quality gate — skip items with empty titles (and warn on
            # absent authors / abstracts) before doing any expensive work.
            ok, reason = _check_item_quality(data)
            if not ok:
                logger.debug("Quality filter: skipping new item %s (%r) — %s", zotero_key, title[:60], reason)
                continue

            # A3: resume check — if a previous run fully processed this item (e.g.
            # ingestion partially completed then was restarted), skip it now.
            progress = state_db.get_brain_build_progress(zotero_key)
            if progress and progress.get("fully_complete"):
                logger.debug("Ingestion resume: %s already fully complete — skipping", doi)
                continue

            # Paper-status dedup: skip items already processed by brain-build or a
            # prior complete ingestion run.
            existing = state_db.get_paper(doi)
            if existing and existing.get("status") not in (None, "discovered"):
                continue

            # A3: register this item in the progress table before doing any work
            # so a crash mid-item is recoverable on the next run.
            state_db.upsert_brain_build_progress(zotero_key, doi)

            try:
                # M1: read markdown attachment from Zotero local storage.
                fulltext = zotero_client.get_markdown_attachment(zotero_key)
                if fulltext is None:
                    logger.debug(
                        "No markdown attachment for new item %s — skipping "
                        "(attach a .md file in Zotero to enable extraction)",
                        doi,
                    )
                    continue
                ocr_heavy = False  # markdown attachments are never OCR-heavy
                authors = ZoteroClient.extract_authors(data)
                year = _parse_year(data.get("date", ""))
                journal = data.get("publicationTitle", "")
                abstract = data.get("abstractNote", "") or ""
                keywords = _extract_keywords(data)
                # L1: pre-extraction S2 enrichment (mirrors brain_build H3/H4).
                # Enrichment happens before extract_paper so detect_review() can
                # set the correct source_type on the first DB write.
                # H1: wrap so a transient S2 outage doesn't waste LLM tokens by
                # marking the paper failed; narrow exception list mirrors
                # brain_build for cross-pipeline consistency.
                try:
                    s2_data = enrich_paper(doi)
                except (
                    requests.exceptions.RequestException,
                    ConnectionError,
                    TimeoutError,
                    json.JSONDecodeError,
                    RuntimeError,
                ) as exc:
                    logger.warning("S2 enrichment failed for %s (non-fatal): %s", doi, exc)
                    s2_data = {}
                s2_pub_types = s2_data.get("s2_publication_types") if s2_data else None
                source_type = "paper"
                if detect_review(s2_pub_types, title=title, abstract=abstract, journal=journal):
                    logger.debug("detect_review: classifying %s as review article", doi)
                    source_type = "review"
                existing_extraction = state_db.get_extraction_json(doi) or {}
                # M3: determine which phases still need running based on progress row.
                _prog = state_db.get_brain_build_progress(zotero_key) or {}
                phases_needed: list[str] = []
                if not _prog.get("simple_complete"):
                    phases_needed.append("simple")
                if not _prog.get("complex_complete"):
                    phases_needed.append("complex")
                if not phases_needed:
                    phases_needed = ["simple", "complex"]
                # M5: forward per-mode chunk_chars from extraction.yaml to simple phase.
                _ing_cfg = getattr(config, "ingestion", None)
                _chunk_chars = getattr(_ing_cfg, "chunk_chars", None)
                extraction = extract_paper(
                    fulltext,
                    llm,
                    content_type=source_type,
                    ocr_heavy=ocr_heavy,
                    phases=tuple(phases_needed),
                    existing_extraction=existing_extraction if existing_extraction else None,
                    chunk_chars_override=int(_chunk_chars) if isinstance(_chunk_chars, (int, float)) else None,
                )
                # Merge S2 data into extraction so it's persisted in the same upsert.
                if s2_data:
                    extraction.update(s2_data)
                # Audit R28 fix: defer phase marking until AFTER the paper-level
                # embed has succeeded.  Marking here would leave the paper stuck
                # at fully_complete=1 + status='error' if add_paper raises, and a
                # subsequent --resume would silently skip the retry.
                _phases_to_mark = [
                    phase for phase in phases_needed
                    if f"_{phase}_error" not in extraction
                ]

                # H1: persist the extraction payload but defer the status flip until
                # after the embed step succeeds.  Pre-fix, status='extraction_complete'
                # was written here and an add_paper failure would leave the row in a
                # 'looks done but has zero embeddings' state that --resume skipped.
                state_db.upsert_paper({
                    "doi": doi,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "journal": journal,
                    "zotero_key": zotero_key,
                    "source_type": source_type,
                    "extraction_json": json.dumps(extraction),
                    "extraction_provider": _llm_provider_str(llm),
                    "extraction_model": _llm_model_str(llm),
                    "keywords_json": keywords,
                    "first_seen_date": str(date.today()),
                    "last_updated": _now(),
                })
                # Assign vocabulary themes from Zotero tags (B2).
                from lit_monitor.vocabulary.normalizer import assign_themes
                themes = assign_themes(keywords)
                paper_record = {
                    "doi": doi,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "journal": journal,
                    "zotero_key": zotero_key,
                    "first_seen_date": str(date.today()),
                    "keywords": keywords,
                    "themes": themes,
                    "tracked_author": item.get("tracked_author", False),
                    "fulltext_analyzed": True,
                    # N8d: pass actual provider/model so the template doesn't fall
                    # back to the (incorrect) global config attribute.
                    "extraction_provider": _llm_provider_str(llm),
                    "extraction_model": _llm_model_str(llm),
                }
                # P10: gate the inline note write on discovery.notes.auto_write_per_paper
                # (default TRUE — no behaviour change on upgrade).  When false, note
                # write is deferred to `lit-monitor obsidian sync`; notes_synced stays 0.
                # R28 GUARD: this block sits BEFORE index_embeddings_and_mark_phases so
                # the embed/graph phase marks are always written regardless of the flag.
                _auto_write = True
                try:
                    _auto_write = bool(
                        getattr(
                            getattr(
                                getattr(config, "discovery", None), "notes", None
                            ),
                            "auto_write_per_paper",
                            True,
                        )
                    )
                except Exception:
                    _auto_write = True  # safe default

                note_path: str = ""
                note_title: str = ""
                if _auto_write:
                    note_path = write_paper_note(paper_record, extraction, config)
                    note_title = Path(note_path).stem
                else:
                    logger.debug(
                        "P10: auto_write_per_paper=false — deferred note for %s; "
                        "run `lit-monitor obsidian sync --all` to render.",
                        doi,
                    )
                embed_text = _paper_embed_text(title, abstract, extraction)
                # Each ingested item embeds into ChromaDB here — this is the "embeddings grow
                # on every run" mechanic the README leans on.
                chunks: list = []
                if fulltext:
                    try:
                        from lit_monitor.core.chunker import chunk_markdown
                        from lit_monitor.core.markdown_processor import strip_end_matter
                        chunks = chunk_markdown(strip_end_matter(fulltext), doi)
                    except Exception as exc:
                        logger.warning("Chunking failed for %s (non-fatal): %s", doi, exc)
                # R28 + G6: helper centralises add_paper + add_chunks + graph write + phase marks.
                # ChromaDB metadata stays scalar-only; graph payload extends on the side.
                _embed_paper_meta: dict[str, Any] = {
                    "source_type": "paper", "title": title, "year": year,
                }
                if graph_db is not None:
                    _graph_paper_meta = dict(_embed_paper_meta)
                    _graph_paper_meta["journal"] = journal
                    _graph_paper_meta["authors"] = authors
                    _graph_paper_meta["keywords_json"] = keywords
                    # N4: use NER-augmented MentionEdge list (schema + BioBERT + cloud-LLM).
                    # Falls back to [] on any failure — graph is enrichment, not a gate.
                    _graph_entities = build_ner_mention_edges(
                        extraction=extraction,
                        paper_metadata=_graph_paper_meta,
                        source_doi=doi,
                        state_db=state_db,
                        fulltext=fulltext or "",
                    )
                    # Relationships: G4 schema-source extraction first, then R4 LLM
                    # augmentation (gated by graph.relationships.llm_enabled).  The
                    # two lists are merged; R3 multi-source merge invariants apply
                    # inside graph_db.add_paper.  R28: any R4 failure returns [].
                    _, _schema_relationships = build_graph_tuples(
                        extraction=extraction,
                        paper_metadata=_graph_paper_meta,
                        source_doi=doi,
                        state_db=state_db,
                    )
                    _llm_relationships = maybe_extract_llm_relationships(
                        paper_doi=doi,
                        fulltext=fulltext,
                        extraction_json=extraction,
                        cfg=config,
                    )
                    _graph_relationships = _schema_relationships + _llm_relationships
                else:
                    _graph_paper_meta = _embed_paper_meta
                    _graph_entities, _graph_relationships = [], []
                embed_ok, embed_err = index_embeddings_and_mark_phases(
                    doi=doi,
                    zotero_key=zotero_key,
                    fulltext=embed_text,
                    paper_metadata=_embed_paper_meta,
                    chunks=chunks,
                    state_db=state_db,
                    embeddings_db=embeddings_db,
                    phases_to_mark=tuple(_phases_to_mark),
                    logger=logger,
                    graph_db=graph_db,
                    graph_entities=_graph_entities,
                    graph_relationships=_graph_relationships,
                    graph_paper_metadata=_graph_paper_meta,
                )
                if not embed_ok:
                    # Skip downstream work (state flag, relink) so the paper isn't
                    # falsely marked complete; outer except will mark it failed.
                    raise RuntimeError(embed_err)
                # H1: single authoritative status flip now that the embed step
                # succeeded — combined with the note path + embeddings flag write.
                state_db.upsert_paper({
                    "doi": doi,
                    "status": "extraction_complete",
                    "note_title": note_title,
                    "note_path": note_path,
                    "embeddings_indexed": 1,
                    "last_updated": _now(),
                })
                # P10: record that the note has been written so `obsidian sync` skips it.
                # Only set when auto_write=true AND write_paper_note succeeded.
                if _auto_write and note_path:
                    state_db.set_notes_synced(doi, 1)
                # Relink this note only (skip when note was deferred — no path to link)
                if note_path:
                    try:
                        relink_note(note_path, embeddings_db, state_db, config=config)
                    except Exception as exc:
                        logger.warning("Relink failed for %s: %s", doi, exc)

                # M4: collect discovered_topics for end-of-run vocabulary merge.
                _raw_topics = extraction.get("discovered_topics")
                if isinstance(_raw_topics, list) and _raw_topics:
                    _topics_batch.append(
                        (doi, [str(t) for t in _raw_topics if t and isinstance(t, str)])
                    )
                summary.papers_ingested += 1
                consecutive_429 = 0
                logger.debug("Ingested new paper: %s", doi)
            except RateLimitError as exc:
                # V-9: exponential back-off; abort after 3 consecutive hits.
                consecutive_429 += 1
                logger.warning(
                    "Rate limit hit on %s (consecutive: %d): %s",
                    doi, consecutive_429, exc,
                )
                if consecutive_429 >= 3:
                    logger.error(
                        "Rate limit persists after %d consecutive hits — "
                        "aborting ingestion cleanly. "
                        "Quota should reset shortly; the next run will resume.",
                        consecutive_429,
                    )
                    # M1: mirror brain_build — record rate_limited status so
                    # cron logs are honest. Paper is NOT marked error; next run
                    # will retry via Zotero poll.
                    state_db.finish_run(
                        run_id,
                        status="rate_limited",
                        processed=summary.papers_ingested,
                        failed=summary.papers_failed,
                        errors=summary.errors,
                    )
                    # P5.5: raise a catchable domain exception (mirrors P4.5 in
                    # brain_build), not SystemExit, so API/web callers can handle
                    # it. The discovery CLI boundary maps it to exit code 2.
                    raise RateLimitExhausted(
                        f"Rate limit persists after {consecutive_429} "
                        "consecutive hits; discovery aborted."
                    )
                wait = 60 * (2 ** (consecutive_429 - 1))
                logger.warning("Sleeping %ds before next paper", wait)
                _time.sleep(wait)
                # Paper not marked error — next run will retry via Zotero poll.
            except Exception as exc:
                consecutive_429 = 0
                logger.error("Ingestion failed for %s: %s", doi, exc, exc_info=True)
                summary.papers_failed += 1
                summary.errors.append(f"{doi}: {exc}")
        # Persist current library version so the next run only fetches newer items
        try:
            current_version = zotero_client.get_current_version()
            state_db.set_kv("zotero_library_version", str(current_version))
            logger.debug("Updated Zotero library version to %d", current_version)
        except Exception as exc:
            logger.warning("Could not update Zotero library version: %s", exc)

        # M4: append novel discovered_topics to topics.yaml and concepts.yaml.
        if _topics_batch:
            try:
                from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
                merged = merge_discovered_topics(_topics_batch)
                if merged:
                    logger.warning(
                        "M4 vocabulary merge: %d novel topic(s) appended — %s",
                        len(merged), merged,
                    )
            except Exception as exc:
                strict_fallback(
                    logger,
                    f"M4 vocabulary merge failed: {exc}. "
                    "discovered_topics from this run were not appended to topics.yaml.",
                    exc,
                )
    finally:
        # G6: release the KuzuDB connection at ingestion end.  Safe to call when None.
        if graph_db is not None:
            try:
                graph_db.close()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("G6: graph_db.close() failed (non-fatal): %s", exc)
