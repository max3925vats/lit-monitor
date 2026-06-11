"""
Obsidian relink — update ## Related Work and ## Referenced By sections
across all notes using similarity search.
Relink per note: Similarity-based — find top-K similar documents in ChromaDB.
Citation-based pass deferred to E1 (citation graph).
Related Work format in paper/chapter notes:
  🔗 [[Author2_2011_Process_Ch04]] — related (0.91)
Persist zones are always preserved — relink only replaces the content
_inside_ the zone markers, not the markers themselves.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz

from lit_monitor.core.atomic_write import atomic_write_text
from lit_monitor.core.strict_mode import strict_fallback
from lit_monitor.llm.prompt_safety import sanitize_for_prompt

if TYPE_CHECKING:
    from lit_monitor.core.state_db import StateDB
    from lit_monitor.output.embeddings import EmbeddingsDB
logger = logging.getLogger(__name__)
_RELATED_WORK_SECTION = "## Related Work\n"
_REFERENCED_BY_SECTION = "## Referenced By\n"
_CITATION_SCORE_CUTOFF = 75  # rapidfuzz token_sort_ratio threshold
def resolve_citations(
    key_citations: list[str],
    state_db: StateDB,
    score_cutoff: int = _CITATION_SCORE_CUTOFF,
) -> list[dict[str, Any]]:
    """
    Fuzzy-match key_citations strings against titles in the state DB.
    Uses rapidfuzz.fuzz.token_sort_ratio so word-order differences don't
    prevent a match (e.g. "Author2 Filtration 2011" vs
    "Filtration of ProteinAs — Author2 2011").
    Returns
    -------
    list[dict]
        Each: {doi, note_title, match_type: 'cited', score}.
        Only entries above score_cutoff are returned.
    """
    # Build title index from state DB
    all_papers = state_db.get_all_by_source_type("paper")
    all_reviews = state_db.get_all_by_source_type("review")
    candidates = all_papers + all_reviews
    matched: list[dict[str, Any]] = []
    for citation_str in key_citations:

        if not citation_str or not citation_str.strip():
            continue
        best_score = 0
        best_match: dict | None = None
        for candidate in candidates:
            title = candidate.get("title") or ""
            score = fuzz.token_sort_ratio(citation_str, title)
            if score > best_score:
                best_score = score
                best_match = candidate
        if best_match and best_score >= score_cutoff:
            matched.append({
                "doi": best_match.get("doi", ""),
                "note_title": best_match.get("note_title", ""),
                "match_type": "cited",
                "score": best_score,
            })
    return matched
def build_related_work_block(
    similar: list[dict[str, Any]],
    cited: list[dict[str, Any]],
) -> str:
    """
    Build the content string for the ## Related Work persist zone.
    Cited items appear first (📖), then similarity-based (🔗).
    Items already in cited are deduped from similar.
    """
    lines: list[str] = [_RELATED_WORK_SECTION]
    cited_titles: set[str] = set()
    for item in cited:
        note_title = item.get("note_title", "")
        if not note_title:
            continue
        lines.append(f"📖 [[{note_title}]] — cited")
        cited_titles.add(note_title)
    for item in similar:
        note_title = item.get("metadata", {}).get("note_title", "")
        if not note_title or note_title in cited_titles:
            continue
        score = item.get("score", 0)
        lines.append(f"🔗 [[{note_title}]] — related ({score:.2f})")
    if len(lines) == 1:
        lines.append("*(populated by `lit-monitor obsidian relink`)*")
    return "\n".join(lines) + "\n"
def _build_referenced_by_block(
    sources: list[dict[str, Any]],
) -> str:
    """
    Build the ## Referenced By block for a cited note.
    sources: list of {note_title} for notes that cite this one.
    """
    lines: list[str] = [_REFERENCED_BY_SECTION]
    for src in sources:
        note_title = src.get("note_title", "")
        if note_title:
            lines.append(f"← [[{note_title}]] cites this")

    if len(lines) == 1:
        lines.append("*(populated by `lit-monitor obsidian relink`)*")
    return "\n".join(lines) + "\n"
def _replace_persist_zone_content(note_content: str, zone_name: str, new_body: str) -> str:
    """
    Replace the content inside a named persist zone.
    Leaves the {% persist %} and {% endpersist %} markers intact.
    """
    pattern = re.compile(
        r'(\{%\s*persist\s+"' + re.escape(zone_name) + r'"\s*%\})'
        r'(.*?)'
        r'(\{%\s*endpersist\s*%\})',
        re.DOTALL,
    )
    def replacer(m: re.Match) -> str:
        return m.group(1) + "\n" + new_body.rstrip("\n") + "\n" + m.group(3)
    updated, count = pattern.subn(replacer, note_content)
    if count == 0:
        logger.warning("Zone %r not found in note — skipping relink for this zone", zone_name)
    return updated
def _resolve_cited_notes(doi: str, state_db: StateDB) -> list[dict[str, Any]]:
    """
    Return cited-note dicts for outgoing citation edges with a resolved target_doi.
    Used to populate the Related Work zone with citation-graph entries.
    """
    edges = state_db.get_citation_edges(doi)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edge in edges:
        target_doi = edge.get("target_doi")
        if not target_doi or target_doi in seen:
            continue
        paper = state_db.get_paper(target_doi)
        if paper and paper.get("note_title"):
            results.append({
                "note_title": paper["note_title"],
                "match_type": "cited",
                "score": 100,
            })
            seen.add(target_doi)
    return results


def relink_note(
    note_path: str | Path,
    embeddings_db: EmbeddingsDB,
    state_db: StateDB,
    top_k: int = 10,
    config=None,
    rag_mode: str | None = None,
    graph_db=None,
) -> None:
    """
    Run two-pass relink on a single note file.
    Pass 1: similarity search → Related Work zone.
      rag_mode='vector' (default): ChromaDB embed search (existing path).
      rag_mode='graph':            graph_db.find_similar_papers().
      rag_mode='hybrid':           both sources fused via RRF, then reranked.
    Pass 2 (E2): citation-graph via citation_edges table:
        - outgoing (this paper cites) → Related Work zone (📖 entries)
        - incoming (papers that cite this) → Referenced By zone
    Persist zones (`related_work`, `referenced_by`) are preserved across
    re-runs because both are now defined in `paper_note.md.j2` (M2).
    config: optional — used to read reranker settings and default rag_mode.
    rag_mode: optional — overrides config.retrieval.default_mode; falls back
        to "vector" when neither is set.
    graph_db: optional GraphDB instance required for graph/hybrid modes.
    """
    note_path = Path(note_path)
    if not note_path.exists():
        logger.warning("Note not found, skipping relink: %s", note_path)
        return
    note_content = note_path.read_text(encoding="utf-8")
    note_text = _strip_frontmatter(note_content)
    doi = (
        _extract_frontmatter_value(note_content, "doi")
        or _extract_frontmatter_value(note_content, "book_id")
    )
    # Pass 1: similarity search.
    # Audit R29: build a FOCUSED embed query (title + abstract + core_finding)
    # rather than the full note text.  Three reasons:
    #   (1) Query distribution matches the corpus distribution — every paper-
    #       level vector in ChromaDB was built from these same three fields by
    #       `EmbeddingsDB.add_paper`, so cosine compares like-with-like.
    #   (2) Full notes are 4000–6000 chars; mxbai-embed-large's 512-token
    #       context truncates ~70% of that and silently degrades retrieval.
    #   (3) The reranker still sees the longer note_text for cross-encoder
    #       scoring, which IS where richer query content helps.
    focused_query = _focused_embed_query(state_db, doi, note_text)
    _reranker_cfg = getattr(config, "reranker", None) if config is not None else None
    # G9: resolve rag_mode; default from config, fallback "vector".
    _cfg_mode = (
        getattr(getattr(config, "retrieval", None), "default_mode", "vector")
        if config is not None else "vector"
    )
    _rag_mode = rag_mode or _cfg_mode or "vector"
    if _rag_mode == "vector":
        # Existing path — fully preserved (bit-for-bit v0.3.3 behaviour).
        similar = embeddings_db.find_similar_to_text(
            focused_query,
            top_k=top_k,
            exclude_id=doi,
            rerank_with_query=sanitize_for_prompt(note_text),
            reranker_config=_reranker_cfg,
        )
    else:
        # G9: graph or hybrid — use branch helper, then rerank if configured.
        # W1: vector leg of hybrid must preserve v0.3.3 reranker contract
        # (exclude self, pass query + reranker_config through).
        from lit_monitor.retrieval.branch import retrieve_doi_candidates
        _reranker_query = sanitize_for_prompt(note_text)
        doi_candidates = retrieve_doi_candidates(
            _rag_mode,
            seed_doi=doi,
            query_text=focused_query,
            embeddings_db=embeddings_db,
            graph_db=graph_db,
            k=top_k,
            exclude_id=doi,
            rerank_with_query=_reranker_query if _rag_mode == "hybrid" else None,
            reranker_config=_reranker_cfg if _rag_mode == "hybrid" else None,
        )
        # Convert (doi, score) pairs → ChromaDB-style result dicts so the
        # downstream build_related_work_block receives the expected shape.
        # Resolve note_title for each candidate DOI via state_db.
        similar = _doi_pairs_to_similar(doi_candidates, state_db, top_k,
                                        reranker_cfg=_reranker_cfg,
                                        reranker_query=_reranker_query)
    # Pass 2a: outgoing citations — papers this note cites (E2)
    cited = _resolve_cited_notes(doi, state_db) if doi else []
    related_work = build_related_work_block(similar, cited)
    updated_content = _replace_persist_zone_content(note_content, "related_work", related_work)
    # Pass 2b: incoming citations — papers that cite this note (E2)
    citing_sources: list[dict[str, Any]] = []
    if doi:
        for source_doi in state_db.get_papers_citing_doi(doi):
            paper = state_db.get_paper(source_doi)
            if paper and paper.get("note_title"):
                citing_sources.append({"note_title": paper["note_title"]})
    if citing_sources:
        referenced_by = _build_referenced_by_block(citing_sources)
        updated_content = _replace_persist_zone_content(updated_content, "referenced_by", referenced_by)
    atomic_write_text(note_path, updated_content)
    logger.info(
        "Relinked: %s (%d similar, %d cited, %d citing)",
        note_path.name, len(similar), len(cited), len(citing_sources),
    )
# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter block from note content."""
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            return content[end + 4:].strip()
    return content

def _extract_frontmatter_value(content: str, key: str) -> str | None:
    """Extract a simple string value from YAML frontmatter."""
    match = re.search(rf'^{re.escape(key)}:\s*"?([^"\n]+)"?', content, re.MULTILINE)
    return match.group(1).strip() if match else None
def _safe_json(s: str) -> dict:
    import json
    try:
        return json.loads(s) if s else {}
    except Exception as exc:
        strict_fallback(
            logger,
            f"_safe_json: could not parse JSON string (len={len(s)}): {exc}. "
            "The stored extraction_json or note metadata may be corrupt.",
            exc,
        )
        return {}


# Cap matches what's used by add_paper's embed text — same shape, same length.
# Keeps focused queries comfortably below any embedder's context window.
_FOCUSED_EMBED_QUERY_MAX_CHARS = 1300


def _focused_embed_query(
    state_db: StateDB,
    doi: str | None,
    fallback_text: str,
) -> str:
    """Build a corpus-shape query for relink's cosine retrieval.

    Mirrors ``_paper_embed_text`` in brain_build / discovery: returns
    ``title + abstract + core_finding`` from the state DB.  Falls back to the
    first ``_FOCUSED_EMBED_QUERY_MAX_CHARS`` of the note text if the DB lookup
    or extraction blob is unavailable.
    """
    if not doi:
        return fallback_text[:_FOCUSED_EMBED_QUERY_MAX_CHARS]
    record = state_db.get_paper(doi) if hasattr(state_db, "get_paper") else None
    if not record:
        return fallback_text[:_FOCUSED_EMBED_QUERY_MAX_CHARS]
    title = (record.get("title") or "").strip()
    extraction = _safe_json(record.get("extraction_json") or "")
    abstract = (extraction.get("abstract") or "").strip()
    core_finding = (extraction.get("core_finding") or "").strip()
    parts = [p for p in (title, abstract, core_finding) if p]
    if not parts:
        return fallback_text[:_FOCUSED_EMBED_QUERY_MAX_CHARS]
    text = " ".join(parts)
    return text[:_FOCUSED_EMBED_QUERY_MAX_CHARS]


def _doi_pairs_to_similar(
    doi_candidates: list[tuple[str, float]],
    state_db: StateDB,
    top_k: int,
    *,
    reranker_cfg=None,
    reranker_query: str = "",
) -> list[dict]:
    """G9: convert (doi, score) pairs from branch helper to similar-result dicts.

    The returned list matches the shape expected by build_related_work_block:
    each entry has 'score' and 'metadata': {'note_title': <str>}.

    G9 C3: ``document`` is hydrated with title + abstract (+ core_finding) from
    state_db.get_paper so the cross-encoder reranker actually has text to score.
    Previously this field was "" and the reranker scored (query, "") — noise.

    G9 C1: when a reranker_cfg is present and enabled, results are re-ranked
    via the cross-encoder using the same code path as the vector branch
    (``_apply_reranker`` in scripts.output.embeddings).
    """
    results: list[dict] = []
    for doi_val, score in doi_candidates[:top_k]:
        record = state_db.get_paper(doi_val) if hasattr(state_db, "get_paper") else None
        record = record or {}
        note_title = record.get("note_title", "")
        # C3: hydrate the reranker passage from the same fields v0.3.3 used to
        # build the paper-level embedding (title + abstract + core_finding).
        # N+1 query: one get_paper per candidate. Acceptable trade-off — top_k
        # is typically 10–30 and state_db is local SQLite. Batching would
        # require a new state_db API and the gain is small at these k values.
        results.append({
            "id": doi_val,
            "score": score,
            "document": _build_reranker_document(record),
            "metadata": {"note_title": note_title, "doi": doi_val},
        })
    # C1: rerank using the same private helper the vector path uses. This is
    # the actual API — the previous ``from lit_monitor.output.reranker import rerank``
    # was a phantom symbol that raised ImportError on every call (silently
    # swallowed by except Exception).
    if reranker_cfg is not None and reranker_query and results:
        from lit_monitor.output.embeddings import _apply_reranker, _reranker_enabled
        if _reranker_enabled(reranker_cfg):
            results = _apply_reranker(reranker_query, results, top_k, reranker_cfg)
    return results


def _build_reranker_document(record: dict) -> str:
    """G9 C3: build the passage string the cross-encoder reranker scores against.

    Mirrors ``_paper_embed_text`` in brain_build: title + abstract + core_finding.
    Returns "" only when every field is missing (e.g. record absent in state_db).
    """
    if not record:
        return ""
    title = (record.get("title") or "").strip()
    abstract = ""
    core_finding = ""
    extraction_raw = record.get("extraction_json") or ""
    if extraction_raw:
        extraction = _safe_json(extraction_raw)
        abstract = (extraction.get("abstract") or "").strip()
        core_finding = (extraction.get("core_finding") or "").strip()
    parts = [p for p in (title, abstract, core_finding) if p]
    return " ".join(parts)
