"""
Brain build pipeline.
Processes all journal papers in the Zotero "lit-monitor" collection:
  - Markdown attachment ingestion from Zotero (Phase M)
  - Simple + complex two-phase LLM extraction
  - Obsidian note writing
  - ChromaDB indexing
Resumable: skips papers where both extraction phases are already complete.
Single-item failures are logged and continue — pipeline never aborts.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from scripts.core.doi_resolver import resolve_doi
from scripts.core.item_router import detect_review, get_route
from scripts.core.strict_mode import strict_fallback
from scripts.core.zotero_client import ZoteroClient
from scripts.graph import safe_graph_db
from scripts.llm.extraction_schema import all_fields_for_schema
from scripts.llm.extractor import extract_fields, extract_paper
from scripts.llm.llm_client import RateLimitError
from scripts.output.obsidian_writer import write_paper_note
from scripts.pipelines._ingest import (
    build_graph_tuples,
    build_ner_mention_edges,
    index_embeddings_and_mark_phases,
    maybe_extract_llm_relationships,
)
from scripts.search.semantic_scholar import enrich_paper

# Bundle C: module-level import so tests can patch brain_build.recompute_clusters.
# Wrapped in try/except so the module loads cleanly even before sklearn is installed.
try:
    from scripts.clustering.recompute import recompute_clusters
except Exception:  # pragma: no cover
    recompute_clusters = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _llm_model_str(llm) -> str:
    """Concise model string for extraction metadata — handles single client and dict (F15).

    Note: uses isinstance(v, str) guards after getattr because MagicMock (used in tests)
    auto-creates attributes and never raises AttributeError, so getattr's default is
    never invoked for mock objects.
    """
    if isinstance(llm, dict):
        parts = []
        for n, c in sorted(llm.items()):
            m = getattr(c, "model", "?")
            parts.append(f"p{n}:{m if isinstance(m, str) else '?'}")
        return " ".join(parts)
    m = getattr(llm, "model", "")
    return m if isinstance(m, str) else ""


def _llm_provider_str(llm) -> str:
    """Provider string for extraction metadata — handles single client and dict (F15).

    See _llm_model_str note on isinstance guard for MagicMock safety.
    """
    if isinstance(llm, dict):
        first = next(iter(llm.values()))
        p = getattr(first, "provider", "ollama")
        return p if isinstance(p, str) else "ollama"
    p = getattr(llm, "provider", "ollama")
    return p if isinstance(p, str) else "ollama"
@dataclass
class BuildSummary:
    papers_processed: int = 0
    papers_skipped: int = 0
    papers_failed: int = 0
    errors: list[str] = field(default_factory=list)
    # H3: DOIs resolved via CrossRef (Zotero items that had no DOI)
    crossref_resolved: list[str] = field(default_factory=list)
    # I2: items with no DOI after CrossRef fallback — deferred for --resolve-no-doi
    no_doi_items: list[dict] = field(default_factory=list)
    # I5: run metadata for report writing
    run_id: str = ""
    started_at: str = ""
def run_brain_build(
    config,
    state_db,
    zotero_client,
    embeddings_db,
    llm,
    batch_size: int = 50,
    resume: bool = True,
    max_papers: int | None = None,
    show_progress: bool = False,
    all_library: bool = False,
) -> BuildSummary:
    """
    Process all papers in the Zotero collection.
    Parameters
    ----------
    config:
        Config object. Must have:
          config.zotero.collection_name     (str)
          config.obsidian.vault_path        (str)
          config.obsidian.papers_folder     (str)
    state_db:
        StateDB instance.
    zotero_client:
        ZoteroClient instance.
    embeddings_db:
        EmbeddingsDB instance.
    llm:
        LLMClient instance.
    batch_size:

        Number of papers per batch.
    resume:
        If True, skip papers already fully extracted.
    all_library:
        N22 — if True, iterate the entire Zotero library via
        ``zotero_client.get_all_library_items()`` instead of the configured
        ``config.zotero.collection_name``.  Useful when you want brain-build
        to walk every paper in the library without manually maintaining a
        meta-collection.  Items without a .md attachment are still skipped.
    Returns
    -------
    BuildSummary
    """
    summary = BuildSummary()
    run_id = str(uuid.uuid4())
    summary.run_id = run_id
    summary.started_at = datetime.now(UTC).isoformat()
    state_db.start_run(run_id, "brain_build")
    # G6: resolve GraphDB once per run.  None when [graph] extra absent or
    # the Kuzu DB can't be opened — vector-only behaviour is unaffected.
    graph_db = safe_graph_db()
    try:
        # M4: (doi, discovered_topics) pairs — mirrors discovery.py:_run_ingestion
        # (parallel collection pattern; same name reused intentionally).
        _topics_batch: list[tuple[str, list[str]]] = []
        # K1: read pass_strategy; warn if per-pass models are configured alongside "all"
        _bb_cfg = getattr(config, "brain_build", None)
        pass_strategy = getattr(_bb_cfg, "pass_strategy", "individual")
        if pass_strategy == "all":
            _per_pass_keys = [k for k in ("pass1_model", "pass2_model", "pass3_model")
                              if getattr(_bb_cfg, k, None)]
            if _per_pass_keys:
                logger.warning(
                    "K1: pass_strategy='all' ignores per-pass model keys %s — "
                    "set pass_all_model instead or switch to pass_strategy='individual'",
                    _per_pass_keys,
                )
        # N22: --all-library bypasses the configured collection and iterates the entire library.
        if all_library:
            items = zotero_client.get_all_library_items()
            scope_label = "entire Zotero library"
        else:
            collection_name = getattr(config.zotero, "collection_name", "lit-monitor")
            items = zotero_client.get_collection_items(collection_name)
            scope_label = f"collection {collection_name!r}"
        # N3: filter out Zotero child items (attachments, notes) — they have no DOI
        # and inflate the denominator while burning CrossRef quota on garbage queries.
        items = [it for it in items if it.get("data", {}).get("itemType") not in ("attachment", "note")]
        logger.warning(
            "Found %d parent items in %s (attachments/notes filtered)",
            len(items), scope_label,
        )
        papers_processed_this_run = 0  # N4: count successfully processed, not attempted
        _limit_reached = False
        consecutive_429 = 0  # V-9: abort after 3 consecutive rate-limit hits
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            disable=not show_progress,
            transient=False,  # keep final state visible after completion
        ) as progress:
            task = progress.add_task("[cyan]Brain build[/cyan]", total=len(items))
            for i in range(0, len(items), batch_size):
                if _limit_reached:
                    break
                batch = items[i : i + batch_size]
                for item in batch:
                    if _limit_reached:
                        break
                    data = item.get("data", {})
                    doi = (data.get("DOI") or "").strip() or None
                    zotero_key = item.get("key", "")
                    title = data.get("title", "")
                    # Update progress description with current paper title (truncated)
                    short_title = title[:55] if title else doi or "unknown"
                    progress.update(task, description=f"[dim]{short_title}[/dim]")
                    if not doi:
                        # D2: try CrossRef before giving up — uses title + up to 3
                        # author last names; min_score=70 keeps false-positive rate low.
                        _authors = ZoteroClient.extract_authors(data)
                        doi = resolve_doi(
                            title, _authors, _parse_year(data.get("date", ""))
                        ) or None
                        if not doi:
                            logger.debug(
                                "No DOI for item %s (%s) — skipping", zotero_key, title
                            )
                            # I2: accumulate for --resolve-no-doi deferred batch
                            summary.no_doi_items.append({
                                "zotero_key": zotero_key,
                                "title": title,
                                "authors": ZoteroClient.extract_authors(data),
                                "year": _parse_year(data.get("date", "")),
                                "_item": item,  # full Zotero dict for re-processing
                            })
                            progress.advance(task)
                            continue
                        logger.debug(
                            "CrossRef resolved DOI for %s (%s): %s", zotero_key, title[:60], doi
                        )
                        summary.crossref_resolved.append(doi)
                    # A2: metadata quality filter
                    ok, reason = _check_item_quality(data)
                    if not ok:
                        logger.debug(
                            "Quality filter: skipping %s (%r) — %s", zotero_key, title[:60], reason
                        )
                        summary.papers_skipped += 1
                        progress.advance(task)
                        continue
                    # I1: item-type routing — skip items that belong to another pipeline
                    item_type = data.get("itemType", "journalArticle")
                    try:
                        route = get_route(item_type)
                    except KeyError:
                        logger.warning(
                            "Unknown Zotero item type %r for %s (%s) — skipping",
                            item_type, zotero_key, title[:60],
                        )
                        summary.papers_skipped += 1
                        progress.advance(task)
                        continue
                    if route.pipeline != "brain_build":
                        logger.debug(
                            "Item type %r routes to pipeline %r — skipping %s (%s)",
                            item_type, route.pipeline, zotero_key, title[:60],
                        )
                        summary.papers_skipped += 1
                        progress.advance(task)
                        continue
                    # Resume
                    if resume:
                        item_progress = state_db.get_brain_build_progress(zotero_key)
                        if item_progress and item_progress.get("fully_complete"):
                            logger.debug("Skipping completed paper: %s", doi)
                            summary.papers_skipped += 1
                            progress.advance(task)
                            continue
                    try:
                        processed, _topics = _process_paper(
                            doi=doi,
                            zotero_key=zotero_key,
                            item=item,
                            config=config,
                            state_db=state_db,
                            zotero_client=zotero_client,
                            embeddings_db=embeddings_db,
                            llm=llm,
                            source_type=route.source_type,
                            pass_strategy=pass_strategy,
                            extract_paper_fn=extract_paper,
                            write_paper_note_fn=write_paper_note,
                            graph_db=graph_db,  # G6
                        )
                        if processed:
                            summary.papers_processed += 1
                            papers_processed_this_run += 1
                            if _topics:
                                _topics_batch.append((doi, _topics))
                            consecutive_429 = 0  # reset on successful paper (symmetric with non-429 failure path)
                            # N4: cap counts successfully processed items, not attempts.
                            if max_papers is not None and papers_processed_this_run >= max_papers:
                                _limit_reached = True
                                break
                        else:
                            # N4 bonus: items skipped inside _process_paper (no .md attachment)
                            # were not previously counted — fix the undercount here.
                            summary.papers_skipped += 1
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
                                "aborting brain-build cleanly. "
                                "Resume with `lit-monitor brain-build --resume` once quota resets.",
                                consecutive_429,
                            )
                            state_db.finish_run(
                                run_id,
                                status="rate_limited",
                                processed=summary.papers_processed,
                                skipped=summary.papers_skipped,
                                failed=summary.papers_failed,
                                errors=summary.errors,
                            )
                            raise SystemExit(2)
                        wait = 60 * (2 ** (consecutive_429 - 1))  # 60 s, 120 s
                        logger.warning("Sleeping %ds before next paper", wait)
                        time.sleep(wait)
                        # Paper not marked error — --resume will retry it next run.
                    except Exception as exc:
                        consecutive_429 = 0  # reset on non-429 failures
                        logger.error("Paper %s failed: %s", doi, exc, exc_info=True)
                        summary.papers_failed += 1
                        summary.errors.append(f"{doi}: {exc}")
                        state_db.mark_status(doi, "error")
                    finally:
                        progress.advance(task)
        state_db.finish_run(
            run_id,
            status="complete",
            processed=summary.papers_processed,
            skipped=summary.papers_skipped,
            failed=summary.papers_failed,
            errors=summary.errors,
        )

        # M4: append novel discovered_topics to topics.yaml and concepts.yaml.
        if _topics_batch:
            try:
                from scripts.vocabulary.topic_merger import merge_discovered_topics
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

        logger.warning(
            "Brain build complete: %d processed, %d skipped, %d failed",
            summary.papers_processed,
            summary.papers_skipped,
            summary.papers_failed,
        )
        # Bundle C: trigger initial clustering when the library just crossed
        # the threshold and no clusters exist yet. Non-fatal — never aborts.
        try:
            _maybe_trigger_initial_clustering(state_db, embeddings_db, config)
        except Exception as exc:
            logger.warning("C: initial clustering hook failed (non-fatal): %s", exc)

        return summary
    finally:
        # G6: release the KuzuDB connection at pipeline end.  ``close()`` is
        # safe to call multiple times and is a no-op when graph_db is None.
        if graph_db is not None:
            try:
                graph_db.close()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("G6: graph_db.close() failed (non-fatal): %s", exc)
# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------
def _process_paper(
    doi: str,
    zotero_key: str,
    item: dict[str, Any],
    config,
    state_db,
    embeddings_db,
    llm,
    *,
    zotero_client=None,
    source_type: str = "paper",
    pass_strategy: str = "individual",
    extract_paper_fn,
    write_paper_note_fn,
    graph_db=None,  # G6
) -> tuple[bool, list[str]]:
    """
    Process a single paper end-to-end.

    Returns (processed, discovered_topics) where:
      processed — True if extracted and indexed; False if skipped.
      discovered_topics — list of topic strings from the complex phase (M4).
    Raises on unrecoverable errors so the caller can count failures.
    """
    data = item.get("data", {})
    title = data.get("title", "")
    authors = ZoteroClient.extract_authors(data)
    year = _parse_year(data.get("date", ""))
    journal = data.get("publicationTitle", "") or data.get("journalAbbreviation", "")
    abstract = data.get("abstractNote", "") or ""
    keywords = _extract_keywords(data)
    zotero_key = item.get("key", zotero_key)
    logger.debug("Processing paper: %s (%s)", doi, title[:60])
    # Ensure progress row
    state_db.upsert_brain_build_progress(zotero_key, doi)
    # M1: read markdown attachment from Zotero local storage.
    # PDF extraction is no longer performed in-pipeline — the user converts
    # offline (Marker / Docling / publisher HTML) and attaches the .md file.
    if zotero_client is None:
        logger.warning("No zotero_client provided for %s — cannot read markdown attachment", doi)
        return False, []
    fulltext = zotero_client.get_markdown_attachment(zotero_key)
    if fulltext is None:
        logger.debug("No markdown attachment for %s — skipping (attach a .md file in Zotero)", doi)
        state_db.upsert_paper({
            "doi": doi,
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "zotero_key": zotero_key,
            "status": "no_markdown",
            "source_type": source_type,
            "first_seen_date": str(date.today()),
            "last_updated": _now(),
        })
        return False, []  # not counted as processed
    ocr_heavy = False  # markdown attachments are never OCR-heavy
    # H3: pre-extraction S2 enrichment — fetched now so publicationTypes is
    # available for H4 detect_review() schema routing before extraction starts.
    # H1: wrap in try/except so a transient S2 outage does not abort the paper.
    # Narrowed to the realistic failure modes for an HTTP API call + JSON
    # parsing + strict_fallback re-raise (RuntimeError).  Programmer errors
    # (NameError, AttributeError, etc.) intentionally still surface.
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
    # H4: review detection — override source_type before extraction so the
    # correct type is written on the first DB upsert.
    s2_pub_types = s2_data.get("s2_publication_types") if s2_data else None
    if detect_review(s2_pub_types, title=title, abstract=abstract, journal=journal):
        logger.debug("detect_review: classifying %s as review article", doi)
        source_type = "review"
    # Determine which passes to run (resume support)
    progress = state_db.get_brain_build_progress(zotero_key) or {}
    existing_extraction = state_db.get_extraction_json(doi) or {}
    if pass_strategy == "all":
        # K1/V-7: single-call extraction — all fields for the resolved schema.
        # source_type determines which schema (paper or review after R-10).
        # Resume: skip if already fully complete; otherwise re-run the whole call.
        if progress.get("fully_complete"):
            return False, []
        _bb_cfg = getattr(config, "brain_build", None)
        max_tokens = getattr(_bb_cfg, "max_tokens_per_call", 12288)
        domain_context = getattr(config, "domain_context", None)
        extraction = extract_fields(
            fulltext,
            all_fields_for_schema(source_type),
            llm,
            domain_context=domain_context,
            ocr_heavy=ocr_heavy,
            max_tokens=max_tokens,
            output_reserve=max_tokens,
            content_type=source_type,
        )
        # Phase flags are committed at the end of _process_paper, after the
        # embed step has succeeded.  Marking them here would leave the paper
        # in a stuck "fully_complete=1 + status='error'" state if add_paper
        # raises — `--resume` would then skip it instead of retrying.
        _phases_to_mark = ["simple", "complex"]
    else:
        # M3: phase-based extraction (simple then complex).
        phases_needed = []
        if not progress.get("simple_complete"):
            phases_needed.append("simple")
        if not progress.get("complex_complete"):
            phases_needed.append("complex")
        if not phases_needed:
            return False, []  # both phases already done
        # M5: forward per-mode chunk_chars from extraction.yaml to simple phase.
        _bb_cfg = getattr(config, "brain_build", None)
        _chunk_chars = getattr(_bb_cfg, "chunk_chars", None)
        extraction = extract_paper_fn(
            fulltext,
            llm,
            content_type=source_type,
            ocr_heavy=ocr_heavy,
            phases=tuple(phases_needed),
            existing_extraction=existing_extraction if existing_extraction else None,
            chunk_chars_override=int(_chunk_chars) if isinstance(_chunk_chars, (int, float)) else None,
        )
        # Defer phase marking to the end of _process_paper (after the embed
        # step) so a downstream failure doesn't leave fully_complete=1 set.
        _phases_to_mark = [
            phase for phase in phases_needed
            if f"_{phase}_error" not in extraction
        ]
    # Merge S2 enrichment into extraction so it's persisted in the same upsert
    # and available in note front matter without a second DB write.
    if s2_data:
        extraction.update(s2_data)
    # I3: flag degraded extraction quality when any pass required >3 chunks
    if extraction.get("_max_n_chunks", 1) > 3:
        extraction["_extraction_quality"] = "degraded_high_chunking"
    # H1: Persist the extraction payload BUT do not flip status to
    # 'extraction_complete' yet — that happens after the embed step succeeds.
    # Leaving status untouched here (COALESCE on the StateDB side preserves
    # the prior value) means a downstream embed failure cleanly leaves the
    # paper marked 'error' by the outer handler instead of producing two
    # conflicting writes (extraction_complete then error).
    state_db.upsert_paper({
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "zotero_key": zotero_key,
        "first_seen_date": str(date.today()),
        "source_type": source_type,
        "extraction_json": json.dumps(extraction),
        "extraction_provider": _llm_provider_str(llm),
        "extraction_model": _llm_model_str(llm),
        "keywords_json": keywords,
        "embeddings_indexed": 0,
        "last_updated": _now(),
    })
    # Assign vocabulary themes from Zotero tags (B2).
    # Gracefully returns [] when no vocabulary file exists — no crash.
    from scripts.vocabulary.normalizer import assign_themes
    themes = assign_themes(keywords)
    # Write Obsidian note
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
        "tracked_author": False,
        "fulltext_analyzed": True,
        # N8d: pass the actual provider/model used so the note template reads
        # from the paper dict rather than the (incorrect) global config object.
        "extraction_provider": _llm_provider_str(llm),
        "extraction_model": _llm_model_str(llm),
    }
    # P10: gate the inline note write on discovery.notes.auto_write_per_paper
    # (default TRUE — no behaviour change on upgrade).  When false, the note
    # write is deferred to `lit-monitor obsidian sync`; notes_synced stays 0.
    # R28 GUARD: this block sits BEFORE index_embeddings_and_mark_phases so
    # the embed/graph phase marks are always written regardless of the flag.
    _auto_write = True
    try:
        _auto_write = bool(
            getattr(
                getattr(getattr(config, "discovery", None), "notes", None),
                "auto_write_per_paper",
                True,
            )
        )
    except Exception:
        _auto_write = True  # safe default — never silently stop writing notes

    note_path: str = ""
    note_title: str = ""
    if _auto_write:
        # H1: mirror discovery's broad protection around note writing so a vault
        # I/O failure or template bug surfaces as a single 'error' write from the
        # outer handler rather than crashing mid-pipeline.  Narrowed to the
        # realistic failure modes from Jinja2 template rendering + filesystem
        # writes; outer handler logs the full traceback.
        try:
            note_path = write_paper_note_fn(paper_record, extraction, config)
            note_title = Path(note_path).stem
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("write_paper_note failed for %s: %s", doi, exc)
            raise
    else:
        logger.debug(
            "P10: auto_write_per_paper=false — deferred note for %s; "
            "run `lit-monitor obsidian sync --all` to render.",
            doi,
        )
    # Index in ChromaDB — paper-level and chunk-level
    embed_text = _paper_embed_text(title, abstract, extraction)
    chunks: list = []
    if fulltext:
        try:
            from scripts.core.chunker import chunk_markdown
            from scripts.core.markdown_processor import strip_end_matter
            chunks = chunk_markdown(strip_end_matter(fulltext), doi)
        except Exception as exc:
            logger.warning("Chunking failed for %s (non-fatal): %s", doi, exc)
    # R28 + G6: helper centralises add_paper + add_chunks + graph write + deferred phase marks.
    # add_paper failure -> phases NOT marked, returns (False, err).
    # graph_db failure -> graph_indexed stays 0; phases STILL marked.
    # NB: paper_metadata feeds BOTH ChromaDB (which only accepts scalar values)
    # AND the graph payload (which needs Zotero authors/keywords for G3).
    # ChromaDB metadata stays unchanged; the graph payload extends it on the
    # side so the embeddings call is not broken by list values.
    _embed_paper_meta: dict[str, Any] = {
        "source_type": "paper",
        "title": title,
        "year": year,
        "note_title": note_title,
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
        # augmentation (gated by graph.relationships.llm_enabled).  The two
        # lists are merged; R3 multi-source merge invariants apply inside
        # graph_db.add_paper.  R28: any R4 failure returns [] (non-fatal).
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
        # Propagate embed failure to caller so the paper isn't marked complete.
        raise RuntimeError(embed_err)
    # H1: single authoritative status flip — now that the embed step has
    # succeeded, mark the paper extraction_complete in the same write that
    # records the note path + embeddings flag.  Pre-fix, status was set to
    # 'extraction_complete' before the embed call; an embed failure then
    # caused the outer handler to overwrite it with 'error', leaving two
    # conflicting writes and embeddings_indexed=0.
    state_db.upsert_paper({
        "doi": doi,
        "status": "extraction_complete",
        "note_title": note_title,
        "note_path": note_path,
        "embeddings_indexed": 1,
        "last_updated": _now(),
    })
    # P10: record that the note has been written so `obsidian sync` skips it.
    # Only set when auto_write=true AND write_paper_note_fn succeeded (no exception
    # was raised above).  If deferred (auto_write=false), stays 0 so sync picks
    # it up.
    if _auto_write and note_path:
        state_db.set_notes_synced(doi, 1)
    logger.debug("Paper complete: %s → %s", doi, note_path or "(deferred)")
    # M4: collect discovered_topics for end-of-run vocabulary merge.
    _raw_topics = extraction.get("discovered_topics")
    _discovered: list[str] = (
        [str(t) for t in _raw_topics if t and isinstance(t, str)]
        if isinstance(_raw_topics, list) else []
    )
    return True, _discovered
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _paper_embed_text(title: str, abstract: str, extraction: dict[str, Any]) -> str:
    parts = [title]
    if abstract:
        parts.append(abstract)
    if extraction.get("core_finding"):
        parts.append(extraction["core_finding"])
    return " ".join(parts)
def _parse_year(date_str: str) -> int:
    if not date_str:
        return 0
    for part in date_str.split("-"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return 0
def _extract_keywords(data: dict[str, Any]) -> list[str]:
    tags = data.get("tags", [])
    return [t.get("tag", "") for t in tags if t.get("tag")]
def _extract_tags(data: dict[str, Any]) -> list[str]:
    return _extract_keywords(data)

def _check_item_quality(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate an item's metadata before processing.

    Returns (ok, reason). ok=False means the item should be skipped; the
    caller logs the reason. Non-blocking issues (empty authors, no abstract)
    are logged here as warnings/debug so the caller's loop stays clean.
    """
    title = (data.get("title") or "").strip()
    if not title:
        return False, "empty title"

    creators = data.get("creators", [])
    has_author = any(c.get("creatorType") in ("author", "editor") for c in creators)
    if not has_author:
        logger.warning(
            "Item %r has no author — will still process but extraction quality may suffer",
            title[:60],
        )

    abstract = (data.get("abstractNote") or "").strip()
    if not abstract:
        logger.debug("Item %r has no abstract — proceeding without it", title[:60])

    date_str = data.get("date", "")
    if date_str and _parse_year(date_str) == 0:
        logger.debug("Item %r has unparseable date %r — year will default to 0", title[:60], date_str)

    return True, ""

def _now() -> str:
    return datetime.now(UTC).isoformat()


def write_brain_build_report(config, summary: BuildSummary, model_str: str = "") -> str:
    """
    Write a Brain Build run report to {vault}/{digests_folder}/Brain Build - YYYY-MM-DD.md.

    Returns the path written, or "" if dry_run / vault path is missing.
    Sections: Summary, Model, CrossRef Resolutions, No-DOI Items, Errors,
              Failed Papers, Skipped (count only).
    """
    vault_path = Path(getattr(config.obsidian, "vault_path", ""))
    if not vault_path:
        logger.warning("write_brain_build_report: no vault_path configured — skipping")
        return ""
    digests_folder = vault_path / getattr(config.obsidian, "digests_folder", "Literature/Digests")
    digests_folder.mkdir(parents=True, exist_ok=True)

    run_date = date.today().isoformat()
    started = summary.started_at[:16].replace("T", " ") if summary.started_at else "unknown"
    finished = _now()[:16].replace("T", " ")

    lines: list[str] = [
        f"# Brain Build Report — {run_date}",
        "",
        "## 1. Summary",
        f"- **Date**: {run_date}",
        f"- **Started**: {started} UTC",
        f"- **Finished**: {finished} UTC",
        f"- **Papers processed**: {summary.papers_processed}",
        f"- **Papers skipped**: {summary.papers_skipped}",
        f"- **Papers failed**: {summary.papers_failed}",
        f"- **No-DOI items**: {len(summary.no_doi_items)}",
        f"- **CrossRef resolutions**: {len(summary.crossref_resolved)}",
        "",
        "## 2. Model",
        f"- **Model**: {model_str or 'see extraction.yaml'}",
        f"- **Pass strategy**: {getattr(getattr(config, 'brain_build', None), 'pass_strategy', 'individual')}",
        "",
        "## 3. CrossRef Resolutions",
    ]
    if summary.crossref_resolved:
        for doi in summary.crossref_resolved:
            lines.append(f"- {doi}")
    else:
        lines.append("*None.*")

    lines += ["", "## 4. No-DOI Items"]
    if summary.no_doi_items:
        for item in summary.no_doi_items:
            title = item.get("title", "?")
            key = item.get("zotero_key", "?")
            lines.append(f"- **{title}** (key: `{key}`)")
    else:
        lines.append("*None.*")

    lines += ["", "## 5. Failed Papers"]
    if summary.errors:
        for err in summary.errors:
            lines.append(f"- {err}")
    else:
        lines.append("*None.*")

    lines += [
        "",
        "## 6. Skipped Papers",
        f"*{summary.papers_skipped} papers were skipped (already complete or quality-filtered).*",
        "",
        "## 7. Notes",
        "*Review no-DOI items and use `brain-build --resolve-no-doi` to process them interactively.*",
        "",
    ]

    report_path = digests_folder / f"Brain Build - {run_date}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug("Wrote brain build report: %s", report_path)
    return str(report_path)


# ---------------------------------------------------------------------------
# Bundle C: end-of-brain-build clustering hook
# ---------------------------------------------------------------------------

def _maybe_trigger_initial_clustering(state_db, embeddings_db, cfg) -> None:
    """Trigger one-shot cluster recompute when the library crosses the threshold.

    Idempotent: only runs when clusters table is empty AND the ChromaDB
    collection has at least min_papers_threshold indexed papers.

    Args:
        state_db: StateDB instance.
        embeddings_db: EmbeddingsDB instance.
        cfg: Config object. Must have cfg.clustering block.
    """
    # recompute_clusters is imported at module level so tests can patch
    # scripts.pipelines.brain_build.recompute_clusters. The name lookup at
    # call time uses the module globals dict — patching works correctly.
    clustering_cfg = getattr(cfg, "clustering", None)
    if clustering_cfg is None or not getattr(clustering_cfg, "enabled", True):
        return

    # Check if any clusters already exist — if so, don't re-trigger.
    existing = state_db.list_active_clusters()
    if existing:
        return

    threshold = int(getattr(clustering_cfg, "min_papers_threshold", 100))
    n_papers = embeddings_db._collection.count()

    if n_papers < threshold:
        logger.debug(
            "C: library has %d papers (threshold=%d) — initial clustering deferred.",
            n_papers, threshold,
        )
        return

    logger.info(
        "C: library crossed clustering threshold (%d ≥ %d); running initial recompute.",
        n_papers, threshold,
    )
    try:
        n_clusters = recompute_clusters(state_db, embeddings_db, cfg)
        if n_clusters > 0:
            import click
            click.echo(
                f">> Clustering enabled — {n_clusters} themes identified. "
                "Run `lit-monitor cluster view` to explore."
            )
    except Exception as exc:
        logger.warning("C: initial clustering failed (non-fatal): %s", exc)
