"""
Unit tests for the brain build pipeline.
All I/O, LLM, and Zotero calls are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lit_monitor.pipelines.brain_build import (
    _paper_embed_text,
    _parse_year,
    run_brain_build,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_config(tmp_path: Path) -> SimpleNamespace:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Literature" / "Papers").mkdir(parents=True)
    return SimpleNamespace(
        zotero=SimpleNamespace(
            collection_name="lit-monitor",
            local_storage_path=str(tmp_path / "zotero"),
        ),
        obsidian=SimpleNamespace(
            vault_path=str(vault),
            papers_folder="Literature/Papers",
        ),
        extraction_provider="ollama",
        extraction_model="gemma4:e4b",
    )
def _make_state_db(tmp_path: Path):
    from lit_monitor.core.state_db import StateDB
    return StateDB(tmp_path / "state.db")
def _make_llm() -> MagicMock:
    llm = MagicMock()
    llm.model = "gemma4:e4b"
    llm.provider = "ollama"
    llm.complete.return_value = json.dumps({
        "core_finding": "Key result.",
        "core_finding_confidence": "explicit",
        "methods_summary": "Methods used.",
        "methods_summary_confidence": "explicit",
        "results_summary": None,
        "results_summary_confidence": "absent",
        "conclusions": None,
        "conclusions_confidence": "absent",
        "study_type": "experimental",
        "study_type_confidence": "explicit",

    })
    return llm
def _make_zotero_item(doi: str = "10.1000/test", key: str = "ABCDEF01") -> dict:
    return {
        "key": key,
        "data": {
            "DOI": doi,
            "title": "Test Paper on Filtration",
            "date": "2021",
            "publicationTitle": "J Membrane Sci",
            "abstractNote": "We studied ultrafiltration.",
            "tags": [{"tag": "ultrafiltration"}, {"tag": "mAb"}],
            "creators": [
                {"creatorType": "author", "lastName": "Smith", "firstName": "J"},
            ],
        },
        "attachments": [],
    }
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_brain_build_processes_papers(tmp_path):
    """Two items with PDFs → both processed, notes written."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    items = [
        _make_zotero_item(doi="10.1000/paper1", key="KEY001"),
        _make_zotero_item(doi="10.1000/paper2", key="KEY002"),
    ]
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = items
    # M1: pipeline reads markdown attachments
    zotero_client.get_markdown_attachment.return_value = "Full paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021_Test.md")
    with patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "Key result.",
                             "core_finding_confidence": "explicit"}):
        with patch("scripts.pipelines.brain_build.write_paper_note",
                   return_value=note_path):
            summary = run_brain_build(
                config, state_db, zotero_client, embeddings_db, llm
            )
    assert summary.papers_processed == 2
    assert summary.papers_failed == 0
    assert embeddings_db.add_paper.call_count == 2
@pytest.mark.unit
def test_brain_build_resume_skips_completed(tmp_path):
    """Papers with fully_complete=1 in brain_build_progress are skipped."""
    config = _make_config(tmp_path)

    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    item = _make_zotero_item(doi="10.1000/done", key="KEY001")
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    # Pre-mark as complete
    state_db.upsert_brain_build_progress("KEY001", "10.1000/done")
    state_db.mark_brain_build_pass("KEY001", 1)
    state_db.mark_brain_build_pass("KEY001", 2)
    state_db.mark_brain_build_pass("KEY001", 3)
    summary = run_brain_build(
        config, state_db, zotero_client, embeddings_db, llm, resume=True
    )
    assert summary.papers_skipped == 1
    assert summary.papers_processed == 0
    embeddings_db.add_paper.assert_not_called()
@pytest.mark.unit
def test_brain_build_skips_items_without_doi(tmp_path):
    """Items without a DOI are skipped silently."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    item = _make_zotero_item(doi="", key="KEY001")
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)
    assert summary.papers_processed == 0
    assert summary.papers_failed == 0
@pytest.mark.unit
def test_brain_build_no_markdown_marks_status(tmp_path):
    """Items with DOI but no markdown attachment → status='no_markdown', not counted as processed."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    item = _make_zotero_item(doi="10.1000/nopdf", key="KEY001")
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    # M1: return None signals no markdown attachment in Zotero
    zotero_client.get_markdown_attachment.return_value = None
    summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)
    assert summary.papers_processed == 0
    record = state_db.get_paper("10.1000/nopdf")
    assert record is not None
    assert record["status"] == "no_markdown"
@pytest.mark.unit
def test_brain_build_single_failure_continues(tmp_path):

    """A failing paper is counted in papers_failed; remaining papers proceed."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    items = [
        _make_zotero_item(doi="10.1000/bad", key="KEY001"),
        _make_zotero_item(doi="10.1000/good", key="KEY002"),
    ]
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = items
    call_count = [0]
    # M1: first paper raises on markdown read, second returns text
    def md_side_effect(key):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Zotero API error")
        return "Good full text"
    zotero_client.get_markdown_attachment.side_effect = md_side_effect
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Good2021_Test.md")
    with patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}):
        with patch("scripts.pipelines.brain_build.write_paper_note",
                   return_value=note_path):
            summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)
    assert summary.papers_failed == 1
    assert summary.papers_processed == 1
# ---------------------------------------------------------------------------
# I1: item-type routing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_brain_build_skips_textbook_pipeline_items(tmp_path):
    """Items with itemType=book (pipeline=skip per R-10) are skipped."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    book_item = _make_zotero_item(doi="10.1000/book", key="KEY001")
    book_item["data"]["itemType"] = "book"
    paper_item = _make_zotero_item(doi="10.1000/paper", key="KEY002")
    # paper_item has no itemType → defaults to journalArticle → brain_build
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [book_item, paper_item]
    zotero_client.get_markdown_attachment.return_value = "Paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021.md")
    with patch("scripts.pipelines.brain_build.enrich_paper", return_value={}), \
         patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)
    assert summary.papers_processed == 1   # journalArticle processed
    assert summary.papers_skipped == 1     # book skipped


@pytest.mark.unit
def test_brain_build_skips_report_item_type(tmp_path):
    """Items with itemType=report (pipeline=skip) are counted as skipped."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    item = _make_zotero_item(doi="10.1000/report", key="KEY001")
    item["data"]["itemType"] = "report"
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)
    assert summary.papers_skipped == 1
    assert summary.papers_processed == 0
    embeddings_db.add_paper.assert_not_called()


@pytest.mark.unit
def test_brain_build_unknown_item_type_is_skipped(tmp_path):
    """Items with an item type not in item_routing.yaml are logged and skipped."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    item = _make_zotero_item(doi="10.1000/art", key="KEY001")
    item["data"]["itemType"] = "artwork"  # not in item_routing.yaml
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)
    assert summary.papers_skipped == 1
    assert summary.papers_processed == 0


@pytest.mark.unit
def test_brain_build_review_detection_sets_source_type(tmp_path):
    """When S2 classifies a paper as 'Review', source_type is stored as 'review'."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    item = _make_zotero_item(doi="10.1000/review", key="KEY001")
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    zotero_client.get_markdown_attachment.return_value = "Review paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021_Review.md")
    with patch("scripts.pipelines.brain_build.enrich_paper",
               return_value={"s2_publication_types": ["Review"]}), \
         patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)
    assert summary.papers_processed == 1
    record = state_db.get_paper("10.1000/review")
    assert record is not None
    assert record["source_type"] == "review"


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_parse_year():
    assert _parse_year("2021-06-15") == 2021
    assert _parse_year("2019") == 2019
    assert _parse_year("") == 0
    assert _parse_year("not-a-year") == 0
@pytest.mark.unit
def test_paper_embed_text_includes_core_finding():
    extraction = {"core_finding": "Key result.", "core_finding_confidence": "explicit"}
    text = _paper_embed_text("Title", "Abstract", extraction)
    assert "Key result." in text
    assert "Title" in text
    assert "Abstract" in text
@pytest.mark.unit
def test_brain_build_partial_resume_only_reruns_incomplete_phases(tmp_path):
    """
    If simple phase is complete but complex is not, resume should only run complex.
    Validates the phase-based resume logic (M3).
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)

    embeddings_db = MagicMock()
    llm = _make_llm()
    item = _make_zotero_item(doi="10.1000/partial", key="KEY001")
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    # M3: pre-mark simple_complete; complex phase still pending
    state_db.upsert_brain_build_progress("KEY001", "10.1000/partial")
    state_db.mark_brain_build_phase("KEY001", "simple")
    # Store simple-phase results in extraction_json
    state_db.upsert_paper({
        "doi": "10.1000/partial",
        "title": "Partial Paper",
        "authors": json.dumps(["Smith J"]),
        "year": 2021,
        "status": "in_progress",
        "source_type": "paper",
        "extraction_json": json.dumps({
            "core_finding": "Already extracted",
            "core_finding_confidence": "explicit",
        }),
    })
    zotero_client.get_markdown_attachment.return_value = "Full paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021_Test.md")
    captured_phases = []
    def mock_extract_paper(fulltext, llm_arg, **kwargs):
        captured_phases.append(kwargs.get("phases"))
        return {
            "core_finding": "Already extracted",
            "core_finding_confidence": "explicit",
            "novelty_statement": "New complex result",
            "novelty_statement_confidence": "explicit",
        }
    with patch("scripts.pipelines.brain_build.extract_paper",
               side_effect=mock_extract_paper):
        with patch("scripts.pipelines.brain_build.write_paper_note",
                   return_value=note_path):
            summary = run_brain_build(
                config, state_db, zotero_client, embeddings_db, llm,
                resume=True,
            )
    assert summary.papers_processed == 1
    # Only complex phase should have been requested (simple already done)
    assert captured_phases[0] == ("complex",)
# ---------------------------------------------------------------------------
# K1 — pass_strategy: "all" dispatch and WARNING
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_pass_strategy_all_calls_extract_fields_not_extract_paper(tmp_path):
    """
    When brain_build config has pass_strategy='all', _process_paper must call
    extract_fields() (single-call pass-all path) and must NOT call extract_paper().
    """
    config = _make_config(tmp_path)
    config.brain_build = SimpleNamespace(
        pass_strategy="all",
        max_tokens_per_call=12288,
        pass_strategy_value="all",
    )
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    item = _make_zotero_item(doi="10.1/k1test", key="KEY_K1")
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    zotero_client.get_markdown_attachment.return_value = "Paper text."

    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "K1_Test.md")

    with patch("scripts.pipelines.brain_build.extract_fields",
               return_value={"core_finding": "k1 result",
                             "core_finding_confidence": "explicit"}) as mock_ef, \
         patch("scripts.pipelines.brain_build.extract_paper") as mock_ep, \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    mock_ef.assert_called_once()
    mock_ep.assert_not_called()


@pytest.mark.unit
def test_warning_on_pass_strategy_all_with_pass1_model(tmp_path, caplog):
    """
    When pass_strategy='all' is set alongside any pass*_model key, brain_build
    must emit a WARNING so the user knows those per-pass keys are ignored.
    """
    import logging
    config = _make_config(tmp_path)
    config.brain_build = SimpleNamespace(
        pass_strategy="all",
        pass1_model="qwen2.5:7b",   # conflicting key — should trigger WARNING
        max_tokens_per_call=12288,
    )
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = []  # no items — just boot the pipeline

    with caplog.at_level(logging.WARNING, logger="scripts.pipelines.brain_build"):
        run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    assert any("pass_strategy='all'" in r.message and "pass1_model" in r.message
               for r in caplog.records), (
        f"Expected K1 WARNING about per-pass model conflict; got: {[r.message for r in caplog.records]}"
    )
# ---------------------------------------------------------------------------
# I2 — no-DOI items accumulated in summary
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_doi_items_accumulated_when_crossref_fails(tmp_path):
    """
    Items without a Zotero DOI AND where CrossRef returns None must be added to
    summary.no_doi_items so --resolve-no-doi can process them interactively.
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    no_doi_item = _make_zotero_item(doi="", key="NODOI01")
    no_doi_item["data"]["title"] = "Paper Without DOI"
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [no_doi_item]

    # CrossRef fallback returns None → item should land in no_doi_items
    with patch("scripts.pipelines.brain_build.resolve_doi", return_value=None):
        summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    assert len(summary.no_doi_items) == 1
    assert summary.no_doi_items[0]["zotero_key"] == "NODOI01"
    assert summary.no_doi_items[0]["title"] == "Paper Without DOI"
    assert summary.papers_processed == 0
    assert summary.papers_failed == 0  # no-DOI is not a failure


@pytest.mark.unit
def test_crossref_resolved_doi_is_accumulated(tmp_path):
    """
    When CrossRef successfully resolves a DOI, it goes into summary.crossref_resolved
    and the paper proceeds to extraction.
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    no_doi_item = _make_zotero_item(doi="", key="NODOI02")
    no_doi_item["data"]["title"] = "Paper With Resolvable DOI"
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [no_doi_item]

    zotero_client.get_markdown_attachment.return_value = "Full paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Paper2021.md")

    with patch("scripts.pipelines.brain_build.resolve_doi", return_value="10.1/resolved"), \
         patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok",
                             "core_finding_confidence": "explicit"}), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        summary = run_brain_build(
            config, state_db, zotero_client, embeddings_db, llm
        )

    assert "10.1/resolved" in summary.crossref_resolved
    assert summary.no_doi_items == []
    assert summary.papers_processed == 1
# ---------------------------------------------------------------------------
# I3 — _extraction_quality flag set when _max_n_chunks > 3
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_extraction_quality_flagged_when_many_chunks(tmp_path):
    """
    When extract_paper returns _max_n_chunks > 3, brain_build must set
    _extraction_quality='degraded_high_chunking' in the stored extraction_json.
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    item = _make_zotero_item(doi="10.1/chunky", key="CHUNK01")
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]

    zotero_client.get_markdown_attachment.return_value = "Long paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Chunky2021.md")

    # Simulate an extraction that required 5 chunks (> 3 threshold)
    high_chunk_extraction = {
        "core_finding": "Big finding.",
        "core_finding_confidence": "explicit",
        "_max_n_chunks": 5,
    }

    with patch("scripts.pipelines.brain_build.extract_paper",
               return_value=high_chunk_extraction), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    stored = state_db.get_extraction_json("10.1/chunky")
    assert stored is not None
    assert stored.get("_extraction_quality") == "degraded_high_chunking"


@pytest.mark.unit
def test_extraction_quality_not_flagged_when_few_chunks(tmp_path):
    """
    When _max_n_chunks <= 3, _extraction_quality must NOT be set in stored extraction.
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    item = _make_zotero_item(doi="10.1/small", key="SMALL01")
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]

    zotero_client.get_markdown_attachment.return_value = "Short paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Small2021.md")

    small_extraction = {
        "core_finding": "Small finding.",
        "core_finding_confidence": "explicit",
        "_max_n_chunks": 2,
    }

    with patch("scripts.pipelines.brain_build.extract_paper",
               return_value=small_extraction), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    stored = state_db.get_extraction_json("10.1/small")
    assert stored is not None
    assert "_extraction_quality" not in stored


# ---------------------------------------------------------------------------
# V-7 — pass_strategy="all" dispatches with correct schema for reviews
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pass_strategy_all_review_dispatches_with_review_schema(tmp_path):
    """
    When pass_strategy='all' and detect_review() classifies the item as a review,
    _process_paper must call extract_fields(..., content_type='review', ...) — not
    content_type='paper'.
    """
    config = _make_config(tmp_path)
    config.brain_build = SimpleNamespace(
        pass_strategy="all",
        max_tokens_per_call=12288,
    )
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    item = _make_zotero_item(doi="10.1/review-paper", key="KEY_REV")
    item["data"]["abstractNote"] = "This systematic review examined..."
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]

    zotero_client.get_markdown_attachment.return_value = "Review paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Review2021.md")

    captured_kwargs: dict = {}

    def _capture_extract_fields(fulltext, fields, llm, **kwargs):
        captured_kwargs.update(kwargs)
        return {"core_finding": "review result", "core_finding_confidence": "explicit"}

    with patch("scripts.pipelines.brain_build.enrich_paper",
               return_value={"s2_publication_types": ["Review"]}), \
         patch("scripts.pipelines.brain_build.detect_review", return_value=True), \
         patch("scripts.pipelines.brain_build.extract_fields",
               side_effect=_capture_extract_fields), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    assert captured_kwargs.get("content_type") == "review", (
        f"Expected content_type='review' in extract_fields call; got: {captured_kwargs}"
    )


# ---------------------------------------------------------------------------
# V-9 — rate-limit abort after 3 consecutive 429s
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_rate_limit_abort_after_three_consecutive_429s(tmp_path):
    """
    V-9 / P4.5: when 3 consecutive RateLimitErrors are raised, run_brain_build
    must call state_db.finish_run with status='rate_limited' and raise the
    catchable domain exception RateLimitExhausted (NOT SystemExit — that was
    uncatchable in API/web contexts). The failing paper is NOT marked as
    'error' — --resume should retry it.
    """
    from lit_monitor.llm.llm_client import RateLimitError
    from lit_monitor.pipelines.brain_build import RateLimitExhausted
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    items = [
        _make_zotero_item(doi="10.1000/rl1", key="KEY001"),
        _make_zotero_item(doi="10.1000/rl2", key="KEY002"),
        _make_zotero_item(doi="10.1000/rl3", key="KEY003"),
    ]
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = items
    # Each markdown read raises RateLimitError — simulates 3 consecutive 429s.
    zotero_client.get_markdown_attachment.side_effect = RateLimitError("429 Too Many Requests")

    # P4.5: a domain exception, not SystemExit — and definitely not SystemExit.
    with pytest.raises(RateLimitExhausted):
        with patch("scripts.pipelines.brain_build.time.sleep"):  # skip actual sleep
            run_brain_build(config, state_db, zotero_client, embeddings_db, llm)
    # The run log should record status='rate_limited', not 'error'
    import sqlite3
    with sqlite3.connect(tmp_path / "state.db") as conn:
        rows = conn.execute("SELECT status FROM run_log").fetchall()
    statuses = [r[0] for r in rows]
    assert "rate_limited" in statuses, (
        f"Expected 'rate_limited' in run_log.status; got: {statuses}"
    )
    # No paper should be marked as status='error' (V-9 spec: retry on --resume)
    for doi in ["10.1000/rl1", "10.1000/rl2", "10.1000/rl3"]:
        record = state_db.get_paper(doi)
        if record:
            assert record["status"] != "error", (
                f"Paper {doi} should not be marked 'error' after RateLimitError"
            )


# ---------------------------------------------------------------------------
# N3 — attachment/note items filtered before the processing loop
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_brain_build_filters_attachment_and_note_items(tmp_path):
    """N3: get_collection_items returns a mix of parents + attachments + notes;
    only parent items should reach the processing loop.
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    parent_item = _make_zotero_item(doi="10.1000/parent", key="PAR01")
    attachment_item = {
        "key": "ATT01",
        "data": {"itemType": "attachment", "title": "Full Text PDF", "DOI": ""},
    }
    note_item = {
        "key": "NOTE01",
        "data": {"itemType": "note", "title": "My note", "DOI": ""},
    }
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [parent_item, attachment_item, note_item]
    zotero_client.get_markdown_attachment.return_value = "Paper text."

    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021.md")
    with patch("scripts.pipelines.brain_build.enrich_paper", return_value={}), \
         patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    # Only the one parent item should be processed; attachments/notes ignored entirely.
    assert summary.papers_processed == 1
    # Attachments/notes should not appear in papers_skipped (they're just not iterated).
    assert summary.papers_skipped == 0
    assert summary.papers_failed == 0


# ---------------------------------------------------------------------------
# N4 — --max-papers counts successfully processed, not attempted
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_max_papers_counts_processed_not_attempted(tmp_path):
    """N4: when the first N items have no .md attachment, --max-papers 1 should
    still process 1 paper (the first one WITH an attachment) rather than stopping
    after the first attempt regardless of outcome.
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    no_md_item = _make_zotero_item(doi="10.1000/nomd", key="NOMD01")
    has_md_item = _make_zotero_item(doi="10.1000/hasmd", key="HAS01")

    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [no_md_item, has_md_item]

    call_count = [0]

    def md_side_effect(key):
        call_count[0] += 1
        if call_count[0] == 1:
            return None  # no attachment
        return "Paper text."

    zotero_client.get_markdown_attachment.side_effect = md_side_effect

    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021.md")
    with patch("scripts.pipelines.brain_build.enrich_paper", return_value={}), \
         patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        summary = run_brain_build(
            config, state_db, zotero_client, embeddings_db, llm, max_papers=1
        )

    # The second item (with .md) must be processed even though it was the second attempt.
    assert summary.papers_processed == 1, (
        f"Expected 1 processed paper; got {summary.papers_processed}"
    )


@pytest.mark.unit
def test_max_papers_no_md_items_counted_as_skipped(tmp_path):
    """N4 bonus: items where _process_paper returns processed=False (no .md)
    must be counted in papers_skipped so the summary is accurate.
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()

    items = [
        _make_zotero_item(doi="10.1000/nomd1", key="NOMD01"),
        _make_zotero_item(doi="10.1000/nomd2", key="NOMD02"),
    ]
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = items
    zotero_client.get_markdown_attachment.return_value = None  # no attachment for any

    summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    assert summary.papers_processed == 0
    assert summary.papers_skipped == 2, (
        f"Expected 2 skipped (no .md); got {summary.papers_skipped}"
    )


# ---------------------------------------------------------------------------
# N22 — --all-library flag routes through get_all_library_items
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_brain_build_all_library_uses_full_library_endpoint(tmp_path):
    """N22: all_library=True calls get_all_library_items, NOT get_collection_items."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    items = [
        _make_zotero_item(doi="10.1000/lib1", key="LIB001"),
        _make_zotero_item(doi="10.1000/lib2", key="LIB002"),
    ]
    zotero_client = MagicMock()
    zotero_client.get_all_library_items.return_value = items
    zotero_client.get_collection_items.return_value = []  # should NOT be called
    zotero_client.get_markdown_attachment.return_value = "Full paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021_Test.md")
    with patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "Key result.",
                             "core_finding_confidence": "explicit"}):
        with patch("scripts.pipelines.brain_build.write_paper_note",
                   return_value=note_path):
            summary = run_brain_build(
                config, state_db, zotero_client, embeddings_db, llm,
                all_library=True,
            )

    zotero_client.get_all_library_items.assert_called_once()
    zotero_client.get_collection_items.assert_not_called()
    assert summary.papers_processed == 2


@pytest.mark.unit
def test_brain_build_default_uses_collection_endpoint(tmp_path):
    """N22 inverse: without all_library, get_collection_items is called and get_all_library_items isn't."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = _make_llm()
    items = [_make_zotero_item(doi="10.1000/c1", key="C001")]
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = items
    zotero_client.get_all_library_items.return_value = []  # should NOT be called
    zotero_client.get_markdown_attachment.return_value = "Full paper text."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Foo.md")
    with patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "X", "core_finding_confidence": "explicit"}):
        with patch("scripts.pipelines.brain_build.write_paper_note",
                   return_value=note_path):
            run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    zotero_client.get_collection_items.assert_called_once()
    zotero_client.get_all_library_items.assert_not_called()
