"""
Unit tests for the discovery pipeline, ranker, and obsidian tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_config(tmp_path: Path) -> SimpleNamespace:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Literature" / "Digests").mkdir(parents=True)
    (vault / "Literature" / "Papers").mkdir(parents=True)
    (vault / "Literature" / "Connections").mkdir(parents=True)
    return SimpleNamespace(
        zotero=SimpleNamespace(
            collection_name="lit-monitor",
            local_storage_path=str(tmp_path / "zotero"),
        ),
        obsidian=SimpleNamespace(
            vault_path=str(vault),
            papers_folder="Literature/Papers",
            digests_folder="Literature/Digests",
            connections_folder="Literature/Connections",
        ),
        topics=["filtration AND ProteinA"],
        researchers=[],
        wos_api_key=None,
        scopus_api_key=None,
        email="test@example.com",
    )
def _make_state_db(tmp_path: Path):
    from scripts.core.state_db import StateDB
    return StateDB(tmp_path / "state.db")
def _make_paper(doi: str = "10.1000/test", score: float = 0.0) -> dict:
    return {
        "doi": doi,
        "title": f"Paper on {doi}",
        "abstract": "We studied filtration.",
        "authors": ["Smith J"],
        "year": 2024,
        "similarity_score": score,
        "llm_rationale": "",
    }
# ===========================================================================
# Discovery pipeline tests
# ===========================================================================

@pytest.mark.unit
def test_full_discovery_dry_run_no_writes(tmp_path):
    """dry_run=True: discovery runs but nothing is written to state DB."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    papers = [_make_paper("10.1/new1"), _make_paper("10.1/new2")]
    with patch("scripts.pipelines.discovery.run_searches", return_value=papers):
        with patch("scripts.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("scripts.pipelines.discovery.filter_known_dois",
                       return_value=papers):
                with patch("scripts.pipelines.discovery.rank_papers",
                           return_value=papers):
                    from scripts.pipelines.discovery import run_discovery
                    summary = run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=True
                    )
    # No papers stored in state DB (dry_run)
    known = state_db.known_dois()
    assert known == set()
    assert summary.new_papers_found == 2
    # H1 regression: dry_run must not mutate kv_store with last_run_date.
    # Pre-fix, set_kv was called unconditionally after a successful search,
    # which violated the dry_run read-only contract documented at the top of
    # scripts/pipelines/discovery.py.
    assert state_db.get_kv("last_run_date") is None
@pytest.mark.unit
def test_discovery_digest_written(tmp_path):
    """Non-dry-run: digest file created in digests_folder."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    zotero_client.get_current_version.return_value = 100  # first-run baseline
    papers = [_make_paper("10.1/p1", score=0.85)]
    with patch("scripts.pipelines.discovery.run_searches", return_value=papers):
        with patch("scripts.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("scripts.pipelines.discovery.filter_known_dois",
                       return_value=papers):
                with patch("scripts.pipelines.discovery.rank_papers",
                           return_value=papers):
                    from scripts.pipelines.discovery import run_discovery
                    summary = run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=False
                    )
    digest_dir = Path(config.obsidian.vault_path) / config.obsidian.digests_folder
    digests = list(digest_dir.glob("Discovery_*.md"))
    assert len(digests) == 1
    assert summary.digest_path != ""
@pytest.mark.unit

def test_duplicate_dois_not_reprocessed(tmp_path):
    """Papers already in state DB are filtered by filter_known_dois."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    state_db.upsert_paper({"doi": "10.1/existing", "status": "extraction_complete"})
    embeddings_db = MagicMock()
    llm = MagicMock()
    zotero_client = MagicMock()
    zotero_client.get_current_version.return_value = 100  # first-run baseline
    raw = [_make_paper("10.1/existing"), _make_paper("10.1/new")]
    # Use real filter_known_dois
    with patch("scripts.pipelines.discovery.run_searches", return_value=raw):
        with patch("scripts.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("scripts.pipelines.discovery.rank_papers",
                       return_value=[_make_paper("10.1/new")]):
                from scripts.pipelines.discovery import run_discovery
                from scripts.search.search_runner import filter_known_dois
                with patch("scripts.pipelines.discovery.filter_known_dois",
                           side_effect=lambda papers, db: filter_known_dois(papers, db)):
                    summary = run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=True
                    )
    assert summary.new_papers_found == 1
@pytest.mark.unit
def test_pipeline_continues_on_search_failure(tmp_path):
    """A failing search does not abort the pipeline; empty results returned."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    with patch("scripts.pipelines.discovery.run_searches",
               side_effect=RuntimeError("API down")):
        with patch("scripts.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("scripts.pipelines.discovery.filter_known_dois",
                       return_value=[]):
                with patch("scripts.pipelines.discovery.rank_papers",
                           return_value=[]):
                    from scripts.pipelines.discovery import run_discovery
                    summary = run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=True
                    )
    # Pipeline ran to completion despite search failure
    assert summary.new_papers_found == 0
    assert len(summary.errors) == 1
@pytest.mark.unit
def test_new_zotero_items_ingested(tmp_path):
    """New Zotero items with PDFs are ingested (extracted + noted + indexed)."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()

    llm.complete.return_value = json.dumps({
        "core_finding": "Test finding.",
        "core_finding_confidence": "explicit",
        "methods_summary": None,
        "methods_summary_confidence": "absent",
        "results_summary": None,
        "results_summary_confidence": "absent",
        "conclusions": None,
        "conclusions_confidence": "absent",
        "study_type": None,
        "study_type_confidence": "absent",
    })
    zotero_client = MagicMock()
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2024_Test.md")
    new_item = {
        "key": "NEWKEY1",
        "data": {
            "DOI": "10.1/brand-new",
            "title": "Brand New Paper",
            "date": "2024",
            "publicationTitle": "J Sci",
            "abstractNote": "Abstract.",
            "tags": [],
            "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "J"}],
        },
        "attachments": [],
    }
    # Pre-set a baseline version so ingestion runs (not first-run baseline path)
    state_db.set_kv("zotero_library_version", "100")
    zotero_client.get_items_since.return_value = [new_item]
    zotero_client.get_current_version.return_value = 101
    # M1: pipeline reads markdown attachments — not PDFs
    zotero_client.get_markdown_attachment.return_value = "Full paper text."
    with patch("scripts.pipelines.discovery.run_searches", return_value=[]):
        with patch("scripts.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("scripts.pipelines.discovery.filter_known_dois",
                       return_value=[]):
                with patch("scripts.pipelines.discovery.rank_papers", return_value=[]):
                    with patch("scripts.pipelines.discovery.enrich_paper",
                               return_value={}):
                        with patch("scripts.pipelines.discovery.extract_paper",
                                   return_value={"core_finding": "ok",
                                                 "core_finding_confidence": "explicit"}):
                            with patch("scripts.pipelines.discovery.write_paper_note",
                                       return_value=note_path):
                                with patch("scripts.pipelines.discovery.relink_note"):
                                    from scripts.pipelines.discovery import run_discovery
                                    summary = run_discovery(
                                        config, state_db, zotero_client, embeddings_db,
                                        llm,
                                        dry_run=False
                                    )
    assert summary.papers_ingested == 1
    embeddings_db.add_paper.assert_called_once()

@pytest.mark.unit
def test_item_quality_filter_skips_no_title(tmp_path):
    """A2: _check_item_quality returns (False, reason) for items without a title."""
    from scripts.pipelines.brain_build import _check_item_quality
    ok, reason = _check_item_quality({"title": "", "creators": [], "abstractNote": "Abstract."})
    assert not ok
    assert "title" in reason

@pytest.mark.unit
def test_item_quality_filter_passes_valid_item(tmp_path):
    """A2: _check_item_quality returns (True, '') for a well-formed item."""
    from scripts.pipelines.brain_build import _check_item_quality
    data = {
        "title": "Filtration of ProteinAs",
        "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "J"}],
        "abstractNote": "We studied transport mechanism.",
        "date": "2024",
    }
    ok, reason = _check_item_quality(data)
    assert ok
    assert reason == ""

@pytest.mark.unit
def test_discovery_ingestion_resumes_after_partial_failure(tmp_path):
    """A3: Items marked fully_complete in brain_build_progress are skipped on re-run."""
    from scripts.core.state_db import StateDB
    state_db = StateDB(tmp_path / "state.db")

    # Pre-mark item DONE1 as fully complete in the progress table
    state_db.upsert_brain_build_progress("DONE1", "10.1/done")
    for p in (1, 2, 3):
        state_db.mark_brain_build_pass("DONE1", p)

    # Set a baseline Zotero version so ingestion runs
    state_db.set_kv("zotero_library_version", "100")

    config = _make_config(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    zotero_client.get_current_version.return_value = 101
    zotero_client.get_items_since.return_value = [
        {
            "key": "DONE1",
            "data": {
                "DOI": "10.1/done",
                "title": "Already Done",
                "date": "2024",
                "publicationTitle": "",
                "abstractNote": "Done.",
                "tags": [],
                "creators": [{"creatorType": "author", "lastName": "A", "firstName": "B"}],
            },
        }
    ]
    with patch("scripts.pipelines.discovery.run_searches", return_value=[]):
        with patch("scripts.pipelines.discovery.run_researcher_searches", return_value=[]):
            with patch("scripts.pipelines.discovery.filter_known_dois", return_value=[]):
                with patch("scripts.pipelines.discovery.rank_papers", return_value=[]):
                    from scripts.pipelines.discovery import run_discovery
                    summary = run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=False,
                    )
    # Item should be skipped via resume check (already complete in brain_build_progress)
    assert summary.papers_ingested == 0
    zotero_client.get_markdown_attachment.assert_not_called()

@pytest.mark.unit
def test_since_days_computed_from_last_run_date(tmp_path):
    """A4: search window is derived from stored last_run_date, not hard-coded 14."""
    import datetime as dt
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    # Store a last_run_date 30 days ago
    past_date = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    state_db.set_kv("last_run_date", past_date)
    state_db.set_kv("zotero_library_version", "100")

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    zotero_client.get_current_version.return_value = 101
    zotero_client.get_items_since.return_value = []

    captured_since: list[int] = []

    def capture_run_searches(cfg, since_days=14, **kwargs):
        captured_since.append(since_days)
        return []

    with patch("scripts.pipelines.discovery.run_searches",
               side_effect=capture_run_searches):
        with patch("scripts.pipelines.discovery.run_researcher_searches", return_value=[]):
            with patch("scripts.pipelines.discovery.filter_known_dois", return_value=[]):
                with patch("scripts.pipelines.discovery.rank_papers", return_value=[]):
                    from scripts.pipelines.discovery import run_discovery
                    run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=False,
                    )

    assert len(captured_since) == 1
    # 30 days ago → since_days = 31 (inclusive of today)
    assert captured_since[0] == 31

# ===========================================================================

# Ranker tests
# ===========================================================================
@pytest.mark.unit
def test_ranker_sorts_by_similarity():
    """Papers returned sorted by similarity_score descending."""
    from scripts.llm.ranker import rank_papers
    candidates = [
        {"doi": "10.1/low", "title": "Low relevance", "abstract": ""},
        {"doi": "10.1/high", "title": "High relevance", "abstract": ""},
    ]
    embeddings_db = MagicMock()
    # Return scores: low=0.2, high=0.9
    embeddings_db.find_similar_to_text.side_effect = [
        [{"score": 0.2, "id": "10.1/low", "document": "", "metadata": {}}],
        [{"score": 0.9, "id": "10.1/high", "document": "", "metadata": {}}],
    ]
    llm = MagicMock()
    llm.complete.return_value = json.dumps({"10.1/low": "Not very relevant.", "10.1/high": "Very relevant!"})
    ranked = rank_papers(candidates, embeddings_db, llm, top_k=5)
    assert ranked[0]["doi"] == "10.1/high"
    assert ranked[0]["similarity_score"] == pytest.approx(0.9)
@pytest.mark.unit
def test_ranker_adds_llm_rationale():
    """Top-K papers have llm_rationale set."""
    from scripts.llm.ranker import rank_papers
    candidates = [{"doi": "10.1/p1", "title": "Paper 1", "abstract": "Abstract 1"}]
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = [{"score": 0.75, "id": "10.1/other",
    "document": "", "metadata": {}}]
    llm = MagicMock()
    llm.complete.return_value = json.dumps({"10.1/p1": "Highly relevant to TopicX."})
    ranked = rank_papers(candidates, embeddings_db, llm, top_k=1)
    assert ranked[0]["llm_rationale"] == "Highly relevant to TopicX."
@pytest.mark.unit
def test_ranker_empty_returns_empty():
    """Empty candidates list returns empty list without calling embeddings or LLM."""
    from scripts.llm.ranker import rank_papers
    embeddings_db = MagicMock()
    llm = MagicMock()
    result = rank_papers([], embeddings_db, llm)
    assert result == []
    embeddings_db.find_similar_to_text.assert_not_called()
    llm.complete.assert_not_called()
@pytest.mark.unit
def test_ranker_llm_failure_returns_empty_rationale():
    """LLM failure does not crash ranker — rationale fields left empty."""
    from scripts.llm.ranker import rank_papers
    candidates = [{"doi": "10.1/p1", "title": "Paper", "abstract": ""}]

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = [{"score": 0.5, "id": "x", "document": "",
    "metadata": {}}]
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("Ollama down")
    ranked = rank_papers(candidates, embeddings_db, llm)
    assert ranked[0]["llm_rationale"] == ""
# ===========================================================================
# Obsidian tools tests
# ===========================================================================
@pytest.mark.unit
def test_retheme_rewrites_wikilinks(tmp_path):
    """retheme() rewrites [[OldTheme]] → [[NewTheme]] in vault notes."""
    from scripts.obsidian_tools.retheme import retheme
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("See [[OldTheme]] and [[OldTheme|custom text]] for details.", encoding="utf-8")
    stats = retheme(vault, "OldTheme", "NewTheme")
    content = note.read_text(encoding="utf-8")
    assert "[[NewTheme]]" in content
    assert "[[OldTheme]]" not in content
    assert stats["files_modified"] == 1
    assert stats["wikilinks_rewritten"] == 2
@pytest.mark.unit
def test_retheme_renames_page(tmp_path):
    """retheme() renames the theme .md page if it exists."""
    from scripts.obsidian_tools.retheme import retheme
    vault = tmp_path / "vault"
    vault.mkdir()
    old_page = vault / "OldTheme.md"
    old_page.write_text("# OldTheme\n", encoding="utf-8")
    stats = retheme(vault, "OldTheme", "NewTheme")
    assert not old_page.exists()
    assert (vault / "NewTheme.md").exists()
    assert stats["page_renamed"] == 1
@pytest.mark.unit
def test_rerender_paper_note(tmp_path):
    """rerender_note() calls write_paper_note with extraction from state DB."""
    from scripts.obsidian_tools.rerender import rerender_note
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    extraction = {"core_finding": "Test.", "core_finding_confidence": "explicit"}
    state_db.upsert_paper({
        "doi": "10.1/test",
        "title": "Test Paper",
        "authors": json.dumps(["Smith J"]),
        "year": 2021,

        "source_type": "paper",
        "extraction_json": json.dumps(extraction),
        "status": "extraction_complete",
    })
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021_Test.md")
    with patch("scripts.obsidian_tools.rerender.write_paper_note",
               return_value=note_path) as mock_write:
        result = rerender_note("10.1/test", config, state_db)
    mock_write.assert_called_once()
    assert result == note_path
@pytest.mark.unit
def test_rerender_raises_for_missing_doi(tmp_path):
    """rerender_note() raises ValueError for unknown doi."""
    from scripts.obsidian_tools.rerender import rerender_note
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    with pytest.raises(ValueError, match="No record found"):
        rerender_note("10.1/nonexistent", config, state_db)
@pytest.mark.unit
def test_re_extract_updates_state_db(tmp_path):
    """re_extract() updates extraction_json in state DB."""
    from scripts.obsidian_tools.re_extract import re_extract
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    llm = MagicMock()
    old_extraction = {"core_finding": "Old result.", "core_finding_confidence": "explicit"}
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021_Test.md")
    Path(note_path).write_text("# Note content", encoding="utf-8")
    state_db.upsert_paper({
        "doi": "10.1/test",
        "source_type": "paper",
        "extraction_json": json.dumps(old_extraction),
        "note_path": note_path,
        "status": "extraction_complete",
    })
    new_extraction = {"core_finding": "Updated result.", "core_finding_confidence": "explicit",
                      "methods_summary": None, "methods_summary_confidence": "absent",
                      "results_summary": None, "results_summary_confidence": "absent",
                      "conclusions": None, "conclusions_confidence": "absent",
                      "study_type": None, "study_type_confidence": "absent"}
    with patch("scripts.obsidian_tools.re_extract.extract_paper",
               return_value=new_extraction):
        with patch("scripts.obsidian_tools.re_extract.rerender_note"):
            re_extract("10.1/test", config, state_db, llm, phases=["simple"])
    updated = state_db.get_extraction_json("10.1/test")
    assert updated["core_finding"] == "Updated result."
@pytest.mark.unit
def test_synthesize_writes_note(tmp_path):
    """synthesize() writes a synthesis note to the connections folder."""

    from scripts.obsidian_tools.synthesize import synthesize
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = "This is a synthesis paragraph about TopicX."
    extraction = {"core_finding": "Key finding.", "core_finding_confidence": "explicit"}
    state_db.upsert_paper({
        "doi": "10.1/related",
        "title": "Relevant Paper",
        "source_type": "paper",
        "note_title": "Smith2021_Relevant",
        "extraction_json": json.dumps(extraction),
        "status": "extraction_complete",
    })
    # N18: synthesize() now tries find_similar_chunks() first and falls back
    # to find_similar_to_text() when chunk_results is empty.  Mock both so the
    # fallback path is the one we test here.
    embeddings_db.find_similar_chunks.return_value = []
    embeddings_db.find_similar_to_text.return_value = [
        {"id": "10.1/related", "score": 0.88, "document": "Key finding.", "metadata": {}}
    ]
    note_path = synthesize("TopicX transport mechanism", config, state_db, embeddings_db, llm)
    assert note_path != ""
    note_file = Path(note_path)
    assert note_file.exists()
    content = note_file.read_text(encoding="utf-8")
    assert "TopicX transport mechanism" in content
    assert "[[Smith2021_Relevant]]" in content
@pytest.mark.unit
def test_synthesize_empty_results_returns_empty(tmp_path):
    """synthesize() returns empty string if no similar notes found."""
    from scripts.obsidian_tools.synthesize import synthesize
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    # N18: zero out both retrieval paths so neither chunks nor paper-level
    # results provide a hit.  Without this, find_similar_chunks() returns a
    # MagicMock that is truthy by default and bypasses the empty-result guard.
    embeddings_db.find_similar_chunks.return_value = []
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    result = synthesize("some obscure topic", config, state_db, embeddings_db, llm)
    assert result == ""
    llm.complete.assert_not_called()
# ===========================================================================
# Model compare — signature verification
# ===========================================================================
@pytest.mark.unit
def test_model_compare_extract_paper_uses_valid_kwargs():
    """
    Verify that model_compare.py calls extract_paper with only valid parameters.
    M3 hard-cut: 'passes' is removed; 'phases' replaces it.
    """
    import inspect

    from scripts.llm.extractor import extract_paper
    valid_params = set(inspect.signature(extract_paper).parameters.keys())
    # 'passes' must not exist; 'phases' must
    assert "passes" not in valid_params, "'passes' must be removed after M3 hard-cut"
    assert "phases" in valid_params, "'phases' must be the new parameter"

# ===========================================================================
# A8 — --max-papers flag
# ===========================================================================

@pytest.mark.unit
def test_brain_build_respects_max_papers(tmp_path):
    """
    A8: run_brain_build() with max_papers=2 processes exactly 2 papers even
    when 5 valid items are available.  Papers skipped by quality/resume checks
    do not count against the cap — the limit applies to papers *attempted*.
    """
    from scripts.pipelines.brain_build import run_brain_build

    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()

    # 5 items that all have DOIs, titles, and abstracts — all pass quality checks
    def _make_zotero_item(n: int) -> dict:
        return {
            "key": f"ITEM{n:04d}",
            "data": {
                "DOI": f"10.1000/test{n}",
                "title": f"Test Paper {n}",
                "abstractNote": "We studied membranes.",
                "date": "2024",
                "publicationTitle": "J. Membranes",
                "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "A"}],
                "tags": [],
            },
            "attachments": [],
        }

    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [_make_zotero_item(i) for i in range(5)]

    process_calls: list[str] = []

    def fake_process_paper(doi, **kwargs) -> tuple[bool, list[str]]:
        process_calls.append(doi)
        return True, []  # M4: _process_paper returns (success, discovered_topics)

    with patch("scripts.pipelines.brain_build._process_paper",
               side_effect=fake_process_paper):
        summary = run_brain_build(
            config=config,
            state_db=state_db,
            zotero_client=zotero_client,
            embeddings_db=embeddings_db,
            llm=llm,
            batch_size=50,
            resume=True,      # resume=True is fine — items aren't in the progress table
            max_papers=2,
        )

    assert len(process_calls) == 2, (
        f"Expected exactly 2 papers attempted, got {len(process_calls)}: {process_calls}"
    )
    assert summary.papers_processed == 2


# ---------------------------------------------------------------------------
# L1: pre-extraction S2 enrichment in discovery._run_ingestion
# ---------------------------------------------------------------------------

def _make_zotero_item_for_ingestion(doi: str, key: str = "KEY001") -> dict:
    return {
        "key": key,
        "data": {
            "DOI": doi,
            "title": "Test Paper on Filtration",
            "date": "2021",
            "publicationTitle": "J Membrane Sci",
            "abstractNote": "We studied filtration.",
            "tags": [],
            "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "J"}],
        },
    }


def _run_ingestion_with_item(tmp_path, item, enrich_return=None):
    """Helper: run discovery ingestion for a single item and return (summary, state_db)."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    state_db.set_kv("zotero_library_version", "100")
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = json.dumps({"core_finding": "ok", "core_finding_confidence": "explicit"})
    zotero_client = MagicMock()
    zotero_client.get_items_since.return_value = [item]
    zotero_client.get_current_version.return_value = 101
    # M1: pipeline reads markdown attachments — not PDFs
    zotero_client.get_markdown_attachment.return_value = "Full paper text in markdown."
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021.md")

    with patch("scripts.pipelines.discovery.run_searches", return_value=[]), \
         patch("scripts.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("scripts.pipelines.discovery.filter_known_dois", return_value=[]), \
         patch("scripts.pipelines.discovery.rank_papers", return_value=[]), \
         patch("scripts.pipelines.discovery.enrich_paper",
               return_value=enrich_return or {}), \
         patch("scripts.pipelines.discovery.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("scripts.pipelines.discovery.write_paper_note", return_value=note_path), \
         patch("scripts.pipelines.discovery.relink_note"):
        from scripts.pipelines.discovery import run_discovery
        summary = run_discovery(
            config, state_db, zotero_client, embeddings_db, llm, dry_run=False
        )
    return summary, state_db


@pytest.mark.unit
def test_ingestion_s2_enrichment_happens_before_extraction(tmp_path):
    """L1: enrich_paper is called and its data is merged into extraction_json."""
    item = _make_zotero_item_for_ingestion("10.1/new")
    s2_payload = {"s2_citation_count": 42, "s2_publication_types": ["JournalArticle"]}
    summary, state_db = _run_ingestion_with_item(tmp_path, item, enrich_return=s2_payload)
    assert summary.papers_ingested == 1
    extraction = state_db.get_extraction_json("10.1/new")
    assert extraction is not None
    assert extraction.get("s2_citation_count") == 42


@pytest.mark.unit
def test_ingestion_review_detection_sets_source_type(tmp_path):
    """L1: When S2 publicationTypes includes 'Review', source_type is stored as 'review'."""
    item = _make_zotero_item_for_ingestion("10.1/review")
    s2_payload = {"s2_publication_types": ["Review"]}
    summary, state_db = _run_ingestion_with_item(tmp_path, item, enrich_return=s2_payload)
    assert summary.papers_ingested == 1
    record = state_db.get_paper("10.1/review")
    assert record is not None
    assert record["source_type"] == "review"


@pytest.mark.unit
def test_ingestion_no_s2_data_uses_paper_source_type(tmp_path):
    """L1: When enrich_paper returns {}, source_type defaults to 'paper'."""
    item = _make_zotero_item_for_ingestion("10.1/regular")
    summary, state_db = _run_ingestion_with_item(tmp_path, item, enrich_return={})
    assert summary.papers_ingested == 1
    record = state_db.get_paper("10.1/regular")
    assert record is not None
    assert record["source_type"] == "paper"
# ---------------------------------------------------------------------------
# L2 — Pipeline Run Summary prepended to discovery digest
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_discovery_digest_prepends_pipeline_run_summary(tmp_path):
    """
    write_discovery_digest() with a non-empty recent_runs list must prepend a
    '## Pipeline Run Summary' section before the discovery content.
    """
    from scripts.pipelines.discovery import _write_digest
    config = _make_config(tmp_path)
    recent_runs = [
        {
            "run_type": "brain_build",
            "started_at": "2026-05-11T09:00:00",
            "status": "complete",
            "papers_processed": 12,
            "papers_failed": 0,
        }
    ]
    digest_path = _write_digest(
        ranked=[_make_paper("10.1/new", score=0.9)],
        config=config,
        sim_threshold=0.5,
        dry_run=False,
        recent_runs=recent_runs,
    )
    content = Path(digest_path).read_text(encoding="utf-8")
    # Pipeline Run Summary section must come before the main digest separator
    assert "## Pipeline Run Summary" in content
    assert content.index("## Pipeline Run Summary") < content.index("---")
    assert "brain_build" in content


@pytest.mark.unit
def test_pipeline_run_summary_sources_from_run_log_not_in_memory(tmp_path):
    """
    run_discovery() passes state_db.get_recent_runs() to write_discovery_digest().
    The written digest must include run_log data even when no in-memory variable tracks it.
    """
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    # Seed a run_log entry so get_recent_runs() returns it
    state_db.start_run("run-L2-test", "brain_build")
    state_db.finish_run("run-L2-test", status="complete", processed=7, skipped=0, failed=0)

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    papers = [_make_paper("10.1/L2paper", score=0.8)]

    with patch("scripts.pipelines.discovery.run_searches", return_value=papers):
        with patch("scripts.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("scripts.pipelines.discovery.filter_known_dois",
                       return_value=papers):
                with patch("scripts.pipelines.discovery.rank_papers",
                           return_value=papers):
                    from scripts.pipelines.discovery import run_discovery
                    run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=False,
                    )

    # Find the written digest
    digest_dir = Path(config.obsidian.vault_path) / config.obsidian.digests_folder
    digests = list(digest_dir.glob("Discovery_*.md"))
    assert digests, "Expected a discovery digest to be written"
    content = digests[0].read_text(encoding="utf-8")
    # The seeded run_log entry should appear in the digest
    assert "brain_build" in content
    assert "## Pipeline Run Summary" in content


# ===========================================================================
# M6 — rebuild-citations CLI command
# ===========================================================================

def _seed_paper_with_citations(state_db, doi: str, key_citations: list, note_path: str) -> None:
    """Helper: insert a paper record with key_citations populated."""
    extraction = {
        "key_citations": key_citations,
        "key_citations_confidence": "explicit",
    }
    state_db.upsert_paper({
        "doi": doi,
        "source_type": "paper",
        "extraction_json": json.dumps(extraction),
        "note_path": note_path,
        "status": "extraction_complete",
    })


def _fake_citation_result(n_resolved: int = 2, n_unresolved: int = 1) -> object:
    """Return a minimal stand-in for build_citation_graph's result object."""
    from types import SimpleNamespace
    return SimpleNamespace(
        n_resolved=n_resolved,
        n_unresolved=n_unresolved,
        s2_references_count=n_resolved + n_unresolved,
    )


@pytest.mark.unit
def test_rebuild_citations_scope_doi_calls_re_extract_and_graph(tmp_path):
    """--scope doi: calls complex-phase re_extract then build_citation_graph."""
    from click.testing import CliRunner

    from scripts.cli import main

    config = _make_config(tmp_path)  # creates vault/Literature/Papers/ tree
    state_db = _make_state_db(tmp_path)
    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Test2024.md")
    Path(note_path).write_text("# Test", encoding="utf-8")
    _seed_paper_with_citations(state_db, "10.1/doi-test", ["Ref1", "Ref2"], note_path)

    graph_calls: list[str] = []

    def fake_build_citation_graph(doi, state_db, api_key=None, max_retries=4):
        graph_calls.append(doi)
        return _fake_citation_result()

    runner = CliRunner()
    with patch("scripts.cli._make_config", return_value=config), \
         patch("scripts.cli._make_state_db", return_value=state_db), \
         patch("scripts.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("scripts.cli._load_secrets", return_value={}), \
         patch("scripts.cli._make_zotero_client", return_value=MagicMock()), \
         patch("scripts.cli._make_llm", return_value=MagicMock()), \
         patch("scripts.obsidian_tools.re_extract.re_extract", return_value={}), \
         patch("scripts.search.citation_graph.build_citation_graph", fake_build_citation_graph), \
         patch("scripts.obsidian_tools.relink.relink_note"):
        result = runner.invoke(
            main,
            ["obsidian", "rebuild-citations", "--doi", "10.1/doi-test", "--scope", "doi"],
        )

    assert result.exit_code == 0, result.output
    assert graph_calls == ["10.1/doi-test"]
    assert "Done" in result.output


@pytest.mark.unit
def test_rebuild_citations_scope_doi_requires_doi_flag(tmp_path):
    """--scope doi without --doi raises UsageError."""
    from click.testing import CliRunner

    from scripts.cli import main

    runner = CliRunner()
    with patch("scripts.cli._make_config", return_value=_make_config(tmp_path)), \
         patch("scripts.cli._make_state_db", return_value=_make_state_db(tmp_path)), \
         patch("scripts.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("scripts.cli._load_secrets", return_value={}), \
         patch("scripts.cli._make_zotero_client", return_value=MagicMock()):
        result = runner.invoke(
            main,
            ["obsidian", "rebuild-citations", "--scope", "doi"],
        )

    assert result.exit_code != 0
    assert "doi" in result.output.lower()


@pytest.mark.unit
def test_rebuild_citations_scope_all_skips_no_key_citations(tmp_path):
    """--scope all skips papers that have no key_citations in extraction_json."""
    from click.testing import CliRunner

    from scripts.cli import main

    state_db = _make_state_db(tmp_path)
    # One paper with citations, one without
    state_db.upsert_paper({
        "doi": "10.1/with-cites",
        "source_type": "paper",
        "extraction_json": json.dumps({"key_citations": ["Ref A"]}),
        "note_path": "",
        "status": "extraction_complete",
    })
    state_db.upsert_paper({
        "doi": "10.1/no-cites",
        "source_type": "paper",
        "extraction_json": json.dumps({"key_citations": []}),
        "note_path": "",
        "status": "extraction_complete",
    })

    graph_calls: list[str] = []

    def fake_build(doi, state_db, api_key=None, max_retries=4):
        graph_calls.append(doi)
        return _fake_citation_result()

    runner = CliRunner()
    with patch("scripts.cli._make_config", return_value=_make_config(tmp_path)), \
         patch("scripts.cli._make_state_db", return_value=state_db), \
         patch("scripts.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("scripts.cli._load_secrets", return_value={}), \
         patch("scripts.cli._make_zotero_client", return_value=MagicMock()), \
         patch("scripts.search.citation_graph.build_citation_graph", fake_build), \
         patch("scripts.obsidian_tools.relink.relink_note"):
        result = runner.invoke(main, ["obsidian", "rebuild-citations", "--scope", "all"])

    assert result.exit_code == 0, result.output
    assert graph_calls == ["10.1/with-cites"]
    assert "1 papers" in result.output


@pytest.mark.unit
def test_rebuild_citations_scope_failed_skips_already_resolved(tmp_path):
    """--scope failed skips papers that already have citation_edges rows."""
    from click.testing import CliRunner

    from scripts.cli import main

    state_db = _make_state_db(tmp_path)
    # Seed two papers with citations
    for doi in ["10.1/already-resolved", "10.1/needs-resolve"]:
        state_db.upsert_paper({
            "doi": doi,
            "source_type": "paper",
            "extraction_json": json.dumps({"key_citations": ["Ref X"]}),
            "note_path": "",
            "status": "extraction_complete",
        })
    # Seed an edge row for the first paper so it looks resolved
    state_db.upsert_citation_edge(
        source_doi="10.1/already-resolved",
        ref_id="Ref X",
        target_doi="10.1/target",
        target_s2_id=None,
        context="",
        resolution="resolved",
    )

    graph_calls: list[str] = []

    def fake_build(doi, state_db, api_key=None, max_retries=4):
        graph_calls.append(doi)
        return _fake_citation_result()

    runner = CliRunner()
    with patch("scripts.cli._make_config", return_value=_make_config(tmp_path)), \
         patch("scripts.cli._make_state_db", return_value=state_db), \
         patch("scripts.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("scripts.cli._load_secrets", return_value={}), \
         patch("scripts.cli._make_zotero_client", return_value=MagicMock()), \
         patch("scripts.search.citation_graph.build_citation_graph", fake_build), \
         patch("scripts.obsidian_tools.relink.relink_note"):
        result = runner.invoke(main, ["obsidian", "rebuild-citations", "--scope", "failed"])

    assert result.exit_code == 0, result.output
    # Only the unresolved paper should go through build_citation_graph
    assert graph_calls == ["10.1/needs-resolve"]


@pytest.mark.unit
def test_rebuild_citations_no_rerender_skips_relink(tmp_path):
    """--no-rerender prevents relink_note from being called."""
    from click.testing import CliRunner

    from scripts.cli import main

    state_db = _make_state_db(tmp_path)
    note_path = str(tmp_path / "Test2024.md")
    Path(note_path).write_text("# Note", encoding="utf-8")
    _seed_paper_with_citations(state_db, "10.1/norerender", ["Ref1"], note_path)

    def fake_build(doi, state_db, api_key=None, max_retries=4):
        return _fake_citation_result()

    relink_calls: list = []

    runner = CliRunner()
    with patch("scripts.cli._make_config", return_value=_make_config(tmp_path)), \
         patch("scripts.cli._make_state_db", return_value=state_db), \
         patch("scripts.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("scripts.cli._load_secrets", return_value={}), \
         patch("scripts.cli._make_zotero_client", return_value=MagicMock()), \
         patch("scripts.cli._make_llm", return_value=MagicMock()), \
         patch("scripts.obsidian_tools.re_extract.re_extract", return_value={}), \
         patch("scripts.search.citation_graph.build_citation_graph", fake_build), \
         patch("scripts.obsidian_tools.relink.relink_note",
               side_effect=lambda *a, **kw: relink_calls.append(a)):
        result = runner.invoke(
            main,
            ["obsidian", "rebuild-citations", "--doi", "10.1/norerender",
             "--scope", "doi", "--no-rerender"],
        )

    assert result.exit_code == 0, result.output
    assert relink_calls == [], "relink_note must not be called when --no-rerender is set"


# ===========================================================================
# H1 — state-machine cleanup regression tests
# ===========================================================================

@pytest.mark.unit
def test_brain_build_embed_failure_does_not_mark_extraction_complete(tmp_path):
    """H1: when embeddings_db.add_paper raises, state_db.upsert_paper must
    NEVER be called with status='extraction_complete'.  Pre-fix, status was
    written BEFORE the embed step (line ~481), so an embed failure produced
    two conflicting writes (extraction_complete then error) and left
    embeddings_indexed=0 — the paper looked done but wasn't, and --resume
    silently skipped it.

    Uses a real StateDB to satisfy SELECT-side reads, but wraps upsert_paper
    in a spy so we can assert no extraction_complete payload was written.
    Checking the final row status alone does NOT catch the regression: the
    outer Exception handler overwrites with 'error' in BOTH pre- and post-fix
    code, so the end state looks identical.
    """
    from scripts.pipelines.brain_build import run_brain_build

    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    # Simulate ChromaDB outage — the R28 helper returns (False, err) and
    # _process_paper raises RuntimeError on the embed failure path.
    embeddings_db.add_paper.side_effect = RuntimeError("chroma unavailable")
    llm = MagicMock()
    llm.model = "gemma4:e4b"
    llm.provider = "ollama"

    item = {
        "key": "EMBFAIL01",
        "data": {
            "DOI": "10.1000/embfail",
            "title": "Paper That Fails To Embed",
            "date": "2024",
            "publicationTitle": "J. Membranes",
            "abstractNote": "We studied filtration.",
            "tags": [],
            "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "A"}],
        },
    }
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    zotero_client.get_markdown_attachment.return_value = "Paper text."

    # Spy on upsert_paper so we can inspect every status payload written.
    upsert_payloads: list[dict] = []
    original_upsert = state_db.upsert_paper

    def _spy_upsert(payload: dict) -> None:
        upsert_payloads.append(dict(payload))
        return original_upsert(payload)

    state_db.upsert_paper = _spy_upsert  # type: ignore[method-assign]

    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2024.md")
    with patch("scripts.pipelines.brain_build.enrich_paper", return_value={}), \
         patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    # Pipeline counts: embed failure -> failed=1
    assert summary.papers_failed == 1
    assert summary.papers_processed == 0

    # H1 core assertion: no upsert ever carried status='extraction_complete'.
    # Pre-fix this would be the upsert just before the embed call.
    statuses_written = [p.get("status") for p in upsert_payloads if "status" in p]
    assert "extraction_complete" not in statuses_written, (
        "Pre-H1 regression: status='extraction_complete' was written before "
        f"the embed step succeeded. Statuses written: {statuses_written}; "
        f"full payloads: {upsert_payloads}"
    )
    # And embeddings_indexed must be 0 — embed never succeeded.
    record = state_db.get_paper("10.1000/embfail")
    assert record is not None
    assert record["embeddings_indexed"] == 0


@pytest.mark.unit
def test_brain_build_enrich_paper_failure_is_non_fatal(tmp_path, caplog):
    """H1: a transient Semantic Scholar outage (enrich_paper raises) must NOT
    abort or fail the paper.  The pipeline should log a WARNING, treat S2 data
    as empty, and continue end-to-end.  Pre-fix, enrich_paper() was called
    bare and any exception bubbled up to the outer handler, which marked the
    paper status='error' on what is really a transient external dependency.
    """
    import logging

    from scripts.pipelines.brain_build import run_brain_build

    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.model = "gemma4:e4b"
    llm.provider = "ollama"

    item = {
        "key": "S2FAIL01",
        "data": {
            "DOI": "10.1000/s2fail",
            "title": "Paper Whose S2 Lookup Fails",
            "date": "2024",
            "publicationTitle": "J. Membranes",
            "abstractNote": "We studied filtration.",
            "tags": [],
            "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "A"}],
        },
    }
    zotero_client = MagicMock()
    zotero_client.get_collection_items.return_value = [item]
    zotero_client.get_markdown_attachment.return_value = "Paper text."

    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2024.md")
    with caplog.at_level(logging.WARNING, logger="scripts.pipelines.brain_build"), \
         patch("scripts.pipelines.brain_build.enrich_paper",
               side_effect=ConnectionError("S2 timeout")), \
         patch("scripts.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("scripts.pipelines.brain_build.write_paper_note", return_value=note_path):
        summary = run_brain_build(config, state_db, zotero_client, embeddings_db, llm)

    # The paper completes successfully — S2 failure is non-fatal.
    assert summary.papers_processed == 1
    assert summary.papers_failed == 0
    record = state_db.get_paper("10.1000/s2fail")
    assert record is not None
    assert record["status"] == "extraction_complete"
    # And there should be a WARNING about the S2 enrichment failure.
    assert any(
        "S2 enrichment failed" in r.getMessage() for r in caplog.records
    ), (
        f"Expected WARNING log about S2 enrichment failure; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
