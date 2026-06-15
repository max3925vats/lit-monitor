"""
Unit tests for the discovery pipeline, ranker, and obsidian tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lit_monitor.pipelines.discovery import run_discovery
from lit_monitor.search.window import SearchWindow


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
    from lit_monitor.core.state_db import StateDB
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
    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=papers):
        with patch("lit_monitor.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("lit_monitor.pipelines.discovery.filter_known_dois",
                       return_value=papers):
                with patch("lit_monitor.pipelines.discovery.rank_papers",
                           return_value=papers):
                    from lit_monitor.pipelines.discovery import run_discovery
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
    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=papers):
        with patch("lit_monitor.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("lit_monitor.pipelines.discovery.filter_known_dois",
                       return_value=papers):
                with patch("lit_monitor.pipelines.discovery.rank_papers",
                           return_value=papers):
                    from lit_monitor.pipelines.discovery import run_discovery
                    summary = run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=False
                    )
    digest_dir = Path(config.obsidian.vault_path) / config.obsidian.digests_folder
    digests = list(digest_dir.glob("Discovery_*.md"))
    assert len(digests) == 1
    assert summary.digest_path != ""


# ===========================================================================
# Bundle J2: discovery forwards the interest vector into rank_papers
# ===========================================================================

def _config_with_feedback_weight(tmp_path: Path, weight: float) -> SimpleNamespace:
    """Base discovery config + a ranking.weights.feedback weight (J2 opt-in)."""
    cfg = _make_config(tmp_path)
    cfg.ranking = SimpleNamespace(weights=SimpleNamespace(feedback=weight))
    return cfg


@pytest.mark.unit
def test_discovery_forwards_interest_vector_to_ranker(tmp_path):
    """J2: when ranking.weights.feedback>0 and a global interest vector exists,
    run_discovery loads it and forwards interest_vec / n_events / weight into
    rank_papers."""
    import numpy as np

    config = _config_with_feedback_weight(tmp_path, weight=0.3)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    papers = [_make_paper("10.1/new1")]

    fake_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Patch the StateDB method so no real interest_vectors row is needed.
    state_db.get_interest_vector = MagicMock(return_value=(fake_vec, 17))

    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=papers) as mock_rank:
        run_discovery(
            config, state_db, zotero_client, embeddings_db, llm, dry_run=True
        )

    state_db.get_interest_vector.assert_called_once_with("global")
    _, kwargs = mock_rank.call_args
    assert kwargs["interest_vec"] is fake_vec
    assert kwargs["interest_vec_n_events"] == 17
    assert kwargs["interest_weight"] == 0.3


@pytest.mark.unit
def test_discovery_no_interest_vector_passes_none(tmp_path):
    """J2: absent interest vector (weight on, but nothing stored) → rank_papers
    receives interest_vec=None and the run does not crash."""
    config = _config_with_feedback_weight(tmp_path, weight=0.3)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    papers = [_make_paper("10.1/new1")]

    # No row stored → get_interest_vector returns None (real DB, empty table).
    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=papers) as mock_rank:
        run_discovery(
            config, state_db, zotero_client, embeddings_db, llm, dry_run=True
        )

    _, kwargs = mock_rank.call_args
    assert kwargs["interest_vec"] is None
    assert kwargs["interest_vec_n_events"] == 0
    # weight is still forwarded (harmless: rank_papers no-ops when vec is None).
    assert kwargs["interest_weight"] == 0.3


@pytest.mark.unit
def test_discovery_feedback_weight_zero_skips_db_read(tmp_path):
    """J2: ranking.weights.feedback == 0 (default) → no interest DB read, and
    rank_papers receives interest_vec=None (fully inert)."""
    config = _config_with_feedback_weight(tmp_path, weight=0.0)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    papers = [_make_paper("10.1/new1")]

    state_db.get_interest_vector = MagicMock(return_value=None)

    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=papers) as mock_rank:
        run_discovery(
            config, state_db, zotero_client, embeddings_db, llm, dry_run=True
        )

    # Weight off → we must NOT even read the interest vector from the DB.
    state_db.get_interest_vector.assert_not_called()
    _, kwargs = mock_rank.call_args
    assert kwargs["interest_vec"] is None
    assert kwargs["interest_weight"] == 0.0


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
    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=raw):
        with patch("lit_monitor.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("lit_monitor.pipelines.discovery.rank_papers",
                       return_value=[_make_paper("10.1/new")]):
                from lit_monitor.pipelines.discovery import run_discovery
                from lit_monitor.search.search_runner import filter_known_dois
                with patch("lit_monitor.pipelines.discovery.filter_known_dois",
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
    with patch("lit_monitor.pipelines.discovery.run_searches",
               side_effect=RuntimeError("API down")):
        with patch("lit_monitor.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("lit_monitor.pipelines.discovery.filter_known_dois",
                       return_value=[]):
                with patch("lit_monitor.pipelines.discovery.rank_papers",
                           return_value=[]):
                    from lit_monitor.pipelines.discovery import run_discovery
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
    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=[]):
        with patch("lit_monitor.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("lit_monitor.pipelines.discovery.filter_known_dois",
                       return_value=[]):
                with patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]):
                    with patch("lit_monitor.pipelines.discovery.enrich_paper",
                               return_value={}):
                        with patch("lit_monitor.pipelines.discovery.extract_paper",
                                   return_value={"core_finding": "ok",
                                                 "core_finding_confidence": "explicit"}):
                            with patch("lit_monitor.pipelines.discovery.write_paper_note",
                                       return_value=note_path):
                                with patch("lit_monitor.pipelines.discovery.relink_note"):
                                    from lit_monitor.pipelines.discovery import run_discovery
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
    from lit_monitor.pipelines.brain_build import _check_item_quality
    ok, reason = _check_item_quality({"title": "", "creators": [], "abstractNote": "Abstract."})
    assert not ok
    assert "title" in reason

@pytest.mark.unit
def test_item_quality_filter_passes_valid_item(tmp_path):
    """A2: _check_item_quality returns (True, '') for a well-formed item."""
    from lit_monitor.pipelines.brain_build import _check_item_quality
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
    from lit_monitor.core.state_db import StateDB
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
    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=[]):
        with patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]):
            with patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]):
                with patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]):
                    from lit_monitor.pipelines.discovery import run_discovery
                    summary = run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=False,
                    )
    # Item should be skipped via resume check (already complete in brain_build_progress)
    assert summary.papers_ingested == 0
    zotero_client.get_markdown_attachment.assert_not_called()

@pytest.mark.unit
def test_default_window_seeds_frontier_from_last_run_date(tmp_path):
    """Task 6: on first upgrade (last_run_date present, coverage_until absent) the
    DEFAULT window starts at the frontier seeded from last_run_date — NOT the old
    today-since_days computation. seed_coverage_until migrates last_run_date →
    coverage_until, so the window's `since` is exactly the stored last_run_date."""
    import datetime as dt
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    # First-upgrade state: a stored last_run_date, no coverage_until yet.
    past_date = (dt.date.today() - dt.timedelta(days=30))
    state_db.set_kv("last_run_date", past_date.isoformat())
    state_db.set_kv("zotero_library_version", "100")

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    zotero_client.get_current_version.return_value = 101
    zotero_client.get_items_since.return_value = []

    captured_windows: list[SearchWindow] = []

    def capture_run_searches(cfg, window=None, **kwargs):
        captured_windows.append(window)
        return []

    with patch("lit_monitor.pipelines.discovery.run_searches",
               side_effect=capture_run_searches):
        with patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]):
            with patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]):
                with patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]):
                    from lit_monitor.pipelines.discovery import run_discovery
                    run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=False,
                    )

    assert len(captured_windows) == 1
    # Frontier seeded from last_run_date → since == the stored last_run_date.
    assert captured_windows[0].since == past_date
    assert captured_windows[0].until is None

# ===========================================================================

# Ranker tests
# ===========================================================================
@pytest.mark.unit
def test_ranker_sorts_by_similarity():
    """Papers returned sorted by similarity_score descending."""
    from lit_monitor.llm.ranker import rank_papers
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
    from lit_monitor.llm.ranker import rank_papers
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
    from lit_monitor.llm.ranker import rank_papers
    embeddings_db = MagicMock()
    llm = MagicMock()
    result = rank_papers([], embeddings_db, llm)
    assert result == []
    embeddings_db.find_similar_to_text.assert_not_called()
    llm.complete.assert_not_called()
@pytest.mark.unit
def test_ranker_llm_failure_returns_empty_rationale():
    """LLM failure does not crash ranker — rationale fields left empty."""
    from lit_monitor.llm.ranker import rank_papers
    candidates = [{"doi": "10.1/p1", "title": "Paper", "abstract": ""}]

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = [{"score": 0.5, "id": "x", "document": "",
    "metadata": {}}]
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("Ollama down")
    ranked = rank_papers(candidates, embeddings_db, llm)
    assert ranked[0]["llm_rationale"] == ""


@pytest.mark.unit
def test_get_rationales_raises_under_strict():
    """P4.2: rationale-LLM failure must raise in --strict, not return empty."""
    import lit_monitor.core.strict_mode as _sm
    from lit_monitor.llm.ranker import _get_rationales

    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("Ollama down")
    _sm.set_strict(True)
    with pytest.raises(RuntimeError, match="rationale generation failed"):
        _get_rationales([{"doi": "10.1/p1", "title": "P", "abstract": ""}], llm, "")


@pytest.mark.unit
def test_get_rationales_non_strict_logs_and_returns_empty(caplog):
    """P4.2: non-strict logs at WARNING and returns {} (empty rationales)."""
    import logging

    import lit_monitor.core.strict_mode as _sm
    from lit_monitor.llm.ranker import _get_rationales

    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("Ollama down")
    _sm.set_strict(False)
    with caplog.at_level(logging.WARNING, logger="lit_monitor.llm.ranker"):
        result = _get_rationales([{"doi": "10.1/p1", "title": "P", "abstract": ""}], llm, "")
    assert result == {}
    assert any("rationale generation failed" in r.message for r in caplog.records)


@pytest.mark.unit
def test_ranker_propagates_chromadb_connection_error():
    """L3: a chromadb backend error must surface, not silently default to 0.0.

    Previously every exception was swallowed and the paper was scored 0.0 —
    meaning a broken Chroma collection looked identical to "no neighbours".
    """
    from chromadb.errors import ChromaError

    from lit_monitor.llm.ranker import rank_papers

    candidates = [{"doi": "10.1/p1", "title": "Paper", "abstract": ""}]
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.side_effect = ChromaError("backend down")
    llm = MagicMock()
    with pytest.raises(ChromaError):
        rank_papers(candidates, embeddings_db, llm)


@pytest.mark.unit
def test_ranker_non_chroma_exception_logged_and_continues(caplog):
    """L3: per-paper non-Chroma exceptions log WARNING with traceback, score=0.0."""
    import logging

    from lit_monitor.llm.ranker import rank_papers

    candidates = [{"doi": "10.1/p1", "title": "Paper", "abstract": ""}]
    embeddings_db = MagicMock()
    # Generic ValueError — represents a malformed embed input, not a backend
    # outage. Should be tolerated.
    embeddings_db.find_similar_to_text.side_effect = ValueError("bad embed text")
    llm = MagicMock()
    llm.complete.return_value = json.dumps({"10.1/p1": "ok"})

    with caplog.at_level(logging.WARNING, logger="lit_monitor.llm.ranker"):
        ranked = rank_papers(candidates, embeddings_db, llm)

    assert ranked[0]["similarity_score"] == 0.0
    assert any(
        "Similarity lookup failed" in rec.getMessage() and rec.levelno == logging.WARNING
        for rec in caplog.records
    )
# ===========================================================================
# Obsidian tools tests
# ===========================================================================
@pytest.mark.unit
def test_retheme_rewrites_wikilinks(tmp_path):
    """retheme() rewrites [[OldTheme]] → [[NewTheme]] in vault notes."""
    from lit_monitor.obsidian_tools.retheme import retheme
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
    from lit_monitor.obsidian_tools.retheme import retheme
    vault = tmp_path / "vault"
    vault.mkdir()
    old_page = vault / "OldTheme.md"
    old_page.write_text("# OldTheme\n", encoding="utf-8")
    stats = retheme(vault, "OldTheme", "NewTheme")
    assert not old_page.exists()
    assert (vault / "NewTheme.md").exists()
    assert stats["page_renamed"] == 1
@pytest.mark.unit
def test_retheme_survives_replace_failure_per_file(tmp_path, monkeypatch):
    """B1 crash-safety: retheme writes each note atomically. If os.replace fails
    on the SECOND file, the first-written file must already be a complete, valid
    rewrite (atomic per-file) and no .tmp sibling may leak.

    Why this catches a non-atomic regression: os.replace is patched to raise on
    its second call (the final rename step of the second atomic_write_text).
    atomic_write_text's cleanup unlinks its temp file before re-raising, and
    retheme's per-file try/except logs+swallows the OSError so the loop survives.
    With a plain md_file.write_text(...) there is no temp file at all and a crash
    mid-write would leave a half-written (truncated) note on disk — this test
    would then find a leaked temp file and/or a note whose content is neither the
    clean original nor the clean rewrite.

    NOTE: retheme is intentionally non-transactional ACROSS files — after this
    crash the first note IS rewritten while the second is not. That cross-file
    inconsistency is a documented, out-of-scope limitation; B1 only guarantees
    per-file atomicity (no individual note truncated).
    """
    from lit_monitor.core import atomic_write as atomic_write_mod
    from lit_monitor.obsidian_tools.retheme import retheme

    vault = tmp_path / "vault"
    vault.mkdir()
    note_a = vault / "a.md"
    note_b = vault / "b.md"
    note_a.write_text("Link to [[OldTheme]] in note A.\n", encoding="utf-8")
    note_b.write_text("Link to [[OldTheme]] in note B.\n", encoding="utf-8")

    real_replace = atomic_write_mod.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated rename crash on second file")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic_write_mod.os, "replace", flaky_replace)

    # retheme swallows the per-file OSError (logs a warning) and finishes the
    # loop, so it returns normally rather than propagating.
    retheme(vault, "OldTheme", "NewTheme")

    # Every note on disk is either the clean original or the clean rewrite —
    # never a truncated/partial file. Exactly one was rewritten (the first),
    # the other still holds its original content (cross-file non-transactional).
    valid = {
        "Link to [[OldTheme]] in note A.\n",
        "Link to [[OldTheme]] in note B.\n",
        "Link to [[NewTheme]] in note A.\n",
        "Link to [[NewTheme]] in note B.\n",
    }
    text_a = note_a.read_text(encoding="utf-8")
    text_b = note_b.read_text(encoding="utf-8")
    assert text_a in valid, f"note A is truncated/corrupt: {text_a!r}"
    assert text_b in valid, f"note B is truncated/corrupt: {text_b!r}"
    rewritten = [t for t in (text_a, text_b) if "NewTheme" in t]
    untouched = [t for t in (text_a, text_b) if "OldTheme" in t]
    assert len(rewritten) == 1, "exactly one note should have been atomically rewritten"
    assert len(untouched) == 1, "the second note should remain untouched (cross-file)"
    # No temp-file droppings left behind anywhere in the vault.
    leftovers = [
        p for p in vault.rglob("*")
        if p.name.startswith(".") and p.name.endswith(".tmp")
    ]
    assert leftovers == [], f"leaked temp files: {leftovers}"
@pytest.mark.unit
def test_retheme_per_file_failure_raises_under_strict(tmp_path, monkeypatch):
    """P4.4: in --strict, a per-file retheme failure escalates to a raise."""
    import lit_monitor.core.strict_mode as _sm
    from lit_monitor.obsidian_tools.retheme import retheme

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("Link to [[OldTheme]] here.\n", encoding="utf-8")

    _sm.set_strict(True)
    with patch(
        "lit_monitor.obsidian_tools.retheme.atomic_write_text",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(Exception, match="Could not process"):
            retheme(vault, "OldTheme", "NewTheme")
@pytest.mark.unit
def test_rerender_paper_note(tmp_path):
    """rerender_note() calls write_paper_note with extraction from state DB."""
    from lit_monitor.obsidian_tools.rerender import rerender_note
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
    with patch("lit_monitor.obsidian_tools.rerender.write_paper_note",
               return_value=note_path) as mock_write:
        result = rerender_note("10.1/test", config, state_db)
    mock_write.assert_called_once()
    assert result == note_path
@pytest.mark.unit
def test_rerender_raises_for_missing_doi(tmp_path):
    """rerender_note() raises ValueError for unknown doi."""
    from lit_monitor.obsidian_tools.rerender import rerender_note
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    with pytest.raises(ValueError, match="No record found"):
        rerender_note("10.1/nonexistent", config, state_db)
@pytest.mark.unit
def test_re_extract_updates_state_db(tmp_path):
    """re_extract() updates extraction_json in state DB."""
    from lit_monitor.obsidian_tools.re_extract import re_extract
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
    with patch("lit_monitor.obsidian_tools.re_extract.extract_paper",
               return_value=new_extraction):
        with patch("lit_monitor.obsidian_tools.re_extract.rerender_note"):
            re_extract("10.1/test", config, state_db, llm, phases=["simple"])
    updated = state_db.get_extraction_json("10.1/test")
    assert updated["core_finding"] == "Updated result."
@pytest.mark.unit
def test_synthesize_writes_note(tmp_path):
    """synthesize() writes a synthesis note to the connections folder."""

    from lit_monitor.obsidian_tools.synthesize import synthesize
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
def test_synthesize_empty_results_returns_none(tmp_path):
    """Q3.8: synthesize() returns None (the no-result sentinel) when no similar
    notes are found — NOT an empty string. Callers branch on truthiness, so the
    type contract is str | None, and None is the correct 'nothing written'."""
    from lit_monitor.obsidian_tools.synthesize import synthesize
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
    assert result is None
    assert result != ""  # explicit: the empty-string sentinel is gone
    llm.complete.assert_not_called()
@pytest.mark.unit
def test_synthesize_survives_replace_failure(tmp_path, monkeypatch):
    """B1 crash-safety: if os.replace fails mid-write, a pre-existing synthesis
    note must be preserved verbatim and no .tmp sibling may leak.

    Why this catches a non-atomic regression: os.replace is patched to raise at
    the final rename step of atomic_write_text. A plain note_path.write_text(...)
    has no temp file and would truncate/rewrite the destination in place before
    any replace ran, so the 'original intact' assertion would fail.
    """
    from lit_monitor.core import atomic_write as atomic_write_mod
    from lit_monitor.obsidian_tools.synthesize import _slugify, synthesize

    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = "Freshly generated synthesis that must NOT land."
    extraction = {"core_finding": "Key finding.", "core_finding_confidence": "explicit"}
    state_db.upsert_paper({
        "doi": "10.1/related",
        "title": "Relevant Paper",
        "source_type": "paper",
        "note_title": "Smith2021_Relevant",
        "extraction_json": json.dumps(extraction),
        "status": "extraction_complete",
    })
    embeddings_db.find_similar_chunks.return_value = []
    embeddings_db.find_similar_to_text.return_value = [
        {"id": "10.1/related", "score": 0.88, "document": "Key finding.", "metadata": {}}
    ]

    # Seed an existing synthesis note whose content must survive the crash.
    topic = "TopicX transport mechanism"
    connections = Path(config.obsidian.vault_path) / config.obsidian.connections_folder
    existing_note = connections / f"Synthesis_{_slugify(topic)}.md"
    original = "# Synthesis: TopicX\nuser-authored content that must survive crashes\n"
    existing_note.write_text(original, encoding="utf-8")

    def boom(_src, _dst):  # noqa: ANN001 — match os.replace signature
        raise OSError("simulated rename crash")

    monkeypatch.setattr(atomic_write_mod.os, "replace", boom)

    with pytest.raises(OSError, match="simulated rename crash"):
        synthesize(topic, config, state_db, embeddings_db, llm)

    # Original note content untouched.
    assert existing_note.read_text(encoding="utf-8") == original
    # No temp-file droppings left behind in the connections dir.
    leftovers = [
        p for p in connections.iterdir()
        if p.name.startswith(".") and p.name.endswith(".tmp")
    ]
    assert leftovers == [], f"leaked temp files: {leftovers}"
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

    from lit_monitor.llm.extractor import extract_paper
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
    from lit_monitor.pipelines.brain_build import run_brain_build

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

    with patch("lit_monitor.pipelines.brain_build._process_paper",
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

    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.enrich_paper",
               return_value=enrich_return or {}), \
         patch("lit_monitor.pipelines.discovery.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("lit_monitor.pipelines.discovery.write_paper_note", return_value=note_path), \
         patch("lit_monitor.pipelines.discovery.relink_note"):
        from lit_monitor.pipelines.discovery import run_discovery
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
# Pipeline Run Summary section removed from the discovery digest
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_discovery_digest_has_no_pipeline_run_summary(tmp_path):
    """The digest must no longer contain a '## Pipeline Run Summary' section."""
    from lit_monitor.pipelines.discovery import _write_digest
    config = _make_config(tmp_path)
    digest_path = _write_digest(
        ranked=[_make_paper("10.1/new", score=0.9)],
        config=config,
        sim_threshold=0.5,
        dry_run=False,
    )
    content = Path(digest_path).read_text(encoding="utf-8")
    assert "## Pipeline Run Summary" not in content


@pytest.mark.unit
def test_discovery_digest_does_not_fetch_recent_runs(tmp_path):
    """run_discovery() must not call get_recent_runs() for the digest, and the
    written digest must not contain a Pipeline Run Summary section."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    # Seed a run_log entry so get_recent_runs() WOULD return it if called.
    state_db.start_run("run-L2-test", "brain_build")
    state_db.finish_run("run-L2-test", status="complete", processed=7, skipped=0, failed=0)

    # Tripwire: the digest path must never reach get_recent_runs().
    state_db.get_recent_runs = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("get_recent_runs must not be called for the digest")
    )

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    papers = [_make_paper("10.1/L2paper", score=0.8)]

    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=papers):
        with patch("lit_monitor.pipelines.discovery.run_researcher_searches",
                   return_value=[]):
            with patch("lit_monitor.pipelines.discovery.filter_known_dois",
                       return_value=papers):
                with patch("lit_monitor.pipelines.discovery.rank_papers",
                           return_value=papers):
                    from lit_monitor.pipelines.discovery import run_discovery
                    run_discovery(
                        config, state_db, zotero_client, embeddings_db, llm,
                        dry_run=False,
                    )

    digest_dir = Path(config.obsidian.vault_path) / config.obsidian.digests_folder
    digests = list(digest_dir.glob("Discovery_*.md"))
    assert digests, "Expected a discovery digest to be written"
    content = digests[0].read_text(encoding="utf-8")
    assert "## Pipeline Run Summary" not in content


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

    from lit_monitor.cli import main

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
    with patch("lit_monitor.cli._make_config", return_value=config), \
         patch("lit_monitor.cli._make_state_db", return_value=state_db), \
         patch("lit_monitor.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("lit_monitor.cli._load_secrets", return_value={}), \
         patch("lit_monitor.cli._make_zotero_client", return_value=MagicMock()), \
         patch("lit_monitor.cli._make_llm", return_value=MagicMock()), \
         patch("lit_monitor.obsidian_tools.re_extract.re_extract", return_value={}), \
         patch("lit_monitor.search.citation_graph.build_citation_graph", fake_build_citation_graph), \
         patch("lit_monitor.obsidian_tools.relink.relink_note"):
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

    from lit_monitor.cli import main

    runner = CliRunner()
    with patch("lit_monitor.cli._make_config", return_value=_make_config(tmp_path)), \
         patch("lit_monitor.cli._make_state_db", return_value=_make_state_db(tmp_path)), \
         patch("lit_monitor.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("lit_monitor.cli._load_secrets", return_value={}), \
         patch("lit_monitor.cli._make_zotero_client", return_value=MagicMock()):
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

    from lit_monitor.cli import main

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
    with patch("lit_monitor.cli._make_config", return_value=_make_config(tmp_path)), \
         patch("lit_monitor.cli._make_state_db", return_value=state_db), \
         patch("lit_monitor.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("lit_monitor.cli._load_secrets", return_value={}), \
         patch("lit_monitor.cli._make_zotero_client", return_value=MagicMock()), \
         patch("lit_monitor.search.citation_graph.build_citation_graph", fake_build), \
         patch("lit_monitor.obsidian_tools.relink.relink_note"):
        result = runner.invoke(main, ["obsidian", "rebuild-citations", "--scope", "all"])

    assert result.exit_code == 0, result.output
    assert graph_calls == ["10.1/with-cites"]
    assert "1 papers" in result.output


@pytest.mark.unit
def test_rebuild_citations_scope_failed_skips_already_resolved(tmp_path):
    """--scope failed skips papers that already have citation_edges rows."""
    from click.testing import CliRunner

    from lit_monitor.cli import main

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
    with patch("lit_monitor.cli._make_config", return_value=_make_config(tmp_path)), \
         patch("lit_monitor.cli._make_state_db", return_value=state_db), \
         patch("lit_monitor.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("lit_monitor.cli._load_secrets", return_value={}), \
         patch("lit_monitor.cli._make_zotero_client", return_value=MagicMock()), \
         patch("lit_monitor.search.citation_graph.build_citation_graph", fake_build), \
         patch("lit_monitor.obsidian_tools.relink.relink_note"):
        result = runner.invoke(main, ["obsidian", "rebuild-citations", "--scope", "failed"])

    assert result.exit_code == 0, result.output
    # Only the unresolved paper should go through build_citation_graph
    assert graph_calls == ["10.1/needs-resolve"]


@pytest.mark.unit
def test_rebuild_citations_no_rerender_skips_relink(tmp_path):
    """--no-rerender prevents relink_note from being called."""
    from click.testing import CliRunner

    from lit_monitor.cli import main

    state_db = _make_state_db(tmp_path)
    note_path = str(tmp_path / "Test2024.md")
    Path(note_path).write_text("# Note", encoding="utf-8")
    _seed_paper_with_citations(state_db, "10.1/norerender", ["Ref1"], note_path)

    def fake_build(doi, state_db, api_key=None, max_retries=4):
        return _fake_citation_result()

    relink_calls: list = []

    runner = CliRunner()
    with patch("lit_monitor.cli._make_config", return_value=_make_config(tmp_path)), \
         patch("lit_monitor.cli._make_state_db", return_value=state_db), \
         patch("lit_monitor.cli._make_embeddings_db", return_value=MagicMock()), \
         patch("lit_monitor.cli._load_secrets", return_value={}), \
         patch("lit_monitor.cli._make_zotero_client", return_value=MagicMock()), \
         patch("lit_monitor.cli._make_llm", return_value=MagicMock()), \
         patch("lit_monitor.obsidian_tools.re_extract.re_extract", return_value={}), \
         patch("lit_monitor.search.citation_graph.build_citation_graph", fake_build), \
         patch("lit_monitor.obsidian_tools.relink.relink_note",
               side_effect=lambda *a, **kw: relink_calls.append(a)):
        result = runner.invoke(
            main,
            ["obsidian", "rebuild-citations", "--doi", "10.1/norerender",
             "--scope", "doi", "--no-rerender"],
        )

    assert result.exit_code == 0, result.output
    assert relink_calls == [], "relink_note must not be called when --no-rerender is set"


@pytest.mark.unit
def test_build_citation_graph_uses_state_db_path_not_paths(tmp_path):
    """P1.1: `obsidian build-citation-graph` must build StateDB from
    ``config.state_db.path``, not the non-existent ``config.paths.state_db``.

    Pre-fix the handler did ``StateDB(config.paths.state_db)`` which raised
    AttributeError because Config has no ``paths`` attribute. The handler
    constructs ``Config()`` directly, so we patch the source module's Config
    to expose only ``config.state_db.path`` (and explicitly NO ``paths``).
    """
    from click.testing import CliRunner

    from lit_monitor.cli import main

    state_db = _make_state_db(tmp_path)
    captured: dict = {}

    # Config stub exposing state_db.path; accessing .paths raises, matching
    # the real Config (which has no such attribute).
    class _CfgStub:
        def __init__(self) -> None:
            self.state_db = SimpleNamespace(path=str(tmp_path / "state.db"))

        @property
        def paths(self):  # pragma: no cover - asserts the buggy path is dead
            raise AssertionError("config.paths must not be accessed")

    def fake_state_db_ctor(path):
        captured["path"] = path
        return state_db

    runner = CliRunner()
    with patch("lit_monitor.core.config.Config", _CfgStub), \
         patch("lit_monitor.core.state_db.StateDB", fake_state_db_ctor), \
         patch("lit_monitor.cli._load_secrets", return_value={}), \
         patch("lit_monitor.cli._maybe_set_s2_key"):
        result = runner.invoke(
            main,
            ["obsidian", "build-citation-graph", "--scope", "all"],
        )

    # No AttributeError; StateDB built from config.state_db.path.
    assert "AttributeError" not in result.output, result.output
    assert captured.get("path") == str(tmp_path / "state.db")


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
    from lit_monitor.pipelines.brain_build import run_brain_build

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
    with patch("lit_monitor.pipelines.brain_build.enrich_paper", return_value={}), \
         patch("lit_monitor.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("lit_monitor.pipelines.brain_build.write_paper_note", return_value=note_path):
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

    from lit_monitor.pipelines.brain_build import run_brain_build

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
    with caplog.at_level(logging.WARNING, logger="lit_monitor.pipelines.brain_build"), \
         patch("lit_monitor.pipelines.brain_build.enrich_paper",
               side_effect=ConnectionError("S2 timeout")), \
         patch("lit_monitor.pipelines.brain_build.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("lit_monitor.pipelines.brain_build.write_paper_note", return_value=note_path):
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


@pytest.mark.unit
def test_discovery_enrich_paper_failure_is_non_fatal(tmp_path, caplog):
    """H1 (review fix): the discovery ingestion path must also treat a transient
    S2 outage as non-fatal — same contract as brain_build.  Pre-fix, discovery
    called enrich_paper bare and any exception bubbled up to the outer except,
    marking the paper status='error' on what is really an optional enrichment.
    """
    import logging

    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    state_db.set_kv("zotero_library_version", "100")
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = json.dumps({"core_finding": "ok", "core_finding_confidence": "explicit"})
    llm.model = "gemma4:e4b"
    llm.provider = "ollama"

    item = _make_zotero_item_for_ingestion("10.1000/s2fail", key="DISCS2FAIL01")
    zotero_client = MagicMock()
    zotero_client.get_items_since.return_value = [item]
    zotero_client.get_current_version.return_value = 101
    zotero_client.get_markdown_attachment.return_value = "Paper text."

    note_path = str(tmp_path / "vault" / "Literature" / "Papers" / "Smith2021.md")
    with caplog.at_level(logging.WARNING, logger="lit_monitor.pipelines.discovery"), \
         patch("lit_monitor.pipelines.discovery.run_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.enrich_paper",
               side_effect=ConnectionError("S2 timeout")), \
         patch("lit_monitor.pipelines.discovery.extract_paper",
               return_value={"core_finding": "ok", "core_finding_confidence": "explicit"}), \
         patch("lit_monitor.pipelines.discovery.write_paper_note", return_value=note_path), \
         patch("lit_monitor.pipelines.discovery.relink_note"):
        from lit_monitor.pipelines.discovery import run_discovery
        summary = run_discovery(
            config, state_db, zotero_client, embeddings_db, llm, dry_run=False
        )

    # The paper completes successfully — S2 failure is non-fatal.
    assert summary.papers_ingested == 1
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

# ---------------------------------------------------------------------------
# M1 — discovery rate-limit abort after 3 consecutive 429s
# (Mirrors test_brain_build.py::test_rate_limit_abort_after_three_consecutive_429s)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_discovery_aborts_after_three_consecutive_rate_limits(tmp_path):
    """
    M1 / P5.5: when 3 consecutive RateLimitErrors are raised during discovery
    ingestion, run_discovery must call state_db.finish_run with
    status='rate_limited' and raise the catchable domain exception
    RateLimitExhausted (NOT SystemExit — that was uncatchable in API/web
    contexts). Mirrors the P4.5 contract in brain_build; the discovery CLI
    boundary maps the exception to exit code 2 so cron still sees a non-zero
    exit instead of a silent success.
    """
    from lit_monitor.llm.llm_client import RateLimitError
    from lit_monitor.pipelines.brain_build import RateLimitExhausted

    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = "{}"

    # Pre-seed baseline Zotero version so the ingestion loop runs (skipping
    # the first-run baseline-only branch in _run_ingestion).
    state_db.set_kv("zotero_library_version", "100")

    zotero_client = MagicMock()
    zotero_client.get_current_version.return_value = 101
    new_items = [
        {
            "key": f"RLKEY{i}",
            "data": {
                "DOI": f"10.1000/rl{i}",
                "title": f"Rate-Limited Paper {i}",
                "date": "2024",
                "publicationTitle": "J Sci",
                "abstractNote": "Abstract.",
                "tags": [],
                "creators": [
                    {"creatorType": "author", "lastName": "Smith", "firstName": "J"}
                ],
            },
        }
        for i in range(1, 4)
    ]
    zotero_client.get_items_since.return_value = new_items
    zotero_client.get_markdown_attachment.return_value = "Full paper text."

    # Wrap finish_run so we can assert it was called with status='rate_limited'
    # while still letting the real DB UPDATE happen for downstream inspection.
    real_finish_run = state_db.finish_run
    finish_run_spy = MagicMock(side_effect=real_finish_run)
    state_db.finish_run = finish_run_spy  # type: ignore[method-assign]

    # extract_paper raises RateLimitError on every call — simulates the LLM
    # being throttled across 3 consecutive papers.
    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.enrich_paper", return_value={}), \
         patch("lit_monitor.pipelines.discovery.extract_paper",
               side_effect=RateLimitError("429 Too Many Requests")), \
         patch("lit_monitor.pipelines.discovery._time.sleep"):  # skip actual back-off
        with pytest.raises(RateLimitExhausted):
            run_discovery(
                config, state_db, zotero_client, embeddings_db, llm,
                dry_run=False,
            )

    # finish_run must have been called with status='rate_limited' (not 'complete')
    rate_limited_calls = [
        call for call in finish_run_spy.call_args_list
        if call.kwargs.get("status") == "rate_limited"
    ]
    assert rate_limited_calls, (
        f"Expected finish_run(status='rate_limited'); got calls: "
        f"{finish_run_spy.call_args_list}"
    )

    # And the raised RateLimitExhausted must have short-circuited the trailing
    # finish_run(status='complete') call in run_discovery.
    complete_calls = [
        call for call in finish_run_spy.call_args_list
        if call.kwargs.get("status") == "complete"
    ]
    assert not complete_calls, (
        f"Expected no finish_run(status='complete') after rate-limit abort; "
        f"got: {complete_calls}"
    )

    # Run log should record status='rate_limited' (mirrors brain_build test).
    import sqlite3
    with sqlite3.connect(tmp_path / "state.db") as conn:
        rows = conn.execute("SELECT status FROM run_log").fetchall()
    statuses = [r[0] for r in rows]
    assert "rate_limited" in statuses, (
        f"Expected 'rate_limited' in run_log.status; got: {statuses}"
    )


# ===========================================================================
# G9 — rag-mode wiring: discovery + synthesize + branch helper + query expansion
# ===========================================================================

@pytest.mark.unit
def test_discovery_rag_mode_graph_uses_graph_ranking(tmp_path):
    """G9: run_discovery with rag_mode='graph' calls _rank_papers_graph, not rank_papers."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    papers = [_make_paper("10.1/g9a"), _make_paper("10.1/g9b")]

    rank_papers_calls: list = []
    graph_rank_calls: list = []

    def fake_rank_papers(candidates, emb, llm_, **kwargs):
        rank_papers_calls.append(candidates)
        return [dict(p, similarity_score=0.5) for p in candidates]

    def fake_rank_graph(candidates, *, embeddings_db, llm, top_k, rag_mode, **kwargs):
        # **kwargs absorbs Bundle D additions (graph_signals, graph_weights, etc.)
        # so this mock stays forward-compatible with rank_papers signature growth.
        graph_rank_calls.append(rag_mode)
        return [dict(p, similarity_score=0.3) for p in candidates]

    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.rank_papers", side_effect=fake_rank_papers), \
         patch("lit_monitor.pipelines.discovery._rank_papers_graph", side_effect=fake_rank_graph):
        from lit_monitor.pipelines.discovery import run_discovery
        summary = run_discovery(
            config, state_db, zotero_client, embeddings_db, llm,
            dry_run=True, rag_mode="graph",
        )

    assert not rank_papers_calls, "rank_papers should NOT be called in graph mode"
    assert graph_rank_calls == ["graph"], f"Expected graph ranking; got: {graph_rank_calls}"
    assert summary.new_papers_found == 2


@pytest.mark.unit
def test_discovery_rag_mode_vector_uses_rank_papers(tmp_path):
    """G9: run_discovery with rag_mode='vector' uses existing rank_papers (unchanged path)."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    papers = [_make_paper("10.1/v1")]

    rank_papers_calls: list = []

    def fake_rank_papers(candidates, emb, llm_, **kwargs):
        rank_papers_calls.append(True)
        return [dict(p, similarity_score=0.8) for p in candidates]

    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=papers), \
         patch("lit_monitor.pipelines.discovery.rank_papers", side_effect=fake_rank_papers):
        from lit_monitor.pipelines.discovery import run_discovery
        run_discovery(
            config, state_db, zotero_client, embeddings_db, llm,
            dry_run=True, rag_mode="vector",
        )

    assert rank_papers_calls, "rank_papers MUST be called for vector mode"


# G9 tests for retrieve_doi_candidates, _expand_query, and Config.retrieval
# are in tests/unit/test_retrieval_branch.py (avoids habanero import issue
# that blocks collection of this file when the full dependency set is absent).


# ===========================================================================
# G9 C2 — graph-mode discovery rationales must be populated
# ===========================================================================

@pytest.mark.unit
def test_rank_papers_graph_produces_rationales(tmp_path):
    """G9 C2: _rank_papers_graph LLM rationales must be populated.

    Pre-fix, _get_rationales(_top, llm) raised TypeError because the required
    positional arg ``domain_context: str`` was missing. The broad
    ``except Exception`` swallowed it, leaving every paper with
    ``llm_rationale=""``.
    """
    from lit_monitor.pipelines.discovery import _rank_papers_graph

    candidates = [
        {"doi": "10.1/a", "title": "Paper A", "abstract": "About A."},
        {"doi": "10.1/b", "title": "Paper B", "abstract": "About B."},
    ]

    llm = MagicMock()
    # Match the format parse_llm_json returns — a JSON dict of doi -> rationale.
    llm.complete.return_value = json.dumps({
        "10.1/a": "A is relevant because of X.",
        "10.1/b": "B is relevant because of Y.",
    })

    # Mock graph_db so it returns at least one neighbour per candidate (gives
    # a non-zero similarity_score so we exercise the rationale path on the
    # top-K which is the WHOLE candidate set here).
    graph_db = MagicMock()
    graph_db.find_similar_papers.return_value = [("10.1/other", 2.0)]
    graph_db.close = MagicMock()

    # safe_graph_db is called inside _rank_papers_graph — patch it to return
    # our mock so we don't depend on the [graph] extra.
    with patch("lit_monitor.pipelines.discovery.safe_graph_db", return_value=graph_db), \
         patch("lit_monitor.retrieval.branch.retrieve_doi_candidates",
               return_value=[("10.1/neigh", 0.8)]):
        ranked = _rank_papers_graph(
            candidates, embeddings_db=MagicMock(), llm=llm,
            top_k=10, rag_mode="graph",
        )

    # Every paper must have a NON-EMPTY rationale. Pre-fix, all would be "".
    rationales = [p.get("llm_rationale", "") for p in ranked]
    assert all(r for r in rationales), (
        "G9 C2 regression: _rank_papers_graph produced empty llm_rationale for "
        "at least one paper. Most likely _get_rationales is being called "
        f"without domain_context, raising TypeError. Got rationales: {rationales!r}"
    )
    assert any("relevant" in r for r in rationales), (
        f"Rationales should contain LLM output; got: {rationales!r}"
    )


# ===========================================================================
# P1: discovery pipeline writes discovery_runs + discovery_paper_results
# ===========================================================================


@pytest.mark.unit
def test_discovery_writes_run_row_and_paper_rows(tmp_path):
    """P1: run_discovery writes one discovery_runs row + >=1 discovery_paper_results row."""
    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    llm = MagicMock()
    llm.complete.return_value = "{}"
    zotero_client = MagicMock()
    zotero_client.get_current_version.return_value = 100

    paper = _make_paper("10.9/p1_test", score=0.9)
    paper["llm_rationale"] = "highly relevant"

    with patch("lit_monitor.pipelines.discovery.run_searches", return_value=[paper]), \
         patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
         patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[paper]), \
         patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[paper]), \
         patch("lit_monitor.pipelines.discovery._run_ingestion", return_value=None):
        run_discovery(config, state_db, zotero_client, embeddings_db, llm, dry_run=False)

    with state_db._connect() as conn:
        runs = conn.execute("SELECT id, status FROM discovery_runs").fetchall()
        papers = conn.execute("SELECT doi FROM discovery_paper_results").fetchall()

    assert len(runs) == 1, f"Expected 1 discovery_runs row, got {len(runs)}"
    assert runs[0][1] in ("success", "complete", "completed", "done"), (
        f"Unexpected run status: {runs[0][1]!r}"
    )
    assert len(papers) >= 1, f"Expected >=1 discovery_paper_results rows, got {len(papers)}"


# ---------------------------------------------------------------------------
# P10b: digest auto-write flag
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestDigestAutoWriteFlag:
    """P10b: digest .md auto-write gated on discovery.digest.auto_write."""

    def test_flag_default_true_in_example_yaml(self):
        """P10b: extraction.example.yaml ships with auto_write: true (backward compat)."""
        from importlib.resources import files

        import yaml
        example = files("lit_monitor._data.config_examples") / "extraction.example.yaml"
        raw = example.read_text()
        data = yaml.safe_load(raw)
        assert data["discovery"]["digest"]["auto_write"] is True

    def test_config_exposes_flag_via_namespace(self):
        """P10b: cfg.discovery.digest.auto_write is reachable via the _Namespace path.

        Skips when config can't load (CI has no paths.yaml). The attribute-path
        contract is also covered by the other tests in this class via the
        _resolve_digest_auto_write fallback, which defaults to True on missing
        attribute chains.

        The config cache is reset automatically by the ``_reset_config_cache``
        autouse fixture (tests/unit/conftest.py), so get_config() always returns
        a fresh Config (or raises FileNotFoundError → skip) — never a stale mock.
        """
        from lit_monitor.core import config as config_mod

        try:
            cfg = config_mod.get_config()
        except FileNotFoundError:
            pytest.skip("config not loadable (no paths.yaml in this environment)")
        # Either real value or default-True sentinel; None means downstream default of True
        digest_ns = getattr(getattr(cfg, "discovery", None), "digest", None)
        val = getattr(digest_ns, "auto_write", None) if digest_ns is not None else None
        assert val is True or val is None  # None means defaults to True downstream

    def test_digest_skipped_when_flag_false(self, tmp_path, caplog, monkeypatch):
        """P10b: when _resolve_digest_auto_write returns False, _write_digest is NOT called."""
        import logging

        config = _make_config(tmp_path)
        state_db = _make_state_db(tmp_path)

        # Monkeypatch the resolver to return False
        monkeypatch.setattr(
            "lit_monitor.pipelines.discovery._resolve_digest_auto_write",
            lambda cfg: False,
        )

        with patch("lit_monitor.pipelines.discovery._write_digest") as mock_write, \
             patch("lit_monitor.pipelines.discovery.run_searches", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]):
            with caplog.at_level(logging.INFO):
                run_discovery(config, state_db, MagicMock(), MagicMock(), MagicMock())

        mock_write.assert_not_called()
        # INFO log emitted for skipped digest
        assert any(
            "digest auto-write disabled" in r.message.lower()
            for r in caplog.records
        )

    def test_digest_written_when_flag_true(self, tmp_path, monkeypatch):
        """P10b: when _resolve_digest_auto_write returns True, _write_digest IS called."""
        config = _make_config(tmp_path)
        state_db = _make_state_db(tmp_path)

        monkeypatch.setattr(
            "lit_monitor.pipelines.discovery._resolve_digest_auto_write",
            lambda cfg: True,
        )

        with patch("lit_monitor.pipelines.discovery._write_digest") as mock_write, \
             patch("lit_monitor.pipelines.discovery.run_searches", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]):
            run_discovery(config, state_db, MagicMock(), MagicMock(), MagicMock())

        mock_write.assert_called_once()

    def test_status_correct_regardless_of_flag(self, tmp_path, monkeypatch):
        """P10b: discovery_runs.status is set correctly even when digest is skipped."""
        config = _make_config(tmp_path)
        state_db = _make_state_db(tmp_path)

        monkeypatch.setattr(
            "lit_monitor.pipelines.discovery._resolve_digest_auto_write",
            lambda cfg: False,
        )

        with patch("lit_monitor.pipelines.discovery.run_searches", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.run_researcher_searches", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.filter_known_dois", return_value=[]), \
             patch("lit_monitor.pipelines.discovery.rank_papers", return_value=[]):
            run_discovery(config, state_db, MagicMock(), MagicMock(), MagicMock())

        with state_db._connect() as conn:
            rows = conn.execute("SELECT status FROM discovery_runs").fetchall()
        assert rows, "Expected at least one discovery_runs row"
        # With correctly-shaped (list[dict]) search mocks, _run_discovery completes
        # cleanly and summary.errors stays empty → status MUST be "success".
        # Under the old ([], []) tuple mocks, the dedup loop raised AttributeError,
        # the discovery block hit its catch path, appended an error, and status was
        # "error" — so pinning "success" here would have failed pre-fix.
        assert rows[0][0] == "success", (
            f"Expected 'success' from the real discovery flow; got {rows[0][0]!r}. "
            "A non-success status means search results were the wrong shape and the "
            "pipeline fell into the exception-catch path."
        )
