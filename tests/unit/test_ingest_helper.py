"""R28 contract tests for index_embeddings_and_mark_phases."""
import logging
from unittest.mock import MagicMock


def test_happy_path_marks_all_phases():
    from scripts.pipelines._ingest import index_embeddings_and_mark_phases
    state_db = MagicMock()
    embeddings_db = MagicMock()
    ok, err = index_embeddings_and_mark_phases(
        doi="10.test/happy",
        zotero_key="ZKEY123",
        fulltext="body",
        paper_metadata={"title": "T", "year": 2020, "source_type": "paper"},
        chunks=[],
        state_db=state_db,
        embeddings_db=embeddings_db,
        phases_to_mark=("simple", "complex"),
        logger=logging.getLogger("test"),
    )
    assert ok and err is None
    embeddings_db.add_paper.assert_called_once()
    embeddings_db.add_chunks.assert_called_once()
    # mark_brain_build_phase must be called with the ZOTERO_KEY, not the DOI
    calls = state_db.mark_brain_build_phase.call_args_list
    assert len(calls) == 2
    assert calls[0].args == ("ZKEY123", "simple")
    assert calls[1].args == ("ZKEY123", "complex")


def test_add_paper_failure_skips_phase_marks():
    from scripts.pipelines._ingest import index_embeddings_and_mark_phases
    state_db = MagicMock()
    embeddings_db = MagicMock()
    embeddings_db.add_paper.side_effect = RuntimeError("chromadb down")
    ok, err = index_embeddings_and_mark_phases(
        doi="10.test/embed-fail",
        zotero_key="ZKEY999",
        fulltext="body",
        paper_metadata={"title": "T", "year": 2020, "source_type": "paper"},
        chunks=[],
        state_db=state_db,
        embeddings_db=embeddings_db,
        phases_to_mark=("simple", "complex"),
        logger=logging.getLogger("test"),
    )
    assert not ok
    assert "add_paper_failed" in err
    state_db.mark_brain_build_phase.assert_not_called()


def test_add_chunks_failure_still_marks_phases():
    """Non-fatal — chunks failing should NOT block phase progress (preserves prior behavior)."""
    from scripts.pipelines._ingest import index_embeddings_and_mark_phases
    state_db = MagicMock()
    embeddings_db = MagicMock()
    embeddings_db.add_chunks.side_effect = RuntimeError("chunks down")
    ok, err = index_embeddings_and_mark_phases(
        doi="10.test/chunks-fail",
        zotero_key="ZKEY555",
        fulltext="body",
        paper_metadata={"title": "T", "year": 2020, "source_type": "paper"},
        chunks=[],
        state_db=state_db,
        embeddings_db=embeddings_db,
        phases_to_mark=("simple",),
        logger=logging.getLogger("test"),
    )
    assert ok and err is None
    state_db.mark_brain_build_phase.assert_called_once_with("ZKEY555", "simple")
