"""R28 contract tests for index_embeddings_and_mark_phases."""
import logging
from unittest.mock import MagicMock

import pytest


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


# ---------------------------------------------------------------------------
# G6 — graph dual-write integration
# ---------------------------------------------------------------------------

class TestIngestWithGraph:
    """G6: index_embeddings_and_mark_phases accepts a graph_db kwarg.

    R28 invariant: ``papers.graph_indexed = 1`` ONLY when BOTH the vector
    index AND the graph write succeed.  Graph failure is ENRICHMENT — like
    chunks — and must NEVER block phase marking.
    """

    def _call(self, *, graph_db=None, graph_entities=None, graph_relationships=None,
              embed_fail=False, graph_fail=False):
        from scripts.pipelines._ingest import index_embeddings_and_mark_phases
        state_db = MagicMock()
        embeddings_db = MagicMock()
        if embed_fail:
            embeddings_db.add_paper.side_effect = RuntimeError("embed down")
        if graph_db is not None and graph_fail:
            graph_db.add_paper.side_effect = RuntimeError("kuzu down")
        ok, err = index_embeddings_and_mark_phases(
            doi="10.g6/x",
            zotero_key="ZKEY6",
            fulltext="body",
            paper_metadata={"title": "T", "year": 2024, "source_type": "paper"},
            chunks=[],
            state_db=state_db,
            embeddings_db=embeddings_db,
            phases_to_mark=("simple", "complex"),
            logger=logging.getLogger("test"),
            graph_db=graph_db,
            graph_entities=graph_entities or [],
            graph_relationships=graph_relationships or [],
        )
        return ok, err, state_db, embeddings_db

    def test_graph_db_none_does_not_attempt_write(self):
        """G6: graph_db=None -> no graph attempt, phase marks fire, graph_indexed stays untouched."""
        ok, err, state_db, _embeddings_db = self._call(graph_db=None)
        assert ok and err is None
        # set_graph_indexed must NOT have been called when graph_db is None.
        assert not state_db.set_graph_indexed.called
        # Phase marks still fire — vector-only ingest is unaffected.
        assert state_db.mark_brain_build_phase.call_count == 2

    def test_graph_success_sets_graph_indexed(self):
        """G6: graph add_paper success -> set_graph_indexed(doi, 1) AND phase marks fire."""
        graph_db = MagicMock()
        ok, err, state_db, _embeddings_db = self._call(graph_db=graph_db)
        assert ok and err is None
        # graph_db.add_paper was called with the right doi & paper_metadata
        assert graph_db.add_paper.call_count == 1
        kwargs = graph_db.add_paper.call_args.kwargs
        assert kwargs["doi"] == "10.g6/x"
        assert kwargs["prompt_version"] == "phase1.0"
        # set_graph_indexed flipped to 1
        state_db.set_graph_indexed.assert_called_once_with("10.g6/x", 1)
        # Phase marks fired
        assert state_db.mark_brain_build_phase.call_count == 2

    def test_graph_failure_does_not_block_phase_marks(self):
        """G6 R28 INVARIANT: graph add_paper raises -> phase marks STILL fire,
        graph_indexed is NEVER flipped to 1.

        This is the test that proves the dual-write invariant holds: the graph
        is enrichment, never a correctness gate for the vector pipeline.
        """
        graph_db = MagicMock()
        ok, err, state_db, _embeddings_db = self._call(
            graph_db=graph_db, graph_fail=True,
        )
        # Vector ingest still reported success.
        assert ok and err is None
        # graph add_paper was attempted exactly once.
        assert graph_db.add_paper.call_count == 1
        # set_graph_indexed was NEVER called — graph_indexed stays 0.
        assert not state_db.set_graph_indexed.called
        # Phase marks still fired — this is the invariant proof.
        assert state_db.mark_brain_build_phase.call_count == 2

    def test_embed_failure_with_graph_db_does_not_attempt_graph(self):
        """G6: when ChromaDB add_paper fails, graph_db.add_paper is NOT attempted.

        Graph write is gated on vector add_paper success.  This matches the
        chunks-style precedence — the vector index is the prerequisite, the
        graph is the enrichment.
        """
        graph_db = MagicMock()
        ok, err, state_db, _embeddings_db = self._call(
            graph_db=graph_db, embed_fail=True,
        )
        assert not ok
        assert "add_paper_failed" in err
        # Graph write must NOT be attempted on embed failure.
        assert not graph_db.add_paper.called
        # set_graph_indexed must NOT have been called.
        assert not state_db.set_graph_indexed.called
        # Phase marks must NOT have fired — embed failure is the only gate.
        assert not state_db.mark_brain_build_phase.called

    def test_graph_db_passes_entities_and_relationships(self):
        """G6: graph_entities and graph_relationships flow through to graph_db.add_paper."""
        graph_db = MagicMock()
        ents = ["e1", "e2"]
        rels = ["r1"]
        ok, err, _state_db, _embeddings_db = self._call(
            graph_db=graph_db, graph_entities=ents, graph_relationships=rels,
        )
        assert ok and err is None
        kwargs = graph_db.add_paper.call_args.kwargs
        assert kwargs["entities"] == ents
        assert kwargs["relationships"] == rels

    @pytest.mark.parametrize("phase_count", [0, 1, 2])
    def test_no_phases_to_mark_but_graph_still_runs(self, phase_count):
        """G6: graph write happens even when phases_to_mark is empty (e.g. re-ingest)."""
        from scripts.pipelines._ingest import index_embeddings_and_mark_phases
        graph_db = MagicMock()
        state_db = MagicMock()
        embeddings_db = MagicMock()
        phases = ("simple", "complex")[:phase_count]
        ok, err = index_embeddings_and_mark_phases(
            doi="10.g6/y",
            zotero_key="ZKEY7",
            fulltext="body",
            paper_metadata={"title": "T", "year": 2024, "source_type": "paper"},
            chunks=[],
            state_db=state_db,
            embeddings_db=embeddings_db,
            phases_to_mark=phases,
            logger=logging.getLogger("test"),
            graph_db=graph_db,
            graph_entities=[],
            graph_relationships=[],
        )
        assert ok and err is None
        assert graph_db.add_paper.call_count == 1
        state_db.set_graph_indexed.assert_called_once_with("10.g6/y", 1)
        assert state_db.mark_brain_build_phase.call_count == phase_count
