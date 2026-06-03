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
# P4.6 — strict-mode escalation of enrichment boundaries
# ---------------------------------------------------------------------------

def test_add_chunks_failure_raises_under_strict():
    """P4.6: an enrichment-site failure (add_chunks) raises in --strict but is
    downgraded to WARN + continue in non-strict (preserves prior behaviour)."""
    import scripts.core.strict_mode as _sm
    from scripts.pipelines._ingest import index_embeddings_and_mark_phases
    state_db = MagicMock()
    embeddings_db = MagicMock()
    embeddings_db.add_chunks.side_effect = RuntimeError("chunks down")
    _sm.set_strict(True)
    with pytest.raises(RuntimeError, match="add_chunks failed"):
        index_embeddings_and_mark_phases(
            doi="10.test/strict-chunks",
            zotero_key="ZK",
            fulltext="body",
            paper_metadata={"title": "T", "year": 2020, "source_type": "paper"},
            chunks=[],
            state_db=state_db,
            embeddings_db=embeddings_db,
            phases_to_mark=("simple",),
            logger=logging.getLogger("test"),
        )


def test_add_paper_failure_raises_under_strict():
    """P4.6: the vector add_paper correctness gate also escalates in --strict."""
    import scripts.core.strict_mode as _sm
    from scripts.pipelines._ingest import index_embeddings_and_mark_phases
    state_db = MagicMock()
    embeddings_db = MagicMock()
    embeddings_db.add_paper.side_effect = RuntimeError("chromadb down")
    _sm.set_strict(True)
    with pytest.raises(RuntimeError, match="add_paper failed"):
        index_embeddings_and_mark_phases(
            doi="10.test/strict-paper",
            zotero_key="ZK",
            fulltext="body",
            paper_metadata={"title": "T", "year": 2020, "source_type": "paper"},
            chunks=[],
            state_db=state_db,
            embeddings_db=embeddings_db,
            phases_to_mark=("simple",),
            logger=logging.getLogger("test"),
        )


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


# ---------------------------------------------------------------------------
# N4 — build_mention_edges (ner_pipeline) unit tests
# ---------------------------------------------------------------------------

class TestBuildMentionEdgesFullPath:
    """N4: all three sources (schema + BioBERT + LLM) produce merged MentionEdge list."""

    def test_schema_plus_biobert_plus_llm(self):
        from scripts.graph.ner import NerSpan
        from scripts.graph.ner_pipeline import build_mention_edges
        from scripts.graph.normalizer import EntityNormalizer

        normalizer = EntityNormalizer(aliases={})

        class FakeBiobert:
            def extract(self, text: str):  # noqa: ARG002
                # P3.3: use a label that maps to a known type (DISEASE→topic).
                # An unmappable label like "GENE" is now dropped (no longer
                # silently bucketed as "material"), so it would not produce a
                # biobert edge and this test would no longer exercise that leg.
                return [NerSpan(text="lung cancer", label="DISEASE", start=0, end=11, confidence=0.95)]

        def fake_llm(text, low_conf):  # noqa: ARG001
            return {
                "new_entities": [
                    {"surface": "perfusion bioreactor", "type": "method",
                     "span_start": 0, "span_end": 19},
                ],
                "validations": [],
            }

        edges = build_mention_edges(
            paper_id="10.0/a",
            text="EGFR mutation in mAb cells. perfusion bioreactor used.",
            extraction_json={"methods_summary": "ion exchange chromatography"},
            paper_metadata={"authors": "[]", "journal": ""},
            normalizer=normalizer,
            biobert_factory=lambda: FakeBiobert(),
            llm_long_tail_fn=fake_llm,
        )
        sources = {e.source for e in edges}
        assert "schema" in sources
        assert "biobert" in sources
        assert "llm_cloud" in sources


class TestBuildMentionEdgesNoNlp:
    """N4: ImportError from BioBERT factory falls back gracefully to schema-only."""

    def test_falls_back_to_schema_when_biobert_import_fails(self, caplog):
        import logging

        import scripts.graph.ner_pipeline as np_mod
        from scripts.graph.ner_pipeline import build_mention_edges
        from scripts.graph.normalizer import EntityNormalizer

        # Reset once-flag so we see the WARN in this test run.
        np_mod._NLP_MISSING_LOGGED = False

        normalizer = EntityNormalizer(aliases={})

        def failing_biobert():
            raise ImportError("transformers not installed")

        with caplog.at_level(logging.WARNING):
            edges = build_mention_edges(
                paper_id="10.0/b",
                text="some text",
                extraction_json={"methods_summary": "ion exchange"},
                paper_metadata={"authors": "[]", "journal": ""},
                normalizer=normalizer,
                biobert_factory=failing_biobert,
                llm_long_tail_fn=lambda t, lc: {"new_entities": [], "validations": []},
            )

        sources = {e.source for e in edges}
        # Schema edges still present; BioBERT absent.
        assert "schema" in sources
        assert "biobert" not in sources
        # WARN was logged.
        assert any("nlp" in r.message.lower() for r in caplog.records)

    def test_warn_logged_only_once_across_calls(self):
        """The [nlp]-missing WARN must fire at most once per process."""
        import scripts.graph.ner_pipeline as np_mod
        from scripts.graph.ner_pipeline import build_mention_edges
        from scripts.graph.normalizer import EntityNormalizer

        np_mod._NLP_MISSING_LOGGED = False

        normalizer = EntityNormalizer(aliases={})

        def failing():
            raise ImportError("nope")

        build_mention_edges(
            paper_id="10.0/c", text="t",
            extraction_json={}, paper_metadata={"authors": "[]"},
            normalizer=normalizer, biobert_factory=failing,
            llm_long_tail_fn=lambda t, lc: {"new_entities": [], "validations": []},
            use_cloud_llm=False,
        )
        assert np_mod._NLP_MISSING_LOGGED is True
        # Second call must NOT reset the flag (it's already True — stays True).
        build_mention_edges(
            paper_id="10.0/d", text="t",
            extraction_json={}, paper_metadata={"authors": "[]"},
            normalizer=normalizer, biobert_factory=failing,
            llm_long_tail_fn=lambda t, lc: {"new_entities": [], "validations": []},
            use_cloud_llm=False,
        )
        assert np_mod._NLP_MISSING_LOGGED is True


class TestBuildMentionEdgesLlmDisabled:
    """N4: use_cloud_llm=False skips the LLM step entirely."""

    def test_use_cloud_llm_false_skips_llm_call(self):
        from scripts.graph.ner_pipeline import build_mention_edges
        from scripts.graph.normalizer import EntityNormalizer

        normalizer = EntityNormalizer(aliases={})
        called = {"n": 0}

        def llm_should_not_run(t, lc):  # noqa: ARG001
            called["n"] += 1
            return {"new_entities": [], "validations": []}

        edges = build_mention_edges(
            paper_id="10.0/e", text="t",
            extraction_json={"methods_summary": "ion exchange"},
            paper_metadata={"authors": "[]"},
            normalizer=normalizer,
            use_biobert=False,
            use_cloud_llm=False,
            llm_long_tail_fn=llm_should_not_run,
        )
        assert called["n"] == 0
        sources = {e.source for e in edges}
        # Only schema edges (or nothing if extraction yielded nothing).
        assert "llm_cloud" not in sources
        assert "biobert" not in sources


class TestBuildMentionEdgesNoText:
    """N4: text=None or empty skips BioBERT and LLM (nothing to process)."""

    def test_no_text_skips_biobert_and_llm(self):
        from scripts.graph.ner_pipeline import build_mention_edges
        from scripts.graph.normalizer import EntityNormalizer

        normalizer = EntityNormalizer(aliases={})

        def biobert_should_not_run():
            raise AssertionError("BioBERT must not be called with no text")

        edges = build_mention_edges(
            paper_id="10.0/f", text="",
            extraction_json={"methods_summary": "ion exchange"},
            paper_metadata={"authors": "[]"},
            normalizer=normalizer,
            biobert_factory=biobert_should_not_run,
            # LLM also gated by text — any call would fail the test via the
            # factory, but we pass a no-op to avoid unrelated failure modes.
            llm_long_tail_fn=lambda t, lc: {"new_entities": [], "validations": []},
        )
        sources = {e.source for e in edges}
        assert "biobert" not in sources
        assert "llm_cloud" not in sources


# ---------------------------------------------------------------------------
# R4 — maybe_extract_llm_relationships unit tests
# ---------------------------------------------------------------------------

class TestMaybeExtractLlmRelationships:
    """R4: conditionally run R2 LLM relationship extraction inside ingest.

    R28 invariant: any failure → empty list + WARN; ingest continues.
    """

    def _make_rel(self, doi: str = "10.0/a"):
        """Build a single RelationshipTuple for use as a canned return value."""
        from scripts.graph.relationship_extractor import RelationshipTuple
        return RelationshipTuple(
            source_doi=doi,
            predicate="EXTENDS",
            target_id="10.0/b",
            target_kind="Paper",
            evidence="builds on prior work",
            confidence=0.9,
            field="llm_extracted",
        )

    def test_disabled_returns_empty_no_llm_call(self, monkeypatch):
        """R4: when graph.relationships.llm_enabled=False, no LLM call, empty result."""
        called = {"n": 0}

        def fake_extract(*args, **kwargs):
            called["n"] += 1
            return [self._make_rel()]

        monkeypatch.setattr(
            "scripts.pipelines._ingest.extract_llm_relationships",
            fake_extract,
        )

        cfg = MagicMock()
        cfg.graph.relationships.llm_enabled = False

        from scripts.pipelines._ingest import maybe_extract_llm_relationships
        result = maybe_extract_llm_relationships(
            paper_doi="10.0/a",
            fulltext="paper text",
            extraction_json={},
            cfg=cfg,
        )
        assert result == []
        assert called["n"] == 0

    def test_enabled_calls_extractor_and_returns_triples(self, monkeypatch):
        """R4: when llm_enabled=True, calls R2 and returns its triples."""
        canned = [self._make_rel("10.0/a")]
        called = {"n": 0, "doi": None}

        def fake_extract(fulltext, paper_doi, extraction_json, **kwargs):
            called["n"] += 1
            called["doi"] = paper_doi
            return canned

        monkeypatch.setattr(
            "scripts.pipelines._ingest.extract_llm_relationships",
            fake_extract,
        )

        cfg = MagicMock()
        cfg.graph.relationships.llm_enabled = True

        from scripts.pipelines._ingest import maybe_extract_llm_relationships
        result = maybe_extract_llm_relationships(
            paper_doi="10.0/a",
            fulltext="paper text",
            extraction_json={"methods_summary": "ion exchange"},
            cfg=cfg,
        )
        assert result == canned
        assert called["n"] == 1
        assert called["doi"] == "10.0/a"

    def test_extractor_exception_returns_empty_with_warn(self, monkeypatch, caplog):
        """R4: R2 raises → caught + WARN log + empty result (R28 invariant: ingest continues)."""
        def failing(*args, **kwargs):
            raise RuntimeError("network exploded")

        monkeypatch.setattr(
            "scripts.pipelines._ingest.extract_llm_relationships",
            failing,
        )

        cfg = MagicMock()
        cfg.graph.relationships.llm_enabled = True

        import logging

        from scripts.pipelines._ingest import maybe_extract_llm_relationships
        with caplog.at_level(logging.WARNING):
            result = maybe_extract_llm_relationships(
                paper_doi="10.0/a",
                fulltext="paper text",
                extraction_json={},
                cfg=cfg,
            )
        assert result == []
        assert any(
            "llm" in r.message.lower() or "relationship" in r.message.lower()
            for r in caplog.records
        )

    def test_extractor_exception_raises_under_strict(self, monkeypatch):
        """P4.6: in --strict, an R2 extraction failure escalates to a raise
        instead of being downgraded to an empty list."""
        import scripts.core.strict_mode as _sm

        def failing(*args, **kwargs):
            raise RuntimeError("network exploded")

        monkeypatch.setattr(
            "scripts.pipelines._ingest.extract_llm_relationships",
            failing,
        )
        cfg = MagicMock()
        cfg.graph.relationships.llm_enabled = True

        from scripts.pipelines._ingest import maybe_extract_llm_relationships
        _sm.set_strict(True)
        with pytest.raises(RuntimeError, match="relationship extraction failed"):
            maybe_extract_llm_relationships(
                paper_doi="10.0/a",
                fulltext="paper text",
                extraction_json={},
                cfg=cfg,
            )

    def test_no_fulltext_returns_empty(self, monkeypatch):
        """R4: empty fulltext skips the LLM call entirely."""
        called = {"n": 0}

        def fake_extract(*args, **kwargs):
            called["n"] += 1
            return [self._make_rel()]

        monkeypatch.setattr(
            "scripts.pipelines._ingest.extract_llm_relationships",
            fake_extract,
        )

        cfg = MagicMock()
        cfg.graph.relationships.llm_enabled = True

        from scripts.pipelines._ingest import maybe_extract_llm_relationships
        result = maybe_extract_llm_relationships(
            paper_doi="10.0/a",
            fulltext="",
            extraction_json={},
            cfg=cfg,
        )
        assert result == []
        assert called["n"] == 0

    def test_missing_config_section_treated_as_disabled(self, monkeypatch):
        """R4: when cfg.graph.relationships raises AttributeError, default to disabled."""
        cfg = MagicMock()
        # Make .graph.relationships raise AttributeError on attribute access.
        # NOTE: type(cfg.graph) is the shared MagicMock class — setting the
        # property directly leaks onto every MagicMock in the process. Use
        # monkeypatch.setattr so it auto-reverts at teardown.
        monkeypatch.setattr(
            type(cfg.graph),
            "relationships",
            property(
                lambda self: (_ for _ in ()).throw(AttributeError("no relationships key"))
            ),
            raising=False,
        )

        from scripts.pipelines._ingest import maybe_extract_llm_relationships
        # Must not raise; must return empty list.
        result = maybe_extract_llm_relationships(
            paper_doi="10.0/a",
            fulltext="text",
            extraction_json={},
            cfg=cfg,
        )
        assert result == []
