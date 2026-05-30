"""G10 / N5 / R5: backfill + rebuild + NER-backfill + relationship-backfill tests."""
from __future__ import annotations

from datetime import datetime

from scripts.core.state_db import StateDB
from scripts.graph import GraphDB
from scripts.graph.backfill import (
    backfill_papers,
    rebuild_aliases_only,
    rebuild_all,
)


class TestBackfillPapers:
    def test_backfill_processes_only_graph_indexed_zero(self, tmp_path):
        """G10: backfill_papers walks only rows where graph_indexed=0."""
        state_db_path = tmp_path / "state.db"
        state_db = StateDB(state_db_path)
        # 3 papers — 1 already indexed
        for doi in ("10.0/a", "10.0/b", "10.0/c"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        state_db.set_graph_indexed("10.0/a", 1)

        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        # backfill_papers should process /b and /c, not /a
        processed = backfill_papers(state_db, graph_db, filter_doi=None, since=None)
        assert processed == 2
        # Verify graph_indexed flipped for all
        with state_db._connect() as conn:
            rows = conn.execute("SELECT doi, graph_indexed FROM papers").fetchall()
        assert {r[0]: r[1] for r in rows} == {"10.0/a": 1, "10.0/b": 1, "10.0/c": 1}

    def test_backfill_single_doi(self, tmp_path):
        """G10: filter_doi processes just one paper."""
        state_db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        processed = backfill_papers(state_db, graph_db, filter_doi="10.0/a", since=None)
        assert processed == 1
        with state_db._connect() as conn:
            rows = conn.execute("SELECT doi, graph_indexed FROM papers").fetchall()
        d = {r[0]: r[1] for r in rows}
        assert d == {"10.0/a": 1, "10.0/b": 0}

    def test_backfill_idempotent_on_already_indexed(self, tmp_path):
        """G10: running backfill twice doesn't double-process."""
        state_db = StateDB(tmp_path / "state.db")
        state_db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        first = backfill_papers(state_db, graph_db, filter_doi=None, since=None)
        second = backfill_papers(state_db, graph_db, filter_doi=None, since=None)
        assert first == 1
        assert second == 0

    def test_backfill_since_filter(self, tmp_path):
        """G10: --since filters by papers.last_updated."""
        state_db = StateDB(tmp_path / "state.db")
        state_db.upsert_paper({"doi": "10.0/old", "title": "O", "year": 2020, "source_type": "zotero"})
        state_db.upsert_paper({"doi": "10.0/new", "title": "N", "year": 2024, "source_type": "zotero"})
        # Manually backdate /old
        with state_db._connect() as conn:
            conn.execute(
                "UPDATE papers SET last_updated = ? WHERE doi = ?",
                ("2020-01-01T00:00:00", "10.0/old"),
            )
            conn.commit()
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        since = datetime(2023, 1, 1)
        processed = backfill_papers(state_db, graph_db, filter_doi=None, since=since)
        # Only /new (with current timestamp) processed
        assert processed == 1

    def test_backfill_progress_callback_called(self, tmp_path):
        """G10: progress_callback receives (doi, done, total) for each paper."""
        state_db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        calls: list[tuple[str, int, int]] = []
        backfill_papers(
            state_db, graph_db, filter_doi=None, since=None,
            progress_callback=lambda d, done, total: calls.append((d, done, total)),
        )
        assert len(calls) == 2
        # total should be consistent
        assert all(total == 2 for _, _, total in calls)


class TestRebuildAll:
    def test_rebuild_all_drops_and_repopulates(self, tmp_path):
        """G10: rebuild --all drops all Kuzu data, resets graph_indexed=0, re-backfills."""
        state_db = StateDB(tmp_path / "state.db")
        state_db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # Initial backfill
        backfill_papers(state_db, graph_db, filter_doi=None, since=None)
        # Now rebuild — should process 1 paper again after resetting graph_indexed
        processed = rebuild_all(state_db, graph_db)
        assert processed == 1

    def test_rebuild_all_resets_graph_indexed_flags(self, tmp_path):
        """G10: rebuild_all resets graph_indexed=0 before re-backfilling."""
        state_db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # Mark both indexed
        state_db.set_graph_indexed("10.0/a", 1)
        state_db.set_graph_indexed("10.0/b", 1)
        # rebuild_all should re-process both
        processed = rebuild_all(state_db, graph_db)
        assert processed == 2


class TestRebuildAliasesOnly:
    def test_rebuild_aliases_only_does_not_crash_on_empty_graph(self, tmp_path):
        """G10: rebuild --aliases-only doesn't crash on empty graph."""
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # Even on an empty graph, should not crash
        result = rebuild_aliases_only(graph_db)
        assert result >= 0  # number of entities renormalized

    def test_rebuild_aliases_only_returns_int(self, tmp_path):
        """G10: rebuild_aliases_only returns the count of entities renormalized."""
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        result = rebuild_aliases_only(graph_db)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# N5: backfill_ner tests
# ---------------------------------------------------------------------------

class TestBackfillNer:
    """N5: backfill_ner — NER-augmented entity pipeline over existing corpus."""

    def _fake_edge(self, paper_id: str):
        """Return a minimal MentionEdge for mocking build_mention_edges."""
        from scripts.graph.entity_extractor import MentionEdge
        return MentionEdge(
            paper_id=paper_id,
            entity_key="test_entity",
            type="material",
            source="biobert",
            surface="test",
            field=None,
            confidence=0.9,
            span_start=0,
            span_end=4,
        )

    def test_backfill_ner_processes_unprocessed_papers(self, tmp_path, monkeypatch):
        """N5: backfill_ner runs pipeline on papers where ner_processed_at IS NULL."""
        from scripts.graph.backfill import backfill_ner

        state_db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        fake_edge = self._fake_edge  # bind helper

        def fake_build(*, paper_id: str, **_kwargs):
            return [fake_edge(paper_id)]

        monkeypatch.setattr("scripts.graph.backfill.build_mention_edges", fake_build)

        summary = backfill_ner(state_db, graph_db, with_llm=False)
        assert summary["papers_processed"] == 2
        assert summary["edges_added"] >= 2
        assert summary["failures"] == 0

    def test_backfill_ner_idempotent_on_rerun(self, tmp_path, monkeypatch):
        """N5: re-running backfill_ner after success is a no-op."""
        from scripts.graph.backfill import backfill_ner

        state_db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        monkeypatch.setattr("scripts.graph.backfill.build_mention_edges", lambda **kw: [])

        first = backfill_ner(state_db, graph_db, with_llm=False)
        assert first["papers_processed"] == 2

        # Second run: ner_processed_at is now set → should be a no-op.
        second = backfill_ner(state_db, graph_db, with_llm=False)
        assert second["papers_processed"] == 0

    def test_backfill_ner_respects_limit(self, tmp_path, monkeypatch):
        """N5: limit= caps the number of papers processed."""
        from scripts.graph.backfill import backfill_ner

        state_db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b", "10.0/c"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        monkeypatch.setattr("scripts.graph.backfill.build_mention_edges", lambda **kw: [])

        summary = backfill_ner(state_db, graph_db, with_llm=False, limit=2)
        assert summary["papers_processed"] == 2

    def test_backfill_ner_respects_since_filter(self, tmp_path, monkeypatch):
        """N5: since= filters candidates by papers.last_updated."""
        from scripts.graph.backfill import backfill_ner

        state_db = StateDB(tmp_path / "state.db")
        state_db.upsert_paper({"doi": "10.0/old", "title": "O", "year": 2020, "source_type": "zotero"})
        state_db.upsert_paper({"doi": "10.0/new", "title": "N", "year": 2024, "source_type": "zotero"})
        # Backdate the old paper.
        with state_db._connect() as conn:
            conn.execute(
                "UPDATE papers SET last_updated = ? WHERE doi = ?",
                ("2020-01-01T00:00:00", "10.0/old"),
            )
            conn.commit()
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        monkeypatch.setattr("scripts.graph.backfill.build_mention_edges", lambda **kw: [])

        summary = backfill_ner(state_db, graph_db, with_llm=False, since=datetime(2023, 1, 1))
        # Only the "new" paper (with current timestamp) should be processed.
        assert summary["papers_processed"] == 1

    def test_backfill_ner_with_llm_flag_passed_through(self, tmp_path, monkeypatch):
        """N5: with_llm=True sets use_cloud_llm=True in build_mention_edges."""
        from scripts.graph.backfill import backfill_ner

        state_db = StateDB(tmp_path / "state.db")
        state_db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        captured: dict[str, object] = {}

        def fake_build(*, use_cloud_llm: bool, **kwargs):
            captured["use_cloud_llm"] = use_cloud_llm
            return []

        monkeypatch.setattr("scripts.graph.backfill.build_mention_edges", fake_build)

        backfill_ner(state_db, graph_db, with_llm=True)
        assert captured["use_cloud_llm"] is True

    def test_backfill_ner_without_llm_flag(self, tmp_path, monkeypatch):
        """N5: with_llm=False (default) sets use_cloud_llm=False."""
        from scripts.graph.backfill import backfill_ner

        state_db = StateDB(tmp_path / "state.db")
        state_db.upsert_paper({"doi": "10.0/a", "title": "A", "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        captured: dict[str, object] = {}

        def fake_build(*, use_cloud_llm: bool, **kwargs):
            captured["use_cloud_llm"] = use_cloud_llm
            return []

        monkeypatch.setattr("scripts.graph.backfill.build_mention_edges", fake_build)

        backfill_ner(state_db, graph_db)  # with_llm defaults to False
        assert captured["use_cloud_llm"] is False

    def test_backfill_ner_per_paper_failure_counted(self, tmp_path, monkeypatch):
        """N5: per-paper failures are counted + skipped, not raised."""
        from scripts.graph.backfill import backfill_ner

        state_db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        call_count = {"n": 0}

        def flaky_build(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated NER failure")
            return []

        monkeypatch.setattr("scripts.graph.backfill.build_mention_edges", flaky_build)

        summary = backfill_ner(state_db, graph_db, with_llm=False)
        assert summary["failures"] == 1
        assert summary["papers_processed"] == 1

    def test_backfill_ner_progress_callback_called(self, tmp_path, monkeypatch):
        """N5: progress_callback receives (doi, done, total) for each paper."""
        from scripts.graph.backfill import backfill_ner

        state_db = StateDB(tmp_path / "state.db")
        for doi in ("10.0/a", "10.0/b"):
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        monkeypatch.setattr("scripts.graph.backfill.build_mention_edges", lambda **kw: [])

        calls: list[tuple[str, int, int]] = []
        backfill_ner(
            state_db, graph_db, with_llm=False,
            progress_callback=lambda d, done, total: calls.append((d, done, total)),
        )
        assert len(calls) == 2
        assert all(t == 2 for _, _, t in calls)


# ---------------------------------------------------------------------------
# R5: backfill_relationships tests
# ---------------------------------------------------------------------------

class TestBackfillRelationships:
    """R5: backfill_relationships — G4 schema + optional R2 LLM over existing corpus."""

    def _setup(self, tmp_path, dois=("10.0/a", "10.0/b")):
        """Insert papers and return (state_db, graph_db)."""
        state_db = StateDB(tmp_path / "state.db")
        for doi in dois:
            state_db.upsert_paper({"doi": doi, "title": doi, "year": 2024, "source_type": "zotero"})
        graph_db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        return state_db, graph_db

    def test_processes_unprocessed_papers(self, tmp_path, monkeypatch):
        """R5: backfill_relationships runs over papers where rel_processed_at IS NULL."""
        from scripts.graph.backfill import backfill_relationships

        state_db, graph_db = self._setup(tmp_path)

        monkeypatch.setattr("scripts.graph.backfill.extract_entities", lambda *a, **kw: [])
        monkeypatch.setattr("scripts.graph.backfill.extract_relationships", lambda *a, **kw: [])
        monkeypatch.setattr(
            "scripts.pipelines._ingest.maybe_extract_llm_relationships",
            lambda **kw: [],
        )

        summary = backfill_relationships(state_db, graph_db, with_llm=False)
        assert summary["papers_processed"] == 2
        assert summary["failures"] == 0

    def test_idempotent_on_rerun(self, tmp_path, monkeypatch):
        """R5: re-running after rel_processed_at stamped is a no-op."""
        from scripts.graph.backfill import backfill_relationships

        state_db, graph_db = self._setup(tmp_path)

        monkeypatch.setattr("scripts.graph.backfill.extract_entities", lambda *a, **kw: [])
        monkeypatch.setattr("scripts.graph.backfill.extract_relationships", lambda *a, **kw: [])
        monkeypatch.setattr(
            "scripts.pipelines._ingest.maybe_extract_llm_relationships",
            lambda **kw: [],
        )

        first = backfill_relationships(state_db, graph_db, with_llm=False)
        assert first["papers_processed"] == 2

        # Second run: rel_processed_at is now set → should be a no-op.
        second = backfill_relationships(state_db, graph_db, with_llm=False)
        assert second["papers_processed"] == 0

    def test_respects_limit(self, tmp_path, monkeypatch):
        """R5: limit= caps the number of papers processed."""
        from scripts.graph.backfill import backfill_relationships

        state_db, graph_db = self._setup(tmp_path, dois=("10.0/a", "10.0/b", "10.0/c"))

        monkeypatch.setattr("scripts.graph.backfill.extract_entities", lambda *a, **kw: [])
        monkeypatch.setattr("scripts.graph.backfill.extract_relationships", lambda *a, **kw: [])
        monkeypatch.setattr(
            "scripts.pipelines._ingest.maybe_extract_llm_relationships",
            lambda **kw: [],
        )

        summary = backfill_relationships(state_db, graph_db, with_llm=False, limit=2)
        assert summary["papers_processed"] == 2

    def test_with_llm_flag_propagates_via_cfg_shim(self, tmp_path, monkeypatch):
        """R5: with_llm=True reaches maybe_extract_llm_relationships via the cfg shim."""
        from scripts.graph.backfill import backfill_relationships

        state_db, graph_db = self._setup(tmp_path, dois=("10.0/a",))

        captured: dict[str, object] = {}

        def fake_maybe_llm(*, paper_doi, fulltext, extraction_json, cfg):
            # The shim must expose llm_enabled=True.
            try:
                captured["llm_enabled"] = cfg.graph.relationships.llm_enabled
            except AttributeError:
                captured["llm_enabled"] = None
            return []

        monkeypatch.setattr("scripts.graph.backfill.extract_entities", lambda *a, **kw: [])
        monkeypatch.setattr("scripts.graph.backfill.extract_relationships", lambda *a, **kw: [])
        monkeypatch.setattr(
            "scripts.pipelines._ingest.maybe_extract_llm_relationships",
            fake_maybe_llm,
        )

        backfill_relationships(state_db, graph_db, with_llm=True)
        assert captured.get("llm_enabled") is True

    def test_with_llm_false_propagates_via_cfg_shim(self, tmp_path, monkeypatch):
        """R5: with_llm=False (default) sets cfg.graph.relationships.llm_enabled=False."""
        from scripts.graph.backfill import backfill_relationships

        state_db, graph_db = self._setup(tmp_path, dois=("10.0/a",))

        captured: dict[str, object] = {}

        def fake_maybe_llm(*, paper_doi, fulltext, extraction_json, cfg):
            try:
                captured["llm_enabled"] = cfg.graph.relationships.llm_enabled
            except AttributeError:
                captured["llm_enabled"] = None
            return []

        monkeypatch.setattr("scripts.graph.backfill.extract_entities", lambda *a, **kw: [])
        monkeypatch.setattr("scripts.graph.backfill.extract_relationships", lambda *a, **kw: [])
        monkeypatch.setattr(
            "scripts.pipelines._ingest.maybe_extract_llm_relationships",
            fake_maybe_llm,
        )

        backfill_relationships(state_db, graph_db)  # with_llm defaults to False
        assert captured.get("llm_enabled") is False

    def test_per_paper_failure_counted(self, tmp_path, monkeypatch):
        """R5: per-paper failures are counted + skipped, not raised."""
        from scripts.graph.backfill import backfill_relationships

        state_db, graph_db = self._setup(tmp_path)

        call_count = {"n": 0}

        def flaky_extract(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated extraction failure")
            return []

        monkeypatch.setattr("scripts.graph.backfill.extract_entities", lambda *a, **kw: [])
        monkeypatch.setattr("scripts.graph.backfill.extract_relationships", flaky_extract)
        monkeypatch.setattr(
            "scripts.pipelines._ingest.maybe_extract_llm_relationships",
            lambda **kw: [],
        )

        summary = backfill_relationships(state_db, graph_db, with_llm=False)
        assert summary["failures"] == 1
        assert summary["papers_processed"] == 1

    def test_entities_not_passed_to_graph_db(self, tmp_path, monkeypatch):
        """R5: entities=[] in graph_db.add_paper — R5 never touches MENTIONS edges."""
        from scripts.graph.backfill import backfill_relationships

        state_db, graph_db = self._setup(tmp_path, dois=("10.0/a",))
        captured_entities: list[list] = []

        real_add_paper = graph_db.add_paper

        def spy_add_paper(doi, entities, relationships, paper_metadata, prompt_version):
            captured_entities.append(list(entities))
            return real_add_paper(
                doi=doi,
                entities=entities,
                relationships=relationships,
                paper_metadata=paper_metadata,
                prompt_version=prompt_version,
            )

        monkeypatch.setattr(graph_db, "add_paper", spy_add_paper)
        monkeypatch.setattr("scripts.graph.backfill.extract_entities", lambda *a, **kw: [])
        monkeypatch.setattr("scripts.graph.backfill.extract_relationships", lambda *a, **kw: [])
        monkeypatch.setattr(
            "scripts.pipelines._ingest.maybe_extract_llm_relationships",
            lambda **kw: [],
        )

        backfill_relationships(state_db, graph_db, with_llm=False)
        assert all(e == [] for e in captured_entities), "R5 must not write entity edges"

    def test_progress_callback_called(self, tmp_path, monkeypatch):
        """R5: progress_callback receives (doi, done, total) for each paper."""
        from scripts.graph.backfill import backfill_relationships

        state_db, graph_db = self._setup(tmp_path)

        monkeypatch.setattr("scripts.graph.backfill.extract_entities", lambda *a, **kw: [])
        monkeypatch.setattr("scripts.graph.backfill.extract_relationships", lambda *a, **kw: [])
        monkeypatch.setattr(
            "scripts.pipelines._ingest.maybe_extract_llm_relationships",
            lambda **kw: [],
        )

        calls: list[tuple[str, int, int]] = []
        backfill_relationships(
            state_db, graph_db, with_llm=False,
            progress_callback=lambda d, done, total: calls.append((d, done, total)),
        )
        assert len(calls) == 2
        assert all(t == 2 for _, _, t in calls)
