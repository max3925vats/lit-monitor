"""G10: backfill + rebuild tests."""
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
