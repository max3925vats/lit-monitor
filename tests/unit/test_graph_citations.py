"""G5: CITES mirror tests.

Covers mirror_citations() and safe_graph_db() from
scripts/graph/import_citations.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.graph import GraphDB
from scripts.graph.import_citations import mirror_citations, safe_graph_db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper(db: GraphDB, doi: str) -> None:
    """Insert a minimal Paper node into Kuzu."""
    db._conn.execute(
        "CREATE (p:Paper {doi: $doi, title: $t, year: 2024, journal: 'X'})",
        {"doi": doi, "t": f"Title for {doi}"},
    )


def _count_cites(db: GraphDB) -> int:
    """Return total number of CITES edges in the graph."""
    res = db._conn.execute("MATCH ()-[r:CITES]->() RETURN count(r) AS n")
    row = res.get_next()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# mirror_citations tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMirrorCitations:
    def test_mirror_creates_cites_for_resolved_edges(self, tmp_path):
        """G5: resolved citation edges become CITES edges in Kuzu."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _make_paper(db, "10.0/a")
        _make_paper(db, "10.0/b")

        mock_state = MagicMock()
        # source_doi=None path → get_all_citation_edges
        mock_state.get_all_citation_edges.return_value = [
            {"source_doi": "10.0/a", "target_doi": "10.0/b", "resolution": "exact"},
        ]

        added = mirror_citations(db, mock_state)
        assert added == 1
        assert _count_cites(db) == 1

    def test_mirror_skips_when_target_doi_is_null(self, tmp_path):
        """G5: rows with null target_doi are skipped."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _make_paper(db, "10.0/a")

        mock_state = MagicMock()
        mock_state.get_all_citation_edges.return_value = [
            {"source_doi": "10.0/a", "target_doi": None, "resolution": "unresolved"},
        ]

        added = mirror_citations(db, mock_state)
        assert added == 0
        assert _count_cites(db) == 0

    def test_mirror_skips_when_paper_node_missing(self, tmp_path):
        """G5: edges where source or target Paper doesn't exist are skipped (no crash)."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _make_paper(db, "10.0/a")
        # No Paper node for "10.0/b" — mirror must skip gracefully.

        mock_state = MagicMock()
        mock_state.get_all_citation_edges.return_value = [
            {"source_doi": "10.0/a", "target_doi": "10.0/b", "resolution": "exact"},
        ]

        added = mirror_citations(db, mock_state)
        assert added == 0
        assert _count_cites(db) == 0

    def test_mirror_is_idempotent(self, tmp_path):
        """G5: re-running mirror_citations doesn't create duplicate CITES edges."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _make_paper(db, "10.0/a")
        _make_paper(db, "10.0/b")

        mock_state = MagicMock()
        mock_state.get_all_citation_edges.return_value = [
            {"source_doi": "10.0/a", "target_doi": "10.0/b", "resolution": "exact"},
        ]

        added1 = mirror_citations(db, mock_state)
        added2 = mirror_citations(db, mock_state)

        assert added1 == 1
        assert added2 == 0  # idempotent — edge already present
        assert _count_cites(db) == 1  # still exactly one edge

    def test_mirror_with_source_doi_filter(self, tmp_path):
        """G5: source_doi arg routes through get_citation_edges (not get_all)."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        for doi in ["10.0/a", "10.0/b", "10.0/c"]:
            _make_paper(db, doi)

        mock_state = MagicMock()
        # When called with source_doi="10.0/a", only that paper's edges come back.
        mock_state.get_citation_edges.return_value = [
            {"source_doi": "10.0/a", "target_doi": "10.0/b", "resolution": "exact"},
        ]

        added = mirror_citations(db, mock_state, source_doi="10.0/a")

        # Must route to get_citation_edges, not get_all_citation_edges.
        mock_state.get_citation_edges.assert_called_once_with("10.0/a")
        mock_state.get_all_citation_edges.assert_not_called()
        assert added == 1

    def test_mirror_handles_empty_state_db(self, tmp_path):
        """G5: no citation edges → 0 added, no crash."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        mock_state = MagicMock()
        mock_state.get_all_citation_edges.return_value = []

        added = mirror_citations(db, mock_state)
        assert added == 0

    def test_mirror_returns_zero_when_graph_db_is_none(self):
        """G5: graph_db=None (e.g. [graph] extra missing) returns 0, no raise."""
        mock_state = MagicMock()
        added = mirror_citations(graph_db=None, state_db=mock_state)
        assert added == 0

    def test_mirror_returns_zero_when_state_db_is_none(self, tmp_path):
        """G5: state_db=None returns 0, no raise."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        added = mirror_citations(graph_db=db, state_db=None)
        assert added == 0


# ---------------------------------------------------------------------------
# safe_graph_db tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSafeGraphDB:
    def test_returns_graphdb_when_extra_installed(self, tmp_path):
        """G5: when [graph] is installed, safe_graph_db returns a GraphDB."""
        db = safe_graph_db(persist_dir=str(tmp_path / "safe.kuzu"))
        assert db is not None
        assert isinstance(db, GraphDB)
        db.close()

    # The "extra not installed" path requires sys.modules manipulation — covered
    # by G1's test_module_import_succeeds_without_kuzu.  Not duplicated here.
