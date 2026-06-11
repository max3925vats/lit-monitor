"""G5: CITES mirror tests.

Covers mirror_citations() and safe_graph_db() from
scripts/graph/import_citations.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lit_monitor.graph import GraphDB
from lit_monitor.graph.import_citations import mirror_citations, safe_graph_db

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

    def test_mirror_rolls_back_on_mid_loop_failure(self, tmp_path):
        """P3.4: a failure partway through the CITES loop must leave NO partial
        edges committed (all-or-nothing transaction).

        We let the first CITES CREATE succeed inside the transaction, then make
        the SECOND CITES CREATE raise. The whole batch must roll back, so after
        the (re-raised) failure the graph contains zero CITES edges.
        """
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        for doi in ["10.0/a", "10.0/b", "10.0/c", "10.0/d"]:
            _make_paper(db, doi)

        mock_state = MagicMock()
        mock_state.get_all_citation_edges.return_value = [
            {"source_doi": "10.0/a", "target_doi": "10.0/b", "resolution": "exact"},
            {"source_doi": "10.0/c", "target_doi": "10.0/d", "resolution": "exact"},
        ]

        # Wrap the real connection.execute so the 2nd CITES CREATE raises while
        # everything else (BEGIN, existence checks, 1st CREATE, ROLLBACK) runs
        # for real against Kuzu.
        real_execute = db._conn.execute
        cites_creates = {"n": 0}

        def flaky_execute(query, *args, **kwargs):
            if "CREATE (s)-[r:CITES" in query:
                cites_creates["n"] += 1
                if cites_creates["n"] == 2:
                    raise RuntimeError("simulated mid-loop CITES failure")
            return real_execute(query, *args, **kwargs)

        db._conn.execute = flaky_execute  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="simulated mid-loop"):
                mirror_citations(db, mock_state)
        finally:
            db._conn.execute = real_execute  # type: ignore[assignment]

        # The first edge was created inside the transaction but the batch
        # rolled back, so NO CITES edges may remain.
        assert _count_cites(db) == 0, "ROLLBACK must remove all partial CITES edges"


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


# ---------------------------------------------------------------------------
# E2c: graph-path resolution (config-driven with fallback)
# ---------------------------------------------------------------------------
class TestGraphPathResolution:
    def test_configured_graph_path_returns_persist_dir(self, monkeypatch):
        """_configured_graph_path reads retrieval.graph_db.persist_dir."""
        from types import SimpleNamespace

        import lit_monitor.graph.import_citations as ic

        cfg = SimpleNamespace(
            retrieval=SimpleNamespace(graph_db={"persist_dir": "/tmp/custom.kuzu"})
        )
        monkeypatch.setattr("lit_monitor.core.config.get_config", lambda: cfg)
        assert ic._configured_graph_path() == "/tmp/custom.kuzu"

    def test_configured_graph_path_with_production_namespace_shape(self, monkeypatch):
        """Fidelity: production builds graph_db as a config _Namespace (dot-access
        with .get), not a plain dict — exercise that exact shape too."""
        from types import SimpleNamespace

        import lit_monitor.graph.import_citations as ic
        from lit_monitor.core.config import _Namespace

        cfg = SimpleNamespace(
            retrieval=SimpleNamespace(
                graph_db=_Namespace({"persist_dir": "/ns/graph.kuzu"})
            )
        )
        monkeypatch.setattr("lit_monitor.core.config.get_config", lambda: cfg)
        assert ic._configured_graph_path() == "/ns/graph.kuzu"

    def test_configured_graph_path_none_when_block_absent(self, monkeypatch):
        """A pre-graph config (no graph_db block) yields None, not a crash."""
        from types import SimpleNamespace

        import lit_monitor.graph.import_citations as ic

        cfg = SimpleNamespace(retrieval=SimpleNamespace(graph_db=None))
        monkeypatch.setattr("lit_monitor.core.config.get_config", lambda: cfg)
        assert ic._configured_graph_path() is None

    def test_configured_graph_path_none_on_config_error(self, monkeypatch):
        """Any failure loading config falls through to None (→ default path)."""
        import lit_monitor.graph.import_citations as ic

        def boom():
            raise RuntimeError("config unavailable")

        monkeypatch.setattr("lit_monitor.core.config.get_config", boom)
        assert ic._configured_graph_path() is None

    def test_safe_graph_db_path_precedence(self, monkeypatch):
        """Precedence: explicit persist_dir > configured path > default literal."""
        import lit_monitor.graph.import_citations as ic

        captured: dict[str, str] = {}

        class _FakeGraphDB:
            def __init__(self, persist_dir: str) -> None:
                captured["dir"] = persist_dir

        monkeypatch.setattr("lit_monitor.graph.GraphDB", _FakeGraphDB)
        monkeypatch.setattr(ic, "_configured_graph_path", lambda: "/from/config.kuzu")

        ic.safe_graph_db(persist_dir="/explicit.kuzu")
        assert captured["dir"] == "/explicit.kuzu"  # explicit wins

        ic.safe_graph_db()
        assert captured["dir"] == "/from/config.kuzu"  # config used when no explicit

        monkeypatch.setattr(ic, "_configured_graph_path", lambda: None)
        ic.safe_graph_db()
        assert captured["dir"] == ic._DEFAULT_GRAPH_PATH  # default when neither
