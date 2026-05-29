"""
Unit tests for GraphDB — KuzuDB foundation (G1).

Tests verify:
- GraphDB creates its persist directory when missing.
- After init, _conn is a live kuzu.Connection.
- Re-instantiation with the same directory is a no-op (idempotent DDL).
- Without kuzu installed, importing scripts.graph succeeds but instantiating
  GraphDB raises ImportError with the prescribed install hint.
- All 10 expected node/rel tables exist after init.
"""
from __future__ import annotations

import sys

import pytest  # noqa: I001

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tables(conn) -> set[str]:
    """Return the set of table names in the KuzuDB database via CALL show_tables().

    In KuzuDB 0.11.x, each result row is a list:
    [id (int), name (str), type (str), database (str), comment (str)]
    """
    result = conn.execute("CALL show_tables() RETURN *")
    names: set[str] = set()
    while result.has_next():
        row = result.get_next()
        # row[1] is the table name; row[0] is the numeric table id.
        names.add(row[1])
    return names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGraphDBInit:
    """GraphDB init behaviour."""

    def test_creates_parent_directory(self, tmp_path):
        """GraphDB creates the parent directory of persist_dir if missing.

        KuzuDB stores data as a file at the given path; the parent dir must
        exist before kuzu.Database() is called.  GraphDB ensures this.
        """
        # persist_dir = tmp_path/subdir/graph.kuzu — 'subdir' does not exist yet.
        target = tmp_path / "subdir" / "graph.kuzu"
        assert not (tmp_path / "subdir").exists()

        from scripts.graph import GraphDB
        GraphDB(persist_dir=str(target))

        # Parent dir was created and KuzuDB wrote its file at the target path.
        assert (tmp_path / "subdir").is_dir()
        assert target.exists()  # kuzu created the database file

    def test_conn_is_live(self, tmp_path):
        """After init, _conn is a live kuzu Connection that can execute queries."""
        import kuzu

        from scripts.graph import GraphDB

        g = GraphDB(persist_dir=str(tmp_path / "graph.kuzu"))

        assert isinstance(g._conn, kuzu.Connection)
        # A trivial query must succeed
        result = g._conn.execute("RETURN 1 AS x")
        assert result.has_next()
        row = result.get_next()
        assert row[0] == 1

    def test_all_tables_created(self, tmp_path):
        """All 10 node and rel tables from the G1 DDL must exist after init."""
        from scripts.graph import GraphDB

        g = GraphDB(persist_dir=str(tmp_path / "graph.kuzu"))
        tables = _tables(g._conn)

        expected = {
            "Paper",
            "Entity",
            "MENTIONS",
            "CITES",
            "COMPARES_TO",
            "DEPENDS_ON",
            "PROPOSES",
            "LIMITED_BY",
            "INTRODUCES",
            "RAISES_QUESTION",
        }
        assert expected <= tables, f"Missing tables: {expected - tables}"

    def test_reinit_is_idempotent(self, tmp_path):
        """Re-instantiating GraphDB on an already-initialised directory is a no-op."""
        from scripts.graph import GraphDB

        db_path = str(tmp_path / "graph.kuzu")
        GraphDB(persist_dir=db_path)  # first init
        # Second init must not raise — IF NOT EXISTS guards prevent DDL conflicts.
        g2 = GraphDB(persist_dir=db_path)
        tables = _tables(g2._conn)
        assert "Paper" in tables
        assert "MENTIONS" in tables


@pytest.mark.unit
class TestGraphDB:
    """GraphDB context-manager and resource lifecycle tests."""

    def test_context_manager_closes_handles(self, tmp_path):
        """GraphDB used as a context manager releases its kuzu handles on exit."""
        from scripts.graph import GraphDB

        persist = tmp_path / "ctx.kuzu"
        with GraphDB(persist_dir=str(persist)) as g:
            assert g._conn is not None
            assert g._db is not None
        # After __exit__, handles must be released.
        assert g._conn is None
        assert g._db is None


@pytest.mark.unit
class TestGraphDBEdgeProperties:
    """G14: REL TABLE edges carry confidence, extracted_at, prompt_version defaults."""

    def test_mentions_edge_has_default_confidence_extracted_at_prompt_version(
        self, tmp_path
    ):
        """G14: every MENTIONS edge defaults to confidence=1.0, prompt_version='phase1.0',
        extracted_at=now (set by DB DEFAULT)."""
        from scripts.graph import GraphDB

        db = GraphDB(persist_dir=str(tmp_path / "g14.kuzu"))
        conn = db._conn
        # Insert a Paper, an Entity, and a MENTIONS edge without the 3 new props.
        conn.execute(
            "CREATE (p:Paper {doi: '10.0/a', title: 'A', year: 2024, journal: 'X'})"
        )
        conn.execute(
            "CREATE (e:Entity {canonical_id: 'ion_exchange', type: 'method', "
            "surface: 'ion exchange'})"
        )
        conn.execute(
            "MATCH (p:Paper {doi: '10.0/a'}), (e:Entity {canonical_id: 'ion_exchange'}) "
            "CREATE (p)-[:MENTIONS {source: 'schema', surface: 'ion exchange', "
            "field: 'methods_summary', span_start: 0, span_end: 0}]->(e)"
        )
        # Read back — the three new columns must be populated from DEFAULTs.
        result = conn.execute(
            "MATCH (:Paper)-[m:MENTIONS]->(:Entity) "
            "RETURN m.confidence, m.extracted_at, m.prompt_version LIMIT 1"
        )
        row = result.get_next()
        assert row[0] == 1.0, f"Expected confidence=1.0, got {row[0]!r}"
        assert row[1] is not None, "extracted_at must be set by DB DEFAULT (not None)"
        assert row[2] == "phase1.0", f"Expected prompt_version='phase1.0', got {row[2]!r}"

    def test_compares_to_edge_has_default_properties(self, tmp_path):
        """G14: Paper→Paper REL COMPARES_TO also carries the three default properties."""
        from scripts.graph import GraphDB

        db = GraphDB(persist_dir=str(tmp_path / "g14_pp.kuzu"))
        conn = db._conn
        conn.execute(
            "CREATE (p1:Paper {doi: '10.0/b', title: 'B', year: 2023, journal: 'Y'})"
        )
        conn.execute(
            "CREATE (p2:Paper {doi: '10.0/c', title: 'C', year: 2024, journal: 'Z'})"
        )
        conn.execute(
            "MATCH (p1:Paper {doi: '10.0/b'}), (p2:Paper {doi: '10.0/c'}) "
            "CREATE (p1)-[:COMPARES_TO {evidence: 'benchmarked on same dataset'}]->(p2)"
        )
        result = conn.execute(
            "MATCH (:Paper)-[r:COMPARES_TO]->(:Paper) "
            "RETURN r.confidence, r.extracted_at, r.prompt_version LIMIT 1"
        )
        row = result.get_next()
        assert row[0] == 1.0, f"Expected confidence=1.0, got {row[0]!r}"
        assert row[1] is not None, "extracted_at must be set by DB DEFAULT (not None)"
        assert row[2] == "phase1.0", f"Expected prompt_version='phase1.0', got {row[2]!r}"


@pytest.mark.unit
class TestGraphDBImportError:
    """GraphDB raises ImportError when kuzu is not installed."""

    def test_import_error_without_kuzu(self, tmp_path, monkeypatch):
        """Instantiating GraphDB without kuzu raises a clear ImportError."""
        # Block the kuzu import via monkeypatch so cleanup is automatic.
        monkeypatch.setitem(sys.modules, "kuzu", None)  # type: ignore[arg-type]
        # Purge any cached scripts.graph modules so the import inside __init__
        # re-runs against the blocked kuzu slot.
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("scripts.graph"):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)

        from scripts.graph.db import GraphDB as FreshGraphDB

        with pytest.raises(ImportError, match="uv sync --extra graph"):
            FreshGraphDB(persist_dir=str(tmp_path / "graph.kuzu"))

    def test_module_import_succeeds_without_kuzu(self, monkeypatch):
        """scripts.graph imports cleanly even when kuzu is absent."""
        monkeypatch.setitem(sys.modules, "kuzu", None)  # type: ignore[arg-type]
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("scripts.graph"):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)
        # Importing the package must NOT raise — only instantiation does.
        import scripts.graph  # noqa: F401
