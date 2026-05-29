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
class TestGraphDBImportError:
    """GraphDB raises ImportError when kuzu is not installed."""

    def test_import_error_without_kuzu(self, tmp_path, monkeypatch):
        """When kuzu is absent, GraphDB.__init__ raises ImportError with install hint."""
        # Use monkeypatch to block kuzu import inside GraphDB
        # We do this by temporarily removing kuzu from sys.modules and blocking it.
        kuzu_backup = sys.modules.pop("kuzu", None)

        # Also remove graph.db so it re-evaluates lazy import
        for mod_name in list(sys.modules.keys()):
            if "scripts.graph" in mod_name:
                del sys.modules[mod_name]

        # Block kuzu from being importable
        sys.modules["kuzu"] = None  # type: ignore[assignment]

        try:
            from scripts.graph.db import GraphDB as FreshGraphDB

            with pytest.raises(ImportError, match="uv sync --extra graph"):
                FreshGraphDB(persist_dir=str(tmp_path / "graph.kuzu"))
        finally:
            # Restore kuzu
            if kuzu_backup is not None:
                sys.modules["kuzu"] = kuzu_backup
            else:
                sys.modules.pop("kuzu", None)
            # Reload scripts.graph to restore correct state
            for mod_name in list(sys.modules.keys()):
                if "scripts.graph" in mod_name:
                    del sys.modules[mod_name]
