"""
Unit tests for lit-monitor schema migrations.

Part 1 — state.db (SQLite) migration: papers.graph_indexed column (G1).
Part 2 — KuzuDB graph schema migration: v1 → v2 provenance columns (G14).
"""
from __future__ import annotations

import sqlite3

import pytest

from scripts.core.state_db import StateDB


def _column_info(db_path: str, table: str) -> dict[str, dict]:
    """Return column metadata from PRAGMA table_info as {name: {type, dflt_value, ...}}."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    # Each row: (cid, name, type, notnull, dflt_value, pk)
    return {r[1]: {"type": r[2], "dflt_value": r[4]} for r in rows}


@pytest.mark.unit
class TestGraphIndexedMigration:
    """state.db additive migration adds papers.graph_indexed."""

    def test_column_exists_after_init(self, tmp_path):
        """graph_indexed column must exist in the papers table after StateDB init."""
        db_path = str(tmp_path / "state.db")
        StateDB(db_path)

        cols = _column_info(db_path, "papers")
        assert "graph_indexed" in cols, (
            "papers.graph_indexed missing — additive migration did not run"
        )

    def test_column_has_default_zero(self, tmp_path):
        """graph_indexed column default must be 0 (INTEGER DEFAULT 0)."""
        db_path = str(tmp_path / "state.db")
        StateDB(db_path)

        cols = _column_info(db_path, "papers")
        col = cols["graph_indexed"]
        # SQLite stores the default as a string "0" in PRAGMA table_info.
        assert col["dflt_value"] == "0", (
            f"Expected DEFAULT 0, got: {col['dflt_value']!r}"
        )

    def test_reopen_does_not_raise(self, tmp_path):
        """Re-opening a DB that already has graph_indexed must not raise."""
        db_path = str(tmp_path / "state.db")
        StateDB(db_path)
        # Second open — additive migration skips because _column_exists returns True.
        StateDB(db_path)

    def test_new_paper_row_defaults_to_zero(self, tmp_path):
        """A freshly inserted paper row has graph_indexed = 0 by default."""
        db_path = str(tmp_path / "state.db")
        db = StateDB(db_path)
        db.upsert_paper({"doi": "10.1234/test.g1", "title": "G1 test paper"})

        row = db.get_paper("10.1234/test.g1")
        assert row is not None
        assert row["graph_indexed"] == 0, (
            f"Expected graph_indexed=0, got {row['graph_indexed']!r}"
        )

    def test_migration_on_existing_db_without_column(self, tmp_path):
        """Migration adds graph_indexed to a DB that was created without it.

        Simulates upgrading from an older schema by creating the papers table
        manually without graph_indexed, then opening StateDB which should run
        the additive migration and add it.
        """
        db_path = str(tmp_path / "state.db")

        # Create the table without graph_indexed — simulates a pre-G1 state.db.
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS papers ("
            "doi TEXT PRIMARY KEY, title TEXT, status TEXT DEFAULT 'pending'"
            ")"
        )
        conn.commit()
        conn.close()

        # Open via StateDB — _init_schema will run CREATE TABLE IF NOT EXISTS (no-op)
        # and then the additive_migrations list should add graph_indexed.
        StateDB(db_path)

        cols = _column_info(db_path, "papers")
        assert "graph_indexed" in cols, (
            "Additive migration failed to add graph_indexed to a pre-existing table"
        )


# ---------------------------------------------------------------------------
# Part 2 — KuzuDB graph schema: v1 → v2 provenance columns (G14)
# ---------------------------------------------------------------------------

def _build_v1_kuzu_db(db_path: str) -> tuple:  # type: ignore[type-arg]
    """Create a v1-schema KuzuDB database and return (db, conn).

    Uses _DDL_V1_STATEMENTS from migrations.py — the verbatim G1 DDL snapshot —
    so this test does not rely on reading old Git commits.
    """
    import kuzu

    from scripts.graph.migrations import _DDL_V1_STATEMENTS

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    for ddl in _DDL_V1_STATEMENTS:
        conn.execute(ddl)
    return db, conn


@pytest.mark.unit
class TestKuzuSchemaMigrationV1ToV2:
    """G14: migrate_v1_to_v2 adds provenance columns while preserving edges."""

    def test_migration_adds_provenance_columns_to_mentions(self, tmp_path):
        """After migrate_v1_to_v2, MENTIONS edges have the three new columns populated."""
        from scripts.graph.migrations import migrate_v1_to_v2

        db_path = str(tmp_path / "v1.kuzu")
        db, conn = _build_v1_kuzu_db(db_path)

        # Insert a Paper, an Entity, and a MENTIONS edge using the v1 schema
        # (no provenance columns yet).
        conn.execute(
            "CREATE (p:Paper {doi: '10.0/v1', title: 'V1 Paper', year: 2023, journal: 'J'})"
        )
        conn.execute(
            "CREATE (e:Entity {canonical_id: 'protein_a', type: 'reagent', "
            "surface: 'Protein A'})"
        )
        conn.execute(
            "MATCH (p:Paper {doi: '10.0/v1'}), (e:Entity {canonical_id: 'protein_a'}) "
            "CREATE (p)-[:MENTIONS {source: 'llm', surface: 'Protein A', "
            "field: 'methods_summary', confidence: 0.9, span_start: 5, span_end: 14}]->(e)"
        )

        # Confirm pre-migration: querying the new columns raises a binder error.
        with pytest.raises(RuntimeError, match="extracted_at"):
            conn.execute(
                "MATCH ()-[m:MENTIONS]->() RETURN m.extracted_at LIMIT 1"
            )

        # Run the migration.
        migrate_v1_to_v2(conn)

        # Now the new columns must exist and carry defaults.
        result = conn.execute(
            "MATCH ()-[m:MENTIONS]->() "
            "RETURN m.confidence, m.extracted_at, m.prompt_version LIMIT 1"
        )
        row = result.get_next()
        # The v1 edge had confidence=0.9; after ALTER the column is replaced
        # (kuzu ADD adds a new column, leaving the old value).  The new
        # confidence column has DEFAULT 1.0 — but note: v1 already had a
        # `confidence` column.  migrate_v1_to_v2 will hit the "already exists"
        # path for confidence and skip it, leaving the original 0.9 intact.
        assert row[0] == 0.9, f"Existing confidence must be preserved: {row[0]!r}"
        assert row[1] is not None, "extracted_at must be set after migration"
        assert row[2] == "phase1.0", f"prompt_version must default to 'phase1.0': {row[2]!r}"

    def test_migration_preserves_existing_edges(self, tmp_path):
        """Old MENTIONS edges survive v1→v2 migration with original field values intact."""
        from scripts.graph.migrations import migrate_v1_to_v2

        db_path = str(tmp_path / "v1_preserve.kuzu")
        db, conn = _build_v1_kuzu_db(db_path)

        conn.execute(
            "CREATE (p:Paper {doi: '10.0/old', title: 'Old Paper', year: 2021, journal: 'Z'})"
        )
        conn.execute(
            "CREATE (e:Entity {canonical_id: 'size_exclusion', type: 'method', "
            "surface: 'size exclusion'})"
        )
        conn.execute(
            "MATCH (p:Paper {doi: '10.0/old'}), (e:Entity {canonical_id: 'size_exclusion'}) "
            "CREATE (p)-[:MENTIONS {source: 'llm', surface: 'size exclusion', "
            "field: 'methods_summary', confidence: 0.85, span_start: 10, span_end: 25}]->(e)"
        )

        migrate_v1_to_v2(conn)

        # Original MENTIONS data must still be there.
        result = conn.execute(
            "MATCH ()-[m:MENTIONS]->() "
            "RETURN m.source, m.surface, m.field, m.span_start, m.span_end LIMIT 1"
        )
        row = result.get_next()
        assert row[0] == "llm"
        assert row[1] == "size exclusion"
        assert row[2] == "methods_summary"
        assert row[3] == 10
        assert row[4] == 25

    def test_migration_idempotent_second_run(self, tmp_path):
        """Running migrate_v1_to_v2 twice on the same DB does not raise."""
        from scripts.graph.migrations import migrate_v1_to_v2

        db_path = str(tmp_path / "v1_idem.kuzu")
        db, conn = _build_v1_kuzu_db(db_path)

        migrate_v1_to_v2(conn)
        # Second call must not raise (columns already exist → skip).
        migrate_v1_to_v2(conn)

    def test_apply_migrations_dispatcher_v1_to_current(self, tmp_path):
        """apply_migrations transitions a v1 DB to SCHEMA_VERSION and returns it."""
        from scripts.graph.migrations import SCHEMA_VERSION, apply_migrations

        db_path = str(tmp_path / "v1_dispatch.kuzu")
        db, conn = _build_v1_kuzu_db(db_path)

        new_version = apply_migrations(conn, current_version=1)
        assert new_version == SCHEMA_VERSION, (
            f"Expected version {SCHEMA_VERSION}, got {new_version}"
        )

    def test_apply_migrations_no_op_at_current_version(self, tmp_path):
        """apply_migrations is a no-op and returns the same version when already current."""
        from scripts.graph import GraphDB
        from scripts.graph.migrations import SCHEMA_VERSION, apply_migrations

        db_path = str(tmp_path / "current.kuzu")
        GraphDB(persist_dir=db_path)  # fresh init at v2

        import kuzu
        db2 = kuzu.Database(db_path)
        conn2 = kuzu.Connection(db2)

        result = apply_migrations(conn2, current_version=SCHEMA_VERSION)
        assert result == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Part 3 — KuzuDB graph schema: Phase 3 R1 — EXTENDS + CONTRADICTS (v2 → v3)
# ---------------------------------------------------------------------------


def _table_names(conn) -> set[str]:  # type: ignore[type-arg]
    """Return the set of table names from CALL show_tables()."""
    result = conn.execute("CALL show_tables() RETURN *")
    names: set[str] = set()
    while result.has_next():
        row = result.get_next()
        # row[1] is the table name in KuzuDB 0.11.x
        names.add(row[1])
    return names


@pytest.mark.unit
class TestExtendsContradictsTablesV3:
    """R1: EXTENDS and CONTRADICTS Paper→Paper REL tables are created in v3 schema."""

    def test_apply_schema_creates_both_new_rel_tables(self, tmp_path):
        """R1: fresh GraphDB has EXTENDS and CONTRADICTS as Paper→Paper REL tables."""
        from scripts.graph import GraphDB

        db = GraphDB(persist_dir=str(tmp_path / "r1.kuzu"))
        names = _table_names(db._conn)
        assert "EXTENDS" in names, f"EXTENDS missing from {names}"
        assert "CONTRADICTS" in names, f"CONTRADICTS missing from {names}"

    def test_apply_schema_idempotent_with_new_tables(self, tmp_path):
        """R1: re-running apply_schema on a v3-fresh DB doesn't raise."""
        from scripts.graph import GraphDB
        from scripts.graph.migrations import apply_schema

        db = GraphDB(persist_dir=str(tmp_path / "r1_idem.kuzu"))
        # Re-apply schema (already applied in __init__) — must not raise.
        apply_schema(db._conn)
        # Confirm tables still exist.
        names = _table_names(db._conn)
        assert "EXTENDS" in names
        assert "CONTRADICTS" in names

    def test_extends_edge_can_be_created_with_provenance(self, tmp_path):
        """R1: EXTENDS edge accepts confidence + extracted_at + prompt_version per G14."""
        from scripts.graph import GraphDB

        db = GraphDB(persist_dir=str(tmp_path / "r1_ext.kuzu"))
        conn = db._conn
        conn.execute(
            "CREATE (p:Paper {doi: '10.0/a', title: 'A', year: 2024, journal: 'X'})"
        )
        conn.execute(
            "CREATE (p:Paper {doi: '10.0/b', title: 'B', year: 2024, journal: 'X'})"
        )
        conn.execute(
            "MATCH (s:Paper {doi: '10.0/a'}), (t:Paper {doi: '10.0/b'}) "
            "CREATE (s)-[r:EXTENDS {evidence: 'paper A builds on B'}]->(t)"
        )
        res = conn.execute(
            "MATCH ()-[r:EXTENDS]->() "
            "RETURN r.evidence, r.confidence, r.prompt_version LIMIT 1"
        )
        assert res.has_next()
        row = res.get_next()
        assert row[0] == "paper A builds on B"
        assert row[1] == 1.0, f"confidence default should be 1.0, got {row[1]!r}"
        assert row[2] == "phase1.0", f"prompt_version default wrong: {row[2]!r}"

    def test_contradicts_edge_can_be_created(self, tmp_path):
        """R1: CONTRADICTS edge is insertable and queryable with count check."""
        from scripts.graph import GraphDB

        db = GraphDB(persist_dir=str(tmp_path / "r1_con.kuzu"))
        conn = db._conn
        conn.execute(
            "CREATE (p:Paper {doi: '10.0/a', title: 'A', year: 2024, journal: 'X'})"
        )
        conn.execute(
            "CREATE (p:Paper {doi: '10.0/b', title: 'B', year: 2024, journal: 'X'})"
        )
        conn.execute(
            "MATCH (s:Paper {doi: '10.0/a'}), (t:Paper {doi: '10.0/b'}) "
            "CREATE (s)-[r:CONTRADICTS {evidence: 'A disputes B'}]->(t)"
        )
        res = conn.execute(
            "MATCH ()-[r:CONTRADICTS]->() RETURN count(r) AS n"
        )
        assert res.has_next()
        assert int(res.get_next()[0]) == 1


@pytest.mark.unit
class TestSchemaVersionV3:
    """R1: SCHEMA_VERSION >= 3; migration dispatcher handles v1/v2→v3 (now v4)."""

    def test_schema_version_is_at_least_4(self):
        """SCHEMA_VERSION floor is 4 — CITES parity must stay present.

        Bundle D (v0.9) bumped this to 5 to add Paper.authors. The floor
        stays at 4 to encode 'Phase 4 CITES fix is permanently in', while
        the assertion now matches the test name's 'at_least_4' wording.
        """
        from scripts.graph.migrations import SCHEMA_VERSION

        assert SCHEMA_VERSION >= 4, (
            f"Expected SCHEMA_VERSION >= 4 (Phase 4 CITES fix), got {SCHEMA_VERSION}"
        )

    def test_migrate_v2_to_v3_adds_new_tables(self, tmp_path):
        """R1: migrate_v2_to_v3 on a v2-shaped DB adds EXTENDS + CONTRADICTS."""
        from scripts.graph import GraphDB
        from scripts.graph.migrations import migrate_v2_to_v3

        db = GraphDB(persist_dir=str(tmp_path / "r1_mig.kuzu"))
        conn = db._conn
        # Drop v3-only tables to simulate a v2 snapshot.
        for name in ("EXTENDS", "CONTRADICTS"):
            try:
                conn.execute(f"DROP TABLE {name}")
            except Exception:
                pass
        # Confirm they're gone.
        before = _table_names(conn)
        assert "EXTENDS" not in before
        assert "CONTRADICTS" not in before

        # Run the v2→v3 migration.
        new_version = migrate_v2_to_v3(conn)
        assert new_version == 3

        # Confirm both tables are back.
        after = _table_names(conn)
        assert "EXTENDS" in after, f"EXTENDS missing after migrate_v2_to_v3: {after}"
        assert "CONTRADICTS" in after, f"CONTRADICTS missing: {after}"

    def test_apply_migrations_v1_reaches_current_version(self, tmp_path):
        """apply_migrations from v1 chains all steps and returns SCHEMA_VERSION."""
        from scripts.graph.migrations import SCHEMA_VERSION, apply_migrations

        _, conn = _build_v1_kuzu_db(str(tmp_path / "v1_to_current.kuzu"))
        new_version = apply_migrations(conn, current_version=1)
        assert new_version == SCHEMA_VERSION

    def test_apply_migrations_v2_reaches_current_version(self, tmp_path):
        """apply_migrations from v2 chains v2→v3→v4 and returns SCHEMA_VERSION."""
        import kuzu

        from scripts.graph import GraphDB
        from scripts.graph.migrations import (  # noqa: F401
            SCHEMA_VERSION,
            apply_migrations,
            migrate_v2_to_v3,
        )

        db = GraphDB(persist_dir=str(tmp_path / "v2_to_current.kuzu"))
        # Simulate v2 by dropping v3 tables.
        for name in ("EXTENDS", "CONTRADICTS"):
            try:
                db._conn.execute(f"DROP TABLE {name}")
            except Exception:
                pass

        # Open a fresh connection to the same on-disk DB (simulate re-open at v2).
        db2 = kuzu.Database(str(tmp_path / "v2_to_current.kuzu"))
        conn2 = kuzu.Connection(db2)

        new_version = apply_migrations(conn2, current_version=2)
        assert new_version == SCHEMA_VERSION
        after = _table_names(conn2)
        assert "EXTENDS" in after
        assert "CONTRADICTS" in after
