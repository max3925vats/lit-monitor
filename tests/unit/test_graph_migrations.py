"""
Unit tests for the G1 state.db migration — papers.graph_indexed column.

Tests verify:
- After StateDB init, papers table has a graph_indexed column with DEFAULT 0.
- Re-opening an existing DB (which already has graph_indexed) does not raise.
- A fresh paper row gets graph_indexed = 0 by default.
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
