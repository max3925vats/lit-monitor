"""A3: execute_cypher + render_rows tests."""
from __future__ import annotations

import json
import logging

import pytest

from scripts.graph import GraphDB
from scripts.graph.ask import execute_cypher, render_rows


@pytest.fixture
def db(tmp_path):
    g = GraphDB(persist_dir=str(tmp_path / "a3.kuzu"))
    # Seed two papers so MATCH (p:Paper) returns rows.
    g.add_paper(
        doi="10.0/a",
        entities=[],
        relationships=[],
        paper_metadata={"title": "Alpha", "year": 2024, "journal": "J"},
    )
    g.add_paper(
        doi="10.0/b",
        entities=[],
        relationships=[],
        paper_metadata={"title": "Beta", "year": 2023, "journal": "J"},
    )
    return g


class TestExecuteCypher:
    def test_happy_path_returns_dicts(self, db):
        rows = execute_cypher(db, "MATCH (p:Paper) RETURN p.doi AS doi, p.title AS title")
        assert isinstance(rows, list)
        assert len(rows) == 2
        dois = {r["doi"] for r in rows}
        assert dois == {"10.0/a", "10.0/b"}

    def test_zero_rows_returns_empty_list(self, db):
        rows = execute_cypher(db, "MATCH (p:Paper) WHERE p.year = 1999 RETURN p.doi")
        assert rows == []

    def test_invalid_cypher_returns_none(self, db, caplog):
        with caplog.at_level(logging.INFO):
            rows = execute_cypher(db, "MATCH NOT_VALID_CYPHER ;;; ")
        assert rows is None
        assert any(
            "A3" in r.message or "execute" in r.message.lower()
            for r in caplog.records
        )

    def test_never_raises(self, db):
        # A wild non-Cypher string must not crash.
        rows = execute_cypher(db, "garbage")
        assert rows is None

    def test_row_cap_enforced(self, db):
        # row_cap=1 should produce at most 1 row even though 2 match.
        rows = execute_cypher(db, "MATCH (p:Paper) RETURN p.doi", row_cap=1)
        assert rows is not None
        assert len(rows) == 1

    def test_jsonable_coercion(self, db):
        # Kuzu supports date literals — result should be JSON-serializable.
        rows = execute_cypher(db, "RETURN date('2024-01-15') AS d, 42 AS n")
        assert rows is not None
        # Must not raise
        json.dumps(rows)


class TestRenderRows:
    def test_empty_returns_placeholder(self):
        assert render_rows([]) == "_(no results)_"

    def test_basic_two_column_table(self):
        rows = [{"doi": "10.0/a", "title": "Alpha"}, {"doi": "10.0/b", "title": "Beta"}]
        out = render_rows(rows)
        assert "| doi | title |" in out
        assert "10.0/a" in out
        assert "Alpha" in out
        assert "10.0/b" in out
        assert "Beta" in out
        # Has the markdown separator
        assert "---" in out

    def test_max_rows_truncation(self):
        rows = [{"doi": f"10.0/{i}", "title": f"P{i}"} for i in range(30)]
        out = render_rows(rows, max_rows=5)
        # Truncation footnote
        assert "5 of 30" in out
        # Only first 5 rows present
        assert "10.0/0" in out
        assert "10.0/4" in out
        assert "10.0/5" not in out

    def test_long_cell_truncation(self):
        rows = [{"x": "a" * 200}]
        out = render_rows(rows)
        assert "…" in out  # ellipsis character
        # Full 200-char string should NOT appear in output
        assert "a" * 100 not in out

    def test_none_values_render_empty(self):
        rows = [{"doi": "10.0/a", "title": None}]
        out = render_rows(rows)
        assert "10.0/a" in out
        # The row line should not contain literal "None"
        lines = [ln for ln in out.split("\n") if "10.0/a" in ln]
        assert lines, "expected a row containing doi"
        assert "None" not in lines[0]

    def test_keys_union_when_rows_differ(self):
        """A3: when rows have different keys, the header is the union (stable order)."""
        rows = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
        out = render_rows(rows)
        assert "a" in out and "b" in out and "c" in out


# ---------------------------------------------------------------------------
# AskResult dataclass shape
#
# Relocated from tests/integration/test_v09_e2e.py (Audit-2 A2): a trivial
# dataclass-field check that needs no live services. The shape-parity contract
# (CLI + HTTP see the same AskResult) is worth pinning, so it runs here in the
# unit suite rather than being gated behind a live-Ollama integration fixture.
# ---------------------------------------------------------------------------


class TestAskResultShape:
    def test_ask_result_dataclass_fields(self):
        """AskResult must expose cypher, rows, rendered, prose fields."""
        from scripts.graph.ask import AskResult

        r = AskResult(
            cypher="MATCH (p) RETURN p",
            rows=[{"doi": "10.0/a"}],
            rendered="| doi |\n|---|\n| 10.0/a |",
            prose="One paper found.",
        )
        assert r.cypher
        assert r.rows
        assert r.rendered
        assert r.prose
