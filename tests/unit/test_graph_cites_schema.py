"""Regression test for the CITES resolution→evidence schema fix.

Pre-fix symptom: GraphDB.add_paper with a CITES relationship raised
RuntimeError because _upsert_typed_edge wrote 'evidence' but CITES had
'resolution'. This test pins the contract: CITES must accept the
generic typed-edge payload like every other Paper→Paper REL
(EXTENDS, CONTRADICTS, COMPARES_TO).
"""
from __future__ import annotations

import pytest

from lit_monitor.graph import GraphDB


@pytest.fixture
def fresh_db(tmp_path):
    """Fresh KuzuDB graph for CITES schema regression tests."""
    return GraphDB(persist_dir=str(tmp_path / "cites.kuzu"))


def _get_cites_columns(db: GraphDB) -> list[str]:
    """Return column names of the CITES REL TABLE via table_info()."""
    res = db._conn.execute("CALL table_info('CITES') RETURN *")
    cols: list[str] = []
    while res.has_next():
        row = res.get_next()
        # row layout: (property_id, name, type, default, direction)
        cols.append(row[1])
    return cols


@pytest.mark.unit
class TestCitesSchemaParity:
    """CITES column must be 'evidence', matching EXTENDS / CONTRADICTS / COMPARES_TO."""

    def test_cites_column_is_evidence_not_resolution(self, fresh_db: GraphDB) -> None:
        """After the fix, CITES uses 'evidence' like all other typed Paper→Paper RELs."""
        cols = _get_cites_columns(fresh_db)
        assert "evidence" in cols, (
            f"Expected 'evidence' column in CITES, got: {cols}"
        )
        assert "resolution" not in cols, (
            f"'resolution' should be absent post-fix, got: {cols}"
        )

    def test_add_paper_with_cites_relationship_does_not_raise(
        self, fresh_db: GraphDB
    ) -> None:
        """Regression: add_paper(..., relationships=[('CITES', ...)]) must not raise.

        Pre-fix this raised a Kuzu RuntimeError because _upsert_typed_edge
        writes 'evidence' but the CITES table only had 'resolution'.
        """
        from lit_monitor.graph.relationship_extractor import RelationshipTuple

        fresh_db.add_paper(
            doi="10.9999/cited",
            paper_metadata={"title": "Cited Paper", "year": 2023, "journal": "J"},
            entities=[],
            relationships=[],
        )
        # The act under test — must not raise:
        fresh_db.add_paper(
            doi="10.9999/citing",
            paper_metadata={"title": "Citing Paper", "year": 2024, "journal": "J"},
            entities=[],
            relationships=[
                RelationshipTuple(
                    source_doi="10.9999/citing",
                    predicate="CITES",
                    target_id="10.9999/cited",
                    target_kind="Paper",
                    evidence="confirms findings of Cited Paper",
                    confidence=0.9,
                    field="abstract",
                ),
            ],
        )

    def test_cites_edge_readable_after_add_paper(self, fresh_db: GraphDB) -> None:
        """A CITES edge written via add_paper is visible in a MATCH query."""
        from lit_monitor.graph.relationship_extractor import RelationshipTuple

        fresh_db.add_paper(
            doi="10.8888/ref",
            paper_metadata={"title": "Reference", "year": 2022, "journal": "J"},
            entities=[],
            relationships=[],
        )
        fresh_db.add_paper(
            doi="10.8888/src",
            paper_metadata={"title": "Source", "year": 2023, "journal": "J"},
            entities=[],
            relationships=[
                RelationshipTuple(
                    source_doi="10.8888/src",
                    predicate="CITES",
                    target_id="10.8888/ref",
                    target_kind="Paper",
                    evidence="builds on reference work",
                    confidence=1.0,
                    field="references",
                ),
            ],
        )
        # Read back via Cypher — should match exactly 1 edge.
        res = fresh_db._conn.execute(
            "MATCH (a:Paper {doi: '10.8888/src'})-[:CITES]->(b:Paper {doi: '10.8888/ref'})"
            " RETURN count(*)"
        )
        assert res.has_next()
        count = res.get_next()[0]
        assert count == 1, f"Expected 1 CITES edge, got {count}"
