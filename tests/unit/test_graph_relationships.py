"""R3: multi-source relationship merge invariant tests.

Pins the four R3 sub-invariants for typed predicate edges:
1. Same (source_doi, predicate, target_id) from two different sources
   (schema vs LLM) → 2 distinct edges, one per source.
2. Different target_id for the same predicate → 2 distinct edges.
3. Idempotent re-run: same exact inputs from the same source → still 1 edge
   per source (no duplicates on repeated ingest).
4. Phase 1 schema-only path remains untouched.
"""
from __future__ import annotations

import pytest


@pytest.mark.unit
class TestMultiSourceRelationshipMerge:
    """R3: typed-edge dedup key includes prompt_version so schema + LLM rows
    produce distinct edges when the (source, predicate, target) triple matches."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_db(self, tmp_path, name: str = "r3.kuzu"):
        """Return a fresh GraphDB and pre-create two Paper nodes + one Entity."""
        from scripts.graph import GraphDB

        db = GraphDB(persist_dir=str(tmp_path / name))
        conn = db._conn
        # Pre-create Paper nodes so typed-edge MATCHes succeed.
        conn.execute(
            "CREATE (p:Paper {doi: '10.0/a', title: 'A', year: 2024, journal: 'X'})"
        )
        conn.execute(
            "CREATE (p:Paper {doi: '10.0/b', title: 'B', year: 2024, journal: 'X'})"
        )
        # An entity for Paper→Entity predicates.
        conn.execute(
            "CREATE (e:Entity {canonical_id: 'novel method', type: 'method', "
            "surface: 'novel method'})"
        )
        return db

    def _count_edges(self, db, predicate: str) -> int:
        """Count all edges of the given predicate label."""
        res = db._conn.execute(
            f"MATCH ()-[r:{predicate}]->() RETURN count(r)"
        )
        return int(res.get_next()[0]) if res.has_next() else 0

    def _rel(self, **overrides):
        """Build a RelationshipTuple with sensible defaults; override as needed."""
        from scripts.graph.relationship_extractor import RelationshipTuple

        defaults = dict(
            source_doi="10.0/a",
            predicate="COMPARES_TO",
            target_id="10.0/b",
            target_kind="Paper",
            evidence="default evidence",
            confidence=1.0,
            field="comparison_to_prior",
        )
        defaults.update(overrides)
        return RelationshipTuple(**defaults)

    # ------------------------------------------------------------------
    # R3 invariant 1: schema + LLM same triple → 2 distinct edges
    # ------------------------------------------------------------------

    def test_schema_then_llm_produces_two_distinct_edges(self, tmp_path):
        """R3 inv-1: same (source_doi, predicate, target_id) from schema + LLM
        → 2 edges with distinct provenance, NOT merged into one."""
        db = self._make_db(tmp_path, "r3_inv1.kuzu")

        schema_rel = self._rel(
            evidence="from extraction.json comparison_to_prior",
            confidence=1.0,
            field="comparison_to_prior",  # schema source marker
        )
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[schema_rel],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
            prompt_version="phase1.0_schema",
        )

        llm_rel = self._rel(
            evidence="from R2 LLM extraction",
            confidence=0.85,
            field="llm_extracted",  # LLM source marker
        )
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[llm_rel],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
            prompt_version="phase3.0_llm",
        )

        # Two distinct edges — one per source.
        assert self._count_edges(db, "COMPARES_TO") == 2

    def test_schema_and_llm_edges_carry_correct_prompt_version(self, tmp_path):
        """R3 inv-1 extension: each edge's prompt_version reflects its source."""
        db = self._make_db(tmp_path, "r3_pv.kuzu")

        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[self._rel(field="comparison_to_prior", confidence=1.0)],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
            prompt_version="phase1.0_schema",
        )
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[self._rel(field="llm_extracted", confidence=0.85)],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
            prompt_version="phase3.0_llm",
        )

        res = db._conn.execute(
            "MATCH ()-[r:COMPARES_TO]->() RETURN r.prompt_version ORDER BY r.prompt_version"
        )
        versions = []
        while res.has_next():
            versions.append(res.get_next()[0])

        assert versions == ["phase1.0_schema", "phase3.0_llm"]

    # ------------------------------------------------------------------
    # R3 invariant 2: different targets, same predicate → 2 edges
    # ------------------------------------------------------------------

    def test_different_targets_same_predicate_two_edges(self, tmp_path):
        """R3 inv-2: schema-source predicate with two different targets → 2 edges."""
        db = self._make_db(tmp_path, "r3_inv2.kuzu")
        # Add a third paper as a second target.
        db._conn.execute(
            "CREATE (p:Paper {doi: '10.0/c', title: 'C', year: 2024, journal: 'X'})"
        )

        rel_to_b = self._rel(target_id="10.0/b", evidence="vs B")
        rel_to_c = self._rel(target_id="10.0/c", evidence="vs C")

        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[rel_to_b, rel_to_c],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
            prompt_version="phase1.0_schema",
        )

        assert self._count_edges(db, "COMPARES_TO") == 2

    # ------------------------------------------------------------------
    # R3 invariant 3: idempotent re-run (same source twice → 1 edge)
    # ------------------------------------------------------------------

    def test_idempotent_rerun_schema_source(self, tmp_path):
        """R3 inv-3: schema source running twice → still 1 edge (idempotent)."""
        db = self._make_db(tmp_path, "r3_idem_schema.kuzu")

        schema_rel = self._rel(field="comparison_to_prior")

        for _ in range(2):
            db.add_paper(
                doi="10.0/a",
                entities=[],
                relationships=[schema_rel],
                paper_metadata={"title": "A", "year": 2024, "journal": "X"},
                prompt_version="phase1.0_schema",
            )

        assert self._count_edges(db, "COMPARES_TO") == 1

    def test_idempotent_rerun_llm_source(self, tmp_path):
        """R3 inv-3: LLM source running twice → still 1 edge (idempotent)."""
        db = self._make_db(tmp_path, "r3_idem_llm.kuzu")

        llm_rel = self._rel(
            predicate="EXTENDS",
            field="llm_extracted",
            evidence="A extends B",
            confidence=0.9,
        )

        for _ in range(2):
            db.add_paper(
                doi="10.0/a",
                entities=[],
                relationships=[llm_rel],
                paper_metadata={"title": "A", "year": 2024, "journal": "X"},
                prompt_version="phase3.0_llm",
            )

        assert self._count_edges(db, "EXTENDS") == 1

    def test_two_sources_each_idempotent_total_two_edges(self, tmp_path):
        """R3 inv-3: schema + LLM each run twice → still exactly 2 edges total."""
        db = self._make_db(tmp_path, "r3_idem_both.kuzu")

        schema_rel = self._rel(field="comparison_to_prior")
        llm_rel = self._rel(field="llm_extracted", confidence=0.85)

        for _ in range(2):
            db.add_paper(
                doi="10.0/a",
                entities=[],
                relationships=[schema_rel],
                paper_metadata={"title": "A", "year": 2024, "journal": "X"},
                prompt_version="phase1.0_schema",
            )
            db.add_paper(
                doi="10.0/a",
                entities=[],
                relationships=[llm_rel],
                paper_metadata={"title": "A", "year": 2024, "journal": "X"},
                prompt_version="phase3.0_llm",
            )

        assert self._count_edges(db, "COMPARES_TO") == 2

    # ------------------------------------------------------------------
    # R3 invariant 4: Phase 1 schema-only paths unchanged
    # ------------------------------------------------------------------

    def test_phase1_schema_only_path_unchanged(self, tmp_path):
        """R3 inv-4: Phase 1 schema-only RelationshipTuple produces the right edge count."""
        db = self._make_db(tmp_path, "r3_p1.kuzu")

        schema_rel = self._rel(
            predicate="LIMITED_BY",
            target_id="novel method",
            target_kind="Entity",
            evidence="limited by novel method",
            confidence=1.0,
            field="limitations",
        )
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[schema_rel],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
            prompt_version="phase1.0_schema",
        )
        assert self._count_edges(db, "LIMITED_BY") == 1

    def test_phase1_default_prompt_version_still_works(self, tmp_path):
        """R3 inv-4: callers using the default prompt_version='phase1.0' still
        produce exactly 1 edge (backward-compat with existing G6 callers)."""
        db = self._make_db(tmp_path, "r3_p1_default.kuzu")

        rel = self._rel(field="comparison_to_prior")
        # Call add_paper without specifying prompt_version → uses default.
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[rel],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
        )
        assert self._count_edges(db, "COMPARES_TO") == 1

    # ------------------------------------------------------------------
    # LLM-only predicate path (EXTENDS, R2 integration)
    # ------------------------------------------------------------------

    def test_extends_llm_only_path(self, tmp_path):
        """R3: EXTENDS (LLM-only predicate) works via R2 RelationshipTuple."""
        db = self._make_db(tmp_path, "r3_extends.kuzu")

        llm_rel = self._rel(
            predicate="EXTENDS",
            evidence="A extends B",
            confidence=0.9,
            field="llm_extracted",
        )
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[llm_rel],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
            prompt_version="phase3.0_llm",
        )
        assert self._count_edges(db, "EXTENDS") == 1

    def test_contradicts_llm_only_path(self, tmp_path):
        """R3: CONTRADICTS (LLM-only predicate) works via R2 RelationshipTuple."""
        db = self._make_db(tmp_path, "r3_contradicts.kuzu")

        llm_rel = self._rel(
            predicate="CONTRADICTS",
            evidence="A contradicts B",
            confidence=0.75,
            field="llm_extracted",
        )
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[llm_rel],
            paper_metadata={"title": "A", "year": 2024, "journal": "X"},
            prompt_version="phase3.0_llm",
        )
        assert self._count_edges(db, "CONTRADICTS") == 1
