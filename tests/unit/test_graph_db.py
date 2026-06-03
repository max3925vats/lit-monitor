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

    def test_close_is_idempotent(self, tmp_path):
        """close() may be called repeatedly without raising (the second call sees
        already-None handles and skips the kuzu .close() calls)."""
        from scripts.graph import GraphDB

        g = GraphDB(persist_dir=str(tmp_path / "idem.kuzu"))
        g.close()
        g.close()  # must not raise
        assert g._conn is None
        assert g._db is None


@pytest.mark.unit
class TestGraphDBSchemaVersion:
    """G14: schema-version sentinel file is written on fresh init."""

    def test_schema_version_file_written_on_fresh_init(self, tmp_path):
        """The .schema_version sentinel file is created on first GraphDB init."""
        from scripts.graph import GraphDB
        from scripts.graph.migrations import SCHEMA_VERSION

        db_path = tmp_path / "fresh.kuzu"
        GraphDB(persist_dir=str(db_path))
        version_file = tmp_path / "fresh.kuzu.schema_version"
        assert version_file.exists(), "sentinel file must be created on first init"
        assert int(version_file.read_text().strip()) == SCHEMA_VERSION


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
class TestAddPaper:
    """G6: GraphDB.add_paper writes Paper + Entity nodes + MENTIONS + typed edges
    atomically (single Kuzu transaction).  All edges carry prompt_version.
    """

    def _entity_tuple(self, **overrides):
        from scripts.graph.entity_extractor import EntityTuple
        defaults = dict(
            canonical_id="ion exchange",
            type="method",
            surface="ion exchange",
            field="methods_summary",
            span_start=0,
            span_end=12,
        )
        defaults.update(overrides)
        return EntityTuple(**defaults)

    def _rel_tuple(self, **overrides):
        from scripts.graph.relationship_extractor import RelationshipTuple
        defaults = dict(
            source_doi="10.0/a",
            predicate="PROPOSES",
            target_id="future method",
            target_kind="Entity",
            evidence="we propose a future method",
            confidence=1.0,
            field="future_work",
        )
        defaults.update(overrides)
        return RelationshipTuple(**defaults)

    def test_creates_paper_node(self, tmp_path):
        """G6: add_paper creates a Paper node with the given metadata."""
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[],
            paper_metadata={"title": "Test", "year": 2024, "journal": "X"},
        )
        res = db._conn.execute(
            "MATCH (p:Paper {doi: '10.0/a'}) RETURN p.title, p.year, p.journal"
        )
        assert res.has_next()
        row = res.get_next()
        assert row[0] == "Test"
        assert int(row[1]) == 2024
        assert row[2] == "X"

    def test_creates_entity_and_mentions(self, tmp_path):
        """G6: each entity tuple -> 1 Entity node + 1 MENTIONS edge with source='schema'."""
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        db.add_paper(
            doi="10.0/a",
            entities=[self._entity_tuple()],
            relationships=[],
            paper_metadata={"title": "T", "year": 2024, "journal": "X"},
        )
        res = db._conn.execute(
            "MATCH (p:Paper)-[m:MENTIONS]->(e:Entity) "
            "RETURN e.canonical_id, e.type, m.source, m.field, m.prompt_version"
        )
        assert res.has_next()
        row = res.get_next()
        assert row[0] == "ion exchange"
        assert row[1] == "method"
        assert row[2] == "schema"
        assert row[3] == "methods_summary"
        assert row[4] == "phase1.0"

    def test_mentions_handles_none_spans(self, tmp_path):
        """G6: Zotero-origin entities have span_start/end = None; write must not crash."""
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        ent = self._entity_tuple(
            canonical_id="smith, jane",
            type="author",
            surface="Smith, Jane",
            field=None,
            span_start=None,
            span_end=None,
        )
        db.add_paper(
            doi="10.0/a",
            entities=[ent],
            relationships=[],
            paper_metadata={"title": "T", "year": 2024, "journal": "X"},
        )
        res = db._conn.execute(
            "MATCH (p:Paper)-[m:MENTIONS]->(e:Entity {canonical_id: 'smith, jane'}) "
            "RETURN m.field, m.span_start"
        )
        assert res.has_next()
        row = res.get_next()
        # field stored as empty string (Kuzu STRING column can't be NULL on CREATE)
        assert row[0] == ""

    def test_creates_typed_predicate_edge(self, tmp_path):
        """G6: each validated relationship -> 1 typed edge with confidence + prompt_version."""
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # The target entity must exist for the MATCH to succeed.
        target_ent = self._entity_tuple(
            canonical_id="future method",
            type="method",
            surface="future method",
            field="future_work",
            span_start=None,
            span_end=None,
        )
        db.add_paper(
            doi="10.0/a",
            entities=[target_ent],
            relationships=[self._rel_tuple()],
            paper_metadata={"title": "T", "year": 2024, "journal": "X"},
        )
        res = db._conn.execute(
            "MATCH (p:Paper)-[r:PROPOSES]->(e:Entity) "
            "RETURN e.canonical_id, r.evidence, r.confidence, r.prompt_version"
        )
        assert res.has_next()
        row = res.get_next()
        assert row[0] == "future method"
        assert row[1] == "we propose a future method"
        assert row[2] == 1.0
        assert row[3] == "phase1.0"

    def test_creates_paper_to_paper_compares_to(self, tmp_path):
        """G6: Paper-target relationships create the target Paper if absent then COMPARES_TO."""
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # Pre-create target paper so the MATCH succeeds.
        db.add_paper(
            doi="10.0/target",
            entities=[],
            relationships=[],
            paper_metadata={"title": "Target", "year": 2023, "journal": "Y"},
        )
        rel = self._rel_tuple(
            predicate="COMPARES_TO",
            target_id="10.0/target",
            target_kind="Paper",
            evidence="we benchmark against …",
            field="comparison_to_prior",
        )
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[rel],
            paper_metadata={"title": "Source", "year": 2024, "journal": "X"},
        )
        res = db._conn.execute(
            "MATCH (p1:Paper {doi: '10.0/a'})-[r:COMPARES_TO]->(p2:Paper) "
            "RETURN p2.doi, r.confidence, r.prompt_version"
        )
        assert res.has_next()
        row = res.get_next()
        assert row[0] == "10.0/target"
        assert row[1] == 1.0
        assert row[2] == "phase1.0"

    def test_is_idempotent(self, tmp_path):
        """G6: re-running add_paper for the same DOI must not duplicate nodes or edges."""
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        ents = [self._entity_tuple()]
        rels = []
        db.add_paper(doi="10.0/a", entities=ents, relationships=rels,
                     paper_metadata={"title": "T1", "year": 2024, "journal": "X"})
        # Second call with same DOI/entity: must be a no-op for existing rows.
        db.add_paper(doi="10.0/a", entities=ents, relationships=rels,
                     paper_metadata={"title": "T2", "year": 2024, "journal": "X"})
        paper_count = int(
            db._conn.execute("MATCH (p:Paper) RETURN count(p)").get_next()[0]
        )
        entity_count = int(
            db._conn.execute("MATCH (e:Entity) RETURN count(e)").get_next()[0]
        )
        mentions_count = int(
            db._conn.execute("MATCH ()-[m:MENTIONS]->() RETURN count(m)").get_next()[0]
        )
        assert paper_count == 1
        assert entity_count == 1
        assert mentions_count == 1

    def test_typed_edge_is_idempotent(self, tmp_path):
        """G6: re-running with the same relationship doesn't duplicate the typed edge."""
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        ents = [self._entity_tuple(
            canonical_id="future method", type="method", surface="future method",
            field="future_work", span_start=None, span_end=None,
        )]
        rels = [self._rel_tuple()]
        db.add_paper(doi="10.0/a", entities=ents, relationships=rels,
                     paper_metadata={"title": "T", "year": 2024, "journal": "X"})
        db.add_paper(doi="10.0/a", entities=ents, relationships=rels,
                     paper_metadata={"title": "T", "year": 2024, "journal": "X"})
        count = int(
            db._conn.execute(
                "MATCH ()-[r:PROPOSES]->() RETURN count(r)"
            ).get_next()[0]
        )
        assert count == 1

    def test_rolls_back_on_error(self, tmp_path):
        """G6 INVARIANT: if any step inside the txn fails, no partial Paper node remains.

        We trigger a downstream failure by stubbing the connection.execute to raise on
        a MENTIONS CREATE — that is mid-transaction after the Paper node has been
        created, exercising the ROLLBACK path.
        """
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # The simplest reliable way to force a mid-txn failure is to pass a bad
        # entity whose Cypher will fail. Easiest: a relationship targeting a Paper
        # that does NOT exist with target_kind='Paper' — the CREATE will succeed
        # (no rows matched, 0 edges created) so that's not enough.  Instead we
        # rely on a stub via a sentinel that raises.  Use a custom EntityTuple-like
        # mock with a non-string field that crashes Cypher param binding.
        class BadEnt:
            canonical_id = object()  # non-string causes parameter-bind error
            type = "method"
            surface = "x"
            field = "methods_summary"
            span_start = 0
            span_end = 1
        with pytest.raises(Exception):
            db.add_paper(
                doi="10.0/rollback",
                entities=[BadEnt()],
                relationships=[],
                paper_metadata={"title": "T", "year": 2024, "journal": "X"},
            )
        # After the rollback, the Paper node must NOT exist.
        count = int(
            db._conn.execute(
                "MATCH (p:Paper {doi: '10.0/rollback'}) RETURN count(p)"
            ).get_next()[0]
        )
        assert count == 0, "ROLLBACK must remove the partial Paper node"

    def test_empty_entities_and_relationships(self, tmp_path):
        """G6: empty lists are valid input — only the Paper node is written."""
        from scripts.graph import GraphDB
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        db.add_paper(
            doi="10.0/lonely",
            entities=[],
            relationships=[],
            paper_metadata={"title": "L", "year": 2024, "journal": "X"},
        )
        count = int(
            db._conn.execute(
                "MATCH (p:Paper {doi: '10.0/lonely'}) RETURN count(p)"
            ).get_next()[0]
        )
        assert count == 1

    def test_extracted_at_records_passed_timestamp_not_default(self, tmp_path):
        """P3.1: edges record the extracted_at passed to add_paper, not the DDL
        current_timestamp() DEFAULT.

        We pass a fixed historical timestamp and assert both the MENTIONS edge
        and the typed PROPOSES edge carry exactly that value (and that it is far
        from 'now', proving the DEFAULT was overridden).
        """
        import datetime as dt

        from scripts.graph import GraphDB

        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        known = dt.datetime(2020, 1, 2, 3, 4, 5)
        target_ent = self._entity_tuple(
            canonical_id="future method",
            type="method",
            surface="future method",
            field="future_work",
            span_start=None,
            span_end=None,
        )
        db.add_paper(
            doi="10.0/a",
            entities=[self._entity_tuple(), target_ent],
            relationships=[self._rel_tuple()],
            paper_metadata={"title": "T", "year": 2024, "journal": "X"},
            extracted_at=known,
        )

        # MENTIONS edge extracted_at == known.
        row = db._conn.execute(
            "MATCH (:Paper)-[m:MENTIONS]->(:Entity {canonical_id: 'ion exchange'}) "
            "RETURN m.extracted_at LIMIT 1"
        ).get_next()
        # Kuzu returns a datetime; compare to the known value.
        assert row[0] == known, f"MENTIONS.extracted_at {row[0]!r} != {known!r}"

        # Typed PROPOSES edge extracted_at == known.
        row = db._conn.execute(
            "MATCH (:Paper)-[r:PROPOSES]->(:Entity) RETURN r.extracted_at LIMIT 1"
        ).get_next()
        assert row[0] == known, f"PROPOSES.extracted_at {row[0]!r} != {known!r}"

        # Sanity: the recorded value is clearly historical, not 'now' (which the
        # DDL DEFAULT would have produced).
        assert row[0].year == 2020

    def test_add_paper_invalidates_query_cache(self, tmp_path):
        """P3.2: after add_paper, a new entity resolves via resolve_query_entity
        without a manual invalidate_query_cache() call.

        We prime the query-time normalizer cache (first resolve builds it),
        then add a NEW entity via add_paper, then resolve it — it must be found
        because add_paper invalidated the cache.
        """
        from scripts.graph import GraphDB

        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))

        # Prime the cache against an empty Entity table so a stale snapshot
        # would NOT contain the entity we add below.
        assert db.resolve_query_entity("ion exchange", type_="method") is None
        assert db._query_normalizer is not None, "cache should be primed now"

        # Add a new entity via add_paper — this must invalidate the cache.
        db.add_paper(
            doi="10.0/a",
            entities=[self._entity_tuple()],
            relationships=[],
            paper_metadata={"title": "T", "year": 2024, "journal": "X"},
        )

        # Without P3.2's automatic invalidation the stale snapshot would miss
        # this entity. With it, the rebuild sees the live Entity node.
        resolved = db.resolve_query_entity("ion exchange", type_="method")
        assert resolved == "ion exchange", (
            f"newly-added entity must resolve without manual invalidate; got {resolved!r}"
        )


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
