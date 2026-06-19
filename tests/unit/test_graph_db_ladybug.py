"""GraphDB round-trips a MENTIONS edge on the LadybugDB backend (X1 migration)."""
from lit_monitor.graph.db import GraphDB


def test_graphdb_roundtrips_mentions_on_ladybug(tmp_path):
    db = GraphDB(persist_dir=str(tmp_path / "graph.kuzu"))
    conn = db._conn  # ladybug.Connection after the swap
    # apply_schema() already created the node/rel tables in GraphDB.__init__.
    conn.execute("CREATE (:Paper {doi: '10.test/1', title: 'T', year: 2020, journal: 'J'})")
    conn.execute("CREATE (:Entity {canonical_id: 'ent-1', type: 'topic', surface: 'membrane fouling'})")
    conn.execute(
        "MATCH (p:Paper {doi: '10.test/1'}), (e:Entity {canonical_id: 'ent-1'}) "
        "CREATE (p)-[:MENTIONS {source: 'test'}]->(e)"
    )
    res = conn.execute(
        "MATCH (:Paper)-[:MENTIONS]->(e:Entity) RETURN e.canonical_id, e.surface"
    )
    row = res.get_next()
    assert row[0] == "ent-1"
    assert row[1] == "membrane fouling"
