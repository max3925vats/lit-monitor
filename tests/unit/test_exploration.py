"""Unit tests for Bundle K-b exploration query generation (Part B).

The graph is mocked (a tiny fake KuzuDB ``Connection``), so these tests pin the
*logic* of central-entity extraction + query building without a real graph.
"""

from __future__ import annotations

from typing import Any

from scripts.clustering.exploration import (
    build_exploration_queries,
    build_exploration_queries_for_cluster,
    central_entities_for_dois,
)


class _FakeResult:
    """Minimal stand-in for a Kuzu QueryResult (has_next / get_next)."""

    def __init__(self, rows: list[list[Any]]) -> None:
        self._rows = list(rows)
        self._i = 0

    def has_next(self) -> bool:
        return self._i < len(self._rows)

    def get_next(self) -> list[Any]:
        row = self._rows[self._i]
        self._i += 1
        return row


class _FakeConn:
    """Routes Cypher by substring to canned rows; records executed queries."""

    def __init__(self, central_rows=None, expansion_rows=None, resolve_cid="C1"):
        self._central_rows = central_rows or []
        self._expansion_rows = expansion_rows or []
        self._resolve_cid = resolve_cid
        self.executed: list[str] = []

    def execute(self, cypher: str, params: dict | None = None):
        self.executed.append(cypher)
        if "count(DISTINCT p.doi)" in cypher:
            # central_entities_for_dois query — honour the LIMIT $k param the way
            # real Kuzu would, so the fake mirrors production truncation.
            rows = self._central_rows
            if params and "k" in params:
                rows = rows[: int(params["k"])]
            return _FakeResult(rows)
        if "LIMIT 1" in cypher and "CONTAINS" in cypher:
            # query_expansion._resolve_entity
            return _FakeResult([[self._resolve_cid]] if self._resolve_cid else [])
        if "co_count" in cypher:
            # query_expansion._find_co_occurring
            return _FakeResult(self._expansion_rows)
        return _FakeResult([])


class _FakeGraph:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn


def test_central_entities_orders_by_count_and_caps() -> None:
    conn = _FakeConn(central_rows=[["ultrafiltration", 9], ["mAb", 5], ["resin", 2]])
    g = _FakeGraph(conn)
    out = central_entities_for_dois(["10.1/a", "10.1/b"], g, top_k=2)
    assert out == ["ultrafiltration", "mAb"]


def test_central_entities_empty_dois_or_no_graph() -> None:
    assert central_entities_for_dois([], _FakeGraph(_FakeConn()), top_k=3) == []
    assert central_entities_for_dois(["10.1/a"], None, top_k=3) == []


def test_central_entities_non_fatal_on_cypher_error() -> None:
    class _Boom:
        def execute(self, *a, **k):
            raise RuntimeError("kuzu down")

    g = _FakeGraph(_Boom())
    # Must not raise — degrades to [].
    assert central_entities_for_dois(["10.1/a"], g, top_k=3) == []


def test_build_queries_for_cluster_includes_centrals_and_expansions() -> None:
    conn = _FakeConn(
        central_rows=[["ultrafiltration", 9]],
        expansion_rows=[["tangential flow"], ["membrane fouling"]],
        resolve_cid="C-uf",
    )
    g = _FakeGraph(conn)
    out = build_exploration_queries_for_cluster(
        ["10.1/a"], g, top_entities=1, max_queries=5, expand=True
    )
    # Central entity first, then its expansions (canonical ids here).
    assert out[0] == "ultrafiltration"
    assert "tangential flow" in out
    assert "membrane fouling" in out


def test_build_queries_for_cluster_no_expand() -> None:
    conn = _FakeConn(central_rows=[["ultrafiltration", 9], ["mAb", 5]])
    g = _FakeGraph(conn)
    out = build_exploration_queries_for_cluster(
        ["10.1/a"], g, top_entities=2, expand=False
    )
    assert out == ["ultrafiltration", "mAb"]


def test_build_queries_for_cluster_empty_when_no_entities() -> None:
    g = _FakeGraph(_FakeConn(central_rows=[]))
    assert build_exploration_queries_for_cluster(["10.1/a"], g) == []


def test_build_queries_dedupes_across_clusters_and_caps() -> None:
    # Both clusters surface the same central entity → de-duplicated once.
    conn = _FakeConn(central_rows=[["ultrafiltration", 9]])
    g = _FakeGraph(conn)
    out = build_exploration_queries(
        {1: ["10.1/a"], 2: ["10.1/b"]},
        g,
        top_entities=1,
        expand=False,
        global_cap=10,
    )
    assert out == ["ultrafiltration"]


def test_build_queries_global_cap_truncates() -> None:
    conn = _FakeConn(central_rows=[["a", 3], ["b", 2], ["c", 1]])
    g = _FakeGraph(conn)
    out = build_exploration_queries(
        {1: ["10.1/a"]}, g, top_entities=3, expand=False, global_cap=2
    )
    assert out == ["a", "b"]


def test_build_queries_no_graph_returns_empty() -> None:
    assert build_exploration_queries({1: ["10.1/a"]}, None) == []


def test_build_queries_empty_mapping_returns_empty() -> None:
    g = _FakeGraph(_FakeConn(central_rows=[["a", 3]]))
    assert build_exploration_queries({}, g) == []
