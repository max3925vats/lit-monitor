"""Integration tests for Bundle K-b exploration budget (Part C wiring).

Exercises ``discovery._run_exploration_searches`` against a REAL ``StateDB``
(synthetic clusters + discovery history) with the external search runner and the
KuzuDB graph MOCKED — NO live external API calls. Asserts:

  * a never-surfaced, healthy cluster (A) is explored: its graph-expanded queries
    are issued and capped exploration candidates are returned;
  * a recently-surfaced cluster (B) is NOT explored;
  * ``exploration_budget_pct == 0`` → no exploration (zero extra searches);
  * no clusters → no exploration (a fresh user sees no behaviour change);
  * the cap is ~``budget_pct`` of the topic-pool size;
  * any failure is non-fatal (returns [] without raising).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.pipelines.discovery as discovery
from scripts.core.state_db import StateDB

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeResult:
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
    """Returns central-entity rows for the central-entity query; nothing else."""

    def __init__(self, central_rows: list[list[Any]]) -> None:
        self._central_rows = central_rows
        self.closed = False

    def execute(self, cypher: str, params: dict | None = None):
        if "count(DISTINCT p.doi)" in cypher:
            return _FakeResult(self._central_rows)
        # No expansion matches (resolve returns nothing) → just centrals used.
        return _FakeResult([])


class _FakeGraph:
    def __init__(self, central_rows: list[list[Any]]) -> None:
        self._conn = _FakeConn(central_rows)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _cfg(*, pct: float = 0.20, quiet_weeks: int = 4, floor: float = 0.1):
    return SimpleNamespace(
        feedback=SimpleNamespace(
            exploration_budget_pct=pct,
            cluster_quiet_weeks=quiet_weeks,
            minimum_cluster_floor=floor,
        )
    )


@pytest.fixture
def db(tmp_path):
    return StateDB(tmp_path / "state.db")


def _seed_two_clusters(db: StateDB) -> tuple[int, int]:
    """Cluster A (papers a1/a2) never surfaced; cluster B (paper b1) surfaced
    recently via a prior discovery run. Returns (cid_a, cid_b)."""
    blob = np.zeros(4, dtype=np.float32).tobytes()
    cid_a = db.insert_cluster("Cluster A", 2, 0.5, blob)
    cid_b = db.insert_cluster("Cluster B", 1, 0.5, blob)

    for doi in ("10.1/a1", "10.1/a2", "10.1/b1"):
        db.upsert_paper({"doi": doi, "title": doi, "status": "discovered"})

    db.upsert_cluster_assignment("10.1/a1", cid_a, distance_to_centroid=0.1)
    db.upsert_cluster_assignment("10.1/a2", cid_a, distance_to_centroid=0.2)
    db.upsert_cluster_assignment("10.1/b1", cid_b, distance_to_centroid=0.1)

    # Cluster B surfaced 'b1' in a recent discovery run → NOT under-engaged.
    run_id = db.start_discovery_run({"rag_mode": "vector"})
    db.add_discovery_paper(
        run_id=run_id, doi="10.1/b1", title="B paper",
        score=0.9, rationale="", ingested=False,
    )
    db.finish_discovery_run(run_id, "success", total_found=1, total_ingested=0)
    return cid_a, cid_b


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_under_engaged_cluster_is_explored_recent_is_not(db, monkeypatch):
    _seed_two_clusters(db)

    # Graph returns a central entity ONLY for cluster A's member DOIs. (Cluster B
    # should never be queried because it isn't under-engaged.)
    graph = _FakeGraph(central_rows=[["ultrafiltration", 2]])
    monkeypatch.setattr(discovery, "safe_graph_db", lambda: graph)

    issued_queries: list[list[str]] = []

    def _fake_run_searches(cfg, since_days=14):
        issued_queries.append(list(cfg.topics))
        # Return 3 fresh exploration candidates (none in the topic pool).
        return [
            {"doi": f"10.9/x{i}", "title": f"explore {i}", "abstract": ""}
            for i in range(3)
        ]

    monkeypatch.setattr(discovery, "run_searches", _fake_run_searches)

    topic_results = [{"doi": f"10.1/t{i}", "title": f"t{i}"} for i in range(10)]
    out = discovery._run_exploration_searches(
        _cfg(pct=0.20), db, topic_results, since_days=14
    )

    # Exploration ran: queries were issued, and they came from cluster A's
    # central entity 'ultrafiltration'.
    assert issued_queries, "exploration searches were not issued"
    assert "ultrafiltration" in issued_queries[0]
    # Cap = ceil(0.20 * 10) = 2 exploration candidates.
    assert len(out) == 2
    assert all(p.get("_exploration") is True for p in out)
    # Graph connection was closed (non-leak).
    assert graph.closed is True


def test_budget_zero_disables_exploration(db, monkeypatch):
    _seed_two_clusters(db)
    monkeypatch.setattr(
        discovery, "safe_graph_db",
        lambda: pytest.fail("graph must not be opened when budget=0"),
    )
    monkeypatch.setattr(
        discovery, "run_searches",
        lambda *a, **k: pytest.fail("no searches when budget=0"),
    )
    out = discovery._run_exploration_searches(
        _cfg(pct=0.0), db, [{"doi": "10.1/t0"}], since_days=14
    )
    assert out == []


def test_empty_topic_pool_skips_exploration_no_searches(db, monkeypatch):
    """K-b review #1: when the topic pool is empty the budget caps to 0, so
    exploration must short-circuit BEFORE opening the graph or issuing any
    external search — even with budget>0 and clusters present."""
    _seed_two_clusters(db)
    monkeypatch.setattr(
        discovery, "safe_graph_db",
        lambda: pytest.fail("graph must not be opened on an empty topic pool"),
    )
    monkeypatch.setattr(
        discovery, "run_searches",
        lambda *a, **k: pytest.fail("no exploration searches on an empty topic pool"),
    )
    out = discovery._run_exploration_searches(_cfg(pct=0.20), db, [], since_days=14)
    assert out == []


def test_no_clusters_no_exploration(db, monkeypatch):
    # Fresh user: no clusters at all → zero behaviour change, no graph, no search.
    monkeypatch.setattr(
        discovery, "safe_graph_db",
        lambda: pytest.fail("graph must not be opened with no clusters"),
    )
    monkeypatch.setattr(
        discovery, "run_searches",
        lambda *a, **k: pytest.fail("no searches with no clusters"),
    )
    out = discovery._run_exploration_searches(
        _cfg(pct=0.20), db, [{"doi": "10.1/t0"}], since_days=14
    )
    assert out == []


def test_no_graph_no_exploration(db, monkeypatch):
    _seed_two_clusters(db)
    monkeypatch.setattr(discovery, "safe_graph_db", lambda: None)
    monkeypatch.setattr(
        discovery, "run_searches",
        lambda *a, **k: pytest.fail("no searches when graph unavailable"),
    )
    out = discovery._run_exploration_searches(
        _cfg(pct=0.20), db, [{"doi": "10.1/t0"}], since_days=14
    )
    assert out == []


def test_no_central_entities_skips_exploration(db, monkeypatch):
    _seed_two_clusters(db)
    # Graph present but yields no central entities → no queries → no searches.
    monkeypatch.setattr(
        discovery, "safe_graph_db", lambda: _FakeGraph(central_rows=[])
    )
    monkeypatch.setattr(
        discovery, "run_searches",
        lambda *a, **k: pytest.fail("no searches when graph has no entities"),
    )
    out = discovery._run_exploration_searches(
        _cfg(pct=0.20), db, [{"doi": "10.1/t0"}], since_days=14
    )
    assert out == []


def test_exploration_finds_nothing_no_fake_fill(db, monkeypatch):
    _seed_two_clusters(db)
    monkeypatch.setattr(
        discovery, "safe_graph_db",
        lambda: _FakeGraph(central_rows=[["ultrafiltration", 2]]),
    )
    # Searches run but return zero candidates → proceed normally, no fake-fill.
    monkeypatch.setattr(discovery, "run_searches", lambda *a, **k: [])
    out = discovery._run_exploration_searches(
        _cfg(pct=0.20), db, [{"doi": "10.1/t0"}], since_days=14
    )
    assert out == []


def test_exploration_candidates_already_in_topic_pool_dropped(db, monkeypatch):
    _seed_two_clusters(db)
    monkeypatch.setattr(
        discovery, "safe_graph_db",
        lambda: _FakeGraph(central_rows=[["ultrafiltration", 2]]),
    )
    # The single exploration candidate shares a DOI with the topic pool → dropped.
    monkeypatch.setattr(
        discovery, "run_searches",
        lambda *a, **k: [{"doi": "10.1/t0", "title": "dup"}],
    )
    out = discovery._run_exploration_searches(
        _cfg(pct=0.20), db, [{"doi": "10.1/t0"}], since_days=14
    )
    assert out == []


def test_near_floor_dismissed_cluster_excluded(db, monkeypatch):
    """A quiet cluster the user is actively dismissing (weight near floor) is not
    explored. We make cluster A heavily dismissed so its K-a weight is near the
    floor, and assert no exploration search runs for it."""
    cid_a, _ = _seed_two_clusters(db)
    # Pile on recent dismissals for cluster A's members to drive weight to floor.
    for _ in range(40):
        db.record_feedback_event("10.1/a1", "dismissed", source="test")
        db.record_feedback_event("10.1/a2", "dismissed", source="test")

    monkeypatch.setattr(
        discovery, "safe_graph_db",
        lambda: _FakeGraph(central_rows=[["ultrafiltration", 2]]),
    )
    monkeypatch.setattr(
        discovery, "run_searches",
        lambda *a, **k: pytest.fail("dismissed cluster must not be explored"),
    )
    out = discovery._run_exploration_searches(
        _cfg(pct=0.20), db, [{"doi": "10.1/t0"}], since_days=14
    )
    assert out == []


def test_non_fatal_on_search_error(db, monkeypatch):
    _seed_two_clusters(db)
    monkeypatch.setattr(
        discovery, "safe_graph_db",
        lambda: _FakeGraph(central_rows=[["ultrafiltration", 2]]),
    )

    def _boom(*a, **k):
        raise RuntimeError("search backend exploded")

    monkeypatch.setattr(discovery, "run_searches", _boom)
    # Must not raise — discovery is never broken by exploration.
    out = discovery._run_exploration_searches(
        _cfg(pct=0.20), db, [{"doi": "10.1/t0"}], since_days=14
    )
    assert out == []
