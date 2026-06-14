"""Tests for scripts.setup.check_graph (FG-2 graph-health check)."""
from __future__ import annotations

from lit_monitor.setup.check_graph import _indexed_total, check_graph


def test_indexed_total_excludes_discovered(tmp_path, monkeypatch) -> None:
    """The total denominator counts INGESTED papers only (status != 'discovered')."""
    from lit_monitor.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    with db._connect() as conn:
        # 2 ingested (graph-indexed + extraction_complete), 3 discovery candidates.
        conn.execute(
            "INSERT INTO papers (doi, title, status, graph_indexed) VALUES "
            "('10/a','A','extraction_complete',1),"
            "('10/b','B','no_markdown',0),"
            "('10/c','C','discovered',0),"
            "('10/d','D','discovered',0),"
            "('10/e','E','discovered',0)"
        )
    monkeypatch.setattr(
        "lit_monitor.core.config.get_config",
        lambda: type("C", (), {"state_db": type("S", (), {"path": str(tmp_path / "state.db")})()})(),
    )
    indexed, total = _indexed_total()
    assert indexed == 1  # only the graph_indexed=1 row
    assert total == 2    # the 3 'discovered' rows are excluded


def test_check_graph_warn_when_graph_absent(monkeypatch) -> None:
    """No graph (or no [graph] extra) → ok=False, severity='warn'."""
    monkeypatch.setattr(
        "lit_monitor.setup.check_graph.safe_graph_db", lambda *a, **k: None
    )
    r = check_graph()
    assert r.ok is False and r.severity == "warn" and "graph" in r.message.lower()


def test_check_graph_ok_when_all_indexed(monkeypatch) -> None:
    """Graph present + every paper indexed → ok=True, severity='ok'."""
    monkeypatch.setattr(
        "lit_monitor.setup.check_graph.safe_graph_db", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        "lit_monitor.setup.check_graph._indexed_total", lambda: (10, 10)
    )
    r = check_graph()
    assert r.ok is True and r.severity == "ok" and "10" in r.message


def test_check_graph_partial_is_not_fail(monkeypatch) -> None:
    """A partially-indexed graph is healthy, not a failure."""
    monkeypatch.setattr(
        "lit_monitor.setup.check_graph.safe_graph_db", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        "lit_monitor.setup.check_graph._indexed_total", lambda: (3, 10)
    )
    r = check_graph()
    assert r.severity != "fail"


def test_run_health_check_includes_graph_section(monkeypatch) -> None:
    """The graph section appears in the aggregated health-check dict."""
    from lit_monitor.setup.health_check import run_health_check

    monkeypatch.setattr(
        "lit_monitor.setup.check_graph.safe_graph_db", lambda *a, **k: None
    )
    out = run_health_check()
    assert "graph" in out
