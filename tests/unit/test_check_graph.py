"""Tests for scripts.setup.check_graph (FG-2 graph-health check)."""
from __future__ import annotations

from lit_monitor.setup.check_graph import check_graph


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
