"""FG-3: rag_mode selector on discovery start form + threaded into spawn.

The discovery start form is rendered by the controls fragment
(``GET /api/discovery/controls``), which loads lazily on the ``/discovery``
dashboard. The start handler (``POST /api/discovery/start``) builds the
subprocess argv in ``_spawn_discovery``; FG-3 appends ``--rag-mode <mode>``.

All tests monkeypatch the spawn so NO real subprocess runs.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app
from lit_monitor.server.runtime import get_runtime, reset_runtime


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def _no_real_spawn(monkeypatch):
    """Replace _spawn_discovery with a coroutine that returns a fake process,
    capturing the argv it was asked to build.
    """
    captured: dict = {}

    async def _fake_spawn(*, dry_run: bool, rag_mode: str):
        argv = ["uv", "run", "lit-monitor", "run"]
        if dry_run:
            argv.append("--dry-run")
        argv += ["--rag-mode", rag_mode]
        captured["argv"] = argv
        captured["rag_mode"] = rag_mode
        fake = MagicMock()
        fake.returncode = None
        fake.pid = 4242
        return fake

    monkeypatch.setattr(
        "lit_monitor.server.routes.discovery._spawn_discovery", _fake_spawn
    )
    return captured


@pytest.mark.unit
def test_discovery_form_has_rag_mode_select(client):
    """Idle controls fragment exposes a rag_mode select with all three modes."""
    r = client.get("/api/discovery/controls")
    assert r.status_code == 200
    assert 'name="rag_mode"' in r.text
    for m in ("vector", "graph", "hybrid"):
        assert m in r.text


@pytest.mark.unit
def test_discovery_start_threads_rag_mode_into_spawn(client, monkeypatch):
    """POST /start with rag_mode=hybrid threads --rag-mode hybrid into argv.

    Capture the real argv that _spawn_discovery builds by patching
    create_subprocess_exec (so the production argv-construction code runs).
    """
    captured: dict = {}

    async def _fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        fake = MagicMock()
        fake.returncode = None
        fake.pid = 4242
        return fake

    # graph extra present in this env, so hybrid is allowed to start.
    monkeypatch.setattr(
        "lit_monitor.server.routes.discovery.asyncio.create_subprocess_exec",
        _fake_exec,
    )
    r = client.post("/api/discovery/start", data={"rag_mode": "hybrid"})
    assert r.status_code == 200, r.text
    flat = " ".join(captured.get("argv", []))
    assert "--rag-mode" in flat and "hybrid" in flat


@pytest.mark.unit
def test_discovery_start_invalid_mode_falls_back_to_config_default(
    client, monkeypatch
):
    """Unknown rag_mode falls back to the config default ('vector' here)."""
    captured: dict = {}

    async def _fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        fake = MagicMock()
        fake.returncode = None
        fake.pid = 4242
        return fake

    monkeypatch.setattr(
        "lit_monitor.server.routes.discovery.asyncio.create_subprocess_exec",
        _fake_exec,
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.discovery._default_rag_mode",
        lambda: "vector",
    )
    r = client.post("/api/discovery/start", data={"rag_mode": "nonsense"})
    assert r.status_code == 200, r.text
    flat = " ".join(captured.get("argv", []))
    assert "--rag-mode vector" in flat


@pytest.mark.unit
def test_discovery_default_mode_from_config(client, monkeypatch):
    """The graph option is pre-selected when config default_mode is 'graph'."""
    monkeypatch.setattr(
        "lit_monitor.server.routes.discovery._default_rag_mode",
        lambda: "graph",
    )
    r = client.get("/api/discovery/controls")
    assert r.status_code == 200
    # Shoelace: pre-selection via value="graph" on the <sl-select> host element.
    # The bare assertion 'value="graph"' is vacuous — it also matches
    # <sl-option value="graph">. Assert the value attribute is on the HOST tag.
    import re
    assert re.search(r'<sl-select\b[^>]*\bvalue="graph"', r.text), \
        "default rag mode must pre-select via value= on the sl-select host"


@pytest.mark.unit
def test_graph_mode_without_extra_refuses_with_warning(client, monkeypatch):
    """W4-mirror: rag_mode=graph without [graph] extra refuses (no crash).

    Patches the lightweight import probe (NOT a prod-DB open): when the extra
    is reported unavailable the route returns 400 and never spawns.
    """
    # Simulate missing [graph] extra.
    monkeypatch.setattr(
        "lit_monitor.server.routes.discovery._graph_extra_available",
        lambda: False,
    )
    # Ensure spawn would NOT be reached; if it is, fail loudly.
    async def _boom(*a, **k):
        raise AssertionError("spawn must not run when [graph] extra is missing")

    monkeypatch.setattr(
        "lit_monitor.server.routes.discovery._spawn_discovery", _boom
    )
    r = client.post("/api/discovery/start", data={"rag_mode": "graph"})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["started"] is False
    assert "[graph] extra" in body["reason"]


@pytest.mark.unit
def test_default_rag_mode_reads_config():
    """_default_rag_mode reads retrieval.default_mode from runtime config."""
    from lit_monitor.server.routes import discovery as disc

    runtime = get_runtime()
    runtime._config = SimpleNamespace(
        retrieval=SimpleNamespace(default_mode="hybrid")
    )
    assert disc._default_rag_mode() == "hybrid"
