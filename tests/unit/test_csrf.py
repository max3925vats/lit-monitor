"""AR-1: CSRF + DNS-rebinding guard for the localhost web server.

The web server is localhost-only and unauthenticated. Without an Origin/Host
check, any webpage the user visits can blind-fire a cross-origin form POST
(``fetch(..., {mode: "no-cors", body: formData})``) at ``127.0.0.1:8765`` and
overwrite credentials, start pipelines, or rewrite the schedule.

These tests pin :class:`scripts.server.csrf.LocalOriginMiddleware`:

  - cross-origin POST (browser sends ``Origin``) → 403
  - same-origin (localhost ``Origin``) POST → not blocked
  - no-Origin POST (curl / CLI / some same-origin browsers) → not blocked
  - GET is never blocked, even cross-origin
  - non-local ``Host`` (DNS-rebinding) → 403
  - TestClient's default ``Host: testserver`` is allowed (so the rest of the
    route suite keeps passing)

All spawn paths are mocked so no real subprocess runs.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app
from scripts.server.runtime import reset_runtime


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient on a freshly-built app with an isolated state DB."""
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    return TestClient(create_app())


def _dummy_proc_factory() -> AsyncMock:
    """Stand-in for ``asyncio.create_subprocess_exec`` returning a fake proc.

    Mirrors the spawn-mock idiom in ``test_dev_graph_panel.py`` so the
    same-origin / no-origin POST tests reach the spawn path without ever
    launching a real ``lit-monitor run`` subprocess.
    """
    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.returncode = None
    fake_proc.stdout = MagicMock()
    return AsyncMock(return_value=fake_proc)


def test_cross_origin_form_post_rejected(client):
    """A cross-origin form POST (browser-style attack) is rejected with 403."""
    r = client.post(
        "/api/discovery/start",
        data={"dry_run": "true"},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403


def test_same_origin_post_allowed(client, monkeypatch):
    """A localhost Origin passes the gate (spawn mocked → nothing real runs)."""
    monkeypatch.setattr(
        "scripts.server.routes.discovery.asyncio.create_subprocess_exec",
        _dummy_proc_factory(),
    )
    r = client.post(
        "/api/discovery/start",
        data={"dry_run": "true"},
        headers={"Origin": "http://127.0.0.1:8765"},
    )
    assert r.status_code != 403


def test_no_origin_post_allowed(client, monkeypatch):
    """A POST with no Origin header (curl / CLI tools) must not be blocked."""
    monkeypatch.setattr(
        "scripts.server.routes.discovery.asyncio.create_subprocess_exec",
        _dummy_proc_factory(),
    )
    r = client.post("/api/discovery/start", data={"dry_run": "true"})
    assert r.status_code != 403


def test_get_never_blocked_even_cross_origin(client):
    """GET is a safe method — never blocked, even with a hostile Origin."""
    r = client.get("/", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200


def test_dns_rebinding_nonlocal_host_rejected(client):
    """A non-local Host header (DNS-rebinding) is rejected before any handler."""
    r = client.post(
        "/api/discovery/start",
        data={},
        headers={"Host": "evil.example"},
    )
    assert r.status_code == 403


def test_testserver_host_allowed(client, monkeypatch):
    """TestClient's default ``Host: testserver`` must pass so the suite stays green."""
    monkeypatch.setattr(
        "scripts.server.routes.discovery.asyncio.create_subprocess_exec",
        _dummy_proc_factory(),
    )
    # No Host override → TestClient sends Host: testserver, Origin absent.
    r = client.post("/api/discovery/start", data={"dry_run": "true"})
    assert r.status_code != 403


def test_wildcard_bind_host_not_allowlisted(tmp_path, monkeypatch):
    """A configured bind host of ``0.0.0.0`` must NOT make ``Host: 0.0.0.0`` pass.

    ``0.0.0.0`` is a wildcard *bind* address, never a value a client sends as
    Host, so ``_resolve_allowed_hosts()`` must skip it. A non-safe request whose
    Host is ``0.0.0.0`` is therefore still a non-local Host → 403.
    """
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    # Simulate a persisted ``[server].host = "0.0.0.0"`` in the secrets TOML.
    monkeypatch.setattr(
        "scripts.server.config_io.load_server_config",
        lambda: {"host": "0.0.0.0"},
    )
    client = TestClient(create_app())
    r = client.post(
        "/api/discovery/start",
        data={},
        headers={"Host": "0.0.0.0"},
    )
    assert r.status_code == 403
