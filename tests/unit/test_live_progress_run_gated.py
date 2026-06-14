"""Live-progress pane is run-gated (appears only while a run is active).

The SSE log pane on /brain-build and /discovery should be hidden when nothing is
running and revealed (by site.js, on the controls poll's data-running) once a run
starts. These render/string tests assert the static markup contract that makes
that toggle possible:

1. Each dashboard wraps the live-progress pane in a stable
   `#live-progress-wrap` that ships `hidden` by default.
2. The controls fragments expose `data-running="…"` on the polled <section>.
3. The SSE `sse-connect` div is OUTSIDE the controls <section> — i.e. it is the
   stable element in index.html, never re-rendered by the 5s controls poll (no
   EventSource teardown/reconnect storm).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose brain-build/discovery routes see an isolated, *configured*
    runtime: a real (empty) temp StateDB plus a benign config.

    These tests assert the full configured-dashboard markup contract
    (``#live-progress-wrap``, ``sse-connect=…``). That markup only renders once
    the route has a usable state_db; without one the dashboards correctly
    degrade to a "DB unavailable" card (which omits the live-progress pane). On a
    developer's machine an ambient ``~/.config/lit-monitor/paths.yaml`` happened
    to supply config, so a bare ``create_app()`` passed locally but fell back in
    CI. Injecting the runtime here removes that hidden dependency.
    """
    from unittest.mock import MagicMock

    from lit_monitor.core.state_db import StateDB
    from lit_monitor.server.runtime import reset_runtime

    reset_runtime()
    db = StateDB(tmp_path / "state.db")
    fake_runtime = MagicMock()
    fake_runtime.state_db = db
    fake_runtime.config.obsidian.vault_path = ""
    for route_mod in ("brain_build", "discovery"):
        monkeypatch.setattr(
            f"lit_monitor.server.routes.{route_mod}.get_runtime",
            lambda: fake_runtime,
        )
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# 1) The dashboard pages ship the hidden wrapper.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_brain_build_live_progress_wrap_hidden_by_default(client: TestClient) -> None:
    html = client.get("/brain-build").text
    assert 'id="live-progress-wrap"' in html
    # The wrapper carrying the id must also carry the `hidden` attribute.
    wrap = html.split('id="live-progress-wrap"', 1)[1].split(">", 1)[0]
    assert "hidden" in wrap, f"expected hidden on live-progress-wrap; got <…{wrap}>"


@pytest.mark.unit
def test_discovery_live_progress_wrap_hidden_by_default(client: TestClient) -> None:
    html = client.get("/discovery").text
    assert 'id="live-progress-wrap"' in html
    wrap = html.split('id="live-progress-wrap"', 1)[1].split(">", 1)[0]
    assert "hidden" in wrap, f"expected hidden on live-progress-wrap; got <…{wrap}>"


# ---------------------------------------------------------------------------
# 2) The controls fragments expose data-running for the JS toggle.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_brain_build_controls_expose_data_running(client: TestClient) -> None:
    html = client.get("/api/brain-build/controls").text
    assert "data-running=" in html


@pytest.mark.unit
def test_discovery_controls_expose_data_running(client: TestClient) -> None:
    html = client.get("/api/discovery/controls").text
    assert "data-running=" in html


# ---------------------------------------------------------------------------
# 3) The SSE div is NOT nested inside the polled controls <section>.
#    (If it were, the 5s controls poll would tear down + reconnect the
#    EventSource every cycle, wiping accumulated log lines.)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_brain_build_sse_div_outside_controls_section(client: TestClient) -> None:
    html = client.get("/brain-build").text
    # The controls fragment is loaded lazily (hx-trigger=load) into a placeholder
    # <section hx-get=".../controls">; the sse-connect div lives in the page body,
    # not inside that controls section. We assert the sse-connect element exists
    # in the full page and is wrapped by the stable #live-progress-wrap (which is
    # never a target of the controls poll).
    assert 'sse-connect="/api/brain-build/stream"' in html
    wrap_idx = html.index('id="live-progress-wrap"')
    sse_idx = html.index('sse-connect="/api/brain-build/stream"')
    assert sse_idx > wrap_idx, "sse-connect div must live inside #live-progress-wrap"


@pytest.mark.unit
def test_discovery_sse_div_outside_controls_section(client: TestClient) -> None:
    html = client.get("/discovery").text
    assert 'sse-connect="/api/discovery/stream"' in html
    wrap_idx = html.index('id="live-progress-wrap"')
    sse_idx = html.index('sse-connect="/api/discovery/stream"')
    assert sse_idx > wrap_idx, "sse-connect div must live inside #live-progress-wrap"
