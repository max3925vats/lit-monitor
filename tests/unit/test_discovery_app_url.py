"""E2b: the discovery notification deep-link URL is derived from the [server]
config block, with a safe fallback to the historical default.

These cover scripts/pipelines/discovery.py::_resolve_app_url.
"""
from __future__ import annotations

from lit_monitor.pipelines.discovery import _DEFAULT_APP_URL, _resolve_app_url


def test_resolve_app_url_uses_configured_host_port(monkeypatch):
    """When [server] is configured, the URL reflects its host and port."""
    monkeypatch.setattr(
        "scripts.server.config_io.load_server_config",
        lambda: {"host": "0.0.0.0", "port": 9000, "open_browser": True},
    )
    assert _resolve_app_url() == "http://0.0.0.0:9000"


def test_resolve_app_url_defaults_when_block_empty(monkeypatch):
    """An empty [server] block falls back to the historical localhost:8765."""
    monkeypatch.setattr("scripts.server.config_io.load_server_config", lambda: {})
    assert _resolve_app_url() == _DEFAULT_APP_URL
    assert _DEFAULT_APP_URL == "http://localhost:8765"


def test_resolve_app_url_falls_back_on_error(monkeypatch):
    """If reading the config raises (e.g. the [server] extra isn't installed),
    the default is returned so the notification still fires."""
    def boom():
        raise ModuleNotFoundError("tomli_w")  # simulate missing [server] extra

    monkeypatch.setattr("scripts.server.config_io.load_server_config", boom)
    assert _resolve_app_url() == _DEFAULT_APP_URL
