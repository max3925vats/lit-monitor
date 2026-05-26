"""Unit tests for ``scripts.server.routes.zotero``.

All Zotero traffic is mocked — these tests never touch the network and
never read ``~/.config/lit-monitor/config.toml``. ``load_secrets`` is
monkeypatched at the route module so the secrets file is irrelevant.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.unit
def test_collections_503_when_creds_missing(client, monkeypatch):
    """No api_key in secrets → 503 with a wizard-friendly detail."""
    monkeypatch.setattr("scripts.server.routes.zotero.load_secrets", lambda: {})
    resp = client.get("/api/zotero/collections")
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"].lower()


@pytest.mark.unit
def test_collections_returns_rows_from_zotero_client(client, monkeypatch):
    """Mock ZoteroClient.get_all_collections — endpoint returns flat key/name list."""
    monkeypatch.setattr(
        "scripts.server.routes.zotero.load_secrets",
        lambda: {"zotero": {"api_key": "k", "library_id": "12345"}},
    )
    mock_client = MagicMock()
    mock_client.get_all_collections.return_value = [
        {"key": "ABC", "name": "Polymers", "parent_collection_key": None},
        {"key": "DEF", "name": "Filtration", "parent_collection_key": "ABC"},
    ]
    monkeypatch.setattr(
        "scripts.server.routes.zotero._build_client",
        lambda: mock_client,
    )
    resp = client.get("/api/zotero/collections")
    assert resp.status_code == 200
    assert resp.json() == {
        "collections": [
            {"key": "ABC", "name": "Polymers"},
            {"key": "DEF", "name": "Filtration"},
        ]
    }


@pytest.mark.unit
def test_test_endpoint_handles_api_error(client, monkeypatch):
    """If the ZoteroClient probe raises, ``/test`` returns ``{ok: False, message}``."""
    monkeypatch.setattr(
        "scripts.server.routes.zotero.load_secrets",
        lambda: {"zotero": {"api_key": "k", "library_id": "12345"}},
    )
    mock_client = MagicMock()
    mock_client._zot.collections.side_effect = RuntimeError("401 Unauthorized")
    monkeypatch.setattr(
        "scripts.server.routes.zotero._build_client",
        lambda: mock_client,
    )
    resp = client.get("/api/zotero/test")
    # Endpoint always returns 200, even when the API rejects creds.
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "401" in body["message"]
