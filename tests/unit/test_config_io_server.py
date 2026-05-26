"""Unit tests for the [server] block helpers in scripts.server.config_io.

Covers:
  1. load_server_config returns {} when the secrets file is absent
  2. load_server_config returns {} when secrets exist but [server] is absent
  3. save_server_config round-trips correctly
  4. save_server_config preserves unrelated top-level tables (e.g. [zotero])
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.server import config_io


@pytest.fixture()
def tmp_secrets(monkeypatch, tmp_path: Path) -> Path:
    """Redirect SECRETS_PATH to a tmp file for the duration of the test."""
    secrets_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_io, "SECRETS_PATH", secrets_path)
    return secrets_path


@pytest.mark.unit
def test_load_server_config_missing_file(tmp_secrets: Path) -> None:
    """When the secrets file does not exist, return {}."""
    assert not tmp_secrets.exists()
    assert config_io.load_server_config() == {}


@pytest.mark.unit
def test_load_server_config_missing_block(tmp_secrets: Path) -> None:
    """When secrets exist but no [server] block, return {}."""
    tmp_secrets.write_text(
        '[zotero]\napi_key = "k"\nlibrary_id = "1"\n', encoding="utf-8",
    )
    assert config_io.load_server_config() == {}


@pytest.mark.unit
def test_save_server_config_round_trip(tmp_secrets: Path) -> None:
    """save_server_config followed by load_server_config returns the same dict."""
    config_io.save_server_config(host="0.0.0.0", port=9000, open_browser=False)
    loaded = config_io.load_server_config()
    assert loaded == {"host": "0.0.0.0", "port": 9000, "open_browser": False}


@pytest.mark.unit
def test_save_server_config_preserves_other_tables(tmp_secrets: Path) -> None:
    """Writing [server] must not clobber [zotero] or other top-level keys."""
    tmp_secrets.write_text(
        '[zotero]\napi_key = "abc"\nlibrary_id = "42"\n'
        '[pubmed]\nemail = "u@example.com"\n',
        encoding="utf-8",
    )
    config_io.save_server_config(host="127.0.0.1", port=8765, open_browser=True)

    full = config_io.load_secrets()
    assert full["zotero"] == {"api_key": "abc", "library_id": "42"}
    assert full["pubmed"] == {"email": "u@example.com"}
    assert full["server"] == {"host": "127.0.0.1", "port": 8765, "open_browser": True}
