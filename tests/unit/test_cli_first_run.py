"""Unit tests for the `lit-monitor first-run` Click command.

Covers:
  1. `first-run --help` exits 0 (Click registration sanity)
  2. End-to-end (mocked): writes [server] block, calls Popen once, does NOT
     re-prompt for credentials when the secrets file already exists.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scripts.cli import main
from scripts.server import config_io


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.unit
def test_first_run_help(runner: CliRunner) -> None:
    """`lit-monitor first-run --help` should print help and exit 0."""
    result = runner.invoke(main, ["first-run", "--help"])
    assert result.exit_code == 0
    assert "first-run" in result.output.lower() or "Interactive" in result.output


@pytest.mark.unit
def test_first_run_writes_server_block_and_spawns_serve(
    runner: CliRunner, monkeypatch, tmp_path: Path,
) -> None:
    """With existing credentials, first-run writes [server] and calls Popen once.

    All blocking I/O is mocked:
      - SECRETS_PATH redirected to tmp_path
      - socket.create_connection raises OSError (port "free")
      - subprocess.Popen + webbrowser.open + time.sleep stubbed
    """
    secrets_path = tmp_path / "config.toml"
    # Pre-existing credentials → first-run must skip credential prompts.
    secrets_path.write_text(
        '[zotero]\napi_key = "k"\nlibrary_id = "1"\n'
        '[pubmed]\nemail = "u@example.com"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_io, "SECRETS_PATH", secrets_path)
    # cli.py imports SECRETS_PATH lazily inside first_run() — patch at source.
    from scripts.setup import _paths as setup_paths
    monkeypatch.setattr(setup_paths, "SECRETS_PATH", secrets_path)

    popen_mock = MagicMock()
    # Port is "free" — socket.create_connection raises.
    sock_mock = MagicMock(side_effect=OSError("refused"))

    # Answers to the three server prompts: host, port, open_browser confirm (n).
    user_input = "127.0.0.1\n8765\nn\n"

    with patch("subprocess.Popen", popen_mock), \
         patch("socket.create_connection", sock_mock), \
         patch("webbrowser.open"), \
         patch("time.sleep"):
        result = runner.invoke(main, ["first-run"], input=user_input)

    assert result.exit_code == 0, result.output
    # Popen called exactly once to spawn detached serve.
    assert popen_mock.call_count == 1
    # And [server] block persisted with the chosen values.
    loaded = config_io.load_server_config()
    assert loaded == {"host": "127.0.0.1", "port": 8765, "open_browser": False}
    # Pre-existing [zotero] table untouched.
    full = config_io.load_secrets()
    assert full["zotero"] == {"api_key": "k", "library_id": "1"}
