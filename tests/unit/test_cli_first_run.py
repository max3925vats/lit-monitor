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


@pytest.mark.unit
def test_first_run_idempotency_mixed(
    runner: CliRunner, monkeypatch, tmp_path: Path,
) -> None:
    """L5: mixed idempotency — credentials are write-once, [server] re-prompted.

    Runs `first-run` twice against the same SECRETS_PATH:
      * Run 1 establishes credentials + an initial [server] block.
      * Run 2 must NOT re-prompt for credentials (zotero/pubmed answers absent
        from stdin) BUT must re-prompt for [server] and persist the new values.
    """
    secrets_path = tmp_path / "config.toml"
    # Pre-existing credentials → both runs must skip credential prompts.
    secrets_path.write_text(
        '[zotero]\napi_key = "k"\nlibrary_id = "1"\n'
        '[pubmed]\nemail = "u@example.com"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_io, "SECRETS_PATH", secrets_path)
    from scripts.setup import _paths as setup_paths
    monkeypatch.setattr(setup_paths, "SECRETS_PATH", secrets_path)

    popen_mock = MagicMock()
    sock_mock = MagicMock(side_effect=OSError("refused"))

    # ---- Run 1: host=127.0.0.1, port=8765, open_browser=n
    run1_input = "127.0.0.1\n8765\nn\n"
    with patch("subprocess.Popen", popen_mock), \
         patch("socket.create_connection", sock_mock), \
         patch("webbrowser.open"), \
         patch("time.sleep"):
        r1 = runner.invoke(main, ["first-run"], input=run1_input)
    assert r1.exit_code == 0, r1.output
    # First run wrote the initial [server] block.
    assert config_io.load_server_config() == {
        "host": "127.0.0.1", "port": 8765, "open_browser": False,
    }
    # Sanity: the "skipping credential prompts" path fired.
    assert "skipping credential prompts" in r1.output

    # Reset mocks between runs so call counts reflect run 2 only.
    popen_mock.reset_mock()
    sock_mock.reset_mock()

    # ---- Run 2: different server answers (host=0.0.0.0, port=9000, open_browser=y).
    # Crucially, we provide ONLY the three [server] answers. If first_run tries
    # to also re-prompt for zotero/pubmed credentials, Click will run out of
    # input and the invocation will fail — which is exactly the regression we
    # want to catch.
    run2_input = "0.0.0.0\n9000\ny\n"
    with patch("subprocess.Popen", popen_mock), \
         patch("socket.create_connection", sock_mock), \
         patch("webbrowser.open"), \
         patch("time.sleep"):
        r2 = runner.invoke(main, ["first-run"], input=run2_input)

    assert r2.exit_code == 0, r2.output
    # Credentials skipped again.
    assert "skipping credential prompts" in r2.output
    # Credential prompts MUST NOT appear in run 2's output.
    assert "Zotero API key" not in r2.output
    assert "PubMed contact email" not in r2.output
    # [server] WAS re-prompted: host/port prompts visible in output.
    assert "Server host" in r2.output
    assert "Server port" in r2.output
    # [server] block updated to the new values — re-prompting is meaningful.
    assert config_io.load_server_config() == {
        "host": "0.0.0.0", "port": 9000, "open_browser": True,
    }
    # Pre-existing credentials still intact (write-once).
    full = config_io.load_secrets()
    assert full["zotero"] == {"api_key": "k", "library_id": "1"}
    assert full["pubmed"] == {"email": "u@example.com"}
