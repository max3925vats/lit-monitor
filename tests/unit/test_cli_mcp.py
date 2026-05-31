"""B4: lit-monitor mcp serve CLI tests.

Acceptance criteria:
- `lit-monitor mcp --help` shows `serve` subcommand.
- `lit-monitor mcp serve` starts the MCP server over stdio.
- Banner prints to stderr (not stdout — stdout is the MCP wire protocol).
- Missing [mcp] extra → friendly error with install hint + exit 1.
- KeyboardInterrupt exits clean (non-zero or zero, but prints shutdown msg).
"""
from __future__ import annotations

import sys

import pytest
from click.testing import CliRunner

from scripts.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    # Plain CliRunner — Click 8.4 separates stderr via result.stderr property.
    return CliRunner()


# ---------------------------------------------------------------------------
# Help visibility
# ---------------------------------------------------------------------------

class TestMcpGroup:
    def test_mcp_group_visible_in_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "mcp" in result.output.lower()

    def test_mcp_serve_in_mcp_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output.lower()


# ---------------------------------------------------------------------------
# Missing [mcp] extra — friendly error
# ---------------------------------------------------------------------------

class TestMcpServeFriendlyError:
    def test_missing_mcp_extra_prints_install_hint(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B4: when 'import mcp' raises ImportError, command exits with helpful hint."""
        import builtins

        orig_import = builtins.__import__

        def fail_mcp_import(name: str, *args, **kwargs):
            if name == "mcp" or name.startswith("mcp."):
                raise ImportError("No module named 'mcp'")
            return orig_import(name, *args, **kwargs)

        # Flush any cached mcp modules so the monkeypatch takes effect.
        for mod_name in list(sys.modules):
            if mod_name == "mcp" or mod_name.startswith("mcp."):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)

        monkeypatch.setattr(builtins, "__import__", fail_mcp_import)

        result = runner.invoke(main, ["mcp", "serve"])

        assert result.exit_code != 0
        # Error message must appear somewhere in the combined output.
        combined = result.output + (result.stderr if result.stderr else "")
        assert "[mcp] extra" in combined
        assert "uv sync --extra mcp" in combined


# ---------------------------------------------------------------------------
# Banner + server.main invocation
# ---------------------------------------------------------------------------

class TestMcpServeBannerAndRun:
    def test_serve_prints_banner_then_calls_server_main(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B4: banner prints to stderr, then graph_server.main() is called once."""
        called: dict[str, int] = {"n": 0}

        def fake_server_main() -> None:
            called["n"] += 1
            # Return immediately instead of blocking in asyncio.run.

        # Patch at the module level so the lazy import inside mcp_serve_command
        # gets the fake.
        import scripts.mcp.graph_server as gs

        monkeypatch.setattr(gs, "main", fake_server_main)

        result = runner.invoke(main, ["mcp", "serve"])

        assert result.exit_code == 0, f"stdout={result.output!r} stderr={result.stderr!r}"
        # Banner must appear on stderr (stdout is the MCP wire protocol).
        assert result.stderr is not None and "MCP" in result.stderr
        assert "stdio" in result.stderr.lower()
        # The config snippet should include the command name.
        assert "lit-monitor" in result.stderr
        # graph_server.main() called exactly once.
        assert called["n"] == 1

    def test_keyboard_interrupt_exits_cleanly(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B4: KeyboardInterrupt from server.main is caught and produces shutdown msg."""
        def raise_keyboard_interrupt() -> None:
            raise KeyboardInterrupt

        import scripts.mcp.graph_server as gs

        monkeypatch.setattr(gs, "main", raise_keyboard_interrupt)

        result = runner.invoke(main, ["mcp", "serve"])

        # Should exit cleanly (exit code 0).
        assert result.exit_code == 0
        combined_err = result.stderr or ""
        assert "shutting down" in combined_err.lower() or "ctrl" in combined_err.lower()
