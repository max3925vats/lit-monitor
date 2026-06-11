"""Tests for the ``lit-monitor reset`` Click command.

These tests focus on the confirmation flow — wiring between Click and the
pure-function deletion layer in ``scripts.setup.reset``. The deletion logic
itself is covered by ``test_reset.py``.

Strategy:
  - Build a tmp ``~/.config/lit-monitor`` and tmp vault on disk.
  - Stub ``_make_config`` so the CLI sees those tmp paths.
  - Stub ``_project_config_dir`` so the auto-generated drafts also land in tmp.
  - Drive ``reset state`` / ``reset vault`` / ``reset all`` via Click's
    CliRunner and inspect stdout + the resulting filesystem state.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from lit_monitor.cli import main
from lit_monitor.setup import reset as reset_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def tmp_env(tmp_path: Path) -> dict:
    """Materialise a complete fake env: state DB, chroma, config dir, vault.

    Every reset target is given a real file/dir so the deletion paths
    actually execute (rather than short-circuiting on the absent branch).
    Returns a dict with the resolved paths plus a fake Config.
    """
    # Production state.
    state_db = tmp_path / "config_home" / "state.db"
    state_db.parent.mkdir(parents=True)
    state_db.write_bytes(b"sqlite dummy")
    chroma_dir = state_db.parent / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "chroma.sqlite3").write_bytes(b"chroma")

    # Credentials file MUST NOT be touched by any reset command.
    credentials = state_db.parent / "config.toml"
    credentials.write_text('[zotero]\napi_key = "secret"\n')

    # Auto-generated config drafts.
    config_dir = tmp_path / "repo" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "concepts_draft.yaml").write_text("a: 1")
    (config_dir / "concepts_draft_refinement_raw.txt").write_text("raw")
    (config_dir / "topics_suggested.yaml").write_text("b: 2")
    # A user-tracked config that must survive.
    (config_dir / "paths.yaml").write_text("obsidian: {vault_path: x}")

    # Vault with one .md per generated folder + a theme page + a dev sandbox file.
    vault = tmp_path / "vault"
    for sub in ("Papers", "Books", "Digests", "Connections"):
        d = vault / "Literature" / sub
        d.mkdir(parents=True)
        (d / "note.md").write_text(f"# {sub}")
    # Theme page at vault root — must survive.
    (vault / "MyTheme.md").write_text("# my theme")
    # Dev sandbox — must survive.
    dev_dir = vault / "Literature" / "_Dev"
    dev_dir.mkdir(parents=True)
    (dev_dir / "sandbox.md").write_text("dev")

    fake_cfg = SimpleNamespace(
        state_db=SimpleNamespace(path=state_db),
        obsidian=SimpleNamespace(
            vault_path=vault,
            papers_folder="Literature/Papers",
            books_folder="Literature/Books",
            digests_folder="Literature/Digests",
            connections_folder="Literature/Connections",
        ),
    )
    return {
        "state_db": state_db,
        "chroma_dir": chroma_dir,
        "credentials": credentials,
        "config_dir": config_dir,
        "vault": vault,
        "dev_dir": dev_dir,
        "fake_cfg": fake_cfg,
    }


def _invoke(runner: CliRunner, tmp_env: dict, argv: list[str], stdin: str):
    """Run a reset subcommand against the tmp env.

    Patches both the CLI's config factory and the reset module's project
    config dir resolver so all reset paths land inside tmp_env. Logging
    setup is also stubbed to avoid creating a real logs directory.
    """
    with patch("lit_monitor.cli._make_config", return_value=tmp_env["fake_cfg"]), patch.object(
        reset_mod, "_project_config_dir", return_value=tmp_env["config_dir"]
    ), patch("lit_monitor.cli._setup_logging"):
        return runner.invoke(main, argv, input=stdin)


# ---------------------------------------------------------------------------
# Group registration
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_reset_group_is_registered(runner: CliRunner) -> None:
    """`lit-monitor --help` should list the reset group."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "reset" in result.output


@pytest.mark.unit
def test_reset_help_lists_subcommands(runner: CliRunner) -> None:
    """`lit-monitor reset --help` should list state / vault / all."""
    result = runner.invoke(main, ["reset", "--help"])
    assert result.exit_code == 0
    for sub in ("state", "vault", "all"):
        assert sub in result.output


# ---------------------------------------------------------------------------
# reset state — confirmation flow
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_reset_state_aborts_on_wrong_phrase(runner: CliRunner, tmp_env: dict) -> None:
    """Typing 'yes' (not the exact phrase) must abort with no file changes."""
    result = _invoke(runner, tmp_env, ["reset", "state"], stdin="yes\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    # Nothing deleted.
    assert tmp_env["state_db"].exists()
    assert tmp_env["chroma_dir"].exists()
    assert (tmp_env["config_dir"] / "concepts_draft.yaml").exists()


@pytest.mark.unit
def test_reset_state_aborts_on_wrong_case(runner: CliRunner, tmp_env: dict) -> None:
    """Case-sensitive: 'RESET STATE' must abort."""
    result = _invoke(runner, tmp_env, ["reset", "state"], stdin="RESET STATE\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert tmp_env["state_db"].exists()


@pytest.mark.unit
def test_reset_state_proceeds_on_exact_phrase(
    runner: CliRunner, tmp_env: dict
) -> None:
    """Typing 'reset state' exactly must delete every state target."""
    result = _invoke(runner, tmp_env, ["reset", "state"], stdin="reset state\n")
    assert result.exit_code == 0, result.output
    assert "Aborted" not in result.output
    # State wiped.
    assert not tmp_env["state_db"].exists()
    assert not tmp_env["chroma_dir"].exists()
    assert not (tmp_env["config_dir"] / "concepts_draft.yaml").exists()
    assert not (tmp_env["config_dir"] / "topics_suggested.yaml").exists()
    # Credentials survive.
    assert tmp_env["credentials"].exists()
    # Tracked config survives.
    assert (tmp_env["config_dir"] / "paths.yaml").exists()
    # Next-step hint shown.
    assert "brain-build" in result.output


@pytest.mark.unit
def test_reset_state_shows_preserved_block(
    runner: CliRunner, tmp_env: dict
) -> None:
    """The 'NOT touched' callout must mention credentials and dev sandbox."""
    result = _invoke(runner, tmp_env, ["reset", "state"], stdin="\n")
    assert "NOT touched" in result.output
    assert "credentials" in result.output
    assert "dev sandbox" in result.output or "chroma_dev" in result.output


# ---------------------------------------------------------------------------
# reset vault — confirmation flow
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_reset_vault_aborts_on_wrong_phrase(
    runner: CliRunner, tmp_env: dict
) -> None:
    result = _invoke(runner, tmp_env, ["reset", "vault"], stdin="yes\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    # Vault files untouched.
    assert (tmp_env["vault"] / "Literature" / "Papers" / "note.md").exists()


@pytest.mark.unit
def test_reset_vault_proceeds_on_exact_phrase(
    runner: CliRunner, tmp_env: dict
) -> None:
    """Typing 'reset vault' exactly must remove .md files from the three managed folders."""
    result = _invoke(runner, tmp_env, ["reset", "vault"], stdin="reset vault\n")
    assert result.exit_code == 0, result.output
    # Papers (also reviews), Digests, Connections — lit-monitor-managed → wiped.
    for sub in ("Papers", "Digests", "Connections"):
        assert not (
            tmp_env["vault"] / "Literature" / sub / "note.md"
        ).exists(), f"{sub} .md should be gone"
        # Folder itself preserved for downstream pipelines.
        assert (tmp_env["vault"] / "Literature" / sub).exists()
    # Books folder is user-managed → note.md must survive.
    assert (
        tmp_env["vault"] / "Literature" / "Books" / "note.md"
    ).exists(), "books_folder content is user-managed and must survive reset vault"
    # Dev sandbox and theme page survive.
    assert (tmp_env["dev_dir"] / "sandbox.md").exists()
    assert (tmp_env["vault"] / "MyTheme.md").exists()
    # Next-step hint shown.
    assert "rerender" in result.output


@pytest.mark.unit
def test_reset_vault_warning_about_persist_zones(
    runner: CliRunner, tmp_env: dict
) -> None:
    """The vault prompt must surface a WARNING about destroying user edits."""
    result = _invoke(runner, tmp_env, ["reset", "vault"], stdin="\n")
    assert "WARNING" in result.output
    # Phrasing should mention user-edit destruction in some form.
    assert "persist-zone" in result.output or "user edits" in result.output


# ---------------------------------------------------------------------------
# reset all
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_reset_all_runs_both_confirmations(
    runner: CliRunner, tmp_env: dict
) -> None:
    """``reset all`` prompts twice — once for state, once for vault."""
    result = _invoke(
        runner,
        tmp_env,
        ["reset", "all"],
        stdin="reset state\nreset vault\n",
    )
    assert result.exit_code == 0, result.output
    # Both halves succeeded.
    assert not tmp_env["state_db"].exists()
    assert not (tmp_env["vault"] / "Literature" / "Papers" / "note.md").exists()
    # Both prompts appeared.
    assert result.output.count("Type '") >= 2


@pytest.mark.unit
def test_reset_never_touches_credentials_file(
    runner: CliRunner, tmp_env: dict
) -> None:
    """Even after a full 'reset all', config.toml must remain on disk."""
    _invoke(
        runner,
        tmp_env,
        ["reset", "all"],
        stdin="reset state\nreset vault\n",
    )
    assert tmp_env["credentials"].exists(), (
        "Credentials file must NEVER be removed by reset."
    )
    # And the contents are intact.
    assert "secret" in tmp_env["credentials"].read_text()


@pytest.mark.unit
def test_reset_all_aborting_state_still_runs_vault(
    runner: CliRunner, tmp_env: dict
) -> None:
    """Declining the state half must not silently skip the vault prompt."""
    result = _invoke(
        runner,
        tmp_env,
        ["reset", "all"],
        stdin="no\nreset vault\n",
    )
    assert result.exit_code == 0, result.output
    # State preserved.
    assert tmp_env["state_db"].exists()
    # Vault wiped.
    assert not (tmp_env["vault"] / "Literature" / "Papers" / "note.md").exists()
