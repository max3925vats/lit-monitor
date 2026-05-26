"""Unit tests for scripts/server/dev_sandbox.py (Task #66 D2).

Covers the four invariants that make the sandbox safe to use from /dev:

1. ``sandbox_status`` returns zeroed counts when the sandbox is empty / not
   yet materialised on disk.
2. Writing through ``sandbox_state_db()`` never touches the production
   ``state.db`` file.
3. ``clear_sandbox()`` without ``confirm=True`` is a no-op.
4. ``clear_sandbox(confirm=True)`` removes the sandbox state DB file.

ChromaDB and the obsidian vault are stubbed where needed via monkeypatch so
the tests stay hermetic and never reach the user's real config.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.server import dev_sandbox


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _stub_get_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vault_path: Path,
    state_db_path: Path,
) -> None:
    """Replace ``dev_sandbox.get_config`` with a stub matching the real shape.

    The sandbox only reads ``cfg.state_db.path`` and ``cfg.obsidian.vault_path``,
    so a SimpleNamespace covers it without dragging in the full Pydantic schema.
    """
    fake_cfg = SimpleNamespace(
        state_db=SimpleNamespace(path=str(state_db_path)),
        obsidian=SimpleNamespace(vault_path=vault_path),
    )
    monkeypatch.setattr(dev_sandbox, "get_config", lambda: fake_cfg)


# ---------------------------------------------------------------------------
# 1) Empty sandbox → zeroed counts
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sandbox_status_empty_when_nothing_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no sandbox state on disk, status should report all-zero counts."""
    # Redirect the sandbox sqlite + vault into tmp_path; chroma uses tmp too.
    sandbox_db = tmp_path / "state_dev.db"
    prod_db = tmp_path / "state.db"  # never gets written
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=prod_db,
    )

    status = dev_sandbox.sandbox_status()

    assert status["state_db_rows"] == 0
    assert status["papers_collection_count"] == 0
    assert status["chunks_collection_count"] == 0
    assert status["vault_file_count"] == 0
    assert status["last_modified"] is None


# ---------------------------------------------------------------------------
# 2) Sandbox state DB is fully separate from production state.db
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sandbox_state_db_separate_from_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write into the sandbox must NOT create/modify production state.db."""
    sandbox_db = tmp_path / "state_dev.db"
    prod_db = tmp_path / "state.db"
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=prod_db,
    )

    assert not sandbox_db.exists()
    assert not prod_db.exists()

    db = dev_sandbox.sandbox_state_db()
    db.upsert_paper(
        {
            "doi": "10.0/dev-sandbox-test",
            "title": "Dev Sandbox Probe",
            "authors": ["Tester"],
            "year": 2025,
            "source_type": "paper",
            "status": "pending",
        }
    )

    # Sandbox file came into existence; production file did NOT.
    assert sandbox_db.exists(), "Sandbox sqlite file should be created on write."
    assert not prod_db.exists(), "Production state.db must remain untouched."

    # And the row landed in the sandbox where we expect.
    status = dev_sandbox.sandbox_status()
    assert status["state_db_rows"] == 1


# ---------------------------------------------------------------------------
# 3) clear_sandbox() without confirm is a no-op
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_clear_sandbox_without_confirm_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``confirm=True``, ``clear_sandbox`` must leave files in place."""
    sandbox_db = tmp_path / "state_dev.db"
    sandbox_db.write_bytes(b"dummy-sqlite-bytes-not-a-real-db")
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=tmp_path / "state.db",
    )

    result = dev_sandbox.clear_sandbox()

    assert result == {"status": "skipped", "reason": "confirm=False"}
    assert sandbox_db.exists(), "Sandbox DB must NOT be deleted without confirm."


# ---------------------------------------------------------------------------
# 4) clear_sandbox(confirm=True) removes the sandbox state DB file
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_clear_sandbox_with_confirm_removes_state_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clear_sandbox(confirm=True)`` deletes the sandbox sqlite file."""
    sandbox_db = tmp_path / "state_dev.db"
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    _stub_get_config(
        monkeypatch,
        vault_path=tmp_path / "vault",
        state_db_path=tmp_path / "state.db",
    )

    # Materialise a real sandbox sqlite file so the unlink branch executes.
    dev_sandbox.sandbox_state_db().upsert_paper(
        {"doi": "10.0/x", "title": "x", "year": 2025, "source_type": "paper"}
    )
    assert sandbox_db.exists()

    result = dev_sandbox.clear_sandbox(confirm=True)

    assert result["status"] == "cleared"
    assert not sandbox_db.exists(), "Sandbox DB file should be deleted."
    # Action log must mention the unlink so partial-wipe failures are visible.
    assert "unlinked" in result["actions"]


# ---------------------------------------------------------------------------
# 5) Partial failure surfaces as status="partial"
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_clear_sandbox_returns_partial_on_failure(monkeypatch, tmp_path):
    """If any cleanup step raises, status should be 'partial', not 'cleared'."""
    from scripts.server import dev_sandbox
    # Point sandbox at a non-existent path so state-db unlink path returns 'no-op'
    # but make chromadb cleanup raise.
    sandbox_db = tmp_path / "state_dev.db"
    sandbox_db.touch()
    monkeypatch.setattr(dev_sandbox, "SANDBOX_STATE_DB_PATH", sandbox_db)
    # Force chromadb cleanup to raise:
    import chromadb
    def _bad_client(path):
        raise RuntimeError("chromadb unreachable")
    monkeypatch.setattr(chromadb, "PersistentClient", _bad_client)
    result = dev_sandbox.clear_sandbox(confirm=True)
    assert result["status"] == "partial", f"expected partial, got {result}"
    assert "FAILED" in result["actions"]
