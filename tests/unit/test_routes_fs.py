"""Unit tests for the ``/api/fs/*`` endpoints.

These tests stub ``Path.home`` to point at ``tmp_path`` so the HOME guard
can be exercised against a sandbox without touching the real home dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.unit
def test_ls_returns_entries_for_real_dir(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ls`` on a real tmp dir returns the expected entries with proper types."""

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.txt").write_text("x")

    resp = client.get("/api/fs/ls", params={"path": str(tmp_path)})
    assert resp.status_code == 200

    body = resp.json()
    names = [e["name"] for e in body["entries"]]
    # Directories sort before files.
    assert names == ["sub", "file.txt"]
    assert {e["type"] for e in body["entries"]} == {"dir", "file"}
    # At the HOME root, parent must be null so the UI can't navigate above it.
    assert body["parent"] is None


@pytest.mark.unit
def test_ls_refuses_path_outside_home(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absolute path outside ``$HOME`` must return 403."""

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    resp = client.get("/api/fs/ls", params={"path": "/etc"})
    assert resp.status_code == 403


@pytest.mark.unit
def test_exists_false_for_missing_path(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``exists`` returns the false-triple for a path that doesn't exist."""

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    resp = client.get("/api/fs/exists", params={"path": str(tmp_path / "nope")})
    assert resp.status_code == 200
    assert resp.json() == {"exists": False, "is_dir": False, "is_writable": False}


@pytest.mark.unit
def test_exists_true_for_writable_tmp(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``exists`` reports tmp_path as existing, a dir, and writable."""

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    resp = client.get("/api/fs/exists", params={"path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json() == {"exists": True, "is_dir": True, "is_writable": True}
