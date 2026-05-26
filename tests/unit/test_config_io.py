"""Unit tests for scripts.server.config_io.

Covers:
  1. load_config prefers the real file over the .example fallback
  2. load_config falls back to .example.yaml when the real file is missing
  3. save_config writes atomically (os.replace called once)
  4. save_secrets writes mode 0600
  5. load_secrets returns {} when the secrets file is absent
  6. save_secrets creates missing parent directories
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from scripts.server import config_io


def _tmp_resolver(tmp_path: Path):
    """Build a _resolve_path replacement that searches under tmp_path only.

    Mirrors prompt_registry._resolve_path semantics: try the real file first,
    then fall back to a sibling *.example.yaml.  Restricting the search to
    tmp_path keeps tests independent of whatever sits in the real repo.
    """
    def _resolve(path: Path) -> Path:
        candidates = [path]
        if path.suffix == ".yaml" and not path.name.endswith(".example.yaml"):
            candidates.append(path.with_suffix(".example.yaml"))
        for cand in candidates:
            full = tmp_path / cand
            if full.exists():
                return full
        raise FileNotFoundError(f"Cannot find {path} under {tmp_path}")
    return _resolve


@pytest.mark.unit
def test_load_config_prefers_real_file_over_example(monkeypatch, tmp_path: Path) -> None:
    """Given both foo.yaml and foo.example.yaml, prefer the real file."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "foo.yaml").write_text("name: real\n", encoding="utf-8")
    (cfg_dir / "foo.example.yaml").write_text("name: example\n", encoding="utf-8")

    monkeypatch.setattr(config_io, "_resolve_path", _tmp_resolver(tmp_path))

    assert config_io.load_config("foo") == {"name": "real"}


@pytest.mark.unit
def test_load_config_falls_back_to_example(monkeypatch, tmp_path: Path) -> None:
    """When only foo.example.yaml exists, fallback should return its contents."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "foo.example.yaml").write_text("name: example\n", encoding="utf-8")

    monkeypatch.setattr(config_io, "_resolve_path", _tmp_resolver(tmp_path))

    assert config_io.load_config("foo") == {"name": "example"}


@pytest.mark.unit
def test_save_config_atomic_replace(monkeypatch, tmp_path: Path) -> None:
    """save_config must call os.replace(tmp, target) exactly once.

    We mock os.replace so the rename doesn't actually happen; the assertion
    is purely about the atomic-rename contract.  We clean up the orphan temp
    file ourselves at the end.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()

    replace_mock = MagicMock()
    monkeypatch.setattr(config_io.os, "replace", replace_mock)

    config_io.save_config("foo", {"a": 1})

    assert replace_mock.call_count == 1
    args, _kwargs = replace_mock.call_args
    tmp_arg, target_arg = args
    # Temp file should be a sibling of the eventual target.
    assert Path(target_arg) == Path("config") / "foo.yaml"
    # mkstemp returns an absolute path; compare just the parent's name.
    assert Path(tmp_arg).parent.name == "config"
    assert Path(tmp_arg).name.startswith(".foo.yaml.")
    assert Path(tmp_arg).name.endswith(".tmp")

    # mkstemp left a real file on disk because we mocked the rename; clean up
    # so the tmp_path teardown doesn't complain.
    try:
        os.unlink(tmp_arg)
    except OSError:
        pass


@pytest.mark.unit
def test_save_secrets_chmod_0600(monkeypatch, tmp_path: Path) -> None:
    """The secrets TOML must be written with mode 0600 (owner read/write only)."""
    secrets_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_io, "SECRETS_PATH", secrets_path)

    config_io.save_secrets({"zotero": {"api_key": "x"}})

    assert secrets_path.exists()
    mode = stat.S_IMODE(secrets_path.stat().st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


@pytest.mark.unit
def test_load_secrets_returns_empty_when_absent(monkeypatch, tmp_path: Path) -> None:
    """If the secrets TOML doesn't exist, load_secrets returns {} (never raises)."""
    missing = tmp_path / "does_not_exist" / "config.toml"
    monkeypatch.setattr(config_io, "SECRETS_PATH", missing)

    assert config_io.load_secrets() == {}


@pytest.mark.unit
def test_save_secrets_creates_parent_dir(monkeypatch, tmp_path: Path) -> None:
    """save_secrets must create any missing parent directories."""
    nested = tmp_path / "nested" / "deeper" / "config.toml"
    assert not nested.parent.exists()
    monkeypatch.setattr(config_io, "SECRETS_PATH", nested)

    config_io.save_secrets({"k": "v"})

    assert nested.parent.is_dir()
    assert nested.exists()
    # Sanity: the content round-trips through tomllib.
    import tomllib

    with nested.open("rb") as fh:
        assert tomllib.load(fh) == {"k": "v"}


# Smoke check for save_config round-trip (not one of the required six, but
# guards the YAML serializer choice — kept tiny on purpose).
@pytest.mark.unit
def test_save_config_roundtrip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"name": "round", "items": [1, 2, 3]}
    final = config_io.save_config("foo", data)

    assert final == Path("config") / "foo.yaml"
    loaded = yaml.safe_load(final.read_text(encoding="utf-8"))
    assert loaded == data
