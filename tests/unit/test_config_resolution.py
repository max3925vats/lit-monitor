"""Three-tier config resolution tests.

Every test here controls HOME + CWD via monkeypatch + tmp_path so it can
never pass by accidentally finding the dev machine's real
``~/.config/lit-monitor/`` or the repo's ``./config/``. This is the #1
packaging-bug risk: a resolution mistake that is invisible on a dev box.
"""
import importlib
from pathlib import Path


def _reload_config(monkeypatch, home: Path, cwd: Path):
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LIT_MONITOR_ROOT", raising=False)
    monkeypatch.chdir(cwd)
    import lit_monitor.core.config as cfg
    return importlib.reload(cfg)


def test_user_dir_is_primary_config_location(monkeypatch, tmp_path):
    home = tmp_path / "home"
    user_cfg = home / ".config" / "lit-monitor"
    user_cfg.mkdir(parents=True)
    (user_cfg / "paths.yaml").write_text("obsidian_vault_path: /v\nzotero_library_id: '1'\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    cfg = _reload_config(monkeypatch, home, elsewhere)
    assert cfg.config_dir() == user_cfg


def test_repo_config_is_dev_fallback(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".config" / "lit-monitor").mkdir(parents=True)
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='lit-monitor'\n")
    (repo / "config" / "paths.yaml").write_text("obsidian_vault_path: /v\nzotero_library_id: '1'\n")
    cfg = _reload_config(monkeypatch, home, repo)
    assert cfg.config_dir() == repo / "config"


def test_explicit_override_wins(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".config" / "lit-monitor").mkdir(parents=True)
    override = tmp_path / "override"
    (override / "config").mkdir(parents=True)
    (override / "config" / "paths.yaml").write_text("obsidian_vault_path: /v\nzotero_library_id: '1'\n")
    monkeypatch.setenv("LIT_MONITOR_ROOT", str(override))
    import lit_monitor.core.config as cfg
    cfg = importlib.reload(cfg)
    assert cfg.config_dir() == override / "config"
