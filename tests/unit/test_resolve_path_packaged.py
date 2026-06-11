"""resolve_path must fall back to packaged ship-defaults when no repo
``config/`` and no user-dir override is present — the pip-installed-wheel case.

These tests exercise the packaged-data fallback added in Phase 2: ship-default
runtime data (extraction/review schemas + prompts) lives under
``lit_monitor/_data/config/`` and must be discoverable from an arbitrary CWD
with an empty user config dir.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import lit_monitor.core.path_utils as pu


def test_resolve_path_finds_packaged_schema(monkeypatch, tmp_path):
    # No repo config/, CWD elsewhere, empty user dir -> must fall back to packaged data.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".config" / "lit-monitor").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    importlib.reload(pu)
    p = pu.resolve_path(Path("config/extraction_schema.yaml"))
    assert p.is_file() and "extraction_schema" in p.name


def test_resolve_path_finds_packaged_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".config" / "lit-monitor").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    importlib.reload(pu)
    p = pu.resolve_path(Path("config/prompts/rationale.yaml"))
    assert p.is_file()


def test_packaged_fallback_maps_top_level_config_to_example():
    # A top-level user config (config/domain_context.yaml) has no relocated
    # _data/config/ copy — its ship-default lives only as
    # _data/config_examples/<name>.example.yaml. The packaged-data tier must map
    # to that location so an unseeded wheel (no repo config/, empty user dir)
    # still resolves it. Tested on the helper directly because the in-repo
    # ancestor walk would otherwise shadow it with the real repo file — in a
    # true wheel there is no repo config/ to shadow it.
    p = pu._packaged_data_path(Path("config/domain_context.yaml"))
    assert p is not None and p.is_file()
    assert p.name == "domain_context.example.yaml"


def test_packaged_fallback_finds_relocated_schema():
    # The relocated ship-defaults resolve via the direct _data/<path> mirror.
    p = pu._packaged_data_path(Path("config/extraction_schema.yaml"))
    assert p is not None and p.is_file() and p.name == "extraction_schema.yaml"
