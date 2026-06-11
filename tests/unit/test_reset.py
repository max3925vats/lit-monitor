"""Unit tests for ``scripts/setup/reset.py``.

These tests cover the pure target-enumeration and deletion functions. The
Click confirmation flow is tested separately in ``test_cli_reset.py``.

Key invariants under test:
  - Target enumeration excludes credentials, dev sandbox, and theme pages.
  - Re-running a reset after a successful wipe is a no-op (idempotent).
  - After the chroma rmtree, ``SharedSystemClient.clear_system_cache`` is
    called so the next ``EmbeddingsDB()`` in the same process doesn't hand
    back a stale handle.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lit_monitor.setup import reset as reset_mod


# ---------------------------------------------------------------------------
# Fake config builder
# ---------------------------------------------------------------------------
def _fake_config(
    *,
    state_db_path: Path,
    vault_path: Path,
    papers_folder: str = "Literature/Papers",
    books_folder: str = "Literature/Books",
    digests_folder: str = "Literature/Digests",
    connections_folder: str = "Literature/Connections",
) -> SimpleNamespace:
    """Build a stand-in for the ``Config`` object.

    Mirrors the shape used elsewhere in the test suite (see
    ``test_dev_sandbox._stub_get_config``). Only the attributes ``reset``
    actually reads are populated; everything else stays missing so an
    accidental new read site immediately raises AttributeError instead of
    silently picking up a default.
    """
    return SimpleNamespace(
        state_db=SimpleNamespace(path=state_db_path),
        obsidian=SimpleNamespace(
            vault_path=vault_path,
            papers_folder=papers_folder,
            books_folder=books_folder,
            digests_folder=digests_folder,
            connections_folder=connections_folder,
        ),
    )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_state_targets_enumerates_correct_paths(tmp_path: Path) -> None:
    """state_targets must return the 5 expected target labels in order."""
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=tmp_path / "vault",
    )
    # Point the project config dir at tmp so we don't accidentally probe
    # the real ``config/`` directory inside the repo.
    with patch.object(
        reset_mod, "_project_config_dir", return_value=tmp_path / "config"
    ):
        targets = reset_mod.state_targets(cfg)

    labels = [t.label for t in targets]
    # G12 adds "graph DB" between "ChromaDB persist dir" and the config drafts.
    assert labels == [
        "state DB",
        "ChromaDB persist dir",
        "graph DB",
        "concepts_draft.yaml",
        "concepts_draft_refinement_raw.txt",
        "topics_suggested.yaml",
    ]
    # chroma should sit alongside state DB; graph default is ~/.config/lit-monitor/graph.kuzu
    paths_by_label = {t.label: t.path for t in targets}
    assert paths_by_label["state DB"] == tmp_path / "state.db"
    assert paths_by_label["ChromaDB persist dir"] == tmp_path / "chroma"
    assert paths_by_label["concepts_draft.yaml"] == tmp_path / "config" / "concepts_draft.yaml"
    # Graph DB target must not be the dev sandbox path.
    assert "graph_dev" not in str(paths_by_label["graph DB"])


@pytest.mark.unit
def test_state_targets_never_includes_credentials_or_dev_sandbox(
    tmp_path: Path,
) -> None:
    """No state target should ever resolve to config.toml or a *_dev path."""
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=tmp_path / "vault",
    )
    with patch.object(
        reset_mod, "_project_config_dir", return_value=tmp_path / "config"
    ):
        targets = reset_mod.state_targets(cfg)
    target_paths = {str(t.path) for t in targets}
    for path in target_paths:
        assert "config.toml" not in path
        assert "state_dev" not in path
        assert "chroma_dev" not in path
        assert "logs" not in path


# ---------------------------------------------------------------------------
# State reset behaviour
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_perform_state_reset_removes_files(tmp_path: Path) -> None:
    """Happy path: every present target is deleted."""
    state_db = tmp_path / "state.db"
    state_db.write_bytes(b"sqlite dummy")
    (tmp_path / "chroma").mkdir()
    (tmp_path / "chroma" / "chroma.sqlite3").write_bytes(b"x")
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "concepts_draft.yaml").write_text("a: 1")
    (cfg_dir / "topics_suggested.yaml").write_text("b: 2")
    # concepts_draft_refinement_raw.txt intentionally absent → skipped

    cfg = _fake_config(state_db_path=state_db, vault_path=tmp_path / "vault")
    with patch.object(reset_mod, "_project_config_dir", return_value=cfg_dir):
        targets = reset_mod.state_targets(cfg)
        results = reset_mod.perform_state_reset(targets)

    deleted = {r.label for r in results if r.deleted}
    assert "state DB" in deleted
    assert "ChromaDB persist dir" in deleted
    assert "concepts_draft.yaml" in deleted
    assert "topics_suggested.yaml" in deleted

    skipped = {r.label for r in results if not r.deleted}
    assert "concepts_draft_refinement_raw.txt" in skipped

    assert not state_db.exists()
    assert not (tmp_path / "chroma").exists()
    assert not (cfg_dir / "concepts_draft.yaml").exists()


@pytest.mark.unit
def test_perform_state_reset_unlinks_sqlite_wal_shm(tmp_path: Path) -> None:
    """state.db-wal and state.db-shm siblings must be removed alongside state.db."""
    state_db = tmp_path / "state.db"
    state_db.write_bytes(b"main")
    (tmp_path / "state.db-wal").write_bytes(b"wal")
    (tmp_path / "state.db-shm").write_bytes(b"shm")

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg = _fake_config(state_db_path=state_db, vault_path=tmp_path / "vault")
    with patch.object(reset_mod, "_project_config_dir", return_value=cfg_dir):
        targets = reset_mod.state_targets(cfg)
        reset_mod.perform_state_reset(targets)

    assert not state_db.exists()
    assert not (tmp_path / "state.db-wal").exists()
    assert not (tmp_path / "state.db-shm").exists()


@pytest.mark.unit
def test_perform_state_reset_idempotent(tmp_path: Path) -> None:
    """Second call on the same paths must produce all 'not present' results."""
    state_db = tmp_path / "state.db"
    state_db.write_bytes(b"x")
    (tmp_path / "chroma").mkdir()
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()

    cfg = _fake_config(state_db_path=state_db, vault_path=tmp_path / "vault")
    with patch.object(reset_mod, "_project_config_dir", return_value=cfg_dir):
        targets1 = reset_mod.state_targets(cfg)
        reset_mod.perform_state_reset(targets1)
        # Re-enumerate (now every path is absent) and run again.
        targets2 = reset_mod.state_targets(cfg)
        results2 = reset_mod.perform_state_reset(targets2)

    assert all(not r.deleted for r in results2)
    assert all(r.skipped_reason == "not present" for r in results2)


@pytest.mark.unit
def test_perform_state_reset_clears_chromadb_cache(tmp_path: Path) -> None:
    """After the chroma rmtree, SharedSystemClient.clear_system_cache must fire.

    Without this, a stale in-process chromadb client would still point at the
    deleted persist dir and the next ``EmbeddingsDB()`` write would fail with
    SQLITE_READONLY_DBMOVED.
    """
    state_db = tmp_path / "state.db"
    state_db.write_bytes(b"x")
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg = _fake_config(state_db_path=state_db, vault_path=tmp_path / "vault")

    with patch.object(reset_mod, "_project_config_dir", return_value=cfg_dir):
        targets = reset_mod.state_targets(cfg)
        with patch.object(reset_mod, "_clear_chromadb_cache") as clear_mock:
            reset_mod.perform_state_reset(targets)
            clear_mock.assert_called_once()


@pytest.mark.unit
def test_perform_state_reset_clears_chromadb_cache_even_if_absent(
    tmp_path: Path,
) -> None:
    """Cache eviction must run even if the chroma dir didn't exist.

    A prior ``EmbeddingsDB()`` instantiation in the same process may have
    populated chromadb's cache against a now-missing path; the next caller
    still needs a clean slate.
    """
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db", vault_path=tmp_path / "vault"
    )
    with patch.object(reset_mod, "_project_config_dir", return_value=cfg_dir):
        targets = reset_mod.state_targets(cfg)
        with patch.object(reset_mod, "_clear_chromadb_cache") as clear_mock:
            reset_mod.perform_state_reset(targets)
            clear_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Vault target enumeration
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_vault_targets_enumerates_three_folders(tmp_path: Path) -> None:
    """vault_targets covers Papers (also reviews) / Digests / Synthesis(=Connections).

    Reviews share ``papers_folder`` with regular papers — see
    ``obsidian_writer.write_paper_note`` — so the Papers entry covers them
    too. ``books_folder`` is intentionally excluded because book items are
    routed to the ``skip`` pipeline and lit-monitor does not write there.
    """
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=tmp_path / "vault",
    )
    targets = reset_mod.vault_targets(cfg)
    labels = [t.label for t in targets]
    assert labels == [
        "Papers folder",
        "Digests folder",
        "Synthesis folder",
    ]


@pytest.mark.unit
def test_vault_targets_excludes_books_folder(tmp_path: Path) -> None:
    """books_folder must NOT appear in the vault target list.

    Books/bookSections are routed to ``pipeline: "skip"`` in
    ``config/item_routing.yaml``; any notes in that folder are user-managed
    content outside lit-monitor's scope.
    """
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=tmp_path / "vault",
        books_folder="Literature/Books",
    )
    targets = reset_mod.vault_targets(cfg)
    for tgt in targets:
        path_str = str(tgt.path)
        assert "Books" not in path_str, (
            f"books_folder leaked into target {tgt.label}: {path_str}"
        )
        assert tgt.label != "Reviews folder", (
            "Reviews are co-located in Papers folder; no standalone Reviews target."
        )


@pytest.mark.unit
def test_vault_targets_excludes_dev_sandbox(tmp_path: Path) -> None:
    """Literature/_Dev/ must NEVER appear in the vault target list."""
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=tmp_path / "vault",
    )
    targets = reset_mod.vault_targets(cfg)
    for tgt in targets:
        assert "_Dev" not in str(tgt.path), f"Dev sandbox leaked into {tgt.label}"


@pytest.mark.unit
def test_vault_targets_excludes_theme_pages(tmp_path: Path) -> None:
    """Top-level vault .md files (theme pages) must not be targets.

    Every target must be a subdirectory of the vault, never a file at the
    vault root.
    """
    vault = tmp_path / "vault"
    cfg = _fake_config(state_db_path=tmp_path / "state.db", vault_path=vault)
    targets = reset_mod.vault_targets(cfg)
    for tgt in targets:
        # No target should equal the vault root itself.
        assert tgt.path != vault
        # And every target path must be inside Literature/, not the vault root.
        rel = tgt.path.relative_to(vault)
        assert rel.parts[0] == "Literature", (
            f"Expected target inside Literature/, got {rel}"
        )


# ---------------------------------------------------------------------------
# Vault reset behaviour
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_perform_vault_reset_removes_files(tmp_path: Path) -> None:
    """Happy path: every .md file in each generated folder is removed.

    The folder itself stays in place — downstream pipelines auto-create +
    write into these folders and shouldn't have to special-case absence.
    """
    vault = tmp_path / "vault"
    papers = vault / "Literature" / "Papers"
    papers.mkdir(parents=True)
    (papers / "p1.md").write_text("# paper 1")
    (papers / "p2.md").write_text("# paper 2")

    digests = vault / "Literature" / "Digests"
    digests.mkdir(parents=True)
    (digests / "d1.md").write_text("# digest")
    # Non-md file in same folder — must NOT be deleted.
    (digests / "keep.txt").write_text("preserve")

    cfg = _fake_config(state_db_path=tmp_path / "state.db", vault_path=vault)
    targets = reset_mod.vault_targets(cfg)
    results = reset_mod.perform_vault_reset(targets, vault_root=vault)

    # Papers and Digests should report deleted=True; Synthesis absent → skipped.
    by_label = {r.label: r for r in results}
    assert by_label["Papers folder"].deleted is True
    assert by_label["Digests folder"].deleted is True
    assert by_label["Synthesis folder"].deleted is False
    # No Reviews folder target — reviews share papers_folder.
    assert "Reviews folder" not in by_label

    # Files gone, folder preserved.
    assert papers.exists()
    assert not (papers / "p1.md").exists()
    assert not (papers / "p2.md").exists()
    assert digests.exists()
    assert not (digests / "d1.md").exists()
    # Non-md preserved
    assert (digests / "keep.txt").exists()


@pytest.mark.unit
def test_perform_vault_reset_idempotent(tmp_path: Path) -> None:
    """Re-running vault reset after a clean wipe must produce all skipped."""
    vault = tmp_path / "vault"
    papers = vault / "Literature" / "Papers"
    papers.mkdir(parents=True)
    (papers / "p1.md").write_text("x")

    cfg = _fake_config(state_db_path=tmp_path / "state.db", vault_path=vault)
    reset_mod.perform_vault_reset(reset_mod.vault_targets(cfg), vault_root=vault)
    # Second pass: every folder exists but contains no .md → all skipped.
    results = reset_mod.perform_vault_reset(
        reset_mod.vault_targets(cfg), vault_root=vault
    )
    assert all(not r.deleted for r in results)


@pytest.mark.unit
def test_perform_vault_reset_does_not_touch_dev_sandbox(tmp_path: Path) -> None:
    """A markdown file inside Literature/_Dev/ must survive vault reset."""
    vault = tmp_path / "vault"
    papers = vault / "Literature" / "Papers"
    papers.mkdir(parents=True)
    (papers / "p1.md").write_text("x")
    dev = vault / "Literature" / "_Dev"
    dev.mkdir(parents=True)
    dev_note = dev / "sandbox_paper.md"
    dev_note.write_text("dev-only content")

    cfg = _fake_config(state_db_path=tmp_path / "state.db", vault_path=vault)
    reset_mod.perform_vault_reset(reset_mod.vault_targets(cfg), vault_root=vault)

    assert not (papers / "p1.md").exists()
    assert dev_note.exists(), "Dev sandbox note must survive vault reset."


# ---------------------------------------------------------------------------
# Audit-2 B2: containment guard for vault .md deletion
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_is_safely_within_real_subfolder_true(tmp_path: Path) -> None:
    """A genuine subfolder of the vault root is safe."""
    root = tmp_path / "vault"
    child = root / "Literature" / "Papers"
    assert reset_mod._is_safely_within(child, root) is True


@pytest.mark.unit
def test_is_safely_within_vault_root_itself_false(tmp_path: Path) -> None:
    """The vault root itself is NOT a safe deletion target (depth 0)."""
    root = tmp_path / "vault"
    assert reset_mod._is_safely_within(root, root) is False


@pytest.mark.unit
def test_is_safely_within_sibling_outside_false(tmp_path: Path) -> None:
    """A sibling/outside path is rejected."""
    root = tmp_path / "vault"
    outside = tmp_path / "outside"
    assert reset_mod._is_safely_within(outside, root) is False


@pytest.mark.unit
def test_is_safely_within_dotdot_escape_false(tmp_path: Path) -> None:
    """A ``..``-escaping path that resolves outside the root is rejected."""
    root = tmp_path / "vault"
    root.mkdir()
    # root / "../outside" resolves to tmp_path/outside, outside the vault.
    escaping = root / ".." / "outside"
    assert reset_mod._is_safely_within(escaping, root) is False


@pytest.mark.unit
def test_is_safely_within_nested_deep_true(tmp_path: Path) -> None:
    """A deeply nested subfolder is still safe."""
    root = tmp_path / "vault"
    deep = root / "a" / "b" / "c" / "d"
    assert reset_mod._is_safely_within(deep, root) is True


@pytest.mark.unit
def test_perform_vault_reset_happy_path_with_guard(tmp_path: Path) -> None:
    """Happy path: normal subfolders pass the guard and .md files are deleted.

    Exercises the guarded code path (``vault_root`` supplied) to prove the
    containment check does not interfere with legitimate deletions.
    """
    vault = tmp_path / "vault"
    papers = vault / "Literature" / "Papers"
    papers.mkdir(parents=True)
    (papers / "p1.md").write_text("# paper 1")
    (papers / "p2.md").write_text("# paper 2")

    cfg = _fake_config(state_db_path=tmp_path / "state.db", vault_path=vault)
    targets = reset_mod.vault_targets(cfg)
    results = reset_mod.perform_vault_reset(targets, vault_root=vault)

    by_label = {r.label: r for r in results}
    assert by_label["Papers folder"].deleted is True
    assert not (papers / "p1.md").exists()
    assert not (papers / "p2.md").exists()
    # No target should be flagged unsafe on the happy path.
    assert all(
        r.skipped_reason != "unsafe path (outside vault root)" for r in results
    )


@pytest.mark.unit
def test_perform_vault_reset_skips_empty_folder_config_attack(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Empty-string folder config resolves to the vault root → must be SKIPPED.

    ``vault_root / "" == vault_root``; deleting the vault root's *.md
    recursively would erase the whole vault. The guard must skip it, leave a
    sentinel at the vault root untouched, and record the skip with a warning.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    # Sentinel directly at the vault root — must survive.
    sentinel = vault / "ROOT_NOTE.md"
    sentinel.write_text("important user note at vault root")
    # A theme page in a subfolder too, to make the recursive blast radius real.
    other = vault / "Themes"
    other.mkdir()
    other_note = other / "theme.md"
    other_note.write_text("theme content")

    # papers_folder="" → resolves to vault root.
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=vault,
        papers_folder="",
    )
    targets = reset_mod.vault_targets(cfg)
    with caplog.at_level("WARNING"):
        results = reset_mod.perform_vault_reset(targets, vault_root=vault)

    papers_result = next(r for r in results if r.label == "Papers folder")
    assert papers_result.deleted is False
    assert papers_result.skipped_reason == "unsafe path (outside vault root)"
    # Loud warning recorded.
    assert any("unsafe" in rec.message.lower() for rec in caplog.records)
    # Sentinels at/under the vault root SURVIVE the attack.
    assert sentinel.exists(), "Vault-root note must NOT be deleted by the attack."
    assert other_note.exists(), "Sibling note must NOT be deleted by the attack."


@pytest.mark.unit
def test_perform_vault_reset_skips_absolute_escape_attack(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An absolute folder path outside the vault → must be SKIPPED.

    Builds a sibling ``outside/`` dir holding a sentinel .md and points
    ``papers_folder`` at it via an absolute path. The guard must refuse to
    touch it; the sentinel survives.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "outside_note.md"
    sentinel.write_text("file outside the vault")

    # vault_root / "/abs/path" → absolute path wins, lands outside the vault.
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=vault,
        papers_folder=str(outside),
    )
    targets = reset_mod.vault_targets(cfg)
    with caplog.at_level("WARNING"):
        results = reset_mod.perform_vault_reset(targets, vault_root=vault)

    papers_result = next(r for r in results if r.label == "Papers folder")
    assert papers_result.deleted is False
    assert papers_result.skipped_reason == "unsafe path (outside vault root)"
    assert any("unsafe" in rec.message.lower() for rec in caplog.records)
    assert sentinel.exists(), "File outside the vault must survive the attack."


@pytest.mark.unit
def test_perform_vault_reset_skips_dotdot_escape_attack(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A ``../outside`` folder config that escapes the vault → must be SKIPPED."""
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "escaped_note.md"
    sentinel.write_text("file reached via .. traversal")

    # papers_folder="../outside" → vault/../outside == tmp_path/outside.
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=vault,
        papers_folder="../outside",
    )
    targets = reset_mod.vault_targets(cfg)
    with caplog.at_level("WARNING"):
        results = reset_mod.perform_vault_reset(targets, vault_root=vault)

    papers_result = next(r for r in results if r.label == "Papers folder")
    assert papers_result.deleted is False
    assert papers_result.skipped_reason == "unsafe path (outside vault root)"
    assert any("unsafe" in rec.message.lower() for rec in caplog.records)
    assert sentinel.exists(), "Escaped file must survive the .. traversal attack."


@pytest.mark.unit
def test_perform_vault_reset_mixed_safe_and_unsafe_continues(tmp_path: Path) -> None:
    """An unsafe target is skipped but safe targets in the same run still delete."""
    vault = tmp_path / "vault"
    vault.mkdir()
    # Safe Digests folder with a real note.
    digests = vault / "Literature" / "Digests"
    digests.mkdir(parents=True)
    (digests / "d1.md").write_text("digest")
    # Unsafe Papers folder ("" → vault root) with a root sentinel.
    sentinel = vault / "ROOT.md"
    sentinel.write_text("survive")

    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=vault,
        papers_folder="",
    )
    targets = reset_mod.vault_targets(cfg)
    results = reset_mod.perform_vault_reset(targets, vault_root=vault)

    by_label = {r.label: r for r in results}
    # Unsafe one skipped...
    assert by_label["Papers folder"].deleted is False
    assert by_label["Papers folder"].skipped_reason == (
        "unsafe path (outside vault root)"
    )
    # ...but the safe one still got cleaned.
    assert by_label["Digests folder"].deleted is True
    assert not (digests / "d1.md").exists()
    assert sentinel.exists(), "Root sentinel must survive."


# ---------------------------------------------------------------------------
# Size formatter
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_format_size_bytes_renders_human_readable() -> None:
    """The size formatter should match the diagnose-style B/KB/MB output."""
    assert reset_mod._format_size_bytes(0) == "0 B"
    assert reset_mod._format_size_bytes(512) == "512 B"
    assert reset_mod._format_size_bytes(2048).endswith("KB")
    assert reset_mod._format_size_bytes(5 * 1024 * 1024).endswith("MB")
    # Exact-value pin so future refactors of the unit-walking loop can't
    # silently regress the boundary math.
    assert reset_mod._format_size_bytes(1024 * 1024 * 1024) == "1.0 GB"
    assert reset_mod._format_size_bytes(2 * 1024 * 1024) == "2.0 MB"


# ---------------------------------------------------------------------------
# G12: graph.kuzu reset tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_state_targets_includes_graph_kuzu(tmp_path: Path) -> None:
    """G12: state_targets includes the production graph.kuzu path."""
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=tmp_path / "vault",
    )
    with patch.object(reset_mod, "_project_config_dir", return_value=tmp_path / "config"):
        targets = reset_mod.state_targets(cfg)

    labels = [t.label for t in targets]
    assert "graph DB" in labels, (
        f"Expected 'graph DB' label in state_targets; got labels: {labels}"
    )
    paths_by_label = {t.label: t.path for t in targets}
    graph_path = paths_by_label["graph DB"]
    assert "graph" in str(graph_path).lower(), (
        f"graph DB target path does not look like a graph path: {graph_path}"
    )
    # Must NOT be the production ChromaDB path
    assert "chroma" not in str(graph_path).lower()


@pytest.mark.unit
def test_state_targets_graph_path_not_sandbox(tmp_path: Path) -> None:
    """G12: state_targets graph path must not be the dev sandbox path."""
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=tmp_path / "vault",
    )
    with patch.object(reset_mod, "_project_config_dir", return_value=tmp_path / "config"):
        targets = reset_mod.state_targets(cfg)

    for tgt in targets:
        assert "graph_dev" not in str(tgt.path), (
            f"state_targets must not target the dev sandbox; found: {tgt.path}"
        )


@pytest.mark.unit
def test_graph_persist_dir_returns_string(tmp_path: Path) -> None:
    """G12: _graph_persist_dir returns a non-empty string."""
    cfg = _fake_config(
        state_db_path=tmp_path / "state.db",
        vault_path=tmp_path / "vault",
    )
    result = reset_mod._graph_persist_dir(cfg)
    assert isinstance(result, str)
    assert "graph" in result.lower()


@pytest.mark.unit
def test_perform_state_reset_removes_graph_dir(tmp_path: Path) -> None:
    """G12: perform_state_reset rmtrees the graph dir when present."""
    state_db = tmp_path / "state.db"
    state_db.write_bytes(b"x")
    (tmp_path / "chroma").mkdir()
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()

    # Create a fake graph dir.
    fake_graph = tmp_path / "graph.kuzu"
    fake_graph.mkdir()
    (fake_graph / "data").write_bytes(b"placeholder")

    cfg = _fake_config(state_db_path=state_db, vault_path=tmp_path / "vault")
    with patch.object(reset_mod, "_project_config_dir", return_value=cfg_dir):
        with patch.object(reset_mod, "_graph_persist_dir", return_value=str(fake_graph)):
            targets = reset_mod.state_targets(cfg)
            results = reset_mod.perform_state_reset(targets)

    deleted_labels = {r.label for r in results if r.deleted}
    assert "graph DB" in deleted_labels, (
        f"graph DB was not deleted; results: {results}"
    )
    assert not fake_graph.exists(), "graph dir must be removed by perform_state_reset"


@pytest.mark.unit
def test_perform_state_reset_handles_missing_graph_dir(tmp_path: Path) -> None:
    """G12: missing graph.kuzu does not crash perform_state_reset (idempotent)."""
    state_db = tmp_path / "state.db"
    # No graph dir created — it simply doesn't exist.
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    missing_graph = tmp_path / "nonexistent.kuzu"

    cfg = _fake_config(state_db_path=state_db, vault_path=tmp_path / "vault")
    with patch.object(reset_mod, "_project_config_dir", return_value=cfg_dir):
        with patch.object(reset_mod, "_graph_persist_dir", return_value=str(missing_graph)):
            targets = reset_mod.state_targets(cfg)
            # Should not raise.
            results = reset_mod.perform_state_reset(targets)

    # Graph target should be listed as skipped (not present).
    graph_results = [r for r in results if r.label == "graph DB"]
    assert len(graph_results) == 1
    assert not graph_results[0].deleted
    assert graph_results[0].skipped_reason == "not present"


@pytest.mark.unit
def test_perform_state_reset_removes_graph_sentinel(tmp_path: Path) -> None:
    """G12: perform_state_reset also removes the .schema_version sentinel."""
    state_db = tmp_path / "state.db"
    state_db.write_bytes(b"x")
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()

    # Create both the kuzu file and its sentinel.
    fake_graph = tmp_path / "graph.kuzu"
    fake_graph.write_bytes(b"kuzu-placeholder")
    sentinel = Path(str(fake_graph) + ".schema_version")
    sentinel.write_text("3")

    cfg = _fake_config(state_db_path=state_db, vault_path=tmp_path / "vault")
    with patch.object(reset_mod, "_project_config_dir", return_value=cfg_dir):
        with patch.object(reset_mod, "_graph_persist_dir", return_value=str(fake_graph)):
            targets = reset_mod.state_targets(cfg)
            reset_mod.perform_state_reset(targets)

    assert not fake_graph.exists(), "graph file must be removed"
    assert not sentinel.exists(), "graph .schema_version sentinel must be removed"
