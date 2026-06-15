"""
Unit tests for scripts/vocabulary/topic_merger.py (M4).

Covers: _is_novel() threshold, novel-vs-existing detection,
dual-file append, comment-preservation via raw-text append,
and edge cases (empty input, missing files, duplicate topics).
All tests are marked @pytest.mark.unit and require no external services.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_topics_yaml(path: Path, names: list[str]) -> None:
    searches = [{"name": n, "query": n.lower(), "databases": ["pubmed"]} for n in names]
    path.write_text(yaml.dump({"searches": searches}), encoding="utf-8")


def _make_concepts_yaml(path: Path, names: list[str]) -> None:
    themes = [{"name": n, "keywords": [n.lower()]} for n in names]
    path.write_text("themes:\n" + yaml.dump(themes, default_flow_style=False), encoding="utf-8")


def _read_raw(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# _is_novel
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_is_novel_returns_true_for_new_topic():
    from lit_monitor.vocabulary.topic_merger import _is_novel
    assert _is_novel("system fouling", ["filtration"], []) is True


@pytest.mark.unit
def test_is_novel_returns_false_for_exact_match():
    from lit_monitor.vocabulary.topic_merger import _is_novel
    assert _is_novel("filtration", ["filtration"], []) is False


@pytest.mark.unit
def test_is_novel_returns_false_for_fuzzy_match_above_threshold():
    from lit_monitor.vocabulary.topic_merger import _is_novel
    # "diafiltration" vs "diafiltrations" — high ratio, should be not novel
    assert _is_novel("diafiltrations", ["diafiltration"], []) is False


@pytest.mark.unit
def test_is_novel_returns_true_for_dissimilar_topic():
    from lit_monitor.vocabulary.topic_merger import _is_novel
    # Completely unrelated topics have low ratio → novel
    assert _is_novel("protein aggregation", ["membrane flux"], []) is True


@pytest.mark.unit
def test_is_novel_checks_already_appended_list():
    from lit_monitor.vocabulary.topic_merger import _is_novel
    # Not in existing_names but in already_appended — should NOT be novel
    assert _is_novel("IgG purification", [], ["IgG purification"]) is False


@pytest.mark.unit
def test_is_novel_case_insensitive():
    from lit_monitor.vocabulary.topic_merger import _is_novel
    assert _is_novel("Filtration", ["filtration"], []) is False


# ---------------------------------------------------------------------------
# merge_discovered_topics — return value
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_doi_topics_returns_empty(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    result = merge_discovered_topics(
        [],
        topics_path=tmp_path / "topics.yaml",
        concepts_path=tmp_path / "concepts.yaml",
    )
    assert result == []


@pytest.mark.unit
def test_novel_topic_returned_in_list(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, ["filtration"])
    result = merge_discovered_topics(
        [("10.1/test", ["system fouling"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    assert result == ["system fouling"]


@pytest.mark.unit
def test_existing_topic_not_returned(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, ["filtration"])
    result = merge_discovered_topics(
        [("10.1/test", ["filtration"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    assert result == []


# ---------------------------------------------------------------------------
# topics.yaml append
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_novel_topic_appended_to_topics_yaml(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, ["filtration"])
    merge_discovered_topics(
        [("10.1/doi", ["system fouling"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    raw = _read_raw(topics_path)
    assert "system fouling" in raw


@pytest.mark.unit
def test_topic_entry_contains_auto_comment(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, [])
    merge_discovered_topics(
        [("10.1/doi", ["protein aggregation"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    raw = _read_raw(topics_path)
    assert "# auto-added" in raw
    assert "10.1/doi" in raw


@pytest.mark.unit
def test_missing_topics_yaml_skips_gracefully(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    # topics_path does not exist — should not raise
    result = merge_discovered_topics(
        [("10.1/doi", ["IgG retention"])],
        topics_path=tmp_path / "missing_topics.yaml",
        concepts_path=tmp_path / "missing_concepts.yaml",
    )
    # Topic is still returned as appended (attempted), but files not created
    # since neither file exists — so nothing was written, and the topic should
    # appear in the return list (it was "novel").
    assert result == ["IgG retention"]


@pytest.mark.unit
def test_existing_topics_yaml_not_overwritten(tmp_path):
    """Existing content is preserved after append (raw-text append, no full dump)."""
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, ["filtration"])
    original_content = _read_raw(topics_path)
    merge_discovered_topics(
        [("10.1/doi", ["system fouling"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    raw = _read_raw(topics_path)
    assert original_content in raw  # original lines still present


# ---------------------------------------------------------------------------
# concepts.yaml append
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_novel_topic_appended_to_concepts_yaml(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    concepts_path = tmp_path / "concepts.yaml"
    _make_concepts_yaml(concepts_path, ["Biologics"])
    merge_discovered_topics(
        [("10.1/doi", ["system fouling"])],
        topics_path=tmp_path / "topics.yaml",
        concepts_path=concepts_path,
    )
    raw = _read_raw(concepts_path)
    assert "system fouling" in raw


@pytest.mark.unit
def test_concepts_yaml_append_lands_as_sibling_theme(tmp_path):
    """Audit R28 regression — the appended theme must parse as a SIBLING of
    existing themes, not as a dict-shaped keyword nested inside the last
    theme.  Prior implementation hard-coded 2-space indent and broke when
    existing themes used 0-space indent (the default of yaml.dump).
    """
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    concepts_path = tmp_path / "concepts.yaml"
    _make_concepts_yaml(concepts_path, ["Biologics"])
    merge_discovered_topics(
        [("10.1/doi", ["IgG aggregation"])],
        topics_path=tmp_path / "topics.yaml",
        concepts_path=concepts_path,
    )

    # Reload via yaml.safe_load — this is what the consumer (assign_themes)
    # actually does, so it's the strongest correctness check.
    parsed = yaml.safe_load(concepts_path.read_text(encoding="utf-8"))
    theme_names = [t["name"] for t in parsed["themes"]]
    assert "Biologics" in theme_names
    assert "IgG aggregation" in theme_names, (
        f"New theme must be a sibling of existing themes; got: {theme_names}"
    )
    # And every theme's keywords must be plain strings, not dicts.
    for t in parsed["themes"]:
        for kw in t.get("keywords", []):
            assert isinstance(kw, str), (
                f"Theme {t['name']!r} has non-string keyword {kw!r} — "
                "indicates the append landed at the wrong indent"
            )


# ---------------------------------------------------------------------------
# A3-3 (TF-4) — appends must be atomic
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_topics_append_uses_atomic_write(tmp_path, monkeypatch):
    """A3-3: appending to topics.yaml must go through atomic_write_text, not a
    raw write_text — a crash mid-write must not corrupt the user's config.

    We spy on the module-level atomic_write_text symbol (mirroring the
    obsidian_writer atomicity tests) and assert it is invoked with the topics
    path. The content assertions in the other tests still guarantee the write
    actually lands.
    """
    import lit_monitor.vocabulary.topic_merger as tm

    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, [])
    calls: list[Path] = []
    real = tm.atomic_write_text

    def _spy(path, content, *a, **kw):
        calls.append(Path(path))
        return real(path, content, *a, **kw)

    monkeypatch.setattr(tm, "atomic_write_text", _spy)

    merge_result = tm.merge_discovered_topics(
        [("10.1/doi", ["system fouling"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",  # absent → no concepts write
    )
    assert merge_result == ["system fouling"]
    # atomic_write_text was used for the topics file (and not bypassed).
    assert topics_path in calls, (
        f"topics.yaml append did not route through atomic_write_text; calls={calls}"
    )
    # And the content actually landed.
    assert "system fouling" in topics_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_concepts_append_uses_atomic_write(tmp_path, monkeypatch):
    """A3-3: appending to concepts.yaml must also route through atomic_write_text."""
    import lit_monitor.vocabulary.topic_merger as tm

    concepts_path = tmp_path / "concepts.yaml"
    _make_concepts_yaml(concepts_path, ["Biologics"])
    calls: list[Path] = []
    real = tm.atomic_write_text

    def _spy(path, content, *a, **kw):
        calls.append(Path(path))
        return real(path, content, *a, **kw)

    monkeypatch.setattr(tm, "atomic_write_text", _spy)

    tm.merge_discovered_topics(
        [("10.1/doi", ["IgG aggregation"])],
        topics_path=tmp_path / "topics.yaml",  # absent → no topics write
        concepts_path=concepts_path,
    )
    assert concepts_path in calls, (
        f"concepts.yaml append did not route through atomic_write_text; calls={calls}"
    )
    assert "IgG aggregation" in concepts_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Deduplication across DOIs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_same_topic_from_two_dois_appended_only_once(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, [])
    result = merge_discovered_topics(
        [
            ("10.1/doi1", ["system fouling"]),
            ("10.1/doi2", ["system fouling"]),
        ],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    assert result == ["system fouling"]  # only once in the return list
    # Confirm only one occurrence in the file
    raw = _read_raw(topics_path)
    assert raw.count("system fouling") == 1


@pytest.mark.unit
def test_two_distinct_novel_topics_both_appended(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, [])
    result = merge_discovered_topics(
        [("10.1/doi", ["system fouling", "protein aggregation"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    assert set(result) == {"system fouling", "protein aggregation"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_topic_strings_ignored(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, [])
    result = merge_discovered_topics(
        [("10.1/doi", ["", "  ", None])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    assert result == []


@pytest.mark.unit
def test_empty_topics_list_for_doi_skipped(tmp_path):
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, [])
    result = merge_discovered_topics(
        [("10.1/doi", [])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    assert result == []


# ---------------------------------------------------------------------------
# N6 regression tests — placement bug and scalar encoding
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_append_with_trailing_scalar_keys_produces_valid_yaml(tmp_path):
    """N6: auto-added entries must land inside searches: list even when scalar
    keys follow the list (e.g. date_window_days, screen_all).
    """
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    # Reproduce the real-world topics.yaml layout: searches: list followed by
    # top-level scalar keys at the same indent level.
    topics_path.write_text(
        "searches:\n"
        "- name: existing topic\n"
        "  query: existing AND topic\n"
        "  databases: [pubmed, arxiv]\n"
        "date_window_days: 14\n"
        "discovery_top_k: 20\n"
        "screen_all: false\n",
        encoding="utf-8",
    )
    merge_discovered_topics(
        [("10.1/doi", ["system fouling"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    raw = topics_path.read_text(encoding="utf-8")
    # File must parse cleanly — the main acceptance criterion for N6.
    parsed = yaml.safe_load(raw)
    assert parsed is not None, "topics.yaml did not parse after append"
    searches = parsed.get("searches", [])
    names = [s["name"] for s in searches if isinstance(s, dict)]
    assert "system fouling" in names, "new topic not in searches list"
    # Trailing scalar keys must still be present and correct.
    assert parsed.get("date_window_days") == 14
    assert parsed.get("screen_all") is False


@pytest.mark.unit
def test_append_does_not_produce_document_end_marker(tmp_path):
    """N6: auto-added entries must not contain the YAML document-end marker '...'."""
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    _make_topics_yaml(topics_path, [])
    merge_discovered_topics(
        [("10.1/doi", ["tethered diblock copolymers"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    raw = topics_path.read_text(encoding="utf-8")
    # A bare '...' on its own line is the YAML document-end marker — must be absent.
    lines = raw.splitlines()
    assert "..." not in lines, f"Document-end marker found in topics.yaml after append: {raw!r}"


@pytest.mark.unit
def test_append_with_two_indent_style_produces_valid_yaml(tmp_path):
    """N6: also works when the topics.yaml uses 2-space-indented list items (git HEAD format)."""
    from lit_monitor.vocabulary.topic_merger import merge_discovered_topics
    topics_path = tmp_path / "topics.yaml"
    topics_path.write_text(
        "searches:\n"
        "  - name: existing topic\n"
        "    query: existing AND topic\n"
        "    databases: [pubmed, arxiv]\n"
        "date_window_days: 14\n"
        "screen_all: false\n",
        encoding="utf-8",
    )
    merge_discovered_topics(
        [("10.1/doi", ["protein aggregation"])],
        topics_path=topics_path,
        concepts_path=tmp_path / "concepts.yaml",
    )
    parsed = yaml.safe_load(topics_path.read_text(encoding="utf-8"))
    names = [s["name"] for s in parsed.get("searches", []) if isinstance(s, dict)]
    assert "protein aggregation" in names
    assert parsed.get("date_window_days") == 14


# ---------------------------------------------------------------------------
# P4.3 — all-writes-fail must not report success
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_all_writes_fail_not_reported_as_appended(tmp_path, monkeypatch):
    """P4.3: when both _append_* raise, the topic is NOT reported as appended."""
    import lit_monitor.core.strict_mode as _sm
    import lit_monitor.vocabulary.topic_merger as tm

    topics_path = tmp_path / "topics.yaml"
    concepts_path = tmp_path / "concepts.yaml"
    _make_topics_yaml(topics_path, [])
    _make_concepts_yaml(concepts_path, [])

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(tm, "_append_topics_entry", _boom)
    monkeypatch.setattr(tm, "_append_concepts_entry", _boom)

    _sm.set_strict(False)
    result = tm.merge_discovered_topics(
        [("10.1/doi", ["system fouling"])],
        topics_path=topics_path,
        concepts_path=concepts_path,
    )
    # Topic never reached disk → must not be reported as appended.
    assert result == []


@pytest.mark.unit
def test_all_writes_fail_raises_under_strict(tmp_path, monkeypatch):
    """P4.3: in --strict, an all-writes-fail topic escalates to a raise."""
    import lit_monitor.core.strict_mode as _sm
    import lit_monitor.vocabulary.topic_merger as tm

    topics_path = tmp_path / "topics.yaml"
    concepts_path = tmp_path / "concepts.yaml"
    _make_topics_yaml(topics_path, [])
    _make_concepts_yaml(concepts_path, [])

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(tm, "_append_topics_entry", _boom)
    monkeypatch.setattr(tm, "_append_concepts_entry", _boom)

    _sm.set_strict(True)
    with pytest.raises(Exception, match="all .* write\\(s\\) failed"):
        tm.merge_discovered_topics(
            [("10.1/doi", ["system fouling"])],
            topics_path=topics_path,
            concepts_path=concepts_path,
        )


# ---------------------------------------------------------------------------
# Phase 2 (PyPI packaging): None-default path resolution targets config_dir.
# A wheel run from an arbitrary CWD must read-modify-append the USER's
# topics.yaml/concepts.yaml under config_dir(), NOT a CWD-relative config/.
# LIT_MONITOR_ROOT redirects config_dir() to tmp so this never touches real
# config.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_default_paths_resolve_to_config_dir(tmp_path, monkeypatch):
    import lit_monitor.vocabulary.topic_merger as tm

    # Redirect the active config dir to tmp/config via LIT_MONITOR_ROOT.
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setenv("LIT_MONITOR_ROOT", str(tmp_path))
    # Park the CWD somewhere with NO config/ dir so a CWD-relative fallback
    # would no-op (proving the append really went to config_dir, not CWD).
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    _make_topics_yaml(cfg_dir / "topics.yaml", ["filtration"])
    _make_concepts_yaml(cfg_dir / "concepts.yaml", ["Filtration"])

    # No explicit *_path kwargs → must resolve to config_dir()'s files.
    result = tm.merge_discovered_topics([("10.1/test", ["system fouling"])])

    assert result == ["system fouling"]
    # The novel topic must have been appended to the config_dir copy.
    appended = (cfg_dir / "topics.yaml").read_text(encoding="utf-8")
    assert "system fouling" in appended
    # And nothing was created under the CWD.
    assert not (elsewhere / "config").exists()


@pytest.mark.unit
def test_splice_warns_and_noops_when_block_missing(caplog):
    import logging

    from lit_monitor.vocabulary import topic_merger
    text = "# yaml with no searches block\nother: 1\n"
    with caplog.at_level(logging.WARNING):
        out = topic_merger._splice_into_list_block(text, "searches:", "  - name: foo\n")
    assert "name: foo" not in out
    assert out == text
    assert any("not found" in r.message.lower() for r in caplog.records)
