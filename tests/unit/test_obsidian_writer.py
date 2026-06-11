"""
Phase 6 unit tests — Obsidian output layer.
Tests: paper notes, book notes, chapter notes, persist zones, collision handling.
All I/O goes to tmp_path; no real vault required.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------
def _make_config(tmp_path: Path) -> SimpleNamespace:
    """Minimal config object with obsidian.* attributes."""
    obsidian = SimpleNamespace(
        vault_path=str(tmp_path / "vault"),
        papers_folder="Literature/Papers",
        books_folder="Literature/Books",
        digests_folder="Literature/Digests",
    )
    config = SimpleNamespace(
        obsidian=obsidian,
        extraction_provider="ollama",
        extraction_model="qwen2.5:7b",
    )
    return config
_TEMPLATE_DIR = Path("config/templates")
_PAPER = {
    "doi": "10.1234/test.2024",
    "title": "Filtration of Model Proteins at Scale",
    "authors": ["Smith, John", "Jones, Alice"],
    "year": 2024,
    "journal": "Journal of Mock Domain",
    "zotero_key": "ABC123",
    "first_seen_date": "2024-01-15",
    "keywords": ["filtration", "ProteinA", "TopicX"],
    "themes": ["Mock Theme A"],
    "tracked_author": False,
    "fulltext_analyzed": True,
}
_EXTRACTION = {
    "core_finding": "Flux decline observed above 20 g/L.",
    "core_finding_confidence": "explicit",
    "methods_summary": "MethodB with 30 kDa membrane.",
    "methods_summary_confidence": "explicit",
    "results_summary": None,
    "results_summary_confidence": "absent",
    "study_type": "experimental",
    "study_type_confidence": "explicit",
    "scale": "lab",
    "scale_confidence": "explicit",
    "limitations": None,
    "limitations_confidence": "absent",
}

_BOOK = {
    "book_id": "isbn:9780123456789",
    "title": "Bioseparation Engineering Principles",
    "authors": ["Harrison, Roger"],
    "year": 2015,
    "isbn": "978-0-12-345678-9",
    "zotero_key": "BOOK001",
    "first_seen_date": "2024-02-01",
    "ocr_heavy": False,
    "fulltext_analyzed": True,
}
_BOOK_EXTRACTION = {
    "book_summary": "Comprehensive overview of bioseparation methods.",
    "book_summary_confidence": "explicit",
    "key_topics": "Chromatography, filtration, precipitation",
    "key_topics_confidence": "explicit",
    "target_audience": None,
    "target_audience_confidence": "absent",
    "most_relevant_chapters": "Chapters 4, 7",
    "most_relevant_chapters_confidence": "inferred",
    "relevance_to_domain": "Core TopicX reference.",
    "relevance_to_domain_confidence": "explicit",
}
_CHAPTER = {
    "chapter_number": 4,
    "chapter_title": "Filtration Fundamentals",
    "year": 2015,
    "authors": ["Harrison, Roger"],
    "ocr_heavy": False,
    "ocr_pages": [],
    "fulltext_analyzed": True,
}
_CHAPTER_EXTRACTION = {
    "chapter_summary": "Basics of UF transport mechanism.",
    "chapter_summary_confidence": "explicit",
    "key_concepts": "flux, rejection, TMP",
    "key_concepts_confidence": "explicit",
    "methods_or_frameworks": None,
    "methods_or_frameworks_confidence": "absent",
    "key_results_or_conclusions": "Concentration polarization is dominant.",
    "key_results_or_conclusions_confidence": "explicit",
    "actionable_insights": None,
    "actionable_insights_confidence": "absent",
    "relevance_to_domain": "Foundational TopicX content.",
    "relevance_to_domain_confidence": "explicit",
}
# ---------------------------------------------------------------------------
# Paper note tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_paper_note_written(tmp_path):
    """write_paper_note creates a .md file in the papers folder."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    note_path = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    path = Path(note_path)
    assert path.exists(), f"Note file not found: {path}"
    assert path.suffix == ".md"
    content = path.read_text(encoding="utf-8")
    # YAML frontmatter present

    assert "source_type: \"paper\"" in content
    assert "doi:" in content
    # Extraction fields rendered
    assert "Flux decline observed above 20 g/L." in content
@pytest.mark.unit
def test_paper_note_null_fields_hidden(tmp_path):
    """Fields with null extraction values should not appear as headings."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    note_path = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    # results_summary is None → should not appear as a section
    assert "## Results Summary" not in content
@pytest.mark.unit
def test_paper_note_inferred_flag_shown(tmp_path):
    """Inferred confidence fields should show ⚠️ indicator."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    extraction_with_inferred = dict(_EXTRACTION)
    extraction_with_inferred["methods_summary_confidence"] = "inferred"
    note_path = write_paper_note(_PAPER, extraction_with_inferred, config,
    template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    assert "⚠️" in content
# ---------------------------------------------------------------------------
# Persist zone tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_persist_zones_survive_update(tmp_path):
    """
    When a note is regenerated, user content inside persist zones
    must be preserved unchanged.
    """
    from lit_monitor.output.obsidian_writer import update_note_preserve_persist_zones
    note_path = tmp_path / "TestNote.md"
    # Write initial note with a persist zone that has content
    initial = (
        "# Test\n"
        '{% persist "related_work" %}\n'
        "## Related Work\n"
        "- [[SomeOtherPaper2023_Related_Work]]\n"
        "{% endpersist %}\n"
    )
    note_path.write_text(initial, encoding="utf-8")
    # Regenerate with empty persist zone
    new_content = (
        "# Test (updated)\n"
        '{% persist "related_work" %}\n'
        "## Related Work\n"
        "*(populated by `lit-monitor obsidian relink`)*\n"
        "{% endpersist %}\n"
    )
    update_note_preserve_persist_zones(note_path, new_content)
    result = note_path.read_text(encoding="utf-8")
    # Updated heading present
    assert "# Test (updated)" in result
    # User content in persist zone preserved
    assert "[[SomeOtherPaper2023_Related_Work]]" in result

@pytest.mark.unit
def test_persist_zones_created_fresh_if_no_existing_note(tmp_path):
    """When writing a new note, persist zones are left as-is (no prior content)."""
    from lit_monitor.output.obsidian_writer import update_note_preserve_persist_zones
    note_path = tmp_path / "NewNote.md"
    content = (
        "# New\n"
        '{% persist "synthesis" %}\n'
        "## Synthesis\n"
        "*(empty)*\n"
        "{% endpersist %}\n"
    )
    update_note_preserve_persist_zones(note_path, content)
    result = note_path.read_text(encoding="utf-8")
    assert "## Synthesis" in result
    assert "*(empty)*" in result


@pytest.mark.unit
def test_note_write_survives_replace_failure(tmp_path, monkeypatch):
    """H4 — Atomic-write parity. If os.replace fails partway through, the
    original note content must remain untouched and no .tmp sibling must leak.
    """
    from lit_monitor.core import atomic_write as atomic_write_mod
    from lit_monitor.output.obsidian_writer import update_note_preserve_persist_zones

    note_path = tmp_path / "Atomic.md"
    original = (
        "# Original\n"
        '{% persist "notes" %}\n'
        "user-authored content that must survive crashes\n"
        "{% endpersist %}\n"
    )
    note_path.write_text(original, encoding="utf-8")

    # Inject a crash at the os.replace boundary used by atomic_write_text.
    def boom(_src, _dst):  # noqa: ANN001 — match os.replace signature
        raise OSError("simulated rename crash")

    monkeypatch.setattr(atomic_write_mod.os, "replace", boom)

    new_content = (
        "# Updated\n"
        '{% persist "notes" %}\n'
        "freshly rendered placeholder\n"
        "{% endpersist %}\n"
    )

    with pytest.raises(OSError, match="simulated rename crash"):
        update_note_preserve_persist_zones(note_path, new_content)

    # Original file content is unchanged.
    assert note_path.read_text(encoding="utf-8") == original
    # No temp-file droppings left behind.
    leftovers = [p for p in note_path.parent.iterdir() if p.name.startswith(".") and p.name.endswith(".tmp")]
    assert leftovers == [], f"leaked temp files: {leftovers}"


# ---------------------------------------------------------------------------
# Collision handling tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_title_collision_handled(tmp_path):
    """
    When a note with the generated title already exists in the folder,
    write_paper_note appends 'b' to produce a unique title.
    """
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    # Write the first note
    note_path_1 = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    # Write a second note with the same metadata
    note_path_2 = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    stem_1 = Path(note_path_1).stem
    stem_2 = Path(note_path_2).stem
    # Second note should have a different stem (with collision suffix)
    assert stem_1 != stem_2
    assert stem_2.endswith("_2")
@pytest.mark.unit
def test_note_title_normalises_allcaps_surname():
    """ALL-CAPS surnames are Title-cased in the note title."""
    from lit_monitor.output.obsidian_writer import note_title_for
    title = note_title_for(["HARRISON R"], 2015, "Bioseparation_Principles")
    assert title.startswith("Harrison2015_")
@pytest.mark.unit
def test_note_title_chapter_suffix():
    """Chapter suffix _Ch04 is appended correctly."""
    from lit_monitor.output.obsidian_writer import note_title_for
    title = note_title_for(["Smith, John"], 2024, "Filtration_Fund", suffix="_Ch04")
    assert "_Ch04" in title
# ---------------------------------------------------------------------------
# Path validation tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_write_paper_note_uses_absolute_path(tmp_path):
    """write_paper_note produces an absolute path regardless of platform."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    note_path = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    result_path = Path(note_path)
    assert result_path.is_absolute(), f"Expected absolute path, got: {note_path}"
    assert result_path.exists()
@pytest.mark.unit
def test_write_paper_note_no_backslash_in_content(tmp_path):
    """Note content should not contain Windows backslashes in wikilinks or paths."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    note_path = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    # Wikilinks and internal note references must use forward slashes
    # (Obsidian uses forward slash on all platforms)
    lines_with_wikilinks = [ln for ln in content.split("\n") if "[[" in ln]
    for line in lines_with_wikilinks:
        assert "\\\\" not in line, f"Backslash found in wikilink: {line}"
@pytest.mark.unit
def test_collision_suffix_numeric_sequence(tmp_path):
    """
    Multiple collisions produce _2, _3, _4... suffixes (not alphabet).
    """
    from lit_monitor.output.obsidian_writer import _collision_suffix
    existing = {"base", "base_2", "base_3"}
    result = _collision_suffix("base", existing)
    assert result == "base_4"
@pytest.mark.unit
def test_collision_suffix_no_collision():
    """No collision → returns base unchanged."""
    from lit_monitor.output.obsidian_writer import _collision_suffix
    result = _collision_suffix("unique_title", {"other_title"})
    assert result == "unique_title"


# ---------------------------------------------------------------------------
# C2 — Dataview-compatible YAML front matter
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_obsidian_note_front_matter_dataview_compatible(tmp_path):
    """C2: paper note YAML front matter must be parseable and Dataview-compatible.

    Checks:
    - The YAML front matter block parses without error (no bare keywords).
    - `year` is an int (Dataview requires numeric for range queries).
    - `keywords` is a list of strings (all items quoted in the raw YAML).
    - `tags` is present and is a list of strings (Dataview tag filtering).
    - Tag slugs contain only [a-z0-9/-] characters (no spaces, no special chars).
    - Keywords with spaces or special characters do not cause YAML parse errors.
    """
    import re

    import yaml

    from lit_monitor.output.obsidian_writer import write_paper_note

    paper_with_tricky_keywords = dict(_PAPER)
    paper_with_tricky_keywords["keywords"] = [
        "filtration",
        "ProteinA purification",          # space → should not break YAML
        "TopicX: concentration step",  # colon → would break bare YAML value
        "C4 resin",
    ]
    paper_with_tricky_keywords["themes"] = ["Mock Theme A", "Downstream Processing"]

    config = _make_config(tmp_path)
    note_path = write_paper_note(
        paper_with_tricky_keywords, _EXTRACTION, config, template_dir=_TEMPLATE_DIR
    )
    content = Path(note_path).read_text(encoding="utf-8")

    # Extract the YAML front matter block between the two `---` delimiters
    fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert fm_match is not None, "No YAML front matter found in rendered note"
    front_matter_raw = fm_match.group(1)

    # Must parse without exception
    fm = yaml.safe_load(front_matter_raw)
    assert fm is not None

    # year must be int (not a string like "" or "2024")
    assert isinstance(fm["year"], int), f"year must be int, got {type(fm['year'])}: {fm['year']!r}"
    assert fm["year"] == 2024

    # keywords must be a list; items must all be strings
    assert isinstance(fm["keywords"], list), "keywords must be a YAML list"
    for kw in fm["keywords"]:
        assert isinstance(kw, str), f"keyword item must be str, got {type(kw)}: {kw!r}"

    # tags must be present and be a list
    assert "tags" in fm, "tags field missing from front matter"
    assert isinstance(fm["tags"], list), "tags must be a YAML list"
    assert len(fm["tags"]) > 0, "tags list must not be empty"

    # Every tag slug must match [a-z0-9/-]+ (no spaces, uppercase, or special chars)
    _valid_tag = re.compile(r"^[a-z0-9/\-]+$")
    for tag in fm["tags"]:
        assert isinstance(tag, str)
        assert _valid_tag.match(tag), f"Tag has invalid characters: {tag!r}"


@pytest.mark.unit
def test_coerce_year_variants():
    """_coerce_year handles int, str, None, and unparseable gracefully."""
    from lit_monitor.output.obsidian_writer import _coerce_year

    assert _coerce_year(2024) == 2024
    assert _coerce_year("2024") == 2024
    assert _coerce_year(None) == 0
    assert _coerce_year("") == 0
    assert _coerce_year("not-a-year") == 0


@pytest.mark.unit
def test_make_tags_sanitization():
    """_make_tags produces lowercase hyphenated slugs and deduplicates."""
    from lit_monitor.output.obsidian_writer import _make_tags

    tags = _make_tags(
        keywords=["Filtration", "ProteinA purification", "TopicX: concentration"],
        themes=["Mock Theme A", "Filtration"],  # dupe of keyword
    )
    # All lowercase
    for t in tags:
        assert t == t.lower(), f"Tag not lowercase: {t!r}"
    # No spaces
    for t in tags:
        assert " " not in t, f"Tag contains space: {t!r}"
    # No colons or other special chars
    import re
    _valid = re.compile(r"^[a-z0-9/\-]+$")
    for t in tags:
        assert _valid.match(t), f"Tag has invalid characters: {t!r}"
    # Deduplication: "filtration" from keywords and themes both map to same slug
    assert tags.count("filtration") == 1


@pytest.mark.unit
def test_make_tags_non_string_inputs():
    """_make_tags coerces None and int items rather than raising AttributeError."""
    from lit_monitor.output.obsidian_writer import _make_tags

    # Should not raise — returns whatever sanitization produces from str(raw)
    tags = _make_tags(keywords=[None, 42, "valid-tag"], themes=[])  # type: ignore[list-item]
    # None → "none", 42 → "42", "valid-tag" → "valid-tag"
    assert "none" in tags
    assert "42" in tags
    assert "valid-tag" in tags


@pytest.mark.unit
def test_tags_yaml_reserved_words_parse_correctly(tmp_path):
    """Tags from YAML-reserved keywords ('yes', 'null') must parse as strings, not booleans."""
    import re

    import yaml

    from lit_monitor.output.obsidian_writer import write_paper_note

    paper = dict(_PAPER)
    paper["keywords"] = ["yes", "null", "true", "off"]  # YAML reserved words
    paper["themes"] = []

    config = _make_config(tmp_path)
    note_path = write_paper_note(paper, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")

    fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert fm_match is not None
    fm = yaml.safe_load(fm_match.group(1))

    for tag in fm.get("tags", []):
        assert isinstance(tag, str), f"YAML reserved word parsed as non-string type: {tag!r}"
# ---------------------------------------------------------------------------
# I3 — degraded extraction quality tag and warning callout
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_degraded_extraction_adds_quality_tag(tmp_path):
    """
    When extraction contains _extraction_quality='degraded_high_chunking', the
    written note must include the 'extraction-quality-degraded' YAML tag.
    """
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    extraction = dict(_EXTRACTION)
    extraction["_extraction_quality"] = "degraded_high_chunking"
    note_path = write_paper_note(_PAPER, extraction, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    assert "extraction-quality-degraded" in content


@pytest.mark.unit
def test_degraded_extraction_prepends_warning_callout(tmp_path):
    """
    When extraction contains _extraction_quality='degraded_high_chunking', the
    written note body must contain the Obsidian warning callout block.
    """
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    extraction = dict(_EXTRACTION)
    extraction["_extraction_quality"] = "degraded_high_chunking"
    note_path = write_paper_note(_PAPER, extraction, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    assert "[!warning]" in content
    assert "Extraction Quality Degraded" in content


@pytest.mark.unit
def test_normal_extraction_has_no_quality_callout(tmp_path):
    """
    When _extraction_quality is absent, the note must NOT contain the warning callout.
    """
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    note_path = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    assert "Extraction Quality Degraded" not in content
    assert "extraction-quality-degraded" not in content


@pytest.mark.unit
def test_discovered_topics_rendered_in_front_matter(tmp_path):
    """M4: discovered_topics from extraction must appear as a YAML list in note front matter."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    extraction_with_topics = {
        **_EXTRACTION,
        "discovered_topics": ["system fouling", "protein aggregation"],
    }
    note_path = write_paper_note(_PAPER, extraction_with_topics, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    # Field must be inside the YAML front matter block (before the closing ---)
    fm_end = content.find("\n---", 3)
    front_matter = content[:fm_end] if fm_end != -1 else content
    assert "discovered_topics:" in front_matter, "discovered_topics missing from front matter"
    assert "system fouling" in front_matter
    assert "protein aggregation" in front_matter


@pytest.mark.unit
def test_discovered_topics_empty_when_absent_from_extraction(tmp_path):
    """M4: notes written without discovered_topics must render an empty list, not an error."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    note_path = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    fm_end = content.find("\n---", 3)
    front_matter = content[:fm_end] if fm_end != -1 else content
    assert "discovered_topics: []" in front_matter


@pytest.mark.unit
def test_write_paper_note_backfills_overall_confidence_when_absent(tmp_path):
    """N8c: notes from old DB rows (no _overall_confidence) must compute and render it."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    # _EXTRACTION has no _overall_confidence — simulates pre-fix DB row
    assert "_overall_confidence" not in _EXTRACTION
    note_path = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    content = Path(note_path).read_text(encoding="utf-8")
    fm_end = content.find("\n---", 3)
    front_matter = content[:fm_end] if fm_end != -1 else content
    assert "extraction_confidence:" in front_matter


@pytest.mark.unit
def test_write_paper_note_does_not_mutate_extraction_dict(tmp_path):
    """N8c: write_paper_note must not modify the caller's extraction dict."""
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)
    extraction_copy = dict(_EXTRACTION)
    write_paper_note(_PAPER, extraction_copy, config, template_dir=_TEMPLATE_DIR)
    assert "_overall_confidence" not in _EXTRACTION, "original _EXTRACTION dict was mutated"


@pytest.mark.unit
def test_write_paper_note_uses_extraction_provider_from_paper_dict(tmp_path):
    """N12 Gap 2 / N8d: extraction_provider in the paper dict takes precedence over the config.

    The N8d fix routes through paper.get('extraction_provider') first.
    This test verifies the PRIMARY path (paper dict) is used, not the fallback (config attr).
    """
    from lit_monitor.output.obsidian_writer import write_paper_note
    config = _make_config(tmp_path)  # config has extraction_provider="ollama"
    # Paper dict explicitly carries a different provider (cloud model string).
    paper_with_provider = {
        **_PAPER,
        "extraction_provider": "ollama:gemma4:31b-cloud",
        "extraction_model": "gemma4:31b-cloud",
    }
    note_path = write_paper_note(
        paper_with_provider, _EXTRACTION, config, template_dir=_TEMPLATE_DIR
    )
    content = Path(note_path).read_text(encoding="utf-8")
    fm_end = content.find("\n---", 3)
    front_matter = content[:fm_end] if fm_end != -1 else content
    assert "ollama:gemma4:31b-cloud" in front_matter, (
        "paper dict's extraction_provider must appear in note front matter"
    )
    # Also verify that when the paper dict has no provider, config is used as fallback.
    note_path2 = write_paper_note(_PAPER, _EXTRACTION, config, template_dir=_TEMPLATE_DIR)
    content2 = Path(note_path2).read_text(encoding="utf-8")
    fm_end2 = content2.find("\n---", 3)
    front_matter2 = content2[:fm_end2] if fm_end2 != -1 else content2
    assert "ollama" in front_matter2, "config fallback extraction_provider must appear when paper dict is empty"
