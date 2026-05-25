"""
V-6 test backfill — E2 (relink citation graph).

Tests for relink.py Pass 2b: incoming citations from citation_edges table
populating the ## Referenced By persist zone.

All I/O uses tmp_path; no Zotero or LLM calls needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOTE_TEMPLATE = """\
---
doi: "{doi}"
title: "{title}"
source_type: "paper"
---

# {title}

Body text here.

{{% persist "related_work" %}}
## Related Work
*(populated by `lit-monitor obsidian relink`)*
{{% endpersist %}}

{{% persist "referenced_by" %}}
## Referenced By
*(populated by `lit-monitor obsidian relink`)*
{{% endpersist %}}
"""


def _make_note(tmp_path: Path, doi: str, title: str) -> Path:
    safe_name = doi.replace("/", "_").replace(":", "_")
    path = tmp_path / f"{safe_name}.md"
    path.write_text(_NOTE_TEMPLATE.format(doi=doi, title=title), encoding="utf-8")
    return path


def _make_state_db(tmp_path: Path):
    from scripts.core.state_db import StateDB
    return StateDB(tmp_path / "state.db")


def _make_embeddings_db() -> MagicMock:
    db = MagicMock()
    db.find_similar_to_text.return_value = []
    return db


# ---------------------------------------------------------------------------
# E2 — Referenced By section populated from citation_edges
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_referenced_by_section_populated_from_state_db(tmp_path):
    """
    When citation_edges contains a resolved edge (source_doi → target_doi),
    relink_note must populate the ## Referenced By zone of the target note
    with the source paper's note_title.
    """
    from scripts.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    embeddings_db = _make_embeddings_db()

    # Register target paper (the note being relinked)
    state_db.upsert_paper({
        "doi": "10.1/target",
        "title": "Target Paper",
        "source_type": "paper",
        "note_title": "Smith2021_TargetPaper",
    })
    # Register source paper (the one that cites the target)
    state_db.upsert_paper({
        "doi": "10.1/source",
        "title": "Source Paper",
        "source_type": "paper",
        "note_title": "Jones2022_SourcePaper",
    })
    # Create a resolved citation edge: source → target
    state_db.upsert_citation_edge(
        source_doi="10.1/source",
        ref_id="[1]",
        target_doi="10.1/target",
        target_s2_id=None,
        context="Target Paper is foundational.",
        resolution="numeric_index",
    )

    note_path = _make_note(tmp_path, "10.1/target", "Target Paper")
    relink_note(note_path, embeddings_db, state_db)

    content = note_path.read_text(encoding="utf-8")
    assert "Jones2022_SourcePaper" in content, (
        "Expected source note title in Referenced By zone"
    )
    assert "← [[Jones2022_SourcePaper]] cites this" in content


@pytest.mark.unit
def test_referenced_by_library_only(tmp_path):
    """
    When the source_doi in citation_edges is not in the papers table (no note_title),
    it must NOT appear in the ## Referenced By zone of the target note.
    """
    from scripts.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    embeddings_db = _make_embeddings_db()

    # Register target paper (only the target is in the library)
    state_db.upsert_paper({
        "doi": "10.1/target2",
        "title": "Another Target",
        "source_type": "paper",
        "note_title": "Brown2023_AnotherTarget",
    })
    # Create a citation edge where source is NOT in the papers table
    state_db.upsert_citation_edge(
        source_doi="10.1/not-in-library",
        ref_id="[5]",
        target_doi="10.1/target2",
        target_s2_id=None,
        context="Mentioned in passing.",
        resolution="numeric_index",
    )

    note_path = _make_note(tmp_path, "10.1/target2", "Another Target")
    relink_note(note_path, embeddings_db, state_db)

    content = note_path.read_text(encoding="utf-8")
    # Source is not in the library → Referenced By zone must stay as the placeholder
    assert "← [[" not in content, (
        "An out-of-library source must not appear in Referenced By"
    )


@pytest.mark.unit
def test_referenced_by_not_overwritten_when_empty(tmp_path):
    """
    When no resolved incoming citation edges exist, relink_note must NOT replace
    the placeholder text in ## Referenced By.
    """
    from scripts.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    embeddings_db = _make_embeddings_db()

    state_db.upsert_paper({
        "doi": "10.1/alone",
        "title": "Lonely Paper",
        "source_type": "paper",
        "note_title": "Kim2020_LonelyPaper",
    })

    note_path = _make_note(tmp_path, "10.1/alone", "Lonely Paper")

    relink_note(note_path, embeddings_db, state_db)

    content = note_path.read_text(encoding="utf-8")
    # Referenced By zone should still be intact (no incoming edges added)
    assert "## Referenced By" in content
    assert "← [[" not in content
