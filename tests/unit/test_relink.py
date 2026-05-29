"""
V-6 test backfill — E2 (relink citation graph).

Tests for relink.py Pass 2b: incoming citations from citation_edges table
populating the ## Referenced By persist zone.

G9 additions: rag-mode dispatch (vector/graph/hybrid) tests.

All I/O uses tmp_path; no Zotero or LLM calls needed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
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


# ---------------------------------------------------------------------------
# M2 — Production template has the referenced_by persist zone; round-trip works
# ---------------------------------------------------------------------------

def _make_paper_config(tmp_path: Path) -> SimpleNamespace:
    """Minimal config for write_paper_note used by the M2 round-trip test."""
    obsidian = SimpleNamespace(
        vault_path=str(tmp_path / "vault"),
        papers_folder="Literature/Papers",
        books_folder="Literature/Books",
        digests_folder="Literature/Digests",
    )
    return SimpleNamespace(
        obsidian=obsidian,
        extraction_provider="ollama",
        extraction_model="qwen2.5:7b",
    )


@pytest.mark.unit
def test_referenced_by_persist_zone_round_trip(tmp_path, caplog):
    """
    M2: Production template defines a `referenced_by` persist zone.
    Round-trip:
      1. Render a fresh note via write_paper_note.
      2. relink_note with citing source A → A appears inside markers.
      3. relink_note with citing source B → B appears, A is gone (replaced, not appended).
      4. Persist markers remain intact across both relinks.
      5. No "Zone 'referenced_by' not found" WARNING is logged.
    """
    from scripts.obsidian_tools.relink import relink_note
    from scripts.output.obsidian_writer import write_paper_note

    config = _make_paper_config(tmp_path)
    paper = {
        "doi": "10.1/roundtrip",
        "title": "Round Trip Target",
        "authors": ["Doe, Jane"],
        "year": 2024,
        "journal": "Journal of Roundtrips",
        "zotero_key": "",
        "first_seen_date": "2024-01-01",
        "keywords": [],
        "themes": [],
        "tracked_author": False,
        "fulltext_analyzed": False,
    }
    extraction = {"core_finding": "x", "core_finding_confidence": "explicit"}

    # 1. Render note via production template (no test fixture).
    note_path_str = write_paper_note(
        paper, extraction, config, template_dir=Path("config/templates"),
    )
    note_path = Path(note_path_str)
    rendered = note_path.read_text(encoding="utf-8")

    # Sanity: the production template now emits the referenced_by markers.
    assert '{% persist "referenced_by" %}' in rendered
    assert "{% endpersist %}" in rendered

    # State DB with target paper and two distinct citing sources (A, B).
    state_db = _make_state_db(tmp_path)
    embeddings_db = _make_embeddings_db()
    state_db.upsert_paper({
        "doi": "10.1/roundtrip",
        "title": "Round Trip Target",
        "source_type": "paper",
        "note_title": note_path.stem,
    })
    state_db.upsert_paper({
        "doi": "10.1/sourceA",
        "title": "Source A",
        "source_type": "paper",
        "note_title": "Alpha2023_SourceA",
    })
    state_db.upsert_paper({
        "doi": "10.1/sourceB",
        "title": "Source B",
        "source_type": "paper",
        "note_title": "Beta2024_SourceB",
    })

    # 2. First relink with source A.
    state_db.upsert_citation_edge(
        source_doi="10.1/sourceA",
        ref_id="[1]",
        target_doi="10.1/roundtrip",
        target_s2_id=None,
        context="Cited by A.",
        resolution="numeric_index",
    )
    with caplog.at_level(logging.WARNING):
        relink_note(note_path, embeddings_db, state_db)
    after_a = note_path.read_text(encoding="utf-8")

    # No "Zone 'referenced_by' not found" warning on a freshly rendered note.
    assert not any(
        "referenced_by" in rec.getMessage() and "not found" in rec.getMessage()
        for rec in caplog.records
    ), "relink_note should not warn about missing 'referenced_by' zone"

    # Markers intact, A's note title inside the zone.
    assert '{% persist "referenced_by" %}' in after_a
    assert "{% endpersist %}" in after_a
    assert "Alpha2023_SourceA" in after_a

    # 3. Swap edges: drop A, add B. Then relink again.
    # The simplest way to "replace" is to delete the citation_edges row for A
    # and add B. StateDB has upsert; for a clean swap we use the underlying
    # connection.
    with state_db._connect() as conn:  # type: ignore[attr-defined]
        conn.execute(
            "DELETE FROM citation_edges WHERE source_doi = ?",
            ("10.1/sourceA",),
        )
        conn.commit()
    state_db.upsert_citation_edge(
        source_doi="10.1/sourceB",
        ref_id="[2]",
        target_doi="10.1/roundtrip",
        target_s2_id=None,
        context="Cited by B.",
        resolution="numeric_index",
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        relink_note(note_path, embeddings_db, state_db)
    after_b = note_path.read_text(encoding="utf-8")

    # Still no missing-zone warning on the second relink.
    assert not any(
        "referenced_by" in rec.getMessage() and "not found" in rec.getMessage()
        for rec in caplog.records
    )

    # Markers still intact.
    assert '{% persist "referenced_by" %}' in after_b
    assert "{% endpersist %}" in after_b
    # Content was replaced, not appended: B is present, A is gone.
    assert "Beta2024_SourceB" in after_b
    assert "Alpha2023_SourceA" not in after_b


# ===========================================================================
# G9 — rag-mode dispatch for relink_note
# ===========================================================================

def _make_note_g9(tmp_path: Path, doi: str = "10.0/g9") -> Path:
    """Minimal note with both persist zones for G9 rag-mode tests."""
    content = (
        f'---\ndoi: "{doi}"\ntitle: "G9 Test"\nsource_type: "paper"\n---\n\n'
        '{% persist "related_work" %}\n## Related Work\n{% endpersist %}\n\n'
        '{% persist "referenced_by" %}\n## Referenced By\n{% endpersist %}\n'
    )
    path = tmp_path / f"{doi.replace('/', '_')}.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.unit
def test_relink_vector_mode_calls_embeddings_only(tmp_path):
    """G9: rag_mode='vector' calls embeddings_db.find_similar_to_text, not graph."""
    from scripts.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    state_db.upsert_paper({
        "doi": "10.0/g9", "title": "G9 Paper", "source_type": "paper",
        "note_title": "G9_Paper",
    })
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    graph_db = MagicMock()

    note_path = _make_note_g9(tmp_path, "10.0/g9")
    relink_note(note_path, embeddings_db, state_db, rag_mode="vector", graph_db=graph_db)

    embeddings_db.find_similar_to_text.assert_called()
    graph_db.find_similar_papers.assert_not_called()
    graph_db.find_papers_by_entities.assert_not_called()


@pytest.mark.unit
def test_relink_graph_mode_calls_graph_only(tmp_path):
    """G9: rag_mode='graph' calls graph_db.find_similar_papers, not embeddings."""
    from scripts.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    state_db.upsert_paper({
        "doi": "10.0/g9g", "title": "G9 Graph", "source_type": "paper",
        "note_title": "G9_Graph",
    })
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    graph_db = MagicMock()
    graph_db.find_similar_papers.return_value = [("10.0/other", 3.0)]

    # Register the other paper so the note_title can be resolved.
    state_db.upsert_paper({
        "doi": "10.0/other", "title": "Other", "source_type": "paper",
        "note_title": "Other_Paper",
    })

    note_path = _make_note_g9(tmp_path, "10.0/g9g")
    relink_note(note_path, embeddings_db, state_db, rag_mode="graph", graph_db=graph_db)

    graph_db.find_similar_papers.assert_called()
    embeddings_db.find_similar_to_text.assert_not_called()


@pytest.mark.unit
def test_relink_hybrid_mode_calls_both_and_fuses(tmp_path):
    """G9: rag_mode='hybrid' calls both embeddings and graph_db; both DOIs appear."""
    from scripts.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    doi_seed = "10.0/g9h"
    doi_vec = "10.0/vec"
    doi_graph = "10.0/graph"
    for d, title in [(doi_seed, "Seed"), (doi_vec, "Vec"), (doi_graph, "Graph")]:
        state_db.upsert_paper({
            "doi": d, "title": title, "source_type": "paper",
            "note_title": f"{title.replace(' ', '_')}_Note",
        })

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = [
        {"id": doi_vec, "score": 0.9, "document": "", "metadata": {"note_title": "Vec_Note"}},
    ]
    graph_db = MagicMock()
    graph_db.find_similar_papers.return_value = [(doi_graph, 5.0)]

    note_path = _make_note_g9(tmp_path, doi_seed)
    relink_note(note_path, embeddings_db, state_db, rag_mode="hybrid", graph_db=graph_db)

    embeddings_db.find_similar_to_text.assert_called()
    graph_db.find_similar_papers.assert_called()

    content = note_path.read_text(encoding="utf-8")
    # Both candidates should appear somewhere in the note (Related Work zone).
    assert "Vec_Note" in content or "Graph_Note" in content, (
        "Hybrid mode must surface at least one fused result in the note."
    )


@pytest.mark.unit
def test_relink_default_rag_mode_reads_from_config(tmp_path):
    """G9: when rag_mode is None, default is read from config.retrieval.default_mode."""
    from scripts.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    state_db.upsert_paper({
        "doi": "10.0/cfg", "title": "Cfg Paper", "source_type": "paper",
        "note_title": "Cfg_Paper",
    })
    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = []
    graph_db = MagicMock()
    graph_db.find_similar_papers.return_value = []

    # Config that sets default_mode to "graph".
    config = SimpleNamespace(
        retrieval=SimpleNamespace(default_mode="graph"),
        reranker=None,
    )

    note_path = _make_note_g9(tmp_path, "10.0/cfg")
    relink_note(
        note_path, embeddings_db, state_db,
        rag_mode=None, graph_db=graph_db, config=config,
    )

    # graph mode should have been used.
    graph_db.find_similar_papers.assert_called()
    embeddings_db.find_similar_to_text.assert_not_called()
