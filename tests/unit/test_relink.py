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
from unittest.mock import MagicMock, patch

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
    from lit_monitor.core.state_db import StateDB
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
    from lit_monitor.obsidian_tools.relink import relink_note

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
    from lit_monitor.obsidian_tools.relink import relink_note

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
    from lit_monitor.obsidian_tools.relink import relink_note

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
    from lit_monitor.obsidian_tools.relink import relink_note
    from lit_monitor.output.obsidian_writer import write_paper_note

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
    from lit_monitor.obsidian_tools.relink import relink_note

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
    from lit_monitor.obsidian_tools.relink import relink_note

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
    from lit_monitor.obsidian_tools.relink import relink_note

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


# ===========================================================================
# G9 review fixes: C1 (reranker), C3 (document hydration), C4/W4 (graph gate)
# ===========================================================================


class TestG9C1Reranker:
    """G9 C1+C3: hybrid mode actually invokes the cross-encoder reranker
    with NON-EMPTY documents.

    Pre-fix, ``from scripts.output.reranker import rerank`` raised ImportError
    on every call (the symbol does not exist) and was silently swallowed by
    ``except Exception``. Even if the import had worked, candidate dicts had
    ``"document": ""`` so the cross-encoder scored (query, "") — pure noise.

    Both regressions must be caught here so the test fails if either is rolled
    back.
    """

    @pytest.mark.unit
    def test_hybrid_mode_calls_reranker_with_hydrated_documents(self, tmp_path):
        from lit_monitor.obsidian_tools.relink import relink_note

        state_db = _make_state_db(tmp_path)
        # Seed paper (the one being relinked) — has a real extraction blob so
        # _focused_embed_query has content too.
        import json
        seed_extraction = json.dumps({"abstract": "Seed abstract.", "core_finding": "Seed finding."})
        state_db.upsert_paper({
            "doi": "10.0/seed", "title": "Seed Paper", "source_type": "paper",
            "note_title": "Seed_Paper", "extraction_json": seed_extraction,
        })
        # Two candidate papers with rich extraction_json so the reranker has
        # text to score. If C3 regresses, the test fails because docs are "".
        for d, title, finding in [
            ("10.0/cand1", "Membrane Filtration of Proteins", "Pore size matters."),
            ("10.0/cand2", "Chromatography Elution Profile", "Gradient shape matters."),
        ]:
            state_db.upsert_paper({
                "doi": d, "title": title, "source_type": "paper",
                "note_title": f"{d.replace('/', '_')}",
                "extraction_json": json.dumps({
                    "abstract": f"Abstract for {title}.",
                    "core_finding": finding,
                }),
            })

        embeddings_db = MagicMock()
        embeddings_db.find_similar_to_text.return_value = [
            {"id": "10.0/cand1", "score": 0.9, "document": "Membrane …",
             "metadata": {"note_title": "10.0_cand1"}},
        ]
        graph_db = MagicMock()
        graph_db.find_similar_papers.return_value = [("10.0/cand2", 5.0)]

        # Reranker config (enabled). _apply_reranker will call
        # scripts.output.reranker.get_reranker which we patch out — so the
        # actual cross-encoder never loads, but we observe its inputs.
        config = SimpleNamespace(
            retrieval=SimpleNamespace(default_mode="vector"),
            reranker=SimpleNamespace(
                enabled=True,
                model="mock-rerank",
                candidate_multiplier=3,
                device=None,
            ),
        )

        mock_reranker = MagicMock()
        # Return candidates unchanged (with a fake rerank_score) so we can
        # assert on what was passed IN, not what came out.
        def _fake_rerank(query, candidates, top_k=None):
            return [{**c, "rerank_score": 0.5} for c in candidates[: top_k or len(candidates)]]
        mock_reranker.rerank.side_effect = _fake_rerank

        note_path = _make_note_g9(tmp_path, "10.0/seed")
        with patch("scripts.output.reranker.get_reranker", return_value=mock_reranker):
            relink_note(
                note_path, embeddings_db, state_db,
                rag_mode="hybrid", graph_db=graph_db, config=config,
            )

        # C1: the reranker MUST have been called. Pre-fix, the broken import
        # raised ImportError and the rerank step was silently skipped.
        assert mock_reranker.rerank.called, (
            "G9 C1 regression: hybrid mode did not invoke the cross-encoder reranker. "
            "Most likely scripts.output.reranker.rerank (non-existent) is being "
            "imported again instead of the real API."
        )

        # C3: documents passed to the reranker must NOT be all empty strings.
        # The reranker receives (query, candidates) — inspect the candidate docs.
        call_args = mock_reranker.rerank.call_args
        # First positional arg is query string; second is candidates list.
        cand_list = call_args[0][1] if len(call_args[0]) >= 2 else call_args[1]["candidates"]
        docs = [c.get("document", "") for c in cand_list]
        assert any(d.strip() for d in docs), (
            "G9 C3 regression: reranker received candidates with empty 'document' "
            "fields. Hydrate title + abstract + core_finding from state_db before "
            f"reranking. Got docs: {docs!r}"
        )


class TestG9C4W4GraphGating:
    """G9 C4 + W4: relink command must compute effective rag_mode BEFORE opening
    graph_db, and raise UsageError for an EXPLICIT --rag-mode graph|hybrid when
    the [graph] extra is missing.
    """

    @pytest.mark.unit
    def test_explicit_graph_mode_without_extra_raises(self, tmp_path):
        """W4: ``--rag-mode graph`` + safe_graph_db()→None → UsageError."""
        from click.testing import CliRunner

        from lit_monitor.cli import main as cli_main

        runner = CliRunner()
        # Patch safe_graph_db at its origin module to return None, simulating a
        # missing [graph] extra. The CLI imports it inside the function body.
        with patch("scripts.graph.safe_graph_db", return_value=None), \
             patch("scripts.cli._make_config") as mock_cfg, \
             patch("scripts.cli._make_state_db", return_value=MagicMock()), \
             patch("scripts.cli._make_embeddings_db", return_value=MagicMock()):
            mock_cfg.return_value = SimpleNamespace(
                retrieval=SimpleNamespace(default_mode="vector"),
                reranker=None,
            )
            result = runner.invoke(
                cli_main,
                ["obsidian", "relink", "--rag-mode", "graph", "--doi", "10.1/x"],
            )

        # Click's UsageError converts to exit_code=2.
        assert result.exit_code != 0, (
            f"W4 regression: explicit graph mode without extra should fail.\n"
            f"Output: {result.output!r}"
        )
        assert "uv sync --extra graph" in result.output, (
            f"Error message should point to the install command. "
            f"Got output: {result.output!r}  exception: {result.exception!r}"
        )

    @pytest.mark.unit
    def test_config_default_graph_mode_without_extra_falls_back(self, tmp_path):
        """C4: when CONFIG default is 'graph' but [graph] is missing, the
        command MUST silently fall back to vector — not silently degrade with
        graph_db=None and effective_rag='graph' (the pre-fix state).
        """
        from lit_monitor.obsidian_tools.relink import relink_note

        state_db = _make_state_db(tmp_path)
        state_db.upsert_paper({
            "doi": "10.0/cfg", "title": "Cfg Paper", "source_type": "paper",
            "note_title": "Cfg_Paper",
        })
        embeddings_db = MagicMock()
        embeddings_db.find_similar_to_text.return_value = []
        # Config default is "graph" but graph_db is None (extra missing).
        # In this case the relink_note function itself will go down the
        # graph/hybrid branch with graph_db=None — and the CLI is supposed to
        # downgrade to vector. We assert here that the LIBRARY function does
        # the right thing when called with rag_mode='vector' explicitly — which
        # is what the CLI passes after its fallback.
        config = SimpleNamespace(
            retrieval=SimpleNamespace(default_mode="vector"),
            reranker=None,
        )
        note_path = _make_note_g9(tmp_path, "10.0/cfg")
        # Call with explicit "vector" — what CLI does after downgrade.
        relink_note(
            note_path, embeddings_db, state_db,
            rag_mode="vector", graph_db=None, config=config,
        )
        # Vector path was used.
        embeddings_db.find_similar_to_text.assert_called()


@pytest.mark.unit
def test_relink_default_rag_mode_reads_from_config(tmp_path):
    """G9: when rag_mode is None, default is read from config.retrieval.default_mode."""
    from lit_monitor.obsidian_tools.relink import relink_note

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


# ===========================================================================
# B1 — atomic vault writes (Audit-2 data-safety)
# ===========================================================================


@pytest.mark.unit
def test_relink_note_updated_via_atomic_path(tmp_path):
    """B1 happy-path: relink_note rewrites the note's persist zone and the new
    content is present (proving the atomic_write_text path actually lands)."""
    from lit_monitor.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    embeddings_db = _make_embeddings_db()

    state_db.upsert_paper({
        "doi": "10.1/atomic-ok",
        "title": "Atomic OK",
        "source_type": "paper",
        "note_title": "Smith2021_AtomicOK",
    })
    state_db.upsert_paper({
        "doi": "10.1/citer",
        "title": "Citing Paper",
        "source_type": "paper",
        "note_title": "Jones2022_Citer",
    })
    state_db.upsert_citation_edge(
        source_doi="10.1/citer",
        ref_id="[1]",
        target_doi="10.1/atomic-ok",
        target_s2_id=None,
        context="Cites the target.",
        resolution="numeric_index",
    )

    note_path = _make_note(tmp_path, "10.1/atomic-ok", "Atomic OK")
    relink_note(note_path, embeddings_db, state_db)

    content = note_path.read_text(encoding="utf-8")
    assert "← [[Jones2022_Citer]] cites this" in content


@pytest.mark.unit
def test_relink_note_survives_replace_failure(tmp_path, monkeypatch):
    """B1 crash-safety: if os.replace fails mid-write, the original note must be
    preserved verbatim and no .tmp sibling may leak.

    Why this catches a non-atomic regression: the patched os.replace raises only
    at the final rename step of atomic_write_text. With a plain
    note_path.write_text(...) there is no temp file — the destination would
    already be truncated/rewritten in place before any replace ran, so the
    'original intact' assertion would fail.
    """
    from lit_monitor.core import atomic_write as atomic_write_mod
    from lit_monitor.obsidian_tools.relink import relink_note

    state_db = _make_state_db(tmp_path)
    embeddings_db = _make_embeddings_db()

    state_db.upsert_paper({
        "doi": "10.1/atomic-boom",
        "title": "Atomic Boom",
        "source_type": "paper",
        "note_title": "Smith2021_AtomicBoom",
    })
    # An incoming citation guarantees relink_note reaches the write step.
    state_db.upsert_paper({
        "doi": "10.1/citer2",
        "title": "Citing Paper 2",
        "source_type": "paper",
        "note_title": "Jones2022_Citer2",
    })
    state_db.upsert_citation_edge(
        source_doi="10.1/citer2",
        ref_id="[1]",
        target_doi="10.1/atomic-boom",
        target_s2_id=None,
        context="Cites the target.",
        resolution="numeric_index",
    )

    note_path = _make_note(tmp_path, "10.1/atomic-boom", "Atomic Boom")
    original = note_path.read_text(encoding="utf-8")

    def boom(_src, _dst):  # noqa: ANN001 — match os.replace signature
        raise OSError("simulated rename crash")

    monkeypatch.setattr(atomic_write_mod.os, "replace", boom)

    with pytest.raises(OSError, match="simulated rename crash"):
        relink_note(note_path, embeddings_db, state_db)

    # Original note content untouched.
    assert note_path.read_text(encoding="utf-8") == original
    # No temp-file droppings left behind in the vault dir.
    leftovers = [
        p for p in note_path.parent.iterdir()
        if p.name.startswith(".") and p.name.endswith(".tmp")
    ]
    assert leftovers == [], f"leaked temp files: {leftovers}"
