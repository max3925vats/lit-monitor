"""
Unit tests for StateDB.list_papers (Task FE2-1, Corpus Explorer).

Covers the read-lens query layer over PROCESSED papers: search, source_type
filter, status-gap filter, pagination, sort, and the (rows, total) contract.

All seeding goes through the public API (upsert_paper + the dedicated flag
setters) so the tests exercise the real write path, not a hand-rolled schema.
"""
from __future__ import annotations

import json

import pytest

from lit_monitor.core.state_db import StateDB

# ---------------------------------------------------------------------------
# Fixture — a small, varied corpus
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Seed ~5 papers varying title/year/source_type/status/confidence/flags."""
    state = StateDB(str(tmp_path / "state.db"))

    # (doi, title, year, source_type, status, confidence, embeddings, graph, notes)
    seeds = [
        ("10.1/chrom", "Membrane chromatography study", 2021, "paper",
         "extraction_complete", 0.80, 1, 1, 1),
        ("10.2/filt", "Filtration scaling review", 2019, "review",
         "extraction_complete", 0.30, 1, 0, 0),
        ("10.3/ufdf", "UFDF process optimisation", 2023, "paper",
         "extraction_complete", 0.65, 0, 1, 0),
        ("10.4/CHROMATOGRAPHY-caps", "CHROMATOGRAPHY resin screening", 2020, "paper",
         "extraction_complete", None, 1, 1, 1),
        ("10.5/review2", "Downstream review of viral clearance", 2022, "review",
         "extraction_complete", 0.10, 0, 0, 1),
    ]

    for idx, (doi, title, year, source_type, status, conf, emb, graph, notes) in enumerate(seeds):
        extraction = {"core_finding": "x"}
        if conf is not None:
            extraction["_overall_confidence"] = conf
        state.upsert_paper(
            {
                "doi": doi,
                "title": title,
                "authors": ["A. Author", "B. Author"],
                "year": year,
                "journal": "J Test",
                "source_type": source_type,
                "status": status,
                "embeddings_indexed": emb,
                "extraction_json": json.dumps(extraction),
                # distinct last_updated per row so sort/pagination order is stable
                "last_updated": f"2026-06-0{idx + 1}",
            }
        )
        # graph_indexed / notes_synced are setter-managed flag columns.
        state.set_graph_indexed(doi, graph)
        state.set_notes_synced(doi, notes)

    return state


# ---------------------------------------------------------------------------
# Tests (the plan's 6)
# ---------------------------------------------------------------------------


def test_list_papers_returns_rows_and_total(db):
    rows, total = db.list_papers(limit=50, offset=0)
    assert total == 5 and len(rows) == 5
    r = rows[0]
    assert {"doi", "title", "year", "source_type", "status", "confidence",
            "embeddings_indexed", "graph_indexed", "notes_synced",
            "last_updated"} <= set(r)


def test_list_papers_search_matches_title_or_doi(db):
    rows, total = db.list_papers(search="chromatography")
    assert rows  # non-empty — proves the filter actually returns matches
    assert all("chromatography" in (x["title"] or "").lower()
               or "chromatography" in x["doi"].lower() for x in rows)


def test_list_papers_filter_source_type(db):
    rows, _ = db.list_papers(source_type="review")
    assert rows
    assert all(x["source_type"] == "review" for x in rows)


def test_list_papers_status_gap_missing_graph(db):
    rows, _ = db.list_papers(status_gap="missing_graph")
    assert rows
    assert all(x["graph_indexed"] == 0 for x in rows)


def test_list_papers_pagination_and_total(db):
    rows, total = db.list_papers(limit=2, offset=0)
    assert len(rows) == 2 and total == 5


def test_list_papers_sort_year_desc(db):
    rows, _ = db.list_papers(sort="year", order="desc")
    years = [x["year"] or 0 for x in rows]
    assert years == sorted(years, reverse=True)


# ---------------------------------------------------------------------------
# P4: exclude_status — the corpus list shows only INGESTED papers, never the
# hundreds of un-ingested 'discovered' candidates discovery surfaces per run.
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_status_db(tmp_path):
    """Seed a corpus mixing ingested rows with un-ingested 'discovered' ones."""
    state = StateDB(str(tmp_path / "state.db"))
    seeds = [
        ("10.1/a", "Ingested paper A", "extraction_complete"),
        ("10.2/b", "Ingested review B", "extraction_complete"),
        ("10.3/c", "No-markdown paper C", "no_markdown"),
        ("10.4/d", "Discovery candidate D", "discovered"),
        ("10.5/e", "Discovery candidate E", "discovered"),
        ("10.6/f", "Discovery candidate F", "discovered"),
    ]
    for idx, (doi, title, status) in enumerate(seeds):
        state.upsert_paper(
            {
                "doi": doi,
                "title": title,
                "year": 2020 + idx,
                "source_type": "paper",
                "status": status,
                "extraction_json": json.dumps({"core_finding": "x"}),
                "last_updated": f"2026-06-0{idx + 1}",
            }
        )
    return state


def test_list_papers_default_excludes_discovered(mixed_status_db):
    # Default call (no exclude_status passed) must drop the 3 'discovered' rows
    # from BOTH the rows and the total count.
    rows, total = mixed_status_db.list_papers(limit=50, offset=0)
    statuses = {r["status"] for r in rows}
    assert "discovered" not in statuses
    assert len(rows) == 3 and total == 3


def test_list_papers_explicit_exclude_status(mixed_status_db):
    rows, total = mixed_status_db.list_papers(exclude_status=("discovered",))
    assert all(r["status"] != "discovered" for r in rows)
    assert len(rows) == 3 and total == 3


def test_list_papers_empty_exclude_status_returns_all(mixed_status_db):
    # exclude_status=() disables the filter → every row (incl. discovered).
    rows, total = mixed_status_db.list_papers(exclude_status=(), limit=50)
    assert len(rows) == 6 and total == 6
    assert sum(1 for r in rows if r["status"] == "discovered") == 3
