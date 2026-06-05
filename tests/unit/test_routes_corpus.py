"""FE2-2: GET /corpus list page route tests.

The state DB accessor is patched at the single seam
``scripts.server.routes.corpus._list_papers`` — no live SQLite, embeddings, or
network. Mirrors the ``client`` fixture from tests/unit/test_routes_ask.py.
"""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    from scripts.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


def test_get_corpus_renders_table(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.corpus._list_papers",
        lambda **k: (
            [
                {
                    "doi": "10.1/a",
                    "title": "Chromatography study",
                    "year": 2021,
                    "source_type": "paper",
                    "status": "extraction_complete",
                    "confidence": 0.8,
                    "embeddings_indexed": 1,
                    "graph_indexed": 1,
                    "notes_synced": 1,
                    "last_updated": "2026-06-01",
                }
            ],
            1,
        ),
    )
    r = client.get("/corpus")
    assert r.status_code == 200
    assert "Chromatography study" in r.text and 'href="/corpus/10.1/a"' in r.text


def test_corpus_search_passes_through(client, monkeypatch):
    seen = {}

    def _fake(**k):
        seen.update(k)
        return ([], 0)

    monkeypatch.setattr("scripts.server.routes.corpus._list_papers", _fake)
    client.get("/corpus?search=carta&source_type=review&status_gap=missing_graph")
    assert (
        seen["search"] == "carta"
        and seen["source_type"] == "review"
        and seen["status_gap"] == "missing_graph"
    )


def test_corpus_empty_state(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.corpus._list_papers", lambda **k: ([], 0)
    )
    r = client.get("/corpus")
    assert "brain-build" in r.text.lower()


def test_nav_explore_has_corpus(client):
    r = client.get("/")
    assert 'href="/corpus"' in r.text


# --- FE2-3: GET /corpus/{doi} detail page -------------------------------------


def test_get_corpus_detail_renders_extraction(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.corpus._get_paper_row",
        lambda doi: {
            "doi": doi,
            "title": "Carta 2009",
            "authors": ["A. Carta"],
            "year": 2009,
            "journal": "J Chrom",
            "source_type": "paper",
            "zotero_key": "ABCD1234",
            "note_path": "Literature/Papers/Carta 2009.md",
            "extraction": {
                "core_finding": "membrane chromatography scales",
                "_overall_confidence": 0.7,
            },
        },
    )
    monkeypatch.setattr(
        "scripts.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    assert "Carta 2009" in r.text and "membrane chromatography scales" in r.text
    assert "zotero://select" in r.text and "ABCD1234" in r.text
    assert "Carta 2009.md" in r.text
    assert (
        'hx-post="/api/papers/10.1/carta/relink"' in r.text or "relink" in r.text
    )


def test_corpus_detail_404_when_absent(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.corpus._get_paper_row", lambda doi: None
    )
    r = client.get("/corpus/10.1/missing")
    assert r.status_code == 404


def test_corpus_detail_xss_escaped(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.corpus._get_paper_row",
        lambda doi: {
            "doi": doi,
            "title": "<script>alert(1)</script>",
            "authors": [],
            "year": None,
            "journal": None,
            "source_type": "paper",
            "zotero_key": None,
            "note_path": None,
            "extraction": {"core_finding": "<img src=x onerror=alert(2)>"},
        },
    )
    monkeypatch.setattr(
        "scripts.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/x")
    # Executable forms must NOT appear (real angle brackets = would execute):
    assert "<script>alert(1)</script>" not in r.text
    assert "<img src=x onerror=alert(2)>" not in r.text
    # Positively confirm the payload was HTML-escaped (rendered inert as text):
    assert "&lt;img src=x onerror=alert(2)&gt;" in r.text


# --- FE2-4: lazy HTMX fragments — related work + knowledge graph --------------


def test_corpus_related_fragment_renders(client, monkeypatch):
    # _get_related returns the H1 shape: list of {doi, score} (no title field).
    monkeypatch.setattr(
        "scripts.server.routes.corpus._get_related",
        lambda doi, mode, k: [{"doi": "10.2/b", "title": "Related paper", "score": 0.9}],
    )
    r = client.get("/corpus/10.1/a/related")
    assert r.status_code == 200
    assert "Related paper" in r.text
    assert 'href="/corpus/10.2/b"' in r.text
    # A vector/graph/hybrid mode toggle must be present.
    assert "mode=vector" in r.text
    assert "mode=graph" in r.text
    assert "mode=hybrid" in r.text


def test_corpus_graph_fragment_renders(client, monkeypatch):
    monkeypatch.setattr(
        "scripts.server.routes.corpus._get_paper_snapshot",
        lambda doi: {
            "metadata": {"doi": doi, "title": "Seed", "year": 2009, "journal": "J"},
            "entities_by_type": {
                "method": [
                    {"canonical_id": "ion-exchange", "type": "method", "surface": "IEX"}
                ]
            },
            "relationships_out": [
                {
                    "predicate": "CITES",
                    "target_kind": "Paper",
                    "target_id": "10.3/c",
                    "evidence": "builds on prior work",
                }
            ],
            "relationships_in": [],
        },
    )
    r = client.get("/corpus/10.1/a/graph")
    assert r.status_code == 200
    assert "ion-exchange" in r.text  # entity name
    assert "CITES" in r.text  # relationship predicate
    assert "10.3/c" in r.text  # relationship target


def test_corpus_fragments_graceful_when_graph_absent(client, monkeypatch):
    # safe_graph_db returning None simulates the graph not being built. Both
    # seams resolve the graph via safe_graph_db, so they return None → notice.
    monkeypatch.setattr(
        "scripts.server.routes.corpus.safe_graph_db", lambda: None
    )
    r_rel = client.get("/corpus/10.1/a/related")
    r_graph = client.get("/corpus/10.1/a/graph")
    assert r_rel.status_code == 200 and r_graph.status_code == 200
    # The "graph not built" notice mentions building the graph via backfill —
    # a stable substring that the empty-result / mode-toggle paths do NOT emit.
    assert "graph backfill" in r_rel.text.lower()
    assert "graph backfill" in r_graph.text.lower()


def test_corpus_related_error_is_generic_no_leak(client, monkeypatch, caplog):
    secret = "kuzu://secret/path/db"

    def _boom(doi, mode, k):
        raise RuntimeError(secret)

    monkeypatch.setattr("scripts.server.routes.corpus._get_related", _boom)
    with caplog.at_level(logging.ERROR, logger="scripts.server.routes.corpus"):
        r = client.get("/corpus/10.1/a/related")
    assert r.status_code == 200  # fragment, not a 500 page
    assert secret not in r.text  # no leak to the browser
    assert any(secret in rec.getMessage() for rec in caplog.records)  # but logged
