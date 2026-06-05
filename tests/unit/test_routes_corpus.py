"""FE2-2: GET /corpus list page route tests.

The state DB accessor is patched at the single seam
``scripts.server.routes.corpus._list_papers`` — no live SQLite, embeddings, or
network. Mirrors the ``client`` fixture from tests/unit/test_routes_ask.py.
"""
from __future__ import annotations

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
    assert (
        "<script>alert(1)</script>" not in r.text
        and "onerror=alert(2)" not in r.text
    )
