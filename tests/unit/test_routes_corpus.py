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
