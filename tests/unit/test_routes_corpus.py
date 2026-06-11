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
    from lit_monitor.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


def test_get_corpus_renders_table(client, monkeypatch):
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers",
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

    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_papers", _fake)
    client.get("/corpus?search=carta&source_type=review&status_gap=missing_graph")
    assert (
        seen["search"] == "carta"
        and seen["source_type"] == "review"
        and seen["status_gap"] == "missing_graph"
    )


def test_corpus_empty_state(client, monkeypatch):
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0)
    )
    r = client.get("/corpus")
    assert "brain-build" in r.text.lower()


def test_nav_explore_has_corpus(client):
    r = client.get("/")
    assert 'href="/corpus"' in r.text


# --- PF-3: theme/cluster filter dropdown on the corpus list page -------------


def test_corpus_list_theme_dropdown_populated(client, monkeypatch):
    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_themes",
        lambda: [{"id": 1, "display_name": "Chromatography"},
                 {"id": 2, "display_name": "Filtration"}])
    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0))
    r = client.get("/corpus")
    assert r.status_code == 200
    assert 'name="theme"' in r.text and "Chromatography" in r.text and "Filtration" in r.text


def test_corpus_list_no_themes_first_run_message_not_broken(client, monkeypatch):
    # FIRST RUN: no clusters → friendly message, NOT an empty <select>, page still renders fully
    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_themes", lambda: [])
    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0))
    r = client.get("/corpus")
    assert r.status_code == 200
    # the rest of the filter UI still works (search box present):
    assert 'name="search"' in r.text
    # a no-themes note is shown, and NO populated theme <select> with cluster options:
    assert ("no themes" in r.text.lower()) or ("brain-build" in r.text.lower())


def test_corpus_list_themes_failure_is_graceful(client, monkeypatch):
    # _list_themes raising must not 500 the page
    def _boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_themes", _boom)
    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0))
    r = client.get("/corpus")
    assert r.status_code == 200  # graceful — page renders without the theme filter


# --- FE2-3: GET /corpus/{doi} detail page -------------------------------------


def test_get_corpus_detail_renders_extraction(client, monkeypatch):
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
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
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
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
        "lit_monitor.server.routes.corpus._get_paper_row", lambda doi: None
    )
    r = client.get("/corpus/10.1/missing")
    assert r.status_code == 404


def test_corpus_detail_xss_escaped(client, monkeypatch):
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
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
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/x")
    # Executable forms must NOT appear (real angle brackets = would execute):
    assert "<script>alert(1)</script>" not in r.text
    assert "<img src=x onerror=alert(2)>" not in r.text
    # Positively confirm the payload was HTML-escaped (rendered inert as text):
    assert "&lt;img src=x onerror=alert(2)&gt;" in r.text


# --- PF-2: clickable Obsidian deep-link on the corpus detail page ------------


def test_corpus_detail_obsidian_deeplink(client, monkeypatch):
    monkeypatch.setattr("lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: {"doi":doi,"title":"X","authors":[],"year":None,"journal":None,
                     "source_type":"paper","zotero_key":None,
                     "note_path":"Literature/Papers/X.md","extraction":{}})
    monkeypatch.setattr("lit_monitor.server.routes.corpus._get_score_breakdown", lambda d: None)
    monkeypatch.setattr("lit_monitor.server.routes.corpus._vault_name", lambda: "MyVault")
    r = client.get("/corpus/10.1/x")
    assert r.status_code == 200
    assert "obsidian://open?vault=MyVault" in r.text
    assert "Literature/Papers/X.md" in r.text or "X.md" in r.text  # file in the URI/label


def test_corpus_detail_no_obsidian_link_when_no_note(client, monkeypatch):
    monkeypatch.setattr("lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: {"doi":doi,"title":"X","authors":[],"year":None,"journal":None,
                     "source_type":"paper","zotero_key":None,"note_path":None,"extraction":{}})
    monkeypatch.setattr("lit_monitor.server.routes.corpus._get_score_breakdown", lambda d: None)
    r = client.get("/corpus/10.1/x")
    assert r.status_code == 200 and "obsidian://open" not in r.text  # no note → no link, no crash


def test_corpus_detail_absolute_note_path_made_relative(client, monkeypatch):
    monkeypatch.setattr("lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: {"doi":doi,"title":"X","authors":[],"year":None,"journal":None,
                     "source_type":"paper","zotero_key":None,
                     "note_path":"/Users/me/MyVault/Literature/Papers/X.md","extraction":{}})
    monkeypatch.setattr("lit_monitor.server.routes.corpus._get_score_breakdown", lambda d: None)
    monkeypatch.setattr("lit_monitor.server.routes.corpus._vault_path", lambda: "/Users/me/MyVault")
    monkeypatch.setattr("lit_monitor.server.routes.corpus._vault_name", lambda: "MyVault")
    r = client.get("/corpus/10.1/x")
    # absolute path under the vault → file param is vault-relative, not the absolute path
    assert "file=Literature%2FPapers%2FX.md" in r.text or "file=Literature/Papers/X.md" in r.text


# --- FE2-4: lazy HTMX fragments — related work + knowledge graph --------------


def test_corpus_related_fragment_renders(client, monkeypatch):
    # _get_related returns the H1 shape: list of {doi, score} (no title field).
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_related",
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
        "lit_monitor.server.routes.corpus._get_paper_snapshot",
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
    # safe_graph_db returning None simulates the graph not being built. The
    # GRAPH-mode related fragment and the graph snapshot both resolve via
    # safe_graph_db, so they return None → graph-backfill notice. (Vector mode
    # no longer touches the graph — see PF-1 tests below.)
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus.safe_graph_db", lambda: None
    )
    r_rel = client.get("/corpus/10.1/a/related?mode=graph")
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

    monkeypatch.setattr("lit_monitor.server.routes.corpus._get_related", _boom)
    with caplog.at_level(logging.ERROR, logger="lit_monitor.server.routes.corpus"):
        r = client.get("/corpus/10.1/a/related")
    assert r.status_code == 200  # fragment, not a 500 page
    assert secret not in r.text  # no leak to the browser
    assert any(secret in rec.getMessage() for rec in caplog.records)  # but logged


# --- PF-1: graph-free vector related-work + explicit per-result provenance -----


def test_related_vector_mode_labels_vector_source(client, monkeypatch):
    # Vector mode returns embedding neighbours tagged source="vector". The
    # template must LABEL the provenance so a vector hit is never mistaken for a
    # graph relationship ("no lies" requirement).
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_related",
        lambda doi, mode, k: (
            [{"doi": "10.2/b", "title": "Sim paper", "score": 0.91, "source": "vector"}]
            if mode == "vector"
            else None
        ),
    )
    r = client.get("/corpus/10.1/a/related?mode=vector")
    assert r.status_code == 200
    assert "10.2/b" in r.text
    assert "vector" in r.text.lower()  # provenance label is shown to the user


def test_related_graph_mode_labels_graph_source(client, monkeypatch):
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_related",
        lambda doi, mode, k: [
            {"doi": "10.3/c", "title": "G", "score": 0.8, "source": "graph"}
        ],
    )
    r = client.get("/corpus/10.1/a/related?mode=graph")
    assert r.status_code == 200
    assert "10.3/c" in r.text and "graph" in r.text.lower()


def test_related_graph_mode_without_graph_notice(client, monkeypatch):
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_related", lambda d, m, k: None
    )
    r = client.get("/corpus/10.1/a/related?mode=graph")
    assert r.status_code == 200
    assert "graph" in r.text.lower() and "backfill" in r.text.lower()


def test_related_vector_no_embeddings_shows_neighbour_notice(client, monkeypatch):
    # Vector mode with no embeddings available → distinct empty-state notice,
    # NOT the graph-backfill notice (vector path never touches the graph).
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_related", lambda d, m, k: None
    )
    r = client.get("/corpus/10.1/a/related?mode=vector")
    assert r.status_code == 200
    txt = r.text.lower()
    assert "backfill" not in txt  # graph notice must NOT appear in vector mode
    assert "neighbour" in txt or "embedded" in txt  # distinct no-data state


# --- PF-1: direct unit tests of the real _get_related logic ------------------


def test_get_related_vector_uses_embeddings_no_graph(monkeypatch):
    """Vector mode finds neighbours via ChromaDB and NEVER opens the graph."""
    from lit_monitor.server.routes import corpus as corpus_mod

    graph_opened = {"count": 0}

    def _graph_sentinel():
        graph_opened["count"] += 1
        return object()

    class _FakeEmbeddings:
        def find_similar_to_text(self, text, top_k, exclude_id):
            assert exclude_id == "10.1/a"
            return [
                {"id": "10.2/b", "score": 0.91, "document": "doc", "metadata": {}},
                {"id": "10.3/c", "score": 0.80, "document": "doc2", "metadata": {}},
            ]

    monkeypatch.setattr(corpus_mod, "safe_graph_db", _graph_sentinel)
    monkeypatch.setattr(
        corpus_mod, "_focused_embed_query", lambda state_db, doi, fallback: "q"
    )
    monkeypatch.setattr(corpus_mod, "_get_state_db", lambda: object())
    monkeypatch.setattr(corpus_mod, "_get_embeddings_db", lambda: _FakeEmbeddings())

    rows = corpus_mod._get_related("10.1/a", "vector", 10)
    assert rows is not None
    assert {r["doi"] for r in rows} == {"10.2/b", "10.3/c"}
    assert all(r["source"] == "vector" for r in rows)
    assert graph_opened["count"] == 0  # vector path must NOT open the graph


def test_get_related_vector_none_when_embeddings_absent(monkeypatch):
    from lit_monitor.server.routes import corpus as corpus_mod

    monkeypatch.setattr(corpus_mod, "_get_state_db", lambda: object())
    monkeypatch.setattr(corpus_mod, "_get_embeddings_db", lambda: None)
    assert corpus_mod._get_related("10.1/a", "vector", 10) is None


def test_get_related_graph_none_when_no_graph(monkeypatch):
    from lit_monitor.server.routes import corpus as corpus_mod

    monkeypatch.setattr(corpus_mod, "safe_graph_db", lambda: None)
    assert corpus_mod._get_related("10.1/a", "graph", 10) is None
