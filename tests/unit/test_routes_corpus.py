"""FE2-2: GET /corpus list page route tests.

The state DB accessor is patched at the single seam
``scripts.server.routes.corpus._list_papers`` — no live SQLite, embeddings, or
network. Mirrors the ``client`` fixture from tests/unit/test_routes_ask.py.
"""
from __future__ import annotations

import logging
import re

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIT_MONITOR_STATE_DB", str(tmp_path / "state.db"))
    from lit_monitor.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _reset_corpus_query_store():
    """Clear the process-global last-corpus-query store around every test.

    The state-retention feature (item 5) keeps the last-applied corpus query in a
    module-level dict so a bare ``/corpus`` visit restores the user's filters for
    the life of the server process. That global makes the route tests
    order-dependent (one test's saved query leaks into the next bare visit) unless
    it is reset, so this autouse fixture clears it before AND after each test.
    """
    from lit_monitor.server.routes.corpus import _reset_corpus_state  # noqa: PLC0415

    _reset_corpus_state()
    yield
    _reset_corpus_state()


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


# --- P4: Corpus Health — inline toolbar + auto-apply + status-dot table -------


def test_corpus_page_is_corpus_health(client, monkeypatch):
    # P4: rename to "Corpus Health" — the h1 changes, the /corpus URL stays.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0)
    )
    r = client.get("/corpus")
    assert r.status_code == 200
    assert "<h1>Corpus Health</h1>" in r.text


def test_corpus_list_has_inline_toolbar_no_rail(client, monkeypatch):
    # P4: the left filter rail is GONE — filters live in a single inline toolbar
    # above the table. The toolbar holds Shoelace controls (search input + the
    # select filters + a page-size select). Active values are preserved on the
    # HOST (value="…") per the Shoelace initial-value gotcha. There is NO Apply
    # button — filters auto-apply via the form's hx-trigger.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_themes",
        lambda: [{"id": 1, "display_name": "Chromatography"}],
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0)
    )
    r = client.get(
        "/corpus?search=carta&source_type=review&status_gap=missing_graph"
        "&theme=Chromatography&sort=year&limit=50"
    )
    assert r.status_code == 200
    # The two-column rail layout is removed.
    assert "corpus-layout" not in r.text
    assert "corpus-rail" not in r.text
    # The inline toolbar replaces it.
    assert 'class="corpus-toolbar"' in r.text
    # Shoelace search input bound to the route's EXACT param name (search, not q).
    assert "<sl-input" in r.text
    assert 'name="search"' in r.text and 'value="carta"' in r.text
    # Shoelace selects for the filters, each pre-selected via value= on the HOST.
    assert "<sl-select" in r.text
    assert 'name="source_type"' in r.text and 'value="review"' in r.text
    assert 'name="status_gap"' in r.text and 'value="missing_graph"' in r.text
    assert 'name="theme"' in r.text and 'value="Chromatography"' in r.text
    assert 'name="sort"' in r.text and 'value="year"' in r.text
    # A page-size select driving the route's `limit` param, value preserved.
    assert 'name="limit"' in r.text and 'value="50"' in r.text
    # NO Apply button (filters auto-apply); no native form controls.
    assert "Apply" not in r.text
    assert "<select" not in r.text


def test_corpus_form_auto_applies_on_change(client, monkeypatch):
    # P4: removing Apply means the form must auto-fire on sl-change (selects) and
    # debounced sl-input (search). Assert the hx-trigger wiring is present.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0)
    )
    r = client.get("/corpus")
    assert r.status_code == 200
    assert 'hx-get="/corpus"' in r.text
    assert "sl-change" in r.text
    assert "sl-input" in r.text  # debounced search trigger source


def test_corpus_table_status_dot_and_doi_link(client, monkeypatch):
    # P4: the status column renders .status-dot (color = state, not glyphs) and
    # the DOI links to the internal detail page with a .doi-cell truncation.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers",
        lambda **k: (
            [
                {
                    "doi": "10.1/a",
                    "title": "Chromatography study",
                    "year": 2021,
                    "source_type": "paper",
                    "confidence": 0.8,
                    "embeddings_indexed": 1,
                    "graph_indexed": 0,
                    "notes_synced": 1,
                    "last_updated": "2026-06-01",
                }
            ],
            1,
        ),
    )
    r = client.get("/corpus")
    assert r.status_code == 200
    # Status shown as coloured dots, not the old ✓/✗ pill glyphs.
    assert "status-dot" in r.text
    assert "&check;" not in r.text and "&cross;" not in r.text
    # The DOI cell links to the internal detail page with truncation + tooltip.
    assert "doi-cell" in r.text
    assert 'href="/corpus/10.1/a"' in r.text
    assert 'title="10.1/a"' in r.text


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
    # review #4: the full note-path text is no longer rendered on the page; the
    # note is reachable only via the Obsidian deep-link button (vault-gated, not
    # configured here). The old "Obsidian note: <path>" text line is removed.
    assert "Obsidian note:" not in r.text
    assert (
        'hx-post="/api/papers/10.1/carta/relink"' in r.text or "relink" in r.text
    )


# --- P4: detail page Shoelace collapsibles + capped extraction list ----------


def test_corpus_detail_has_sl_details_sections(client, monkeypatch):
    # P4: the detail body is organised into <sl-details> sections. Extraction is
    # open by default; the rest are collapsed. The action buttons are <sl-button>.
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
            "note_path": None,
            # Indexed paper → the lazy related/graph fragments render (the honest
            # placeholders only kick in when these flags are falsy — see item 1).
            "embeddings_indexed": 1,
            "graph_indexed": 1,
            "extraction": {"core_finding": "x", "_overall_confidence": 0.7},
        },
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown",
        lambda doi: {"doi": doi, "run_id": "r", "breakdown": {"cosine": 0.5}},
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    # Each section is a Shoelace <sl-details> with its heading as the summary.
    assert 'summary="Extraction"' in r.text
    assert 'summary="Score decomposition"' in r.text
    assert 'summary="Related work"' in r.text
    assert 'summary="Knowledge graph"' in r.text
    # Extraction is open by default; verify its <sl-details> carries `open`.
    extraction_tag = re.search(r"<sl-details[^>]*summary=\"Extraction\"[^>]*>", r.text)
    assert extraction_tag is not None and " open" in extraction_tag.group(0)
    # Action controls are Shoelace buttons that still POST the same endpoints.
    assert "<sl-button" in r.text
    assert 'hx-post="/api/papers/10.1/carta/relink"' in r.text
    assert 'hx-post="/api/papers/10.1/carta/re-extract"' in r.text
    # The lazy fragments live inside their collapsed sections.
    assert 'id="corpus-related"' in r.text
    assert 'id="corpus-graph"' in r.text


def test_corpus_detail_extraction_list_height_capped(client, monkeypatch):
    # P4 / rubric #15: the extraction <dl> can have 100+ fields, so its container
    # is height-capped and scrolls inside the card. Assert the CSS rule ships.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: {
            "doi": doi,
            "title": "X",
            "authors": [],
            "year": None,
            "journal": None,
            "source_type": "paper",
            "zotero_key": None,
            "note_path": None,
            "extraction": {"a": "1", "b": "2"},
        },
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/x")
    assert r.status_code == 200
    # The extraction list container carries the cap class.
    assert "extraction-scroll" in r.text
    # And the stylesheet bounds its height with internal scroll.
    css = client.get("/static/site.css").text
    assert ".extraction-scroll" in css
    assert "max-height" in css and "overflow" in css


def test_corpus_results_table_height_capped(client):
    # Review 2026-06-13 / rubric #15: the corpus paper list scrolls inside its
    # own region (like the digest box) instead of stretching the whole page.
    css = client.get("/static/site.css").text
    # Find the .corpus-results rule and assert it bounds height + scrolls.
    idx = css.index(".corpus-results {")
    rule = css[idx : idx + 120]
    assert "max-height" in rule
    assert "overflow" in rule
    # Sticky header keeps the column labels visible while scrolling.
    assert ".corpus-results table thead th" in css
    assert "position: sticky" in css


def test_corpus_table_one_line_rows_css(client):
    # P4: every row stays on ONE line — the corpus table is table-layout:fixed
    # and cells truncate with ellipsis. Assert the CSS rules ship.
    css = client.get("/static/site.css").text
    assert ".corpus-results table" in css
    assert "table-layout: fixed" in css
    # The inline toolbar wraps on narrow screens.
    assert ".corpus-toolbar" in css
    assert "flex-wrap" in css


def test_corpus_table_fragment_markup(client, monkeypatch):
    # P4: the _table.html fragment uses table-layout:fixed, truncates the title
    # with a title= tooltip, and the pager swaps #corpus-table via hx-select.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers",
        lambda **k: (
            [
                {
                    "doi": "10.1/a",
                    "title": "A long chromatography membrane study title",
                    "year": 2021,
                    "source_type": "paper",
                    "confidence": 0.8,
                    "embeddings_indexed": 1,
                    "graph_indexed": 1,
                    "notes_synced": 1,
                    "last_updated": "2026-06-01",
                }
            ],
            45,  # > limit so Next pagination shows
        ),
    )
    # HTMX fragment request returns _table.html standalone.
    r = client.get("/corpus?limit=20", headers={"HX-Request": "true"})
    assert r.status_code == 200
    # Title truncated with a full-title tooltip (same treatment as DOI).
    assert 'title="A long chromatography membrane study title"' in r.text
    # Pager re-targets the swap region in place via hx-select/outerHTML.
    assert 'hx-target="#corpus-table"' in r.text
    assert 'hx-select="#corpus-table"' in r.text
    assert 'hx-swap="outerHTML"' in r.text
    # The "X–Y of Z" count is shown.
    assert "of 45" in r.text
    # Next is reachable (offset 0 + limit 20 < 45); href fallback present.
    assert "offset=20" in r.text


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


# --- P4 review refinements (2026-06-13): list filters/pager + detail page -----


def _detail_paper(_doi, **over):
    """Helper: a baseline _get_paper_row dict for the detail-page tests.

    The positional arg is the route DOI; pass ``doi=...`` in ``over`` to override
    the stored doi value (e.g. doi="" for the no-DOI case).
    """
    base = {
        "doi": _doi,
        "title": "Carta 2009",
        "authors": ["A. Carta"],
        "year": 2009,
        "journal": "J Chrom",
        "source_type": "paper",
        "zotero_key": "ABCD1234",
        "note_path": None,
        # Item 1: index flags drive the honest related/graph placeholders. Default
        # to indexed so existing tests keep the lazy fragments; the placeholder
        # tests pass embeddings_indexed=0 / graph_indexed=0 explicitly.
        "embeddings_indexed": 1,
        "graph_indexed": 1,
        "extraction": {},
    }
    base.update(over)
    return base


def test_corpus_toolbar_holds_only_filters_not_limit(client, monkeypatch):
    # P4 review #2: the per-page selector moves OUT of the top filter toolbar to
    # the bottom near the pager. The toolbar holds only the 4 filters; the limit
    # select renders after the table (it's a different kind of setting).
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0)
    )
    r = client.get("/corpus?limit=50")
    assert r.status_code == 200
    # Split the page at the toolbar form's close + the table region so we can
    # assert the limit select is NOT inside the toolbar but IS after the table.
    toolbar_start = r.text.index('class="corpus-toolbar"')
    toolbar_end = r.text.index("</form>", toolbar_start)
    toolbar = r.text[toolbar_start:toolbar_end]
    # The 4 filters live in the toolbar…
    assert 'name="search"' in toolbar
    assert 'name="source_type"' in toolbar
    assert 'name="status_gap"' in toolbar
    assert 'name="sort"' in toolbar
    # …but the per-page (limit) select does NOT.
    assert 'name="limit"' not in toolbar
    # The limit select still renders on the page (after the table), value kept.
    assert 'name="limit"' in r.text and 'value="50"' in r.text
    # "Per page" label sits with the pager, not the toolbar.
    assert "Per page" in r.text[toolbar_end:]


def test_corpus_limit_select_in_table_fragment_near_pager(client, monkeypatch):
    # P4 review #2: the limit select is part of the _table.html fragment (so it
    # swaps with the pager) and drives the route via the same HTMX swap.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers",
        lambda **k: ([], 0),
    )
    r = client.get("/corpus?limit=50", headers={"HX-Request": "true"})
    assert r.status_code == 200
    # Fragment (table region) carries the limit select with the active value.
    assert 'name="limit"' in r.text and 'value="50"' in r.text
    # It re-fetches via the same #corpus-table swap as the rest of the controls.
    assert 'hx-target="#corpus-table"' in r.text


def test_corpus_table_doi_links_externally(client, monkeypatch):
    # P4 review #3: the DOI cell now links to the EXTERNAL doi.org URL (new tab),
    # while the TITLE keeps linking to the internal /corpus/{doi} detail page.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers",
        lambda **k: (
            [
                {
                    "doi": "10.1/a",
                    "title": "Chromatography study",
                    "year": 2021,
                    "source_type": "paper",
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
    r = client.get("/corpus", headers={"HX-Request": "true"})
    assert r.status_code == 200
    # DOI cell → external doi.org, new tab, noopener, truncation tooltip kept.
    assert 'href="https://doi.org/10.1/a"' in r.text
    assert 'target="_blank"' in r.text and 'rel="noopener"' in r.text
    assert "doi-cell" in r.text and 'title="10.1/a"' in r.text
    # Title still links to the internal detail page.
    assert 'href="/corpus/10.1/a"' in r.text


def test_corpus_table_no_doi_plain_text(client, monkeypatch):
    # P4 review #3: a row with no DOI shows a plain placeholder, no external link.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers",
        lambda **k: (
            [
                {
                    "doi": "",
                    "title": "No-DOI paper",
                    "year": 2021,
                    "source_type": "paper",
                    "confidence": None,
                    "embeddings_indexed": 0,
                    "graph_indexed": 0,
                    "notes_synced": 0,
                    "last_updated": "2026-06-01",
                }
            ],
            1,
        ),
    )
    r = client.get("/corpus", headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert "https://doi.org/" not in r.text  # no external DOI link for a no-DOI row


# --- P4 review: detail page — 3 inline buttons, single cards, extraction table -


def test_corpus_detail_three_inline_buttons_no_text_links(client, monkeypatch):
    # P4 review #4: the DOI text link + "Open in Zotero" anchor + "Obsidian note:
    # path" text are replaced by THREE <sl-button>s with the right hrefs/classes.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, note_path="Literature/Papers/Carta 2009.md"),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._vault_name", lambda: "MyVault"
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    # Open DOI button → external doi.org, new tab.
    assert "Open DOI" in r.text
    assert 'href="https://doi.org/10.1/carta"' in r.text
    # Open in Zotero button → zotero:// deep link, Zotero-red class.
    assert "zotero://select/library/items/ABCD1234" in r.text
    assert "btn-zotero" in r.text
    # Open in Obsidian button → obsidian:// deep link, purple class.
    assert "obsidian://open?vault=MyVault" in r.text
    assert "btn-obsidian" in r.text
    # The OLD text affordances are GONE: no <a class="doi"> text line, no
    # "Obsidian note:" path text.
    assert 'class="doi"' not in r.text
    assert "Obsidian note:" not in r.text


def test_corpus_detail_obsidian_button_absent_without_uri(client, monkeypatch):
    # P4 review #4: the Obsidian button only renders when obsidian_uri is set.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, note_path=None),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    assert "btn-obsidian" not in r.text and "obsidian://open" not in r.text
    # Zotero + DOI buttons still present.
    assert "Open DOI" in r.text and "btn-zotero" in r.text


def test_corpus_detail_doi_button_absent_without_doi(client, monkeypatch):
    # P4 review #4: the Open DOI button only renders when paper.doi is set.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, doi=""),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    assert "Open DOI" not in r.text and "https://doi.org/" not in r.text


def test_corpus_detail_sections_not_double_carded(client, monkeypatch):
    # P4 review #5: each <sl-details> is its own bordered card — class="card" on
    # it adds a SECOND nested card box, so it must be removed from every section.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, extraction={"core_finding": "x"}),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown",
        lambda doi: {"doi": doi, "run_id": "r", "breakdown": {"cosine": 0.5}},
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    # No <sl-details> carries class="card".
    for tag in re.findall(r"<sl-details[^>]*>", r.text):
        assert 'class="card"' not in tag and "card" not in tag


def test_corpus_detail_extraction_renders_table_with_confidence(client, monkeypatch):
    # P4 review #6: the raw <dl class="extraction-list"> dump is replaced by a
    # 3-column Field | Value | Confidence table. Each field's _confidence sibling
    # is folded into its row; metadata (_*) and *_confidence keys are skipped.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(
            doi,
            extraction={
                "limitations": "small sample size",
                "limitations_confidence": "explicit",
                "assumptions": "linear binding",
                "assumptions_confidence": "high",
                "_overall_confidence": 0.7,
            },
        ),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    # The old <dl> dump is gone, a <table> ships instead.
    assert 'class="extraction-list"' not in r.text
    # A "Confidence" column header is present.
    assert "Confidence" in r.text
    # Humanized field name (underscore → space, capitalized first letter).
    assert "Limitations" in r.text
    # The field's value + its folded confidence both render.
    assert "small sample size" in r.text and "explicit" in r.text
    assert "linear binding" in r.text and "high" in r.text
    # The *_confidence keys are NOT surfaced as their own rows / humanized labels.
    assert "Limitations confidence" not in r.text
    # Overall confidence line still present above the table.
    assert "Overall confidence" in r.text
    # Still inside the height-capped scroll container.
    assert "extraction-scroll" in r.text


def test_corpus_detail_extraction_empty_state(client, monkeypatch):
    # P4 review #6: an empty extraction still shows the no-fields notice.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, extraction={}),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    assert "No extraction fields" in r.text


def test_corpus_detail_actions_not_card_buttons_match_size(client, monkeypatch):
    # P4 review #7: the Actions block is a section, not a .card; its Relink /
    # Re-extract buttons match the top Open-in-* button size (medium, not small).
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    # The action block keeps its id + header but is NOT a .card.
    actions_start = r.text.index('id="corpus-actions"')
    # Walk back to the opening tag of the actions container.
    open_tag_start = r.text.rindex("<div", 0, actions_start)
    open_tag = r.text[open_tag_start : r.text.index(">", actions_start) + 1]
    assert "card" not in open_tag
    assert "<h2>Actions</h2>" in r.text
    # Relink / Re-extract buttons are size="medium" (not small) to match the top.
    assert 'hx-post="/api/papers/10.1/carta/relink"' in r.text
    assert 'hx-post="/api/papers/10.1/carta/re-extract"' in r.text
    # No small buttons left in the detail page BODY (the base.html top-bar "Ask"
    # button is size="small" but is shell chrome, not detail content).
    detail_start = r.text.index("paper-detail")
    detail_end = r.text.index("</section>", detail_start)
    body = r.text[detail_start:detail_end]
    assert 'size="small"' not in body


# --- Corpus Health review fixes (2026-06-13): honest placeholders, actions row,
#     back link, inline per-page, in-memory state retention, reset button -------


def test_corpus_detail_back_link_reads_corpus_health(client, monkeypatch):
    # Item 3: the back link reads "Back to Corpus Health" (not "Back to corpus").
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    assert "Back to Corpus Health" in r.text
    # The bare /corpus href is preserved (state restore relies on it being bare).
    assert 'href="/corpus"' in r.text


def test_corpus_detail_related_honest_when_not_embedded(client, monkeypatch):
    # Item 1: a paper with embeddings_indexed=0 can NEVER have related work, so the
    # Related-work section renders an honest note and NOT the lazy hx-trigger div.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, embeddings_indexed=0, graph_indexed=1),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    # The Related-work section is present but shows the honest "not embedded" note…
    assert "isn't embedded yet" in r.text
    # …and does NOT lazy-load (no related #corpus-related hx-trigger).
    rel_start = r.text.index('summary="Related work"')
    rel_end = r.text.index("</sl-details>", rel_start)
    rel_block = r.text[rel_start:rel_end]
    assert "sl-show once" not in rel_block
    assert 'id="corpus-related"' not in rel_block


def test_corpus_detail_related_lazy_when_embedded(client, monkeypatch):
    # Item 1: a paper with embeddings_indexed=1 keeps the lazy related-work div.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, embeddings_indexed=1, graph_indexed=0),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    rel_start = r.text.index('summary="Related work"')
    rel_end = r.text.index("</sl-details>", rel_start)
    rel_block = r.text[rel_start:rel_end]
    assert 'id="corpus-related"' in rel_block
    assert 'hx-trigger="sl-show once"' in rel_block
    assert "isn't embedded yet" not in rel_block


def test_corpus_detail_graph_honest_when_not_graph_indexed(client, monkeypatch):
    # Item 1: a paper with graph_indexed=0 shows the honest "not graph-indexed"
    # note in the Knowledge-graph section and NOT the lazy graph div.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, embeddings_indexed=1, graph_indexed=0),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    assert "isn't graph-indexed yet" in r.text
    g_start = r.text.index('summary="Knowledge graph"')
    g_end = r.text.index("</sl-details>", g_start)
    g_block = r.text[g_start:g_end]
    assert "sl-show once" not in g_block
    assert 'id="corpus-graph"' not in g_block


def test_corpus_detail_graph_lazy_when_graph_indexed(client, monkeypatch):
    # Item 1: a paper with graph_indexed=1 keeps the lazy knowledge-graph div.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi, embeddings_indexed=0, graph_indexed=1),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    g_start = r.text.index('summary="Knowledge graph"')
    g_end = r.text.index("</sl-details>", g_start)
    g_block = r.text[g_start:g_end]
    assert 'id="corpus-graph"' in g_block
    assert 'hx-trigger="sl-show once"' in g_block
    assert "isn't graph-indexed yet" not in g_block


def test_corpus_detail_actions_buttons_in_button_row_after_header(client, monkeypatch):
    # Item 2: the Relink / Re-extract buttons sit on a NEW LINE under the
    # <h2>Actions</h2> header — wrapped in a .button-row that comes AFTER the <h2>.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_paper_row",
        lambda doi: _detail_paper(doi),
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._get_score_breakdown", lambda doi: None
    )
    r = client.get("/corpus/10.1/carta")
    assert r.status_code == 200
    # Inside the actions block, the button-row opens AFTER the </h2>, and the
    # buttons live inside it.
    actions_start = r.text.index('id="corpus-actions"')
    actions_end = r.text.index("</section>", actions_start)
    block = r.text[actions_start:actions_end]
    h2_idx = block.index("<h2>Actions</h2>")
    row_idx = block.index('class="button-row"')
    assert row_idx > h2_idx  # button-row comes AFTER the header
    # Both buttons are inside the button-row (after its opening, before </div>).
    relink_idx = block.index('hx-post="/api/papers/10.1/carta/relink"')
    reextract_idx = block.index('hx-post="/api/papers/10.1/carta/re-extract"')
    assert relink_idx > row_idx and reextract_idx > row_idx


def test_corpus_toolbar_has_reset_button(client, monkeypatch):
    # Item 6: the filter toolbar carries a Reset sl-button → /corpus?reset=1.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers", lambda **k: ([], 0)
    )
    r = client.get("/corpus")
    assert r.status_code == 200
    assert "<sl-button" in r.text and 'href="/corpus?reset=1"' in r.text
    assert ">Reset<" in r.text or "Reset</sl-button>" in r.text


def test_corpus_pager_per_page_label_and_labelless_select(client, monkeypatch):
    # Item 4: the per-page control renders as an inline "Per page" label + a
    # labelLESS sl-select (no label= attribute), with the count to its right.
    monkeypatch.setattr(
        "lit_monitor.server.routes.corpus._list_papers",
        lambda **k: (
            [
                {
                    "doi": "10.1/a",
                    "title": "X",
                    "year": 2021,
                    "source_type": "paper",
                    "confidence": 0.8,
                    "embeddings_indexed": 1,
                    "graph_indexed": 1,
                    "notes_synced": 1,
                    "last_updated": "2026-06-01",
                }
            ],
            14,
        ),
    )
    r = client.get("/corpus", headers={"HX-Request": "true"})
    assert r.status_code == 200
    # An inline "Per page" label sits in the pager.
    assert "Per page" in r.text
    # The limit select has NO label= attribute (that would stack the label above).
    sel = re.search(r'<sl-select[^>]*name="limit"[^>]*>', r.text)
    assert sel is not None and "label=" not in sel.group(0)
    # The count is present.
    assert "of 14" in r.text


# --- Item 5: in-memory corpus query state retention --------------------------


def test_corpus_state_restored_on_bare_visit_after_explicit(client, monkeypatch):
    # Item 5(b)→restore: an explicit filtered request is SAVED; a subsequent BARE
    # /corpus visit RESTORES it (controls reflect it + _list_papers sees it).
    seen: list[dict] = []

    def _fake(**k):
        seen.append(dict(k))
        return ([], 0)

    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_papers", _fake)
    # Explicit request with filters → saved.
    r1 = client.get("/corpus?search=carta&limit=50")
    assert r1.status_code == 200
    # Bare visit (Back link / sidebar) → restores search=carta + limit=50.
    r2 = client.get("/corpus")
    assert r2.status_code == 200
    assert 'value="carta"' in r2.text  # search control restored
    assert 'value="50"' in r2.text  # per-page control restored
    # The effective query passed to _list_papers on the bare visit was restored.
    assert seen[-1]["search"] == "carta"
    assert seen[-1]["limit"] == 50


def test_corpus_reset_clears_state_and_renders_defaults(client, monkeypatch):
    # Item 5(a)→reset: /corpus?reset=1 clears the store AND renders defaults.
    seen: list[dict] = []

    def _fake(**k):
        seen.append(dict(k))
        return ([], 0)

    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_papers", _fake)
    client.get("/corpus?search=carta&limit=50")  # save
    r_reset = client.get("/corpus?reset=1")
    assert r_reset.status_code == 200
    assert 'value="carta"' not in r_reset.text  # search not restored
    assert seen[-1]["search"] is None  # default (no filter)
    # And a subsequent bare visit stays default (store was cleared, not restored).
    r_bare = client.get("/corpus")
    assert r_bare.status_code == 200
    assert 'value="carta"' not in r_bare.text
    assert seen[-1]["search"] is None


def test_corpus_first_bare_visit_uses_defaults(client, monkeypatch):
    # Item 5(c): with an empty store, a first bare /corpus visit uses defaults.
    seen: list[dict] = []

    def _fake(**k):
        seen.append(dict(k))
        return ([], 0)

    monkeypatch.setattr("lit_monitor.server.routes.corpus._list_papers", _fake)
    r = client.get("/corpus")
    assert r.status_code == 200
    assert seen[-1]["search"] is None
    assert seen[-1]["limit"] == 20  # the route default
    assert 'value="carta"' not in r.text
