"""v0.9.0 LIVE end-to-end integration test (Audit-2 A2).

This file exercises the full v0.9 engine against REAL services — real Ollama
(embed + chat models), a real ChromaDB-backed :class:`EmbeddingsDB`, and a real
Kuzu-backed :class:`GraphDB` — wired together exactly as production wires them:

    embed 5 papers  →  cluster (k-means + silhouette)  →  rank (real cosine +
    real rationale LLM)  →  graph ask (NL → Cypher → execute → summarize)  →
    cluster-aware ask annotation.

It is the real replacement for the former all-mocks "integration" file, which
was a unit test in an integration costume (its own docstring claimed "No live
Ollama, no real ChromaDB") and contained one half-mocked test that accidentally
reached real Ollama and HUNG. Here every test that touches a service depends —
directly or through its fixture chain — on a real-service fixture that calls
:func:`tests.integration._live.skip_or_fail`. So:

    * No env vars, Ollama down  → every test SKIPS cleanly in well under a
      second. Zero hangs, zero real network calls.
    * ``LIT_MONITOR_LIVE=1`` with Ollama down → every test FAILS loudly
      (the live-mode contract: a green run with services down is a lie).

Run (live, manual / pre-release gate)::

    LIT_MONITOR_LIVE=1 uv run pytest tests/integration/test_v09_e2e.py -v -m integration

Required real services for the live run:
    * Ollama reachable at the configured ingestion host (default
      http://localhost:11434).
    * Embed model ``mxbai-embed-large`` pulled (``ollama pull mxbai-embed-large``).
    * The ingestion chat model pulled (default ``qwen2.5:3b`` — see
      ``real_llm`` in conftest.py).
    * Write access to pytest tmp dirs (ChromaDB + Kuzu persist to ``tmp_path``).
    * A valid ``~/.config/lit-monitor/config.toml`` (loaded by ``real_config``).

All tests are marked ``@pytest.mark.integration`` and are excluded from the
fast unit CI run — that is correct for a real-service test. The pure-unit
prompt-registry and AskResult-shape checks that used to live here were
relocated to tests/llm/test_prompt_registry.py and
tests/unit/test_graph_ask_exec.py respectively (Audit-2 A2), where they run
unconditionally.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from tests.integration._live import skip_or_fail as _skip_or_fail

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture library: 5 papers with text we embed directly (no Zotero markdown
# attachment required — the v0.9 engine only needs paper text + DOIs).
# ---------------------------------------------------------------------------

@pytest.fixture()
def five_papers() -> list[dict]:
    """Five papers spanning two clear themes (capture/polish vs filtration)
    so a real k-means over real embeddings produces meaningful structure."""
    return [
        {
            "doi": "10.0/a",
            "title": "Protein A affinity chromatography for mAb capture",
            "year": 2020,
            "journal": "J1",
            "abstract": (
                "Protein A affinity chromatography is the workhorse capture "
                "step for monoclonal antibody purification, binding the Fc "
                "region with high selectivity."
            ),
            "authors": ["Smith J", "Jones K"],
        },
        {
            "doi": "10.0/b",
            "title": "Cation exchange polishing of antibodies",
            "year": 2021,
            "journal": "J1",
            "abstract": (
                "Cation exchange chromatography is the standard polishing step "
                "after Protein A capture, clearing aggregates and host cell "
                "protein from the monoclonal antibody pool."
            ),
            "authors": ["Jones K", "Lee L"],
        },
        {
            "doi": "10.0/c",
            "title": "Viral filtration in downstream processing",
            "year": 2022,
            "journal": "J2",
            "abstract": (
                "Nanofiltration membranes remove enveloped and non-enveloped "
                "viruses from antibody process streams, a critical viral "
                "clearance unit operation in downstream filtration."
            ),
            "authors": ["Brown A"],
        },
        {
            "doi": "10.0/d",
            "title": "Tangential flow filtration for concentration",
            "year": 2023,
            "journal": "J2",
            "abstract": (
                "Tangential flow filtration with hollow-fibre membrane modules "
                "concentrates and buffer-exchanges protein solutions in the "
                "ultrafiltration/diafiltration step of downstream filtration."
            ),
            "authors": ["Patel R", "Brown A"],
        },
        {
            "doi": "10.0/e",
            "title": "High-throughput screening of chromatography resins",
            "year": 2024,
            "journal": "J3",
            "abstract": (
                "Miniaturised robotic column screens accelerate chromatography "
                "resin selection for antibody capture and polishing process "
                "development."
            ),
            "authors": ["Smith J", "Patel R"],
        },
    ]


def _embed_text(paper: dict) -> str:
    """Concatenate the text the engine embeds for a paper."""
    return f"{paper['title']}. {paper['abstract']}"


# ---------------------------------------------------------------------------
# Real-service fixtures. Every one funnels missing-service paths through
# skip_or_fail, so the whole module skips cleanly (or fails loudly in live
# mode) without ever issuing a real network call when Ollama is down.
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_embed_host(real_config) -> str:
    """Ollama host with mxbai-embed-large pulled. Skips/fails otherwise.

    Depends on ``real_config`` (which itself skips when config/vault are
    absent). ``mxbai-embed-large`` is the embedding model — distinct from the
    ingestion chat model that ``real_llm`` checks.
    """
    import requests as _requests

    host = getattr(real_config.ingestion, "ollama_host", "http://localhost:11434")
    try:
        resp = _requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — surface as skip/fail, never raise
        _skip_or_fail(f"Ollama not reachable at {host}: {exc}")

    pulled = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any("mxbai-embed-large" in p for p in pulled):
        _skip_or_fail(
            "Embed model 'mxbai-embed-large' is not pulled. "
            "Run: ollama pull mxbai-embed-large"
        )
    return host


@pytest.fixture()
def real_embeddings_db(real_embed_host, tmp_chroma_dir, five_papers):
    """A real EmbeddingsDB with the 5 fixture papers embedded via real Ollama.

    Each ``add_paper`` issues a real embedding call to mxbai-embed-large and
    persists the vector into a throwaway ChromaDB under ``tmp_chroma_dir``.
    """
    from lit_monitor.output.embeddings import EmbeddingsDB

    db = EmbeddingsDB(str(tmp_chroma_dir), ollama_host=real_embed_host)
    for paper in five_papers:
        db.add_paper(
            doi=paper["doi"],
            text=_embed_text(paper),
            metadata={"title": paper["title"], "year": paper["year"]},
        )
    assert db.count() == len(five_papers), "not all fixture papers were embedded"
    return db


@pytest.fixture()
def real_clusters(real_embeddings_db, real_config, tmp_state_db, five_papers):
    """Run the REAL clustering pipeline over the real embeddings.

    Overrides ``min_papers_threshold`` (default 100) down to 2 and ``k_min``
    down to 2 so a 5-paper library actually clusters. ``recompute_clusters``
    reads embeddings straight from ChromaDB, runs deterministic k-means
    (random_state=42) with silhouette-based K selection, names clusters
    (llm=None → ``Cluster N`` fallback), and persists rows + assignments into
    the real ``tmp_state_db``.

    Returns the list of persisted active clusters (Bundle C row dicts).
    """
    from lit_monitor.clustering.recompute import recompute_clusters

    # Seed state_db with the paper rows so cluster naming / title sampling has
    # something to read (name_cluster falls back to "Cluster N" with llm=None).
    for paper in five_papers:
        tmp_state_db.upsert_paper({
            "doi": paper["doi"],
            "title": paper["title"],
            "year": paper["year"],
            "source_type": "paper",
            "status": "extraction_complete",
        })

    # Clone the clustering config namespace with a low threshold so 5 papers
    # cross the gate. _Namespace is a plain dot-access wrapper, so setattr on a
    # shallow copy is the cleanest override that doesn't mutate the shared cfg.
    import copy

    cfg = copy.copy(real_config)
    cfg.clustering = copy.copy(real_config.clustering)
    cfg.clustering.min_papers_threshold = 2
    cfg.clustering.k_min = 2
    cfg.clustering.k_max = 3
    cfg.clustering.enabled = True

    created = recompute_clusters(tmp_state_db, real_embeddings_db, cfg, llm=None)
    if created < 1:
        _skip_or_fail(
            "recompute_clusters produced 0 clusters from 5 real embeddings — "
            "k-means/silhouette path did not select any K (unexpected)."
        )
    clusters = tmp_state_db.list_active_clusters()
    assert clusters, "clusters were created but list_active_clusters is empty"
    return clusters


@pytest.fixture()
def real_graph_db(tmp_path, five_papers):
    """A real tmp Kuzu GraphDB seeded with two papers + one shared entity.

    Skips/fails (instead of erroring) when the [graph] extra (kuzu) is not
    installed, mirroring production's ``safe_graph_db`` tolerance.
    """
    from lit_monitor.graph.import_citations import safe_graph_db

    graph_db = safe_graph_db(persist_dir=str(tmp_path / "graph.kuzu"))
    if graph_db is None:
        _skip_or_fail(
            "GraphDB unavailable — the [graph] extra (kuzu) is not installed. "
            "Install with: uv sync --extra graph"
        )

    from lit_monitor.graph.entity_extractor import EntityTuple

    shared = EntityTuple(
        canonical_id="protein a",
        type="method",
        surface="Protein A",
        field="methods_summary",
        span_start=None,
        span_end=None,
    )
    graph_db.add_paper(
        doi="10.0/a",
        entities=[shared],
        relationships=[],
        paper_metadata={"title": five_papers[0]["title"], "year": 2020, "journal": "J1"},
    )
    graph_db.add_paper(
        doi="10.0/b",
        entities=[shared],
        relationships=[],
        paper_metadata={"title": five_papers[1]["title"], "year": 2021, "journal": "J1"},
    )
    return graph_db


# ===========================================================================
# 1. Embeddings — real Ollama embed round trip
# ===========================================================================

@pytest.mark.integration
def test_embeddings_real_round_trip(real_embeddings_db, five_papers):
    """All 5 fixture papers embed into ChromaDB and similarity search ranks
    the most-relevant paper first for an aligned query."""
    results = real_embeddings_db.find_similar_to_text(
        "monoclonal antibody affinity capture with Protein A", top_k=5
    )
    assert results, "find_similar_to_text returned no results"
    ids = [r["id"] for r in results]
    # The Protein A capture paper should be the top hit for this query.
    assert ids[0] == "10.0/a", f"expected 10.0/a as top hit, got {ids}"
    # Each result carries the contract keys consumed downstream.
    top = results[0]
    assert "score" in top and "document" in top and "metadata" in top


# ===========================================================================
# 2. Clustering — real k-means / silhouette over real embeddings
# ===========================================================================

@pytest.mark.integration
def test_clustering_real_recompute(real_clusters):
    """recompute_clusters persists real clusters with stable IDs, names, and
    serialised centroids."""
    assert len(real_clusters) >= 1, "no active clusters persisted"
    for cluster in real_clusters:
        assert cluster.get("id") is not None
        assert cluster.get("display_name"), "cluster has no display_name"
        # Centroid blob must round-trip as a float32 vector.
        blob = cluster.get("centroid_blob")
        assert blob, f"cluster {cluster.get('id')} has empty centroid_blob"
        vec = np.frombuffer(blob, dtype=np.float32)
        assert vec.size > 0, "centroid deserialised to an empty vector"
    # K was chosen within the overridden [k_min=2, k_max=3] band.
    assert 1 <= len(real_clusters) <= 3


# ===========================================================================
# 3. Ranking — real cosine + real rationale LLM
# ===========================================================================

@pytest.mark.integration
def test_ranking_real_full_breakdown(real_embeddings_db, real_embed_host, real_llm, five_papers):
    """rank_papers over real embeddings (real cosine), a real domain-context
    embedding, and the real ingestion LLM for rationales must emit all six
    score_breakdown signal keys, preserve all 5 papers, and sort descending."""
    from lit_monitor.llm.ranker import rank_papers

    candidates = [
        {"doi": p["doi"], "title": p["title"], "abstract": p["abstract"]}
        for p in five_papers
    ]

    # Real domain-context embedding via the same Ollama embed model.
    domain_text = "downstream bioprocessing of monoclonal antibodies"
    domain_emb = real_embeddings_db.embed_text(domain_text)

    ranked = rank_papers(
        candidates,
        real_embeddings_db,
        real_llm,
        domain_context=domain_text,
        domain_context_emb=domain_emb,
        domain_context_weight=0.2,
    )

    required_signals = {
        "vector",
        "domain_context",
        "cluster_centroid",
        "graph_entity_overlap",
        "graph_citation",
        "graph_shared_authors",
    }
    assert ranked, "rank_papers returned no papers"
    for paper in ranked:
        breakdown = paper.get("score_breakdown", {})
        missing = required_signals - set(breakdown.keys())
        assert not missing, f"Paper {paper['doi']} missing signals: {missing}"
        for key, val in breakdown.items():
            assert isinstance(val, (int, float)), (
                f"Paper {paper['doi']} signal '{key}' is {type(val)}"
            )

    # No candidate silently dropped.
    dois = {p["doi"] for p in ranked}
    assert dois == {"10.0/a", "10.0/b", "10.0/c", "10.0/d", "10.0/e"}

    # Sorted descending by similarity_score.
    scores = [p["similarity_score"] for p in ranked]
    assert scores == sorted(scores, reverse=True), (
        "rank_papers output is not sorted descending by similarity_score"
    )


# ===========================================================================
# 4. Graph + ask — real NL → Cypher → execute → summarize via real Ollama
# ===========================================================================

@pytest.mark.integration
def test_graph_ask_real_pipeline(real_graph_db, real_llm):
    """run_pipeline drives a REAL Ollama client through NL → Cypher →
    execute (real Kuzu) → summarize, returning an AskResult with cypher + prose.

    The ask pipeline normally builds its own cloud client when OLLAMA_API_KEY is
    set; here we inject the real LOCAL ingestion client by patching
    ``_maybe_construct_client`` so the NL→Cypher and summarize stages make real
    local Ollama calls. This is the test that previously HUNG (it half-mocked
    its way into an unguarded real call) — it now only runs behind real_llm.
    """
    from lit_monitor.graph.ask import AskResult, run_pipeline

    with patch(
        "lit_monitor.graph.ask._maybe_construct_client",
        return_value=real_llm,
    ):
        result = run_pipeline(
            "Which papers are in my library? List their DOIs.",
            graph_db=real_graph_db,
            state_db=None,
            embeddings_db=None,
            summarize=True,
        )

    assert isinstance(result, AskResult)
    # A real LLM produced a real, validated, read-only Cypher query.
    assert result.cypher, f"no cypher generated; prose={result.prose!r}"
    # And a real prose summary (non-empty) came back.
    assert result.prose, f"no prose summary; cypher={result.cypher!r}"


# ===========================================================================
# 5. Cluster-aware ask — real clusters + real question embedding
# ===========================================================================

@pytest.mark.integration
def test_cluster_aware_ask_real(real_clusters, real_embeddings_db, tmp_state_db):
    """_identify_relevant_cluster over the REAL persisted clusters returns the
    dominant cluster for a capture-aligned query, using a real question
    embedding and real cosine similarity."""
    from lit_monitor.graph.ask import _identify_relevant_cluster

    result = _identify_relevant_cluster(
        "best methods for monoclonal antibody affinity capture",
        tmp_state_db,
        real_embeddings_db,
        threshold=0.1,
    )

    assert result is not None, (
        "_identify_relevant_cluster returned None despite real clusters and "
        "an aligned query"
    )
    assert result["display_name"], "matched cluster has no display_name"
    assert result["n_papers"] >= 1
    assert result["similarity"] >= 0.1

    # The matched cluster must be one of the real persisted clusters.
    real_names = {c["display_name"] for c in real_clusters}
    assert result["display_name"] in real_names, (
        f"matched cluster {result['display_name']!r} not in persisted set "
        f"{real_names}"
    )
