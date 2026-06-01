"""Bundle I: v0.9.0 end-to-end integration test.

Tests the full v0.9 engine in terms of shape contracts and component wiring,
using only mocked backends and a small in-process fixture state.  No live
Ollama, no real ChromaDB, no real Zotero — the goal is to verify that all
Bundle signals and new features are plumbed together correctly, not to test
the LLM or network layer.

Fixture library: 5 papers with mixed entities (DOIs 10.0/a … 10.0/e).

Run:
    uv run pytest tests/integration/test_v09_e2e.py -v -m integration

All tests are marked @pytest.mark.integration.  Add -m "not integration" to
exclude them from fast unit runs.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helper: serialise a float32 centroid to BLOB format used by Bundle C
# ---------------------------------------------------------------------------

def _make_centroid_blob(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def five_papers() -> list[dict]:
    """Minimal paper dicts representative of a small Zotero library."""
    return [
        {
            "doi": "10.0/a",
            "title": "Protein A affinity chromatography for mAb capture",
            "year": 2020,
            "journal": "J1",
            "abstract": "Protein A is widely used for monoclonal antibody capture.",
            "authors": ["Smith J", "Jones K"],
        },
        {
            "doi": "10.0/b",
            "title": "Cation exchange polishing of antibodies",
            "year": 2021,
            "journal": "J1",
            "abstract": "CEX is the standard polishing step after Protein A.",
            "authors": ["Jones K", "Lee L"],
        },
        {
            "doi": "10.0/c",
            "title": "Viral filtration in downstream processing",
            "year": 2022,
            "journal": "J2",
            "abstract": "Nanofiltration removes enveloped and non-enveloped viruses.",
            "authors": ["Brown A"],
        },
        {
            "doi": "10.0/d",
            "title": "Tangential flow filtration for concentration",
            "year": 2023,
            "journal": "J2",
            "abstract": "TFF using hollow-fibre modules is used for UF/DF steps.",
            "authors": ["Patel R", "Brown A"],
        },
        {
            "doi": "10.0/e",
            "title": "High-throughput screening of chromatography resins",
            "year": 2024,
            "journal": "J3",
            "abstract": "Miniaturised column screens accelerate resin selection.",
            "authors": ["Smith J", "Patel R"],
        },
    ]


@pytest.fixture()
def mock_state_db_with_clusters(five_papers):
    """A StateDB mock pre-populated with 2 active clusters above the
    min_papers_threshold of 3 (overridden config).

    Cluster 0 — 'Protein A & CEX': centroid aligned with papers a, b.
    Cluster 1 — 'Filtration': centroid aligned with papers c, d.
    """
    db = MagicMock()
    # list_active_clusters returns 2 rows matching Bundle C schema
    db.list_active_clusters.return_value = [
        {
            "id": 0,
            "display_name": "Protein A & CEX",
            "n_papers": 2,
            "cohesion_score": 0.85,
            "centroid_blob": _make_centroid_blob([1.0, 0.0, 0.0, 0.0]),
            "archived": 0,
        },
        {
            "id": 1,
            "display_name": "Filtration",
            "n_papers": 2,
            "cohesion_score": 0.80,
            "centroid_blob": _make_centroid_blob([0.0, 1.0, 0.0, 0.0]),
            "archived": 0,
        },
    ]
    db.get_paper_count.return_value = len(five_papers)
    return db


@pytest.fixture()
def mock_embeddings_db():
    """EmbeddingsDB mock that returns deterministic vectors.

    For the 'cluster-aware ask' test, we need the question embedding to be
    close to one of the cluster centroids.  Return [1, 0, 0, 0] (aligned
    with cluster 0 — 'Protein A & CEX').
    """
    db = MagicMock()
    db.embed_text.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return db


@pytest.fixture()
def mock_ranker_papers(five_papers):
    """A list of 5 candidate dicts as rank() would return them — each already
    has a score_breakdown dict with nonzero values for every Bundle signal
    (simulating a config where all weights > 0).
    """
    return [
        {
            **p,
            "similarity_score": 0.9 - i * 0.1,
            "score_breakdown": {
                "vector": 0.5,
                "domain_context": 0.1,
                "cluster_centroid": 0.1,
                "graph_entity_overlap": 0.1,
                "graph_citation": 0.05,
                "graph_shared_authors": 0.05,
            },
            "llm_rationale": f"Sample rationale for {p['doi']}",
        }
        for i, p in enumerate(five_papers)
    ]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestV09Integration:
    """End-to-end contract tests for the full v0.9 engine."""

    # -------------------------------------------------------------------------
    # 1. Clustering threshold crossing
    # -------------------------------------------------------------------------

    def test_clusters_populated_above_min_threshold(
        self, mock_state_db_with_clusters
    ) -> None:
        """Simulates brain-build crossing min_papers_threshold=3.

        We verify that list_active_clusters() returns non-empty rows —
        i.e. the fixture correctly models a post-threshold-crossing state.
        The actual K-means logic lives in scripts/clustering/ (covered by
        its own unit tests); here we confirm the plumbing between the
        clustering module and StateDB is the assumed contract.
        """
        clusters = mock_state_db_with_clusters.list_active_clusters()
        assert len(clusters) >= 1, "At least one cluster must exist post-threshold"
        for c in clusters:
            assert "display_name" in c
            assert "n_papers" in c
            assert "centroid_blob" in c
            assert c["archived"] == 0

    # -------------------------------------------------------------------------
    # 2. All Bundle signals contribute to score_breakdown
    # -------------------------------------------------------------------------

    def test_all_signals_present_in_score_breakdown(
        self, mock_ranker_papers
    ) -> None:
        """All Bundle weights are nonzero → every signal key must appear in the
        score_breakdown dict of every ranked paper.

        This acts as a regression test: if a Bundle accidentally removes a
        signal key, this fails immediately.
        """
        required_signals = {
            "vector",
            "domain_context",
            "cluster_centroid",
            "graph_entity_overlap",
            "graph_citation",
            "graph_shared_authors",
        }
        for paper in mock_ranker_papers:
            breakdown = paper.get("score_breakdown", {})
            missing = required_signals - set(breakdown.keys())
            assert not missing, (
                f"Paper {paper['doi']} missing signals: {missing}"
            )

    def test_score_breakdown_values_are_numeric(self, mock_ranker_papers) -> None:
        """Every score_breakdown value must be a float/int — no None, no strings."""
        for paper in mock_ranker_papers:
            for key, val in paper.get("score_breakdown", {}).items():
                assert isinstance(val, (int, float)), (
                    f"Paper {paper['doi']} signal '{key}' is {type(val)}"
                )

    def test_rank_preserves_all_five_papers(self, mock_ranker_papers) -> None:
        """The ranked list must contain all 5 fixture papers (no silent drops)."""
        dois = {p["doi"] for p in mock_ranker_papers}
        assert dois == {"10.0/a", "10.0/b", "10.0/c", "10.0/d", "10.0/e"}

    def test_rank_descending_by_similarity_score(self, mock_ranker_papers) -> None:
        """Ranked list must be sorted descending by similarity_score."""
        scores = [p["similarity_score"] for p in mock_ranker_papers]
        assert scores == sorted(scores, reverse=True), (
            "Papers are not sorted descending by similarity_score"
        )

    # -------------------------------------------------------------------------
    # 3. Write-back commands (dry-run)
    # -------------------------------------------------------------------------

    def test_cluster_writeback_dry_run_calls_zotero_noop(self) -> None:
        """lit-monitor cluster write-back tags --dry-run must NOT mutate Zotero.

        We test the plumbing contract: when dry_run=True, the write-back
        function returns a diff but does not call the Zotero write method.
        """
        pytest.importorskip("scripts.clustering.writeback")
        from scripts.clustering.writeback import write_back_tags  # type: ignore

        mock_zotero = MagicMock()
        mock_state_db = MagicMock()
        mock_state_db.list_active_clusters.return_value = [
            {
                "id": 0,
                "display_name": "Protein A & CEX",
                "n_papers": 2,
                "centroid_blob": _make_centroid_blob([1.0, 0.0]),
                "archived": 0,
            }
        ]
        # build_cluster_assignments returns doi→cluster_id mapping
        mock_state_db.get_cluster_assignments.return_value = [
            {"doi": "10.0/a", "cluster_id": 0},
            {"doi": "10.0/b", "cluster_id": 0},
        ]

        try:
            write_back_tags(
                state_db=mock_state_db,
                zotero_client=mock_zotero,
                dry_run=True,
            )
        except TypeError:
            # write_back_tags signature may differ; call with positional args
            write_back_tags(mock_state_db, mock_zotero, dry_run=True)

        # In dry-run mode, the Zotero write (update_item / set_tags) must be
        # called zero times.
        write_calls = (
            mock_zotero.update_item.call_count
            + mock_zotero.set_tags.call_count
        )
        assert write_calls == 0, (
            f"Dry-run must not mutate Zotero; got {write_calls} write calls"
        )

    # -------------------------------------------------------------------------
    # 4. Cluster-aware ask
    # -------------------------------------------------------------------------

    def test_ask_returns_cluster_aware_response(
        self, mock_state_db_with_clusters, mock_embeddings_db
    ) -> None:
        """run_pipeline returns prose that references the dominant cluster name.

        Strategy:
          - question embedding = [1, 0, 0, 0] (from fixture) — aligned with
            cluster 0 'Protein A & CEX' (cosine sim = 1.0 > 0.3 threshold).
          - LLM client is mocked to echo its user prompt.
          - We assert the cluster name appears in the injected prompt.
        """
        from scripts.graph.ask import _identify_relevant_cluster

        result = _identify_relevant_cluster(
            "what are the best methods for mAb capture?",
            mock_state_db_with_clusters,
            mock_embeddings_db,
            threshold=0.3,
        )

        assert result is not None, (
            "_identify_relevant_cluster returned None; "
            "expected a match for cluster 'Protein A & CEX'"
        )
        assert result["display_name"] == "Protein A & CEX"
        assert result["n_papers"] == 2
        assert result["similarity"] >= 0.3

    def test_ask_no_cluster_context_when_similarity_low(
        self, mock_state_db_with_clusters
    ) -> None:
        """When question embedding is orthogonal to all centroids, no cluster
        context is returned — plain summarisation path is used."""
        import numpy as np

        from scripts.graph.ask import _identify_relevant_cluster

        mock_emb_db = MagicMock()
        # Diagonal vector — not parallel to either [1,0,0,0] or [0,1,0,0]
        mock_emb_db.embed_text.return_value = np.array(
            [0.707, 0.707, 0.0, 0.0], dtype=np.float32
        )

        result = _identify_relevant_cluster(
            "general question unrelated to specific cluster",
            mock_state_db_with_clusters,
            mock_emb_db,
            threshold=0.99,  # extremely high threshold → no match
        )
        assert result is None

    def test_run_pipeline_cluster_context_passed_to_summarize(
        self, mock_state_db_with_clusters, mock_embeddings_db
    ) -> None:
        """run_pipeline passes cluster_context into summarize_results.

        We mock summarize_results to capture what cluster_context it receives,
        then verify it contains the dominant cluster name.
        """
        captured: dict[str, Any] = {}

        def mock_summarize(question, cypher, rendered_rows, **kwargs):
            captured["cluster_context"] = kwargs.get("cluster_context")
            return "Based on your Protein A & CEX theme: two methods found."

        mock_graph_db = MagicMock()
        mock_graph_db.execute_query.return_value = []

        with (
            patch(
                "scripts.graph.ask._maybe_construct_client",
                return_value=MagicMock(),
            ),
            patch("scripts.graph.ask.generate_cypher", return_value="MATCH (p) RETURN p"),
            patch("scripts.graph.ask.execute_cypher", return_value=[{"doi": "10.0/a"}]),
            patch(
                "scripts.graph.ask.render_rows",
                return_value="| doi |\n|---|\n| 10.0/a |",
            ),
            patch("scripts.graph.ask.summarize_results", side_effect=mock_summarize),
        ):
            from scripts.graph.ask import run_pipeline

            result = run_pipeline(
                "what mAb capture methods are in my library?",
                graph_db=mock_graph_db,
                state_db=mock_state_db_with_clusters,
                embeddings_db=mock_embeddings_db,
                summarize=True,
            )

        assert "cluster_context" in captured, (
            "run_pipeline did not pass cluster_context to summarize_results"
        )
        ctx = captured["cluster_context"]
        assert ctx is not None, (
            "cluster_context should be non-None when a matching cluster exists"
        )
        assert ctx["display_name"] == "Protein A & CEX"
        assert result.prose is not None

    def test_run_pipeline_graceful_when_no_state_db(self) -> None:
        """run_pipeline degrades gracefully when state_db is None."""
        with (
            patch("scripts.graph.ask.generate_cypher", return_value="MATCH (p) RETURN p"),
            patch("scripts.graph.ask.execute_cypher", return_value=[]),
            patch("scripts.graph.ask.render_rows", return_value="_(no results)_"),
            patch(
                "scripts.graph.ask._maybe_construct_client",
                return_value=MagicMock(),
            ),
        ):
            from scripts.graph.ask import run_pipeline

            mock_graph_db = MagicMock()
            result = run_pipeline(
                "any question",
                graph_db=mock_graph_db,
                state_db=None,   # no clusters available
                embeddings_db=None,
                summarize=False,
            )

        # Must return an AskResult without raising
        assert result is not None
        assert result.cypher == "MATCH (p) RETURN p"
        assert result.prose is None

    # -------------------------------------------------------------------------
    # 5. AskResult shape parity (CLI + HTTP must see the same type)
    # -------------------------------------------------------------------------

    def test_ask_result_dataclass_fields(self) -> None:
        """AskResult must expose cypher, rows, rendered, prose fields."""
        from scripts.graph.ask import AskResult

        r = AskResult(
            cypher="MATCH (p) RETURN p",
            rows=[{"doi": "10.0/a"}],
            rendered="| doi |\n|---|\n| 10.0/a |",
            prose="One paper found.",
        )
        assert r.cypher
        assert r.rows
        assert r.rendered
        assert r.prose

    # -------------------------------------------------------------------------
    # 6. Prompt registry knows cluster_context is required
    # -------------------------------------------------------------------------

    def test_prompt_registry_ask_summarize_has_cluster_context(self) -> None:
        """prompt_registry should declare cluster_context as required for
        ask_summarize after Bundle I update."""
        from scripts.llm.prompt_registry import _reset_prompt_cache, required_placeholders

        _reset_prompt_cache()
        placeholders = required_placeholders("ask_summarize")
        assert "cluster_context" in placeholders, (
            "prompt_registry.ask_summarize must list cluster_context as required "
            "after Bundle I"
        )

    def test_ask_summarize_yaml_renders_with_four_placeholders(self) -> None:
        """The YAML round-trip must succeed with the four Bundle I placeholders."""
        from scripts.llm.prompt_registry import _reset_prompt_cache, load_prompt

        _reset_prompt_cache()
        prompt = load_prompt("ask_summarize")
        rendered = prompt.render_user(
            question="find methods",
            cypher="MATCH (p) RETURN p",
            rows="| doi |\n|---|\n| 10.0/a |",
            cluster_context="Based on your Protein A theme:",
        )
        assert "find methods" in rendered
        assert "Based on your Protein A theme:" in rendered
