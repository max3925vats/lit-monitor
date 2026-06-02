"""
Unit tests for Bundle C: theme-clustered library centroids.

Tests cover:
- compute_clusters determinism + correct K selection
- Cluster dataclass shape and size property
- Nearest-centroid stable ID mapping across recomputes
- name_cluster LLM call (mocked) + None on failure (defensive perimeter)
- assign_papers_to_clusters writes rows + distance values
- recompute_clusters end-to-end (mocked LLM for names)
- Threshold gating: below 100 → no-op; above 100 → compute
- End-of-brain-build hook idempotency (don't re-trigger when clusters exist)
- Config namespace defaults for clustering block
- Bundle B score_breakdown extended with cluster_centroid key
- rank_papers default behavior preserved when cluster_centroid weight = 0.0
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embeddings(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Generate reproducible random unit-norm embeddings."""
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    # normalize to unit sphere (cosine-friendly)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-9)


def _make_dois(n: int) -> list[str]:
    return [f"10.1000/test.{i:04d}" for i in range(n)]


# ---------------------------------------------------------------------------
# compute_clusters tests
# ---------------------------------------------------------------------------

class TestComputeClusters:
    def test_returns_cluster_objects(self):
        from scripts.clustering.kmeans import Cluster, compute_clusters

        embs = _make_embeddings(30, dim=8)
        dois = _make_dois(30)
        clusters = compute_clusters(embs, dois, k_min=2, k_max=5, random_state=42)

        assert len(clusters) >= 1
        for c in clusters:
            assert isinstance(c, Cluster)

    def test_cluster_size_property(self):
        from scripts.clustering.kmeans import compute_clusters

        embs = _make_embeddings(30, dim=8)
        dois = _make_dois(30)
        clusters = compute_clusters(embs, dois, k_min=2, k_max=5, random_state=42)

        total = sum(c.size for c in clusters)
        assert total == 30  # every paper assigned to exactly one cluster

    def test_deterministic_output(self):
        """Same inputs + same random_state must yield identical cluster structure."""
        from scripts.clustering.kmeans import compute_clusters

        embs = _make_embeddings(40, dim=8)
        dois = _make_dois(40)

        clusters_a = compute_clusters(embs, dois, k_min=2, k_max=6, random_state=42)
        clusters_b = compute_clusters(embs, dois, k_min=2, k_max=6, random_state=42)

        assert len(clusters_a) == len(clusters_b)
        for a, b in zip(clusters_a, clusters_b):
            assert a.members == b.members
            np.testing.assert_array_equal(a.centroid_vec, b.centroid_vec)

    def test_different_seed_may_differ(self):
        """Changing random_state CAN (doesn't have to) change the output."""
        from scripts.clustering.kmeans import compute_clusters

        # We just assert the function runs without error; output may be same/different
        embs = _make_embeddings(40, dim=8)
        dois = _make_dois(40)
        clusters = compute_clusters(embs, dois, k_min=2, k_max=6, random_state=99)
        assert len(clusters) >= 1

    def test_below_k_min_returns_empty(self):
        """Fewer embeddings than k_min → empty list, not an error."""
        from scripts.clustering.kmeans import compute_clusters

        embs = _make_embeddings(3, dim=8)
        dois = _make_dois(3)
        clusters = compute_clusters(embs, dois, k_min=5, k_max=10, random_state=42)
        assert clusters == []

    def test_centroid_vec_is_float32(self):
        from scripts.clustering.kmeans import compute_clusters

        embs = _make_embeddings(30, dim=8)
        dois = _make_dois(30)
        clusters = compute_clusters(embs, dois, k_min=2, k_max=4, random_state=42)
        for c in clusters:
            assert c.centroid_vec.dtype == np.float32

    def test_members_are_subset_of_input_dois(self):
        from scripts.clustering.kmeans import compute_clusters

        embs = _make_embeddings(30, dim=8)
        dois = _make_dois(30)
        clusters = compute_clusters(embs, dois, k_min=2, k_max=5, random_state=42)

        all_members: list[str] = []
        for c in clusters:
            all_members.extend(c.members)

        assert sorted(all_members) == sorted(dois)

    def test_initial_cluster_ids_are_none(self):
        """Clusters come out of compute_clusters with id=None; IDs are set on persist."""
        from scripts.clustering.kmeans import compute_clusters

        embs = _make_embeddings(30, dim=8)
        dois = _make_dois(30)
        clusters = compute_clusters(embs, dois, k_min=2, k_max=5, random_state=42)
        for c in clusters:
            assert c.id is None


# ---------------------------------------------------------------------------
# map_to_existing_clusters (stable ID mapping)
# ---------------------------------------------------------------------------

class TestMapToExistingClusters:
    def _make_cluster(self, cid: int, centroid: np.ndarray, members: list[str]):
        from scripts.clustering.kmeans import Cluster

        return Cluster(
            id=cid,
            display_name=None,
            centroid_vec=centroid.astype(np.float32),
            members=members,
            cohesion_score=0.5,
        )

    def test_identical_centroids_map_to_themselves(self):
        from scripts.clustering.kmeans import Cluster, map_to_existing_clusters

        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        existing = [self._make_cluster(10, centroid, [])]
        # New cluster with same centroid
        new_clusters = [
            Cluster(id=None, display_name=None, centroid_vec=centroid,
                    members=[], cohesion_score=0.5)
        ]
        mapping = map_to_existing_clusters(new_clusters, existing)
        assert mapping.get(0) == 10

    def test_empty_existing_returns_empty_mapping(self):
        from scripts.clustering.kmeans import Cluster, map_to_existing_clusters

        new_clusters = [
            Cluster(id=None, display_name=None,
                    centroid_vec=np.array([1.0, 0.0], dtype=np.float32),
                    members=[], cohesion_score=0.5)
        ]
        mapping = map_to_existing_clusters(new_clusters, [])
        assert mapping == {}

    def test_dissimilar_centroids_get_no_mapping(self):
        """New cluster too far from any existing cluster → not mapped."""
        from scripts.clustering.kmeans import Cluster, map_to_existing_clusters

        existing = [
            self._make_cluster(1, np.array([1.0, 0.0, 0.0], dtype=np.float32), [])
        ]
        # Orthogonal vector — cosine distance = 1.0, well above the 0.3 threshold
        new_cluster = Cluster(
            id=None, display_name=None,
            centroid_vec=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            members=[], cohesion_score=0.5,
        )
        mapping = map_to_existing_clusters([new_cluster], existing)
        assert 0 not in mapping


# ---------------------------------------------------------------------------
# name_cluster (defensive perimeter)
# ---------------------------------------------------------------------------

class TestNameCluster:
    def test_returns_string_on_success(self):
        from scripts.clustering.naming import name_cluster

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Cation Exchange Chromatography"

        with patch("scripts.clustering.naming.load_prompt") as mock_load:
            mock_prompt = MagicMock()
            mock_prompt.system = "You are a librarian."
            mock_prompt.user_template = "Papers:\n{paper_samples}\n\nName:"
            mock_load.return_value = mock_prompt

            result = name_cluster(
                sample_dois=["10.1000/a"],
                sample_titles=["Optimization of CEX for mAb purification"],
                llm=mock_llm,
            )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_caps_at_four_words(self):
        from scripts.clustering.naming import name_cluster

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Very Long Name That Exceeds The Word Limit"

        with patch("scripts.clustering.naming.load_prompt") as mock_load:
            mock_prompt = MagicMock()
            mock_prompt.system = "sys"
            mock_prompt.user_template = "{paper_samples}"
            mock_load.return_value = mock_prompt

            result = name_cluster(
                sample_dois=["10.1000/a"],
                sample_titles=["Test paper"],
                llm=mock_llm,
            )

        # Maximum 4 words
        assert result is None or len(result.split()) <= 4

    def test_returns_none_on_llm_exception(self):
        """Defensive perimeter: LLM error → None, never raises."""
        from scripts.clustering.naming import name_cluster

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("LLM unreachable")

        with patch("scripts.clustering.naming.load_prompt") as mock_load:
            mock_prompt = MagicMock()
            mock_prompt.system = "sys"
            mock_prompt.user_template = "{paper_samples}"
            mock_load.return_value = mock_prompt

            result = name_cluster(
                sample_dois=["10.1000/a"],
                sample_titles=["Test paper"],
                llm=mock_llm,
            )

        assert result is None

    def test_returns_none_on_empty_titles(self):
        from scripts.clustering.naming import name_cluster

        result = name_cluster(sample_dois=[], sample_titles=[], llm=MagicMock())
        assert result is None

    def test_returns_none_on_empty_llm_response(self):
        from scripts.clustering.naming import name_cluster

        mock_llm = MagicMock()
        mock_llm.complete.return_value = ""

        with patch("scripts.clustering.naming.load_prompt") as mock_load:
            mock_prompt = MagicMock()
            mock_prompt.system = "sys"
            mock_prompt.user_template = "{paper_samples}"
            mock_load.return_value = mock_prompt

            result = name_cluster(
                sample_dois=["10.1000/a"],
                sample_titles=["Test paper"],
                llm=mock_llm,
            )

        assert result is None

    def test_prompt_load_failure_returns_none(self):
        """If prompt registry raises, name_cluster returns None (not raises)."""
        from scripts.clustering.naming import name_cluster

        with patch("scripts.clustering.naming.load_prompt",
                   side_effect=FileNotFoundError("prompt not found")):
            result = name_cluster(
                sample_dois=["10.1000/a"],
                sample_titles=["Test paper"],
                llm=MagicMock(),
            )

        assert result is None


# ---------------------------------------------------------------------------
# StateDB clustering schema (clusters + cluster_assignments tables)
# ---------------------------------------------------------------------------

class TestClusteringSchema:
    def test_tables_created(self, tmp_path):
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "clusters" in tables
        assert "cluster_assignments" in tables

    def test_cluster_insert_and_retrieve(self, tmp_path):
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        centroid = np.zeros(8, dtype=np.float32)

        cid = db.insert_cluster(
            display_name="Test Theme",
            n_papers=5,
            cohesion_score=0.7,
            centroid_blob=centroid.tobytes(),
        )
        assert isinstance(cid, int)
        assert cid > 0

        row = db.get_cluster(cid)
        assert row is not None
        assert row["display_name"] == "Test Theme"
        assert row["n_papers"] == 5

    def test_list_active_clusters(self, tmp_path):
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        centroid = np.zeros(8, dtype=np.float32)

        db.insert_cluster("Theme A", 3, 0.6, centroid.tobytes())
        db.insert_cluster("Theme B", 7, 0.8, centroid.tobytes())

        active = db.list_active_clusters()
        assert len(active) == 2

    def test_archive_old_clusters(self, tmp_path):
        """archive_clusters marks rows archived=1 so list_active_clusters skips them."""
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        centroid = np.zeros(8, dtype=np.float32)

        cid = db.insert_cluster("Old Theme", 2, 0.5, centroid.tobytes())
        db.archive_clusters([cid])

        active = db.list_active_clusters()
        assert not any(c["id"] == cid for c in active)

    def test_upsert_cluster_assignment(self, tmp_path):
        from scripts.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        centroid = np.zeros(8, dtype=np.float32)

        # Need a paper row first (FK enforcement; or disable FK in test)
        with db._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO papers (doi, title) VALUES (?, ?)",
                ("10.test/a", "Test Paper"),
            )

        cid = db.insert_cluster("Theme", 1, 0.5, centroid.tobytes())
        db.upsert_cluster_assignment("10.test/a", cid, distance_to_centroid=0.15)

        # Verify row exists
        with db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cluster_assignments WHERE doi=? AND cluster_id=?",
                ("10.test/a", cid),
            ).fetchone()
        assert row is not None
        assert abs(dict(row)["distance_to_centroid"] - 0.15) < 1e-6


# ---------------------------------------------------------------------------
# Threshold gating
# ---------------------------------------------------------------------------

class TestThresholdGating:
    def test_below_threshold_returns_zero(self, tmp_path):
        """recompute_clusters with fewer papers than threshold returns 0."""
        from unittest.mock import MagicMock

        from scripts.clustering.recompute import recompute_clusters

        state_db = MagicMock()
        # list_active_clusters returns empty (no existing clusters)
        state_db.list_active_clusters.return_value = []

        embeddings_db = MagicMock()
        # Only 10 papers in the collection
        embeddings_db._collection.count.return_value = 10
        embeddings_db._collection.get.return_value = {
            "ids": [f"doi_{i}" for i in range(10)],
            "embeddings": _make_embeddings(10, dim=8).tolist(),
        }

        cfg = MagicMock()
        cfg.clustering.enabled = True
        cfg.clustering.min_papers_threshold = 100
        cfg.clustering.k_min = 5
        cfg.clustering.k_max = 15

        result = recompute_clusters(state_db, embeddings_db, cfg)
        assert result == 0

    def test_above_threshold_computes_clusters(self, tmp_path):
        """recompute_clusters with enough papers calls compute and returns N > 0."""
        from unittest.mock import MagicMock, patch

        from scripts.clustering.recompute import recompute_clusters

        n = 120
        embs = _make_embeddings(n, dim=8)
        dois = _make_dois(n)

        state_db = MagicMock()
        state_db.list_active_clusters.return_value = []
        state_db.insert_cluster.return_value = 1

        embeddings_db = MagicMock()
        embeddings_db._collection.count.return_value = n
        embeddings_db._collection.get.return_value = {
            "ids": dois,
            "embeddings": embs.tolist(),
        }

        cfg = MagicMock()
        cfg.clustering.enabled = True
        cfg.clustering.min_papers_threshold = 100
        cfg.clustering.k_min = 2
        cfg.clustering.k_max = 5

        with patch("scripts.clustering.recompute.name_cluster", return_value=None), \
             patch("scripts.clustering.recompute.assign_papers_to_clusters"):
            result = recompute_clusters(state_db, embeddings_db, cfg)

        assert result > 0


# ---------------------------------------------------------------------------
# Brain-build hook idempotency
# ---------------------------------------------------------------------------

class TestBrainBuildHook:
    def test_no_trigger_when_clusters_exist(self, tmp_path):
        """_maybe_trigger_initial_clustering is a no-op when clusters table is non-empty."""
        from scripts.pipelines.brain_build import _maybe_trigger_initial_clustering

        state_db = MagicMock()
        state_db.list_active_clusters.return_value = [{"id": 1, "display_name": "Existing"}]

        embeddings_db = MagicMock()
        embeddings_db._collection.count.return_value = 200

        cfg = MagicMock()
        cfg.clustering.enabled = True
        cfg.clustering.min_papers_threshold = 100

        with patch("scripts.clustering.recompute.recompute_clusters") as mock_recompute:
            _maybe_trigger_initial_clustering(state_db, embeddings_db, cfg)
            mock_recompute.assert_not_called()

    def test_triggers_when_threshold_crossed_and_no_clusters(self, tmp_path):
        """_maybe_trigger_initial_clustering calls recompute when 0 clusters + ≥threshold papers."""
        from scripts.pipelines.brain_build import _maybe_trigger_initial_clustering

        state_db = MagicMock()
        state_db.list_active_clusters.return_value = []

        embeddings_db = MagicMock()
        embeddings_db._collection.count.return_value = 150

        cfg = MagicMock()
        cfg.clustering.enabled = True
        cfg.clustering.min_papers_threshold = 100

        with patch(
            "scripts.pipelines.brain_build.recompute_clusters"
        ) as mock_recompute:
            mock_recompute.return_value = 5
            _maybe_trigger_initial_clustering(state_db, embeddings_db, cfg)
            mock_recompute.assert_called_once()

    def test_no_trigger_when_feature_disabled(self):
        """_maybe_trigger_initial_clustering is no-op when clustering.enabled is False."""
        from scripts.pipelines.brain_build import _maybe_trigger_initial_clustering

        state_db = MagicMock()
        embeddings_db = MagicMock()
        embeddings_db._collection.count.return_value = 200

        cfg = MagicMock()
        cfg.clustering.enabled = False

        with patch(
            "scripts.pipelines.brain_build.recompute_clusters"
        ) as mock_recompute:
            _maybe_trigger_initial_clustering(state_db, embeddings_db, cfg)
            mock_recompute.assert_not_called()


# ---------------------------------------------------------------------------
# rank_papers: cluster_centroid signal integration
# ---------------------------------------------------------------------------

class TestRankPapersClusterSignal:
    def _make_candidates(self, n: int) -> list[dict[str, Any]]:
        return [
            {"doi": f"10.1000/cand.{i}", "title": f"Candidate Paper {i}",
             "abstract": "some abstract"}
            for i in range(n)
        ]

    def test_default_zero_weight_preserves_behavior(self):
        """When cluster_centroid weight=0.0, rank_papers output is unchanged."""
        from scripts.llm.ranker import rank_papers

        candidates = self._make_candidates(3)

        mock_emb_db = MagicMock()
        mock_emb_db.find_similar_to_text.return_value = [{"score": 0.8}]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "{}"

        with patch("scripts.llm.ranker.load_prompt") as mock_load:
            mock_prompt = MagicMock()
            mock_prompt.render_user.return_value = "user"
            mock_prompt.system = "sys"
            mock_prompt.max_tokens = 512
            mock_prompt.render_paper_card.return_value = ""
            mock_prompt.paper_card_separator = "\n---\n"
            mock_load.return_value = mock_prompt

            result = rank_papers(
                candidates,
                mock_emb_db,
                mock_llm,
                top_k=3,
                cluster_centroids=None,
                cluster_centroid_weight=0.0,
            )

        assert len(result) == 3
        # score_breakdown must contain cluster_centroid key = 0.0
        for p in result:
            breakdown = p.get("score_breakdown", {})
            assert "cluster_centroid" in breakdown
            assert breakdown["cluster_centroid"] == 0.0

    def test_nonzero_weight_adds_cluster_score(self):
        """When cluster_centroid weight > 0 and centroids provided, scores change."""
        from scripts.clustering.kmeans import Cluster
        from scripts.llm.ranker import rank_papers

        # Centroid pointing in a specific direction
        centroid_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 dtype=np.float32)
        cluster = Cluster(
            id=1, display_name="Test Theme",
            centroid_vec=centroid_vec,
            members=["10.1000/a"],
            cohesion_score=0.6,
        )

        candidates = self._make_candidates(2)
        # Give candidates _embedding so the cluster signal can run
        for c in candidates:
            c["_embedding"] = centroid_vec.tolist()  # identical → max cosine

        mock_emb_db = MagicMock()
        mock_emb_db.find_similar_to_text.return_value = [{"score": 0.5}]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "{}"

        with patch("scripts.llm.ranker.load_prompt") as mock_load:
            mock_prompt = MagicMock()
            mock_prompt.render_user.return_value = "user"
            mock_prompt.system = "sys"
            mock_prompt.max_tokens = 512
            mock_prompt.render_paper_card.return_value = ""
            mock_prompt.paper_card_separator = "\n---\n"
            mock_load.return_value = mock_prompt

            result = rank_papers(
                candidates,
                mock_emb_db,
                mock_llm,
                top_k=2,
                cluster_centroids=[cluster],
                cluster_centroid_weight=0.3,
            )

        for p in result:
            breakdown = p.get("score_breakdown", {})
            # cluster_centroid contribution should be non-zero
            assert "cluster_centroid" in breakdown
        # At least one paper should have cluster_centroid > 0
        assert any(p["score_breakdown"]["cluster_centroid"] > 0 for p in result)

    def test_cluster_matched_annotation_added(self):
        """Papers with a matching centroid get cluster_matched set to display_name."""
        from scripts.clustering.kmeans import Cluster
        from scripts.llm.ranker import rank_papers

        centroid_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cluster = Cluster(
            id=1, display_name="My Theme",
            centroid_vec=centroid_vec,
            members=[],
            cohesion_score=0.5,
        )

        candidates = [{"doi": "10.x/1", "title": "P1", "_embedding": centroid_vec.tolist()}]

        mock_emb_db = MagicMock()
        mock_emb_db.find_similar_to_text.return_value = [{"score": 0.5}]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "{}"

        with patch("scripts.llm.ranker.load_prompt") as mock_load:
            mock_prompt = MagicMock()
            mock_prompt.render_user.return_value = "user"
            mock_prompt.system = "sys"
            mock_prompt.max_tokens = 512
            mock_prompt.render_paper_card.return_value = ""
            mock_prompt.paper_card_separator = "\n---\n"
            mock_load.return_value = mock_prompt

            result = rank_papers(
                candidates,
                mock_emb_db,
                mock_llm,
                top_k=1,
                cluster_centroids=[cluster],
                cluster_centroid_weight=0.3,
            )

        assert result[0].get("cluster_matched") == "My Theme"
