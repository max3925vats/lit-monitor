"""
Bundle C: k-means clustering of the paper library into thematic centroids.

Uses scikit-learn KMeans + silhouette score for automatic K selection.
All randomness is seeded at random_state=42 for reproducibility.

Key invariant: Cluster.id is None until persisted to state_db.
                map_to_existing_clusters preserves IDs across recomputes so
                user-applied tags (e.g. lm/<theme>) don't drift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """A single thematic cluster from k-means over the library embeddings.

    id is None until the cluster is persisted to the clusters table.
    All distance computations use cosine similarity.
    """
    id: int | None          # set by state_db.insert_cluster; None before persist
    display_name: str | None
    centroid_vec: np.ndarray  # float32, shape (dim,)
    members: list[str]        # list of DOIs assigned to this cluster
    cohesion_score: float     # mean silhouette score (global or per-cluster)

    @property
    def size(self) -> int:
        """Number of papers in this cluster."""
        return len(self.members)


def compute_clusters(
    embeddings: np.ndarray,
    paper_dois: list[str],
    *,
    k_min: int = 5,
    k_max: int = 15,
    random_state: int = 42,
) -> list[Cluster]:
    """Cluster library embeddings into k thematic groups using k-means.

    K is chosen automatically in [k_min, k_max] by maximising silhouette score.
    Deterministic: same inputs + random_state=42 → same output.

    Args:
        embeddings: (N, dim) float32 array of paper embeddings.
        paper_dois: Ordered list of DOIs matching row index of embeddings.
        k_min: Minimum number of clusters to try.
        k_max: Maximum number of clusters to try (capped at N-1).
        random_state: Seed for KMeans. Fixed at 42 in production for stability.

    Returns:
        List of Cluster objects. Empty list when N < k_min (graceful no-op).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n < k_min:
        logger.info(
            "C: %d papers < k_min=%d — clustering skipped (graceful no-op).",
            n, k_min,
        )
        return []

    k_max_actual = min(k_max, n - 1)
    if k_max_actual < k_min:
        logger.info("C: k_max_actual=%d < k_min=%d — skipping.", k_max_actual, k_min)
        return []

    best_k = k_min
    best_silhouette = -1.0
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None

    for k in range(k_min, k_max_actual + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        # Silhouette requires at least 2 unique labels and ≥2 samples per cluster
        unique = set(labels)
        if len(unique) < 2:
            continue
        try:
            score = float(silhouette_score(embeddings, labels))
        except ValueError:
            # Can occur when a cluster has only 1 sample
            continue
        if score > best_silhouette:
            best_k = k
            best_silhouette = score
            best_labels = labels
            best_centers = km.cluster_centers_

    if best_labels is None or best_centers is None:
        logger.warning("C: silhouette selection found no valid k in [%d, %d].", k_min, k_max_actual)
        return []

    logger.info("C: selected K=%d (silhouette=%.4f) for %d papers.", best_k, best_silhouette, n)

    clusters: list[Cluster] = []
    for cluster_idx in range(best_k):
        members = [
            paper_dois[i]
            for i in range(n)
            if best_labels[i] == cluster_idx
        ]
        # Per-cluster cohesion: mean silhouette of members in this cluster.
        # We use the global silhouette as a proxy here; it's informative
        # and avoids a second O(N^2) silhouette call.
        clusters.append(Cluster(
            id=None,
            display_name=None,
            centroid_vec=best_centers[cluster_idx].astype(np.float32),
            members=members,
            cohesion_score=float(best_silhouette),
        ))

    return clusters


def map_to_existing_clusters(
    new_clusters: list[Cluster],
    existing_clusters: list[Cluster],
    *,
    cosine_distance_threshold: float = 0.3,
) -> dict[int, int]:
    """Stable ID mapping for incremental recomputes.

    For each new cluster, find the nearest existing cluster centroid.
    If the cosine distance is below threshold, the new cluster inherits
    the existing cluster's ID — so user-applied tags don't drift.

    Args:
        new_clusters: Freshly computed clusters (all with id=None).
        existing_clusters: Previously persisted clusters (with integer ids).
        cosine_distance_threshold: Max distance for an ID to be inherited.
            Default 0.3 (cosine distance, not similarity). At 0.3 the clusters
            are still clearly in the same neighbourhood.

    Returns:
        Dict mapping {new_cluster_index → existing_cluster_id} for clusters
        that are close enough to an existing one. Unmapped new clusters get
        fresh AUTOINCREMENT IDs at insert time.
    """
    if not existing_clusters or not new_clusters:
        return {}

    existing_centers = np.array(
        [c.centroid_vec for c in existing_clusters], dtype=np.float32
    )  # (E, dim)
    existing_ids = [c.id for c in existing_clusters]

    mapping: dict[int, int] = {}
    used_existing: set[int] = set()  # prevent many-to-one mapping

    for new_idx, nc in enumerate(new_clusters):
        nc_vec = nc.centroid_vec  # (dim,)
        nc_norm = float(np.linalg.norm(nc_vec))
        if nc_norm < 1e-9:
            continue

        # Cosine distance = 1 - cosine_similarity
        norms = np.linalg.norm(existing_centers, axis=1) + 1e-9  # (E,)
        dots = existing_centers @ nc_vec                          # (E,)
        dists = 1.0 - dots / (norms * nc_norm)                   # (E,)

        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])

        if nearest_dist < cosine_distance_threshold and nearest_idx not in used_existing:
            eid = existing_ids[nearest_idx]
            if eid is not None:
                mapping[new_idx] = eid
                used_existing.add(nearest_idx)

    return mapping
