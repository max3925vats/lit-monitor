"""
Bundle C: assign library papers to their nearest cluster centroid.

Writes cluster_assignments rows with distance-to-centroid.
Called after compute_clusters + persist, and can be called standalone
via `lit-monitor cluster assign` to re-assign without recomputing clusters.
"""
from __future__ import annotations

import logging

import numpy as np

from lit_monitor.clustering.kmeans import Cluster

logger = logging.getLogger(__name__)


def assign_papers_to_clusters(
    state_db,
    embeddings_db,
    clusters: list[Cluster],
) -> int:
    """Assign every embedded library paper to its nearest cluster centroid.

    For each paper in ChromaDB, compute cosine distance to all centroids
    and write the nearest assignment into cluster_assignments.

    Args:
        state_db: StateDB instance.
        embeddings_db: EmbeddingsDB instance (used to retrieve all embeddings).
        clusters: List of persisted Cluster objects (must have integer ids).

    Returns:
        Number of assignments written.
    """
    if not clusters:
        return 0

    # Fetch all library embeddings in one ChromaDB get() call
    collection_count = embeddings_db._collection.count()
    if collection_count == 0:
        logger.info("C: no embeddings in collection — skipping assignment.")
        return 0

    raw = embeddings_db._collection.get(
        limit=collection_count,
        include=["embeddings"],
    )
    ids: list[str] = raw.get("ids", [])
    raw_embs = raw.get("embeddings") or []

    if not ids or not raw_embs:
        logger.info("C: collection returned no ids/embeddings.")
        return 0

    embeddings = np.array(raw_embs, dtype=np.float32)  # (N, dim)

    # Build centroid matrix — (K, dim)
    centroid_matrix = np.array(
        [c.centroid_vec for c in clusters], dtype=np.float32
    )
    centroid_ids = [c.id for c in clusters]

    # Cosine distance: 1 - (emb @ centroids^T) / (|emb| * |centroid|)
    # Normalize both
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    cen_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-9
    emb_unit = embeddings / emb_norms           # (N, dim)
    cen_unit = centroid_matrix / cen_norms      # (K, dim)

    # Cosine similarities (N, K); distance = 1 - similarity
    similarities = emb_unit @ cen_unit.T        # (N, K)
    distances = 1.0 - similarities              # (N, K)

    # Nearest centroid index per paper
    nearest_idxs = np.argmin(distances, axis=1)  # (N,)
    nearest_dists = distances[np.arange(len(ids)), nearest_idxs]  # (N,)

    # P3.5: build the full assignment batch first, then write it in a SINGLE
    # transaction via state_db.replace_cluster_assignments(). Previously each
    # assignment was its own upsert call (its own transaction), so a failure
    # partway through left the table half-updated. Batching all-or-nothing
    # means a mid-write crash rolls back the whole batch.
    batch: list[tuple[str, int, float | None]] = []
    for i, doi in enumerate(ids):
        cid = centroid_ids[nearest_idxs[i]]
        dist = float(nearest_dists[i])
        if cid is None:
            continue
        batch.append((doi, cid, dist))

    count = state_db.replace_cluster_assignments(batch)
    logger.info("C: wrote %d cluster assignments.", count)
    return count
