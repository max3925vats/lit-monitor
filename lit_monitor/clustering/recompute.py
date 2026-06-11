"""
Bundle C: full clustering pipeline — fetch → cluster → name → assign → persist.

recompute_clusters() is the main entry point. It:
1. Fetches all library embeddings from ChromaDB.
2. Checks the min_papers_threshold gate.
3. Runs k-means (deterministic, random_state=42).
4. Maps new clusters to existing cluster IDs (stable IDs for user-applied tags).
5. Calls name_cluster() for each cluster (defensive perimeter).
6. Persists clusters + assignments atomically.
7. Returns the number of clusters created.
"""
from __future__ import annotations

import logging

import numpy as np

from lit_monitor.clustering.assign import assign_papers_to_clusters
from lit_monitor.clustering.kmeans import Cluster, compute_clusters, map_to_existing_clusters
from lit_monitor.clustering.naming import name_cluster
from lit_monitor.core.state_db import NewClusterSpec

logger = logging.getLogger(__name__)


def recompute_clusters(
    state_db,
    embeddings_db,
    cfg,
    *,
    llm=None,
) -> int:
    """Full clustering pipeline.

    Args:
        state_db: StateDB instance.
        embeddings_db: EmbeddingsDB instance.
        cfg: Config object. Must have cfg.clustering block.
        llm: Optional LLM client passed through to name_cluster.

    Returns:
        Number of new cluster rows created. 0 when below threshold.
    """
    clustering_cfg = getattr(cfg, "clustering", None)
    if clustering_cfg is None or not getattr(clustering_cfg, "enabled", True):
        logger.info("C: clustering disabled in config — skipping recompute.")
        return 0

    threshold = int(getattr(clustering_cfg, "min_papers_threshold", 100))
    k_min = int(getattr(clustering_cfg, "k_min", 5))
    k_max = int(getattr(clustering_cfg, "k_max", 15))

    # Check library size via ChromaDB (authoritative for indexed papers)
    collection_count = embeddings_db._collection.count()
    if collection_count < threshold:
        logger.info(
            "C: %d papers < threshold=%d — clustering is a no-op.",
            collection_count, threshold,
        )
        return 0

    # Fetch all embeddings + DOIs from ChromaDB
    raw = embeddings_db._collection.get(
        limit=collection_count,
        include=["embeddings"],
    )
    ids: list[str] = raw.get("ids", [])
    raw_embs = raw.get("embeddings") or []

    if not ids or not raw_embs:
        logger.warning("C: ChromaDB returned empty get() despite count=%d.", collection_count)
        return 0

    embeddings = np.array(raw_embs, dtype=np.float32)

    # Run k-means (deterministic random_state=42)
    new_clusters = compute_clusters(
        embeddings, ids,
        k_min=k_min, k_max=k_max,
        random_state=42,
    )

    if not new_clusters:
        logger.info("C: compute_clusters returned empty list — not enough distinct papers.")
        return 0

    # Embedding dimensionality from the freshly-fetched library embeddings.
    # Every stored centroid must match this width; map_to_existing_clusters
    # stacks centroids into one (E, dim) array and does centroid @ new_vec, so a
    # wrong-width centroid would raise or silently corrupt the distance matrix.
    embedding_dim = int(embeddings.shape[1])

    # Load existing clusters for stable ID mapping
    existing_raw = state_db.list_active_clusters()
    existing_clusters: list[Cluster] = []
    for row in existing_raw:
        blob = row.get("centroid_blob")
        if blob:
            try:
                vec = np.frombuffer(blob, dtype=np.float32).copy()
                # Validate the centroid shape before it reaches the distance
                # math. A malformed/stale-dimension blob is skipped with a
                # warning (same warn-and-skip style as the parse failure below)
                # rather than poisoning map_to_existing_clusters.
                if vec.ndim != 1 or vec.shape[0] != embedding_dim:
                    logger.warning(
                        "C: skipping cluster %d — centroid blob shape %s does not "
                        "match embedding dim %d (malformed or stale-dimension blob).",
                        row["id"], vec.shape, embedding_dim,
                    )
                    continue
                existing_clusters.append(Cluster(
                    id=row["id"],
                    display_name=row.get("display_name"),
                    centroid_vec=vec,
                    members=[],
                    cohesion_score=row.get("cohesion_score") or 0.0,
                ))
            except Exception as exc:
                logger.warning("C: could not parse centroid blob for cluster %d: %s", row["id"], exc)

    # Map new clusters → existing IDs where centroids are still close
    id_mapping: dict[int, int] = map_to_existing_clusters(new_clusters, existing_clusters)

    # Name each cluster (defensive — never raises)
    for i, cluster in enumerate(new_clusters):
        titles = _sample_titles(state_db, cluster.members, max_titles=5)
        name = name_cluster(
            sample_dois=cluster.members[:5],
            sample_titles=titles,
            llm=llm,
        )
        cluster.display_name = name  # None when LLM unavailable → fallback later

    # Persist: archive old clusters + insert/upsert new ones ATOMICALLY.
    # B3: replace_clusters runs the archive AND every insert inside ONE SQLite
    # transaction.  Previously these were separate state_db calls (each its own
    # commit), so a crash between archive and the inserts left the library with
    # ZERO active clusters.  Now a crash mid-persist rolls the whole thing back:
    # either the old clusters survive or the new ones land fully — never zero.
    old_ids = [c.id for c in existing_clusters if c.id is not None]

    specs: list[NewClusterSpec] = [
        NewClusterSpec(
            inherited_id=id_mapping.get(i),
            display_name=cluster.display_name,  # may be None → fallback in txn
            n_papers=cluster.size,
            cohesion_score=cluster.cohesion_score,
            centroid_blob=cluster.centroid_vec.tobytes(),
        )
        for i, cluster in enumerate(new_clusters)
    ]

    assigned_ids = state_db.replace_clusters(old_ids, specs)

    # Mirror the assigned ids + resolved fallback names back onto the Cluster
    # objects so downstream assignment uses the real persisted ids/names.
    for cluster, cid in zip(new_clusters, assigned_ids):
        cluster.id = cid
        if cluster.display_name is None:
            cluster.display_name = f"Cluster {cid}"

    created = len(assigned_ids)
    logger.info("C: persisted %d clusters.", created)

    # Assign all library papers to their nearest centroid
    assign_papers_to_clusters(state_db, embeddings_db, new_clusters)

    return created


def _sample_titles(state_db, dois: list[str], max_titles: int = 5) -> list[str]:
    """Fetch paper titles for a sample of DOIs from state_db."""
    titles: list[str] = []
    for doi in dois[:max_titles]:
        try:
            row = state_db.get_paper(doi)
            if row and row.get("title"):
                titles.append(row["title"])
        except Exception:
            pass
    return titles
