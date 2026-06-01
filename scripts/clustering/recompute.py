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

from scripts.clustering.assign import assign_papers_to_clusters
from scripts.clustering.kmeans import Cluster, compute_clusters, map_to_existing_clusters
from scripts.clustering.naming import name_cluster

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

    # Load existing clusters for stable ID mapping
    existing_raw = state_db.list_active_clusters()
    existing_clusters: list[Cluster] = []
    for row in existing_raw:
        blob = row.get("centroid_blob")
        if blob:
            try:
                vec = np.frombuffer(blob, dtype=np.float32).copy()
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

    # Persist: archive old clusters, insert new ones.
    # Do this in a single logical transaction via sequential state_db calls.
    old_ids = [c.id for c in existing_clusters if c.id is not None]
    if old_ids:
        state_db.archive_clusters(old_ids)

    created = 0
    for i, cluster in enumerate(new_clusters):
        inherited_id = id_mapping.get(i)
        # Display name: use LLM result or fallback placeholder
        # The fallback uses the inherited_id or a temp marker resolved after insert
        display = cluster.display_name  # may be None

        if inherited_id is not None:
            # Reactivate the existing row with updated metadata
            state_db.upsert_cluster_by_id(
                cluster_id=inherited_id,
                display_name=display,
                n_papers=cluster.size,
                cohesion_score=cluster.cohesion_score,
                centroid_blob=cluster.centroid_vec.tobytes(),
            )
            cluster.id = inherited_id
        else:
            new_id = state_db.insert_cluster(
                display_name=display,
                n_papers=cluster.size,
                cohesion_score=cluster.cohesion_score,
                centroid_blob=cluster.centroid_vec.tobytes(),
            )
            cluster.id = new_id

        # Fill in fallback display name using now-known id
        if cluster.display_name is None:
            fallback = f"Cluster {cluster.id}"
            state_db.update_cluster_display_name(cluster.id, fallback)
            cluster.display_name = fallback

        created += 1

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
