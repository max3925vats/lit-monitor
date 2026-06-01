"""
Bundle C (v0.9 engine plan): theme-clustered library centroids.

Replaces "max cosine to any single library paper" with
"max cosine to nearest theme centroid, weighted by cluster size."

Public surface:
  compute_clusters  — k-means + silhouette K selection
  name_cluster      — single LLM call for human-readable label (defensive)
  assign_papers_to_clusters — write cluster_assignments rows
  recompute_clusters — full pipeline (fetch → cluster → name → assign → persist)
  push_tags_to_zotero      — additive tag write-back (opt-in)
  push_collections_to_zotero — additive sub-collection write-back (opt-in)
"""
from __future__ import annotations

from scripts.clustering.kmeans import Cluster, compute_clusters, map_to_existing_clusters
from scripts.clustering.naming import name_cluster

__all__ = [
    "Cluster",
    "compute_clusters",
    "map_to_existing_clusters",
    "name_cluster",
]
