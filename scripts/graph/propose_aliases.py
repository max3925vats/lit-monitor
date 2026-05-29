"""G11: fuzzy-only alias proposal helper.

Phase 1 is LLM-free. Surveys the live KuzuDB Entity vocab, clusters likely-
equivalent surfaces within type using rapidfuzz, and writes a YAML proposal
file the operator reviews and merges into entity_aliases.yaml by hand.

Phase 2 will add a --with-llm flag for cloud-Ollama validation of borderline
clusters.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz

from scripts.graph.aliases import load_aliases

logger = logging.getLogger(__name__)

_DEFAULT_MIN_RATIO = 80


def _cluster_surfaces(
    surfaces: list[str],
    min_ratio: int,
) -> list[list[str]]:
    """Greedy fuzzy clustering.

    For each surface, attach to the highest-scoring existing cluster
    if score ≥ min_ratio; else start a new cluster.
    Returns clusters as list of [surfaces], dropping singletons.

    Args:
        surfaces: Unique surface forms (sorted for determinism).
        min_ratio: Minimum fuzz.ratio score (0-100) to attach to a cluster.

    Returns:
        List of clusters with two or more surfaces.
    """
    clusters: list[list[str]] = []
    for surface in surfaces:
        best_cluster_idx = -1
        best_score = 0
        for i, cluster in enumerate(clusters):
            # Score against the first (representative) surface in the cluster.
            # Using the representative keeps the comparison stable across
            # additions and avoids drift as cluster membership grows.
            score = fuzz.ratio(surface, cluster[0])
            if score >= min_ratio and score > best_score:
                best_cluster_idx = i
                best_score = score
        if best_cluster_idx >= 0:
            clusters[best_cluster_idx].append(surface)
        else:
            clusters.append([surface])
    # Drop singletons — no pairing means no alias to propose.
    return [c for c in clusters if len(c) > 1]


def propose_aliases(
    graph_db: Any,
    *,
    min_ratio: int = _DEFAULT_MIN_RATIO,
) -> dict[str, dict[str, str]]:
    """Survey the graph and propose alias mappings for fuzzy-similar surfaces.

    Queries every Entity node from the live KuzuDB, groups surfaces by type,
    then applies greedy fuzzy clustering within each type group.

    Surfaces already present in entity_aliases.yaml (loaded via G2's
    load_aliases) are filtered out before clustering so they are never
    re-proposed.

    Args:
        graph_db: A live GraphDB instance (KuzuDB-backed).
        min_ratio: Minimum fuzz.ratio score for two surfaces to be placed
                   in the same cluster. Higher values = stricter matching.

    Returns:
        ``{type: {surface: canonical}}`` where each surface maps to the
        cluster representative (alphabetically first surface in the cluster,
        since input surfaces are unique and sorted).
    """
    existing_aliases = load_aliases()  # {type: {lower(surface): canonical}}

    # Survey all Entity nodes — group surfaces by type.
    conn = graph_db._conn
    res = conn.execute("MATCH (e:Entity) RETURN e.type, e.surface")
    by_type: dict[str, list[str]] = {}
    while res.has_next():
        row = res.get_next()
        type_, surface = str(row[0]), str(row[1])
        by_type.setdefault(type_, []).append(surface)

    proposals: dict[str, dict[str, str]] = {}
    for type_, surfaces in by_type.items():
        existing_set = {s.lower() for s in existing_aliases.get(type_, {})}
        # Deduplicate and exclude already-aliased surfaces.
        # Sort for determinism — greedy cluster order matters.
        unique = sorted(set(
            s for s in surfaces
            if s.lower() not in existing_set
        ))
        if len(unique) < 2:
            # Need at least two surfaces to form any cluster.
            continue

        clusters = _cluster_surfaces(unique, min_ratio)
        if not clusters:
            continue

        type_proposals: dict[str, str] = {}
        for cluster in clusters:
            # Representative: use Counter to pick most-frequent surface.
            # Since input is already deduplicated (unique), all counts are 1,
            # and most_common() returns the first element alphabetically via
            # Python's stable sort — deterministic.
            counts = Counter(cluster)
            representative = counts.most_common(1)[0][0]
            for surface in cluster:
                if surface == representative:
                    # The canonical is not an alias of itself — skip.
                    continue
                type_proposals[surface] = representative

        if type_proposals:
            proposals[type_] = type_proposals

    return proposals


def write_proposal_file(
    out_path: Path,
    proposals: dict[str, dict[str, str]],
) -> None:
    """Write the YAML proposal with a dated header comment.

    Args:
        out_path: Destination file path (created with parent dirs if needed).
        proposals: ``{type: {surface: canonical}}`` mapping to serialise.
    """
    header = (
        f"# Proposed by `lit-monitor graph propose-aliases` on "
        f"{datetime.now().isoformat(timespec='seconds')}.\n"
        f"# Review and merge by hand into config/entity_aliases.yaml,\n"
        f"# then run `lit-monitor graph rebuild --aliases-only` to apply.\n\n"
    )
    body = yaml.safe_dump(proposals, default_flow_style=False, sort_keys=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body)
