"""G11 + N6: alias proposal helper.

G11 (Phase 1) is LLM-free: survey the live KuzuDB Entity vocab, cluster
likely-equivalent surfaces within type using rapidfuzz, write a YAML proposal
file the operator reviews and merges into entity_aliases.yaml by hand.

N6 (Phase 2) adds the optional ``with_llm=True`` path: each fuzzy cluster is
sent to cloud-Ollama for validation. Rejected clusters are dropped; accepted
clusters may have their canonical overridden by the LLM. On ANY LLM failure
(no key, network error, malformed JSON), output is byte-identical to the
fuzzy-only path. The propose-review-accept loop is the safety net — N6 just
makes the proposals cleaner when the LLM is reachable.

Defensive perimeter: ``_consult_llm`` NEVER raises; it returns ``None`` on
any failure and the caller falls back to fuzzy-only output unchanged.

The LLM call site uses LAZY imports of ``scripts.llm.*`` so the module-level
import surface stays free of LLM clients — the G11 test
``test_no_llm_imports_in_module`` continues to pass.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz

from scripts.graph.aliases import load_aliases

logger = logging.getLogger(__name__)

_DEFAULT_MIN_RATIO = 80


# ---------------------------------------------------------------------------
# G11: fuzzy clustering primitives
# ---------------------------------------------------------------------------
def _cluster_surfaces(
    surfaces: list[str],
    min_ratio: int,
) -> list[list[str]]:
    """Greedy fuzzy clustering.

    For each surface, attach to the highest-scoring existing cluster
    if score >= min_ratio; else start a new cluster.
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


def _collect_clusters_by_type(
    graph_db: Any,
    min_ratio: int,
) -> dict[str, list[list[str]]]:
    """Survey the graph, group surfaces by type, fuzzy-cluster within type.

    Surfaces already present in ``entity_aliases.yaml`` (loaded via G2's
    ``load_aliases``) are filtered out before clustering so they are never
    re-proposed.

    Returns:
        ``{entity_type: [[member, ...], ...]}`` — only types with at least one
        multi-member cluster are present.
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

    out: dict[str, list[list[str]]] = {}
    for type_, surfaces in by_type.items():
        existing_set = {s.lower() for s in existing_aliases.get(type_, {})}
        # Deduplicate and exclude already-aliased surfaces.
        # Sort for determinism — greedy cluster order matters.
        unique = sorted(
            {s for s in surfaces if s.lower() not in existing_set}
        )
        if len(unique) < 2:
            continue
        clusters = _cluster_surfaces(unique, min_ratio)
        if clusters:
            out[type_] = clusters
    return out


def _build_proposals_from_clusters(
    clusters_by_type: dict[str, list[list[str]]],
) -> dict[str, dict[str, str]]:
    """G11 fuzzy-only proposal builder.

    The representative is the most-frequent surface in the cluster (and since
    inputs are deduplicated, Counter.most_common returns the first member by
    Python's stable sort — alphabetical because the upstream caller sorts).

    Returns:
        ``{type: {surface: canonical}}``, with the representative excluded
        from its own alias map (a canonical is not an alias of itself).
    """
    proposals: dict[str, dict[str, str]] = {}
    for type_, clusters in clusters_by_type.items():
        type_proposals: dict[str, str] = {}
        for cluster in clusters:
            counts = Counter(cluster)
            representative = counts.most_common(1)[0][0]
            for surface in cluster:
                if surface == representative:
                    continue
                type_proposals[surface] = representative
        if type_proposals:
            proposals[type_] = type_proposals
    return proposals


# ---------------------------------------------------------------------------
# N6: LLM consensus pass
# ---------------------------------------------------------------------------
def _flatten_clusters_for_llm(
    clusters_by_type: dict[str, list[list[str]]],
) -> list[dict[str, Any]]:
    """Flatten {type: [[members]]} into the prompt's input record shape.

    cluster_id format: ``"{type}__{idx}"`` — opaque to the LLM, decoded by
    ``_apply_llm_verdicts`` to look up the original member list.
    """
    out: list[dict[str, Any]] = []
    for type_, clusters in clusters_by_type.items():
        for idx, members in enumerate(clusters):
            out.append({
                "cluster_id": f"{type_}__{idx}",
                "type": type_,
                "members": members,
            })
    return out


def _consult_llm(
    clusters_by_type: dict[str, list[list[str]]],
    *,
    client: Any = None,
    prompt: Any = None,
) -> dict[str, dict[str, Any]] | None:
    """N6: send fuzzy clusters to cloud-Ollama for validation.

    Args:
        clusters_by_type: ``{entity_type: [[member, ...], ...]}`` — the
            fuzzy-clustered surfaces produced by ``_collect_clusters_by_type``.
        client: optional pre-built LLM client (test injection).
        prompt: optional pre-loaded Prompt instance (test injection).

    Returns:
        ``{cluster_id: {"accept": bool, "canonical": str, "rationale": str}}``
        for each cluster the LLM validated. Returns an empty dict when the
        input has no clusters. Returns ``None`` on ANY failure (missing key,
        client/prompt construction error, network error, malformed JSON,
        non-dict response shape) so the caller can fall back to the
        byte-identical fuzzy-only output.

    The function NEVER raises.
    """
    cluster_records = _flatten_clusters_for_llm(clusters_by_type)
    if not cluster_records:
        # Nothing to validate — distinguish from failure with empty dict.
        return {}

    # ---- Gate: only construct a client when explicitly enabled AND key present.
    # When the caller passed a client, skip the gate (test injection path and
    # the future "pipeline already built the client" path).
    if client is None:
        if not os.environ.get("OLLAMA_API_KEY"):
            logger.warning("N6: OLLAMA_API_KEY not set; skipping LLM consensus")
            return None
        try:
            # Lazy imports — keep module-level imports free of LLM clients so
            # the G11 test_no_llm_imports_in_module assertion still passes.
            from scripts.core.config import load_config  # noqa: PLC0415
            from scripts.llm.llm_client import OllamaClient  # noqa: PLC0415

            cfg = load_config()
            # Prefer graph.aliases.cloud_model if the user set one; otherwise
            # reuse the NER cloud model so a single env tuning controls both
            # phase-2 cloud call sites.
            cloud_model = (
                getattr(
                    getattr(getattr(cfg, "graph", None), "aliases", None),
                    "cloud_model",
                    None,
                )
                or getattr(
                    getattr(getattr(cfg, "graph", None), "ner", None),
                    "cloud_model",
                    "gemma2:27b-cloud",
                )
            )
            cloud_host = getattr(
                getattr(getattr(cfg, "graph", None), "ner", None),
                "cloud_host",
                "https://ollama.com",
            )
            client = OllamaClient(model=cloud_model, host=cloud_host)
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("N6: LLM client construction failed: %s", exc)
            return None

    # ---- Load the prompt (registry or override).
    if prompt is None:
        try:
            from scripts.llm.prompt_registry import load_prompt  # noqa: PLC0415

            prompt = load_prompt("alias_consensus")
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("N6: alias_consensus prompt load failed: %s", exc)
            return None

    # ---- Render placeholders. Accept either Prompt model OR a plain dict
    # (test override path uses {"system": ..., "user_template": ...} or
    # {"system": ..., "user": ...}).
    clusters_json = json.dumps(cluster_records)
    try:
        if hasattr(prompt, "render_user"):
            system_msg = prompt.system
            user_msg = prompt.render_user(clusters_json=clusters_json)
        elif isinstance(prompt, dict):
            system_msg = prompt["system"]
            user_template = prompt.get("user_template") or prompt["user"]
            user_msg = user_template.format(clusters_json=clusters_json)
        else:
            logger.warning(
                "N6: unexpected prompt type %s; expected Prompt or dict",
                type(prompt).__name__,
            )
            return None
    except (KeyError, AttributeError, TypeError) as exc:
        logger.warning("N6: prompt render failed: %s", exc)
        return None

    # ---- ONE LLM call.
    try:
        max_tokens = int(getattr(prompt, "max_tokens", 2000))
        response = client.complete(
            system=system_msg, user=user_msg, max_tokens=max_tokens
        )
    except TypeError:
        # Allow simple test clients whose .complete() signature is just
        # (system, user) without the max_tokens kw.
        try:
            response = client.complete(system=system_msg, user=user_msg)
        except Exception as exc:  # noqa: BLE001 — defensive perimeter
            logger.warning("N6: LLM call failed: %s", exc)
            return None
    except Exception as exc:  # noqa: BLE001 — defensive perimeter
        logger.warning("N6: LLM call failed: %s", exc)
        return None

    # ---- Parse the JSON response. Tolerate ```json ... ``` fences.
    try:
        cleaned = (response or "").strip()
        if cleaned.startswith("```"):
            # Drop the opening fence (may be ```json\n or just ```\n).
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning(
            "N6: JSON parse failed: %s (response prefix: %r)",
            exc,
            str(response)[:120],
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning(
            "N6: LLM response is not a dict (got %s)", type(parsed).__name__
        )
        return None

    # ---- Per-item validation. Drop malformed entries but keep the rest.
    out: dict[str, dict[str, Any]] = {}
    for cid, entry in parsed.items():
        if not isinstance(entry, dict):
            continue
        accept = entry.get("accept")
        canonical = entry.get("canonical")
        if not isinstance(accept, bool):
            continue
        if not isinstance(canonical, str) or not canonical:
            continue
        out[str(cid)] = {
            "accept": accept,
            "canonical": canonical,
            "rationale": str(entry.get("rationale", ""))[:120],
        }
    return out


def _apply_llm_verdicts(
    clusters_by_type: dict[str, list[list[str]]],
    verdicts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Build the final proposals dict using accepted LLM verdicts.

    For each accepted cluster, the canonical comes from the verdict. If the
    LLM-supplied canonical is not one of the cluster's members, it is still
    honoured — the prompt allows a normalised form derived from the members
    (e.g. expanding ``mAb`` to ``monoclonal antibody``). Every original
    member that differs from the canonical is mapped to it.

    Clusters whose cluster_id is missing from ``verdicts`` are treated as
    accept=false (the prompt contract).

    Args:
        clusters_by_type: original fuzzy clusters (input to ``_consult_llm``).
        verdicts: validated verdicts returned by ``_consult_llm``.

    Returns:
        ``{type: {surface: canonical}}`` in the same shape as the G11
        fuzzy-only output.
    """
    proposals: dict[str, dict[str, str]] = {}
    for type_, clusters in clusters_by_type.items():
        type_proposals: dict[str, str] = {}
        for idx, members in enumerate(clusters):
            cid = f"{type_}__{idx}"
            verdict = verdicts.get(cid)
            if verdict is None or not verdict.get("accept"):
                continue
            canonical = verdict["canonical"]
            for surface in members:
                if surface == canonical:
                    continue
                type_proposals[surface] = canonical
        if type_proposals:
            proposals[type_] = type_proposals
    return proposals


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def propose_aliases(
    graph_db: Any,
    *,
    min_ratio: int = _DEFAULT_MIN_RATIO,
    with_llm: bool = False,
) -> dict[str, dict[str, str]]:
    """Survey the graph and propose alias mappings for fuzzy-similar surfaces.

    G11 path (``with_llm=False``, default): fuzzy clusters become proposals
    with the most-frequent member as the canonical.

    N6 path (``with_llm=True``): the fuzzy clusters are first validated by
    cloud-Ollama. Rejected clusters are dropped; accepted clusters may have
    a different canonical assigned by the LLM. On ANY LLM failure (no key,
    network error, malformed JSON), the output is byte-identical to the G11
    path — N6 is enrichment, not a gate.

    Args:
        graph_db: A live GraphDB instance (KuzuDB-backed).
        min_ratio: Minimum fuzz.ratio score for two surfaces to be placed in
            the same cluster. Higher values = stricter matching.
        with_llm: When True, run the N6 consensus pass. Requires
            ``OLLAMA_API_KEY`` in the environment.

    Returns:
        ``{type: {surface: canonical}}`` where each surface maps to the
        cluster's canonical.
    """
    clusters_by_type = _collect_clusters_by_type(graph_db, min_ratio)
    fuzzy_proposals = _build_proposals_from_clusters(clusters_by_type)

    if not with_llm:
        return fuzzy_proposals

    # N6 LLM consensus pass.
    verdicts = _consult_llm(clusters_by_type)
    if verdicts is None:
        # Fallback: BYTE-IDENTICAL to fuzzy-only.
        logger.warning(
            "N6: LLM unavailable or returned malformed output; "
            "falling back to fuzzy-only proposals"
        )
        return fuzzy_proposals

    return _apply_llm_verdicts(clusters_by_type, verdicts)


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
