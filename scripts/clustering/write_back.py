"""
Bundle C: Zotero write-back for cluster themes.

Strictly additive — never deletes tags, items, or collections.

push_tags_to_zotero:
  Adds lm/<theme-name> tags to Zotero items for each paper in a cluster.
  Default: dry_run=True. Pass dry_run=False (or --confirm on CLI) to write.

push_collections_to_zotero:
  Creates a parent collection (default "lit-monitor") and one sub-collection
  per theme, then adds each paper to its sub-collection.
  Default: dry_run=True.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TAG_NAMESPACE = "lm"


def push_tags_to_zotero(
    state_db,
    zotero_client,
    *,
    dry_run: bool = True,
) -> dict:
    """Add lm/<theme-name> tags to Zotero items for clustered papers.

    Args:
        state_db: StateDB instance.
        zotero_client: ZoteroClient instance.
        dry_run: When True (default), log what would be written but make no changes.

    Returns:
        Report dict: {"dry_run": bool, "tags_added": int, "papers_processed": int,
                      "papers_skipped": int, "operations": [...]}.
    """
    clusters = state_db.list_active_clusters()
    report: dict = {
        "dry_run": dry_run,
        "tags_added": 0,
        "tags_failed": 0,
        "papers_processed": 0,
        "papers_skipped": 0,
        "operations": [],
    }

    for cluster in clusters:
        cid = cluster["id"]
        theme = cluster.get("display_name") or f"Cluster {cid}"
        tag_value = f"{_TAG_NAMESPACE}/{theme}"

        assignments = state_db.get_cluster_assignments(cid)
        for assignment in assignments:
            doi = assignment["doi"]
            paper_row = state_db.get_paper(doi)
            if not paper_row or not paper_row.get("zotero_key"):
                logger.debug("C write-back: no Zotero key for %s — skipping.", doi)
                report["papers_skipped"] += 1
                continue

            zkey = paper_row["zotero_key"]
            op = {"doi": doi, "zotero_key": zkey, "tag": tag_value}
            report["operations"].append(op)

            if not dry_run:
                try:
                    _add_tag_to_item(zotero_client, zkey, tag_value)
                    report["tags_added"] += 1
                    report["papers_processed"] += 1
                except Exception as exc:
                    # Op failed: surface it as a failure, do NOT count it as
                    # written/processed (accounting must reflect real outcomes).
                    report["tags_failed"] += 1
                    logger.warning(
                        "C write-back: tag update failed for %s (%s): %s",
                        doi, zkey, exc,
                    )
            else:
                report["tags_added"] += 1  # count planned ops
                report["papers_processed"] += 1

    if dry_run:
        logger.info(
            "C write-back (dry-run): would add %d tags across %d papers.",
            report["tags_added"], report["papers_processed"],
        )
    else:
        logger.info(
            "C write-back: added %d tags across %d papers (%d failed).",
            report["tags_added"], report["papers_processed"], report["tags_failed"],
        )

    return report


def push_collections_to_zotero(
    state_db,
    zotero_client,
    *,
    parent_name: str = "lit-monitor",
    dry_run: bool = True,
    use_existing_priors: bool = False,
) -> dict:
    """Add Zotero sub-collections under parent_name for each cluster theme.

    Args:
        state_db: StateDB instance.
        zotero_client: ZoteroClient instance.
        parent_name: Name of the top-level Zotero collection to create/use.
        dry_run: When True (default), report what would be done, no changes.
        use_existing_priors: When True, papers are also added to any existing
            collections they belong to (duplicate-membership).

    Returns:
        Report dict with collections_to_create, items_to_add counts.
    """
    clusters = state_db.list_active_clusters()
    report: dict = {
        "dry_run": dry_run,
        "collections_to_create": 0,
        "items_to_add": 0,
        "items_failed": 0,
        "operations": [],
    }

    if not clusters:
        return report

    # Resolve or plan parent collection
    all_collections = zotero_client.get_all_collections()
    parent_key: str | None = None
    for col in all_collections:
        if col.get("name") == parent_name and col.get("parent_collection_key") is None:
            parent_key = col["key"]
            break

    if parent_key is None:
        report["collections_to_create"] += 1
        report["operations"].append({"action": "create_collection", "name": parent_name, "parent": None})
        if not dry_run:
            parent_key = _create_collection(zotero_client, parent_name, parent_key=None)

    # Resolve existing sub-collections under parent
    existing_sub: dict[str, str] = {}  # name → key
    for col in all_collections:
        if col.get("parent_collection_key") == parent_key:
            existing_sub[col["name"]] = col["key"]

    for cluster in clusters:
        cid = cluster["id"]
        theme = cluster.get("display_name") or f"Cluster {cid}"

        sub_key = existing_sub.get(theme)
        if sub_key is None:
            report["collections_to_create"] += 1
            report["operations"].append({
                "action": "create_collection", "name": theme, "parent": parent_name,
            })
            if not dry_run and parent_key:
                sub_key = _create_collection(zotero_client, theme, parent_key=parent_key)

        assignments = state_db.get_cluster_assignments(cid)
        for assignment in assignments:
            doi = assignment["doi"]
            paper_row = state_db.get_paper(doi)
            if not paper_row or not paper_row.get("zotero_key"):
                continue
            zkey = paper_row["zotero_key"]
            report["operations"].append({
                "action": "add_to_collection", "doi": doi, "collection": theme,
            })
            if not dry_run and sub_key:
                try:
                    zotero_client._zot.addto_collection(sub_key, zkey)
                    report["items_to_add"] += 1
                except Exception as exc:
                    # Only count an item as added when the op actually succeeded;
                    # track failures separately rather than reporting all-success.
                    report["items_failed"] += 1
                    logger.warning(
                        "C write-back: failed to add %s to collection %r: %s",
                        doi, theme, exc,
                    )
            else:
                # dry-run (or no resolved sub-collection): count as planned.
                report["items_to_add"] += 1

    if dry_run:
        logger.info(
            "C write-back collections (dry-run): %d to create, %d items to add.",
            report["collections_to_create"], report["items_to_add"],
        )
    else:
        logger.info(
            "C write-back collections: created %d, added %d items (%d failed).",
            report["collections_to_create"], report["items_to_add"], report["items_failed"],
        )

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_tag_to_item(zotero_client, zkey: str, tag_value: str) -> None:
    """Fetch the item, add tag if not already present, and update (additive only)."""
    item = zotero_client._zot.item(zkey)
    existing_tags: list[dict] = item.get("data", {}).get("tags", [])
    existing_values = {t.get("tag", "") for t in existing_tags}

    if tag_value in existing_values:
        # Already tagged — nothing to do (idempotent)
        return

    existing_tags.append({"tag": tag_value})
    item["data"]["tags"] = existing_tags
    zotero_client._zot.update_item(item)


def _create_collection(zotero_client, name: str, *, parent_key: str | None) -> str | None:
    """Create a Zotero collection. Returns the new collection key or None on failure."""
    try:
        payload: dict = {"name": name}
        if parent_key:
            payload["parentCollection"] = parent_key
        result = zotero_client._zot.create_collections([payload])
        # pyzotero returns a list of created collection dicts
        if isinstance(result, list) and result:
            return result[0].get("key")
        return None
    except Exception as exc:
        logger.warning("C: failed to create Zotero collection %r: %s", name, exc)
        return None
