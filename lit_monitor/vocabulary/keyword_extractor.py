"""
Keyword extraction from Zotero library items.
Pulls tags from all items in the library's collections and returns
(keyword, zotero_key) pairs for downstream normalisation.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lit_monitor.core.zotero_client import ZoteroClient
logger = logging.getLogger(__name__)


def extract_all_keywords(
    zot_client: ZoteroClient,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[tuple[str, str]]:
    """
    Return all (keyword, zotero_key) pairs from the Zotero library.

    Pulls the full library (top-level items) via pyzotero's `everything()`
    paginator, collects every tag string, and returns them paired with the
    item's Zotero key so provenance can be traced back to source items.

    Duplicate (keyword, zotero_key) pairs within the same item are
    deduplicated. The same keyword string from different items appears
    multiple times — normalisation is the job of normalizer.py.

    Args:
        zot_client: Authenticated ZoteroClient wrapper.
        max_retries: Number of attempts before giving up (default 3).
            Transient network timeouts from Zotero's API are retried
            automatically with `retry_delay` seconds between attempts.
        retry_delay: Seconds to wait between retry attempts (default 5).

    Returns:
        List of (tag_string, zotero_item_key) pairs.

    Raises:
        RuntimeError: If all retry attempts fail.  The caller is responsible
            for surfacing this as a user-visible error rather than silently
            treating it as an empty result.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            # `everything()` paginates the full library across multiple HTTP
            # requests to api.zotero.org — transient timeouts are expected on
            # large libraries (1000+ items).
            items = zot_client._zot.everything(
                zot_client._zot.items(itemType="-attachment")
            )
            # Process items inside the same try block so any streaming error
            # is caught by the same retry handler.
            pairs: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for item in items:
                data = item.get("data", {})
                zotero_key = data.get("key", "")
                tags = data.get("tags", [])
                for tag_entry in tags:
                    tag = (
                        tag_entry.get("tag", "")
                        if isinstance(tag_entry, dict)
                        else str(tag_entry)
                    )
                    tag = tag.strip()
                    if not tag:
                        continue
                    pair = (tag, zotero_key)
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)
            logger.info("Extracted %d keyword-item pairs from Zotero library", len(pairs))
            return pairs

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Failed to retrieve Zotero items (attempt %d/%d): %s — "
                    "retrying in %.0f s",
                    attempt, max_retries, exc, retry_delay,
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    "Failed to retrieve Zotero items after %d attempt(s): %s",
                    max_retries, exc,
                )

    raise RuntimeError(
        f"Zotero keyword extraction failed after {max_retries} attempt(s): {last_exc}"
    ) from last_exc
