"""
Zotero API client wrapper (pyzotero).
Read-only for library items. create_item() is available for adding
discovery digests as Zotero notes, but never modifies existing items.
API keys and library ID are read from ~/.config/lit-monitor/config.toml
by the caller — this module accepts them as constructor arguments.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from pyzotero import zotero

from scripts.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)
class ZoteroClient:
    """Thin wrapper around pyzotero.Zotero."""
    def __init__(
        self,
        library_id: str,
        api_key: str,
        library_type: str = "user",
        local_storage_path: str | Path = "~/Zotero/storage",
    ) -> None:
        self._zot = zotero.Zotero(library_id, library_type, api_key)
        self._local_storage = Path(os.path.expanduser(str(local_storage_path)))
    # ------------------------------------------------------------------ #
    # Collection helpers
    # ------------------------------------------------------------------ #
    def get_collection_key(self, collection_name: str) -> str:
        """Return the Zotero collection key for the given collection name."""
        collections = self._zot.collections()
        for col in collections:
            if col["data"]["name"] == collection_name:
                return col["key"]
        raise ValueError(
            f"Zotero collection '{collection_name}' not found. "
            "Check config/paths.yaml collection_name setting."
        )
    def get_all_collections(self) -> list[dict]:
        """Return all collections in the library, flattened.

        Each entry: ``{"key": str, "name": str, "parent_collection_key": str | None}``.
        Used by the setup wizard's collection-name dropdown — the parent key is
        included so the UI can render nested collections later if it wants to.
        ``parentCollection`` from pyzotero is either ``False`` (top-level) or
        a string key; this helper normalizes the ``False`` case to ``None`` so
        the wizard's JSON is unambiguous.
        """
        raw = self._paginate(lambda start, lim: self._zot.collections(start=start, limit=lim))
        out: list[dict] = []
        for col in raw:
            data = col.get("data", {})
            parent = data.get("parentCollection") or None
            out.append({
                "key": col.get("key", ""),
                "name": data.get("name", ""),
                "parent_collection_key": parent if parent else None,
            })
        return out

    def get_collection_items(self, collection_name: str, limit: int | None = None) -> list[dict]:
        """Return all items in a Zotero collection (paginated)."""
        key = self.get_collection_key(collection_name)
        items = self._paginate(lambda start, lim: self._zot.collection_items(key, start=start, limit=lim))
        return items[:limit] if limit is not None else items

    def get_book_collection_items(self, collection_name: str, limit: int | None = None) -> list[dict]:
        """Return book-type items from a Zotero collection."""
        key = self.get_collection_key(collection_name)
        items = self._paginate(
            lambda start, lim: self._zot.collection_items(key, itemType="book", start=start, limit=lim)
        )
        return items[:limit] if limit is not None else items

    def get_all_library_items(self, limit: int | None = None) -> list[dict]:
        """Return all items in the user's Zotero library (paginated).

        Use this when you want everything in the library rather than a specific
        collection. Fetches via GET /users/<id>/items with 100-item pages.
        """
        items = self._paginate(lambda start, lim: self._zot.items(start=start, limit=lim))
        logger.info("get_all_library_items: fetched %d total items", len(items))
        return items[:limit] if limit is not None else items

    # ------------------------------------------------------------------ #
    # Internal pagination helper
    # ------------------------------------------------------------------ #
    def _paginate(self, fetch_fn, page_size: int = 100) -> list[dict]:
        """Walk through Zotero API pages until an empty page is returned."""
        all_items: list[dict] = []
        start = 0
        while True:
            batch = fetch_fn(start, page_size)
            if not batch:
                break
            all_items.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
        return all_items
    # ------------------------------------------------------------------ #
    # Attachment / PDF helpers
    # ------------------------------------------------------------------ #
    def get_children(self, item_key: str) -> list[dict]:
        """Return child items (attachments, notes) for an item key."""
        return self._zot.children(item_key)

    def get_attachment_local_path(
        self, attachment_key: str, filename: str
    ) -> Path:
        """
        Construct the local filesystem path for a Zotero attachment.
        Zotero stores attachments at:
          {local_storage_path}/{attachment_key}/{filename}
        The path is constructed — not confirmed to exist here.
        Use Path.exists() to check before reading.
        """
        return self._local_storage / attachment_key / filename
    def get_markdown_attachment(self, item_key: str) -> str | None:
        """Return the text content of the first .md/.markdown attachment for an item.

        Looks for local files via the Zotero storage path first; if the file is
        not synced locally, returns None (no cloud download for text files).
        Returns None if no markdown attachment exists.

        Phase M ingestion path: brain-build and weekly_monitor call this instead
        of resolving a PDF. If None is returned, the item is skipped silently.
        """
        children = self.get_children(item_key)
        for child in children:
            data = child.get("data", {})
            if data.get("itemType") != "attachment":
                continue
            filename = data.get("filename", "")
            if not filename.lower().endswith((".md", ".markdown")):
                continue
            attachment_key = data["key"]
            local_path = self.get_attachment_local_path(attachment_key, filename)
            if local_path.exists():
                try:
                    return local_path.read_text(encoding="utf-8")
                except Exception as exc:
                    strict_fallback(
                        logger,
                        f"Could not read markdown attachment {local_path} "
                        f"for item {item_key}: {exc}",
                        exc,
                    )
                    return None
        return None

    def download_attachment(self, attachment_key: str) -> bytes | None:
        """Download attachment file bytes from Zotero API (cloud sync). Returns None on failure."""
        try:
            content = self._zot.file(attachment_key)
            return content if content else None
        except Exception as exc:
            logger.warning("Zotero API download failed for attachment %s: %s", attachment_key, exc)
            return None

    # ------------------------------------------------------------------ #
    # Library version / polling
    # ------------------------------------------------------------------ #
    def get_current_version(self) -> int:
        """Return the current Zotero library version number.

        Calls pyzotero's last_modified_version() method, which makes a minimal
        items(limit=1) request internally and reads the Last-Modified-Version
        response header.

        Note: in pyzotero >=1.11, last_modified_version is a callable method,
        not an attribute. Calling it as an attribute raised TypeError in earlier
        versions of this codebase.
        """
        version = self._zot.last_modified_version()
        # Guard: if pyzotero's API changes again, surface it with a clear message
        # rather than a cryptic downstream error. An empty library returning None
        # from the header would also be caught here — callers should treat 0 as
        # "no baseline yet" rather than relaxing this check.
        if not isinstance(version, int):
            raise RuntimeError(
                f"pyzotero last_modified_version() returned {type(version).__name__!r}, "
                f"expected int. The pyzotero API may have changed; check "
                f"scripts/core/zotero_client.py:get_current_version()."
            )
        return version

    def get_items_since(self, library_version: int) -> list[dict]:
        """
        Return all items modified after library_version.
        Used by weekly ingestion to detect new items.
        """
        return self._zot.items(since=library_version)
    # ------------------------------------------------------------------ #
    # Write (digest only — never modifies existing records)
    # ------------------------------------------------------------------ #
    def create_note(self, parent_key: str, note_html: str) -> str:
        """
        Create a child note under an existing Zotero item.
        Returns the new item key.
        Used only for attaching discovery digest notes to Zotero items
        after human curation. Never called automatically.
        """

        template = self._zot.item_template("note")
        template["note"] = note_html
        result = self._zot.create_items([template], parent_key)
        return result["successful"]["0"]["key"]
    # ------------------------------------------------------------------ #
    # Author extraction helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_authors(item_data: dict) -> list[str]:
        """
        Extract author display names from a Zotero item data dict.
        Returns list of "LastName, FirstName" strings.
        """
        creators = item_data.get("creators", [])
        authors = []
        for c in creators:
            if c.get("creatorType") in ("author", "editor"):
                last = c.get("lastName", "")
                first = c.get("firstName", "")
                if last:
                    authors.append(f"{last}, {first}".strip(", "))
        return authors
