"""Unit tests for ``EmbeddingsDB`` collection-name override (Task #68 prereq).

The sandbox needs ``dev-*`` ChromaDB collections without rewriting the
embed/upsert logic. ``EmbeddingsDB`` was extended with optional
``papers_collection`` / ``chunks_collection`` keyword arguments whose
defaults preserve the historical production names exactly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.output.embeddings import EmbeddingsDB


@pytest.mark.unit
def test_embeddings_db_uses_default_collection_names_when_unspecified(
    tmp_path: Path,
) -> None:
    """No kwargs → production collection names are used (zero behaviour drift)."""
    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
    # The instance-bound names are the single source of truth for clear() /
    # clear_chunks(); verify they match the historic defaults.
    assert db._papers_collection_name == "lit_monitor_v1"
    assert db._chunks_collection_name == "lit_monitor_chunks_v1"
    # And the actual ChromaDB collections were created with those names.
    assert db._collection.name == "lit_monitor_v1"
    assert db._chunks_collection.name == "lit_monitor_chunks_v1"


@pytest.mark.unit
def test_embeddings_db_accepts_collection_overrides(tmp_path: Path) -> None:
    """Sandbox path: explicit kwargs route writes to ``dev-*`` collections."""
    db = EmbeddingsDB(
        persist_dir=str(tmp_path / "chroma"),
        papers_collection="dev-papers",
        chunks_collection="dev-chunks",
    )
    assert db._papers_collection_name == "dev-papers"
    assert db._chunks_collection_name == "dev-chunks"
    assert db._collection.name == "dev-papers"
    assert db._chunks_collection.name == "dev-chunks"
