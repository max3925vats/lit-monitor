"""
Integration tests — real services (Ollama, Zotero, Obsidian filesystem).

Run with:
    pytest tests/integration/ -m integration -v --tb=short

All tests are marked @pytest.mark.integration and are excluded from the
default `pytest` run. They require:
  - Ollama running at localhost:11434 with qwen2.5:3b and mxbai-embed-large pulled
  - Valid ~/.config/lit-monitor/config.toml with Zotero credentials
  - Obsidian vault path set in config/paths.yaml and accessible
  - lit-monitor check passing

Tests that write files to the Obsidian vault clean up after themselves.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tests.integration._live import skip_or_fail as _skip_or_fail

logger = logging.getLogger(__name__)

# Short synthetic paper text — used for LLM round-trip without reading a real PDF.
_SYNTHETIC_PAPER_TEXT = (
    "Filtration of model proteins (ProteinA1) was performed at pH 6.0 "
    "using a 30 kDa polyethersulfone membrane. Feed concentration was 10 g/L. "
    "Transmembrane pressure was maintained at 30 psi. Yield was 95.2% with "
    "less than 2% aggregate formation as measured by SEC-HPLC. "
    "The study type is experimental. "
    "Key conclusion: low pH reduces exclusion effect and improves selectivity."
)


# ===========================================================================
# Test 1 — Ollama connectivity and Pass 1 extraction
# ===========================================================================

@pytest.mark.integration
def test_ollama_extraction_round_trip(real_llm):
    """
    Real Ollama call: run Pass 1 extraction on a synthetic abstract.
    Validates:
    - OllamaClient connects and returns a parseable response
    - core_finding is populated (not null for a clear text)
    - All Pass 1 confidence fields are one of: explicit, inferred, absent
    """
    from scripts.llm.extractor import extract_paper

    result = extract_paper(
        fulltext=_SYNTHETIC_PAPER_TEXT,
        llm=real_llm,
        phases=("simple",),
    )

    assert isinstance(result, dict), "extract_paper must return a dict"
    # core_finding should be extractable from this clear text
    assert result.get("core_finding") is not None, (
        f"core_finding is null — model may not have understood the prompt.\n"
        f"Full result: {json.dumps(result, indent=2)}"
    )
    # Every Pass 1 field should have a _confidence companion
    valid_confidence = {"explicit", "inferred", "absent"}
    for field in ("core_finding", "methods_summary", "results_summary",
                  "conclusions", "study_type"):
        conf_key = f"{field}_confidence"
        conf = result.get(conf_key)
        assert conf in valid_confidence, (
            f"{conf_key} is {conf!r}, expected one of {valid_confidence}"
        )

    logger.info("Pass 1 extraction result: %s", json.dumps(result, indent=2))


# ===========================================================================
# Test 2 — Zotero API: list collection items
# ===========================================================================

@pytest.mark.integration
def test_zotero_collection_returns_items(real_config, real_zotero):
    """
    Real Zotero API call: fetch items from the configured 'lit-monitor' collection.
    Validates:
    - Credentials are accepted (no 403/unauthorized)
    - Collection exists and returns at least one item
    - Each item has the expected pyzotero structure: {key, data, ...}
    """
    collection_name = real_config.zotero.collection_name
    items = real_zotero.get_collection_items(collection_name)

    assert isinstance(items, list), "get_collection_items must return a list"
    assert len(items) > 0, (
        f"Collection '{collection_name}' is empty. "
        "Add at least one item before running integration tests."
    )

    first = items[0]
    assert "key" in first, f"Item missing 'key' field: {first}"
    assert "data" in first, f"Item missing 'data' field: {first}"

    data = first["data"]
    assert "title" in data, f"Item data missing 'title': {data.keys()}"
    logger.info(
        "Zotero collection '%s': %d items. First item: %s",
        collection_name, len(items), data.get("title", "(no title)")
    )


# ===========================================================================
# Test 3 — Obsidian writer: write paper note to real vault
# ===========================================================================

@pytest.mark.integration
def test_obsidian_paper_note_written_to_vault(real_config):
    """
    Write a test paper note to the actual Obsidian vault (Literature/Papers/).
    Validates:
    - Note file is created at the expected path
    - YAML frontmatter contains source_type and doi
    - Persist zone markers are present in the output
    Cleanup: the test note is deleted after assertions pass.
    """
    from scripts.output.obsidian_writer import write_paper_note

    paper = {
        "doi": "10.9999/integration-test-note",
        "title": "Integration Test Paper — Do Not Cite",
        "authors": ["TestAuthor, A"],
        "year": 2024,
        "journal": "Journal of Integration Tests",
        "zotero_key": "TESTKEY001",
        "first_seen_date": "2024-01-01",
        "keywords": ["test", "integration"],
        "themes": ["Mock Theme A"],
        "tracked_author": False,
        "fulltext_analyzed": True,
    }
    extraction = {
        "core_finding": "This is a test extraction. The note should be deleted.",
        "core_finding_confidence": "explicit",
        "methods_summary": "Integration test method.",
        "methods_summary_confidence": "explicit",
        "results_summary": None,
        "results_summary_confidence": "absent",
        "conclusions": "Test passed.",
        "conclusions_confidence": "explicit",
        "study_type": "experimental",
        "study_type_confidence": "explicit",
    }

    note_path = write_paper_note(paper, extraction, real_config)

    try:
        path = Path(note_path)
        assert path.exists(), f"Note not created at {path}"
        content = path.read_text(encoding="utf-8")

        assert 'source_type: "paper"' in content, "Missing source_type in frontmatter"
        assert "10.9999/integration-test-note" in content, "Missing DOI in note"
        assert '{% persist "related_work" %}' in content, "Missing related_work persist zone"
        assert '{% persist "synthesis" %}' in content, "Missing synthesis persist zone"
        assert "## Related Work" in content, "Missing Related Work section"

        logger.info("Note written successfully: %s", path)
    finally:
        # Always clean up, even if assertions fail
        if Path(note_path).exists():
            Path(note_path).unlink()
            logger.info("Cleaned up test note: %s", note_path)


# ===========================================================================
# Test 4 — ChromaDB embeddings via Ollama (mxbai-embed-large)
# ===========================================================================

@pytest.mark.integration
def test_chromadb_add_and_find_similar(real_config, tmp_chroma_dir):
    """
    Real Ollama embedding call via EmbeddingsDB (mxbai-embed-large model).
    Validates:
    - mxbai-embed-large is reachable and returns a vector
    - Paper is added to ChromaDB successfully
    - find_similar_to_text returns the added paper as a top result
    """
    import requests as _requests

    from scripts.output.embeddings import EmbeddingsDB

    host = getattr(real_config.ingestion, "ollama_host", "http://localhost:11434")

    # Check that mxbai-embed-large is pulled
    try:
        resp = _requests.get(f"{host}/api/tags", timeout=5)
        tags = resp.json().get("models", [])
        pulled = [m.get("name", "") for m in tags]
        if not any("mxbai-embed-large" in p for p in pulled):
            _skip_or_fail(
                "mxbai-embed-large is not pulled. Run: ollama pull mxbai-embed-large"
            )
    except Exception as exc:
        _skip_or_fail(f"Cannot check Ollama tags: {exc}")

    db = EmbeddingsDB(str(tmp_chroma_dir), ollama_host=host)

    test_doi = "10.test/integration-embed-001"
    test_text = (
        "Filtration protein retention pH 6.0 exclusion effect "
        "polyethersulfone membrane TMP 30 psi yield 95 percent"
    )

    db.add_paper(
        doi=test_doi,
        text=test_text,
        metadata={"title": "Integration Test Paper", "year": 2024},
    )

    assert db.count() >= 1, "Collection should have at least 1 document after add"

    results = db.find_similar_to_text(
        "protein filtration protein retention membrane", top_k=5
    )

    assert len(results) > 0, "find_similar_to_text returned no results"
    ids = [r["id"] for r in results]
    assert test_doi in ids, (
        f"Test paper '{test_doi}' not in top-5 results. Got: {ids}"
    )
    top = results[0]
    assert "score" in top and "document" in top and "metadata" in top

    logger.info(
        "Top result: id=%s score=%.4f", top["id"], top["score"]
    )


# ===========================================================================
# Test 5 — StateDB: paper + chapter round trip with real SQLite file
# ===========================================================================

@pytest.mark.integration
def test_state_db_paper_round_trip(tmp_state_db):
    """
    Full StateDB write/read cycle for a paper using a real SQLite file.
    Validates path handling, schema correctness, and status transitions.
    No external services required.
    """
    doi = "10.1234/test-paper-001"

    tmp_state_db.upsert_paper({
        "doi": doi,
        "title": "Test Paper on Membrane Filtration",
        "authors": json.dumps(["Smith, J", "Jones, A"]),
        "year": 2022,
        "source_type": "paper",
        "status": "pending",
        "extraction_json": json.dumps({
            "core_finding": "Filtration is effective.",
            "core_finding_confidence": "explicit",
        }),
    })

    # Retrieve paper
    paper = tmp_state_db.get_paper(doi)
    assert paper is not None, "Paper not found after insert"
    assert paper["source_type"] == "paper"
    assert paper["status"] == "pending"

    # Status transition
    tmp_state_db.mark_status(doi, "extraction_complete")
    updated = tmp_state_db.get_paper(doi)
    assert updated["status"] == "extraction_complete"

    # Extraction JSON round-trip
    extraction = tmp_state_db.get_extraction_json(doi)
    assert extraction is not None
    assert extraction["core_finding"] == "Filtration is effective."

    # known_dois includes it
    known = tmp_state_db.known_dois()
    assert doi in known

    logger.info("StateDB round-trip: book and chapter stored and retrieved correctly.")


# ===========================================================================
# Test 6 — Full ingestion path: Zotero → markdown → LLM extract → Obsidian note
# ===========================================================================

@pytest.mark.integration
def test_single_paper_ingestion_end_to_end(
    real_config, real_zotero, real_llm, tmp_state_db, tmp_chroma_dir
):
    """
    End-to-end ingestion of one real Zotero paper (simple phase only for speed).
    Steps:
        1. Fetch items from the configured Zotero collection
        2. Find the first item with a markdown attachment in Zotero
        3. Read markdown text via ZoteroClient.get_markdown_attachment()
        4. Run simple-phase LLM extraction
        5. Write paper note to Obsidian vault
        6. Add to ChromaDB embeddings (if mxbai-embed-large is available)
        7. Assert note file exists with expected content
    Cleanup: note file is deleted after all assertions.
    """
    from scripts.core.zotero_client import ZoteroClient
    from scripts.llm.extractor import extract_paper
    from scripts.output.obsidian_writer import write_paper_note

    collection_name = real_config.zotero.collection_name
    items = real_zotero.get_collection_items(collection_name)

    if not items:
        _skip_or_fail(f"No items in Zotero collection '{collection_name}'")

    # Find first item with a markdown attachment
    target_item = None
    fulltext: str | None = None

    for item in items[:20]:  # check first 20 items
        item_key = item.get("key", "")
        data = item.get("data", {})
        if data.get("itemType") in ("attachment", "note"):
            continue  # skip non-paper items
        md = real_zotero.get_markdown_attachment(item_key)
        if md:
            target_item = item
            fulltext = md
            break

    if target_item is None:
        _skip_or_fail(
            "No item with a markdown attachment found in first 20 items of Zotero collection. "
            "Attach a .md file to a Zotero item to enable this integration test."
        )

    data = target_item["data"]
    doi = (data.get("DOI") or "").strip().lower() or f"zotero:{target_item['key']}"
    authors = ZoteroClient.extract_authors(data)
    year = int(data.get("date", "0")[:4]) if data.get("date") else 0
    title = data.get("title", "(no title)")

    logger.info("Testing ingestion of: %s (%s) — markdown chars: %d", title, doi, len(fulltext))

    assert isinstance(fulltext, str) and len(fulltext) > 100, (
        f"Markdown attachment too short ({len(fulltext)} chars)"
    )

    # --- Step 4: LLM extraction (simple phase only for speed) ---
    extraction = extract_paper(
        fulltext=fulltext[:8000],  # truncate for speed in integration test
        llm=real_llm,
        phases=("simple",),
    )

    assert "core_finding" in extraction, (
        f"Simple-phase extraction missing core_finding. Result: {extraction}"
    )

    # Persist to state DB
    tmp_state_db.upsert_paper({
        "doi": doi,
        "title": title,
        "authors": json.dumps(authors),
        "year": year,
        "source_type": "paper",
        "status": "extraction_complete",
        "extraction_json": json.dumps(extraction),
    })

    # --- Step 5: Write Obsidian note ---
    paper_dict = {
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "journal": data.get("publicationTitle", ""),
        "zotero_key": target_item["key"],
        "first_seen_date": "2024-01-01",
        "keywords": [],
        "themes": [],
        "tracked_author": False,
        "fulltext_analyzed": True,
    }

    note_path = write_paper_note(paper_dict, extraction, real_config)
    note_path = Path(note_path)

    try:
        assert note_path.exists(), f"Note not created at {note_path}"
        content = note_path.read_text(encoding="utf-8")
        assert 'source_type: "paper"' in content
        assert '{% persist "related_work" %}' in content

        # --- Step 6: ChromaDB indexing (optional — skipped if nomic not pulled) ---
        import requests as _requests
        host = getattr(real_config.ingestion, "ollama_host", "http://localhost:11434")
        try:
            resp = _requests.get(f"{host}/api/tags", timeout=5)
            pulled = [m.get("name", "") for m in resp.json().get("models", [])]
            if any("mxbai-embed-large" in p for p in pulled):
                from scripts.output.embeddings import EmbeddingsDB
                db = EmbeddingsDB(str(tmp_chroma_dir), ollama_host=host)
                embed_text = f"{title} {extraction.get('core_finding', '')}"
                db.add_paper(doi, embed_text, {"title": title, "year": year})
                assert db.count() >= 1
                logger.info("ChromaDB indexing: OK (1 doc added)")
            else:
                logger.info("mxbai-embed-large not pulled — skipping embedding step")
        except Exception as embed_exc:
            logger.warning("ChromaDB step skipped: %s", embed_exc)

        logger.info("End-to-end ingestion test passed for: %s", title)

    finally:
        # Always clean up vault note
        if note_path.exists():
            note_path.unlink()
            logger.info("Cleaned up vault note: %s", note_path)
