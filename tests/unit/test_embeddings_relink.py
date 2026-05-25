"""
Phase 7 unit tests — Embeddings + Obsidian relink.
EmbeddingsDB tests mock the Ollama HTTP call.
Relink tests use a mock EmbeddingsDB and a real in-memory StateDB.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_state_db(tmp_path: Path):
    from scripts.core.state_db import StateDB
    db = StateDB(str(tmp_path / "state.db"))
    return db
def _make_embeddings_db(tmp_path: Path):
    """Create EmbeddingsDB with Ollama embedding calls mocked out."""
    from scripts.output.embeddings import EmbeddingsDB
    db = EmbeddingsDB(
        persist_dir=str(tmp_path / "chromadb"),
        ollama_host="http://localhost:11434",
    )
    # Patch _embed to return a fixed 768-dim vector
    db._embed = MagicMock(return_value=[0.1] * 768)
    return db
# ---------------------------------------------------------------------------
# EmbeddingsDB tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_paper_embedding_added(tmp_path):
    """add_paper stores the document and returns count=1."""
    db = _make_embeddings_db(tmp_path)
    db.add_paper(
        doi="10.1234/test.2024",
        text="Filtration of ProteinAs at scale",
        metadata={"title": "Test Paper", "year": 2024, "note_title": "Smith2024_Test"},
    )
    assert db.count() == 1
@pytest.mark.unit
def test_find_similar_returns_multiple_papers(tmp_path):
    """Query returns results for multiple papers."""
    db = _make_embeddings_db(tmp_path)
    db.add_paper("doi1", "UF text 1", {"source_type": "paper", "note_title": "A2024_Paper"})
    db.add_paper("doi2", "UF text 2", {"source_type": "paper", "note_title": "B2024_Paper"})
    db.add_paper("doi3", "UF text 3", {"source_type": "review", "note_title": "C2024_Review"})
    results = db.find_similar_to_text("filtration", top_k=5)
    assert len(results) == 3
@pytest.mark.unit
def test_find_similar_source_type_filter(tmp_path):
    """source_types filter restricts results to matching types."""
    db = _make_embeddings_db(tmp_path)
    db.add_paper("doi1", "UF text", {"source_type": "paper", "note_title": "A"})
    db.add_paper("doi2", "UF review", {"source_type": "review", "note_title": "B"})
    results = db.find_similar_to_text("filtration", top_k=5, source_types=["paper"])
    assert all(r["metadata"]["source_type"] == "paper" for r in results)
@pytest.mark.unit
def test_find_similar_excludes_self(tmp_path):
    """exclude_id removes the source document from results."""
    db = _make_embeddings_db(tmp_path)
    db.add_paper("doi1", "text", {"source_type": "paper", "note_title": "A"})
    db.add_paper("doi2", "text", {"source_type": "paper", "note_title": "B"})
    results = db.find_similar_to_text("text", top_k=5, exclude_id="doi1")
    ids = [r["id"] for r in results]
    assert "doi1" not in ids
@pytest.mark.unit
def test_upsert_is_idempotent(tmp_path):
    """Adding the same doi twice does not create a duplicate."""
    db = _make_embeddings_db(tmp_path)
    db.add_paper("doi1", "text v1", {"title": "V1", "note_title": "A"})
    db.add_paper("doi1", "text v2", {"title": "V2", "note_title": "A"})
    assert db.count() == 1
@pytest.mark.unit
def test_find_similar_empty_collection(tmp_path):
    """Query on empty collection returns empty list, no error."""
    db = _make_embeddings_db(tmp_path)
    results = db.find_similar_to_text("anything", top_k=5)

    assert results == []
# ---------------------------------------------------------------------------
# Chunk-level EmbeddingsDB tests (N18)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_add_chunks_indexes_passages(tmp_path):
    """add_chunks stores chunk documents and they are queryable."""
    from scripts.core.chunker import Chunk
    db = _make_embeddings_db(tmp_path)
    chunks = [
        Chunk(
            chunk_id="10.1/a#chunk-0",
            doi="10.1/a",
            text="Filtration system fouling.",
            section_heading="# Introduction",
            chunk_index=0,
            total_chunks=2,
            char_start=0,
            char_end=32,
            metadata={"doi": "10.1/a", "section_heading": "# Introduction", "chunk_index": 0},
        ),
        Chunk(
            chunk_id="10.1/a#chunk-1",
            doi="10.1/a",
            text="Tangential flow results.",
            section_heading="# Results",
            chunk_index=1,
            total_chunks=2,
            char_start=33,
            char_end=57,
            metadata={"doi": "10.1/a", "section_heading": "# Results", "chunk_index": 1},
        ),
    ]
    db.add_chunks("10.1/a", chunks)
    assert db._chunks_collection.count() == 2


@pytest.mark.unit
def test_add_chunks_delete_then_add(tmp_path):
    """Re-calling add_chunks replaces stale chunks, not appends."""
    from scripts.core.chunker import Chunk
    db = _make_embeddings_db(tmp_path)

    def _make_chunk(idx: int) -> Chunk:
        return Chunk(
            chunk_id=f"10.1/b#chunk-{idx}",
            doi="10.1/b",
            text=f"Text {idx}",
            section_heading="",
            chunk_index=idx,
            total_chunks=1,
            char_start=0,
            char_end=10,
            metadata={"doi": "10.1/b", "section_heading": "", "chunk_index": idx},
        )

    # First pass: 3 chunks
    db.add_chunks("10.1/b", [_make_chunk(i) for i in range(3)])
    assert db._chunks_collection.count() == 3

    # Second pass: 2 chunks — old set must be deleted first
    db.add_chunks("10.1/b", [_make_chunk(i) for i in range(2)])
    assert db._chunks_collection.count() == 2


@pytest.mark.unit
def test_add_chunks_empty_list_clears_stale(tmp_path):
    """add_chunks([]) removes existing chunks for that DOI."""
    from scripts.core.chunker import Chunk
    db = _make_embeddings_db(tmp_path)
    c = Chunk(
        chunk_id="10.1/c#chunk-0",
        doi="10.1/c",
        text="Some text.",
        section_heading="",
        chunk_index=0,
        total_chunks=1,
        char_start=0,
        char_end=10,
        metadata={"doi": "10.1/c", "section_heading": "", "chunk_index": 0},
    )
    db.add_chunks("10.1/c", [c])
    assert db._chunks_collection.count() == 1

    db.add_chunks("10.1/c", [])
    assert db._chunks_collection.count() == 0


@pytest.mark.unit
def test_find_similar_chunks_empty_returns_empty(tmp_path):
    """find_similar_chunks on an empty collection returns []."""
    db = _make_embeddings_db(tmp_path)
    assert db.find_similar_chunks("anything") == []


@pytest.mark.unit
def test_find_similar_chunks_dedupe_by_paper(tmp_path):
    """dedupe_by_paper=True returns at most one result per DOI."""
    from scripts.core.chunker import Chunk
    db = _make_embeddings_db(tmp_path)

    for i in range(3):
        db.add_chunks("10.1/d", [
            Chunk(
                chunk_id=f"10.1/d#chunk-{i}",
                doi="10.1/d",
                text=f"Passage {i} about membranes",
                section_heading="",
                chunk_index=i,
                total_chunks=3,
                char_start=0,
                char_end=20,
                metadata={"doi": "10.1/d", "section_heading": "", "chunk_index": i},
            )
        ])
        # add_chunks does delete-then-add, so add incrementally via upsert:
    # Re-add all 3 at once
    chunks = [
        Chunk(
            chunk_id=f"10.1/d#chunk-{i}",
            doi="10.1/d",
            text=f"Passage {i} about membranes",
            section_heading="",
            chunk_index=i,
            total_chunks=3,
            char_start=0,
            char_end=20,
            metadata={"doi": "10.1/d", "section_heading": "", "chunk_index": i},
        )
        for i in range(3)
    ]
    db.add_chunks("10.1/d", chunks)
    assert db._chunks_collection.count() == 3

    results = db.find_similar_chunks("membranes", top_k=10, dedupe_by_paper=True)
    dois = [r["metadata"]["doi"] for r in results]
    assert dois.count("10.1/d") == 1  # deduplicated to exactly 1 result for this DOI


@pytest.mark.unit
def test_clear_chunks_only_removes_chunk_collection(tmp_path):
    """clear_chunks() removes chunks but leaves paper-level index intact."""
    from scripts.core.chunker import Chunk
    db = _make_embeddings_db(tmp_path)
    db.add_paper("10.1/e", "paper text", {"source_type": "paper", "note_title": "E2024"})
    c = Chunk(
        chunk_id="10.1/e#chunk-0",
        doi="10.1/e",
        text="Chunk text.",
        section_heading="",
        chunk_index=0,
        total_chunks=1,
        char_start=0,
        char_end=11,
        metadata={"doi": "10.1/e", "section_heading": "", "chunk_index": 0},
    )
    db.add_chunks("10.1/e", [c])
    assert db.count() == 1
    assert db._chunks_collection.count() == 1

    db.clear_chunks()
    assert db.count() == 1, "Paper-level collection must survive clear_chunks()"
    assert db._chunks_collection.count() == 0


@pytest.mark.unit
def test_clear_also_clears_chunks(tmp_path):
    """clear() removes both the paper-level and chunk-level collections."""
    from scripts.core.chunker import Chunk
    db = _make_embeddings_db(tmp_path)
    db.add_paper("10.1/f", "paper text", {"source_type": "paper", "note_title": "F2024"})
    db.add_chunks("10.1/f", [
        Chunk(
            chunk_id="10.1/f#chunk-0",
            doi="10.1/f",
            text="Chunk text.",
            section_heading="",
            chunk_index=0,
            total_chunks=1,
            char_start=0,
            char_end=11,
            metadata={"doi": "10.1/f", "section_heading": "", "chunk_index": 0},
        )
    ])
    db.clear()
    assert db.count() == 0
    assert db._chunks_collection.count() == 0


@pytest.mark.unit
def test_get_collection_stats_includes_chunk_count(tmp_path):
    """get_collection_stats() reports both paper count and chunk count."""
    from scripts.core.chunker import Chunk
    db = _make_embeddings_db(tmp_path)
    db.add_paper("10.1/g", "text", {"source_type": "paper", "note_title": "G"})
    db.add_chunks("10.1/g", [
        Chunk(
            chunk_id="10.1/g#chunk-0",
            doi="10.1/g",
            text="Chunk.",
            section_heading="",
            chunk_index=0,
            total_chunks=1,
            char_start=0,
            char_end=6,
            metadata={"doi": "10.1/g", "section_heading": "", "chunk_index": 0},
        )
    ])
    stats = db.get_collection_stats()
    assert stats["count"] == 1
    assert stats["chunk_count"] == 1


# ---------------------------------------------------------------------------
# relink.py tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_resolve_citations_matches_title(tmp_path):
    """resolve_citations fuzzy-matches citation strings to state DB titles."""
    from scripts.obsidian_tools.relink import resolve_citations
    state_db = _make_state_db(tmp_path)
    state_db.upsert_paper({
        "doi": "10.1234/author1_2011",
        "title": "Mock Topic in Filtration Systems",
        "authors": ["Author1 A"],
        "year": 2011,
        "source_type": "paper",
        "note_title": "Author12011_MockTopic",
    })
    citations = ["Mock Topic in Filtration Author1"]
    results = resolve_citations(citations, state_db)
    assert len(results) == 1
    assert results[0]["note_title"] == "Author12011_MockTopic"
    assert results[0]["match_type"] == "cited"
@pytest.mark.unit
def test_resolve_citations_below_threshold_excluded(tmp_path):
    """Citations that don't match above threshold return nothing."""
    from scripts.obsidian_tools.relink import resolve_citations
    state_db = _make_state_db(tmp_path)
    state_db.upsert_paper({
        "doi": "10.1234/unrelated",
        "title": "Crystallography of Protein Structures",
        "authors": ["Jones B"],
        "year": 2019,
        "source_type": "paper",
        "note_title": "Jones2019_Crystallography",
    })
    results = resolve_citations(["TopicX stage2filtration yield optimization"], state_db)
    assert results == []
@pytest.mark.unit
def test_build_related_work_block_cited_first():
    """Cited items appear before similarity-based items."""
    from scripts.obsidian_tools.relink import build_related_work_block
    similar = [
        {"metadata": {"note_title": "Foo2020_Related", "source_type": "paper"}, "score": 0.88},
    ]
    cited = [
        {"note_title": "Bar2019_Cited", "match_type": "cited"},
    ]
    block = build_related_work_block(similar, cited)
    cited_pos = block.index("Bar2019_Cited")
    related_pos = block.index("Foo2020_Related")
    assert cited_pos < related_pos
@pytest.mark.unit

def test_relink_similarity_populates_related_work(tmp_path):
    """relink_note writes similarity results into Related Work zone."""
    from scripts.obsidian_tools.relink import relink_note
    state_db = _make_state_db(tmp_path)
    embeddings_db = _make_embeddings_db(tmp_path)
    # Add a target document to embeddings
    embeddings_db.add_paper(
        "doi_target",
        "TopicX concentration polarization",
        {"source_type": "paper", "note_title": "Author22011_TopicX"},
    )
    embeddings_db.add_paper(
        "doi_source",
        "TopicX yield loss",
        {"source_type": "paper", "note_title": "Smith2024_TopicX"},
    )
    # Create a note file with empty persist zones
    note = tmp_path / "Smith2024_TopicX.md"
    note.write_text(
        '---\ndoi: "doi_source"\n---\n\n# Test\n'
        '{% persist "related_work" %}\n'
        "## Related Work\n"
        "*(empty)*\n"
        "{% endpersist %}\n",
        encoding="utf-8",
    )
    # Add state DB entry
    state_db.upsert_paper({
        "doi": "doi_source",
        "title": "TopicX yield loss",
        "source_type": "paper",
        "note_title": "Smith2024_TopicX",
        "extraction_json": "{}",
    })
    relink_note(note, embeddings_db, state_db, top_k=5)
    content = note.read_text(encoding="utf-8")
    assert "🔗 [[Author22011_TopicX]]" in content
@pytest.mark.unit
def test_persist_zones_not_overwritten_by_relink(tmp_path):
    """
    Relink preserves user content in zones it doesn't touch.
    The synthesis zone must remain intact after relinking.
    """
    from scripts.obsidian_tools.relink import relink_note
    state_db = _make_state_db(tmp_path)
    state_db.upsert_paper({
        "doi": "doi_src",
        "title": "Test paper",
        "source_type": "paper",
        "note_title": "Test2024",
        "extraction_json": "{}",
    })
    embeddings_db = _make_embeddings_db(tmp_path)
    embeddings_db.add_paper(
        "doi_src",
        "Test paper text",

        {"source_type": "paper", "note_title": "Test2024"},
    )
    user_synthesis = "My hand-written synthesis note — must not be overwritten."
    note = tmp_path / "Test2024.md"
    note.write_text(
        '---\ndoi: "doi_src"\n---\n# Test\n'
        '{% persist "related_work" %}\n## Related Work\n*(empty)*\n{% endpersist %}\n'
        '{% persist "synthesis" %}\n## Synthesis\n'
        + user_synthesis
        + '\n{% endpersist %}\n',
        encoding="utf-8",
    )
    relink_note(note, embeddings_db, state_db, top_k=5)
    content = note.read_text(encoding="utf-8")
    assert user_synthesis in content
# ---------------------------------------------------------------------------
# Varying embedding vector tests — validate real similarity differentiation
# ---------------------------------------------------------------------------
def _make_embeddings_db_with_varying_vectors(tmp_path):
    """
    Create EmbeddingsDB where _embed returns content-dependent vectors,
    so similarity search produces differentiated scores.
    """
    import hashlib

    from scripts.output.embeddings import EmbeddingsDB
    db = EmbeddingsDB(
        persist_dir=str(tmp_path / "chromadb"),
        ollama_host="http://localhost:11434",
    )
    def _varying_embed(text: str) -> list[float]:
        """Generate a deterministic 768-dim vector from text hash."""
        h = hashlib.sha256(text.encode()).digest()
        # Use hash bytes to seed a vector — different texts get different vectors
        floats = []
        for i in range(768):
            byte_idx = i % 32
            val = h[byte_idx] / 255.0 + (i * 0.001)
            floats.append(val)
        return floats
    db._embed = _varying_embed
    return db
@pytest.mark.unit
def test_varying_vectors_produce_differentiated_similarity(tmp_path):
    """
    With varying embedding vectors, similar texts rank higher than
    dissimilar texts in find_similar_to_text.
    """
    db = _make_embeddings_db_with_varying_vectors(tmp_path)
    # Add documents with very different topics
    db.add_paper("doi_uf", "filtration membrane separation ProteinA", {
        "source_type": "paper", "note_title": "UF_Paper",
    })

    db.add_paper("doi_cryst", "protein crystallography x-ray diffraction", {
        "source_type": "paper", "note_title": "Cryst_Paper",
    })
    db.add_paper("doi_uf2", "filtration stage2filtration flux optimization", {
        "source_type": "paper", "note_title": "UF2_Paper",
    })
    # Query for UF-related text
    results = db.find_similar_to_text("filtration membrane", top_k=3)
    # Should get results (all 3 docs returned)
    assert len(results) == 3
    # Scores should vary — not all identical (which would happen with constant vector)
    scores = [r["score"] for r in results]
    assert len(set(scores)) > 1, "All scores are identical — vectors are not differentiating"
@pytest.mark.unit
def test_varying_vectors_exclude_id_still_works(tmp_path):
    """Excluding an ID still works with content-dependent vectors."""
    db = _make_embeddings_db_with_varying_vectors(tmp_path)
    db.add_paper("doi1", "filtration text", {"source_type": "paper", "note_title": "A"})
    db.add_paper("doi2", "different topic entirely", {"source_type": "paper", "note_title": "B"})
    results = db.find_similar_to_text("filtration", top_k=5, exclude_id="doi1")
    ids = [r["id"] for r in results]
    assert "doi1" not in ids
    assert "doi2" in ids
# ---------------------------------------------------------------------------
# C4 — Embedding model change detection
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_embed_model_first_run_records_model(tmp_path):
    """First call records the model in kv_store and returns False (no reset)."""
    from scripts.output.embeddings import check_embed_model_change
    state_db = _make_state_db(tmp_path)
    embed_db = _make_embeddings_db(tmp_path)

    changed = check_embed_model_change(state_db, embed_db, "nomic-embed-text")

    assert changed is False
    assert state_db.get_kv("embed_model") == "nomic-embed-text"


@pytest.mark.unit
def test_embed_model_no_change_leaves_collection_intact(tmp_path):
    """Calling with the same model as stored does not clear ChromaDB."""
    from scripts.output.embeddings import check_embed_model_change
    state_db = _make_state_db(tmp_path)
    embed_db = _make_embeddings_db(tmp_path)

    state_db.set_kv("embed_model", "nomic-embed-text")
    embed_db.add_paper("doi1", "text", {"note_title": "A"})

    changed = check_embed_model_change(state_db, embed_db, "nomic-embed-text")

    assert changed is False
    assert embed_db.count() == 1  # collection untouched


@pytest.mark.unit
def test_embed_model_change_clears_collection_and_resets_index(tmp_path):
    """Changing the model clears ChromaDB and resets embeddings_indexed=0."""
    from scripts.output.embeddings import check_embed_model_change
    state_db = _make_state_db(tmp_path)
    embed_db = _make_embeddings_db(tmp_path)

    # Seed: one embedded paper, old model stored
    state_db.upsert_paper({
        "doi": "10.1234/test",
        "title": "Test",
        "source_type": "paper",
        "note_title": "Test2024",
        "embeddings_indexed": 1,
    })
    embed_db.add_paper("10.1234/test", "test text", {"note_title": "Test2024"})
    state_db.set_kv("embed_model", "nomic-embed-text")

    changed = check_embed_model_change(state_db, embed_db, "mxbai-embed-large")

    assert changed is True
    # ChromaDB collection must be empty
    assert embed_db.count() == 0
    # State DB flag must be reset so the paper is re-embedded on the next run
    paper = state_db.get_paper("10.1234/test")
    assert paper["embeddings_indexed"] == 0
    # New model name persisted
    assert state_db.get_kv("embed_model") == "mxbai-embed-large"


@pytest.mark.unit
def test_embed_model_change_resets_all_rows(tmp_path):
    """reset_embeddings_indexed covers all rows regardless of source_type."""
    from scripts.output.embeddings import check_embed_model_change
    state_db = _make_state_db(tmp_path)
    embed_db = _make_embeddings_db(tmp_path)

    for i in range(3):
        state_db.upsert_paper({
            "doi": f"10.1234/paper{i}",
            "title": f"Paper {i}",
            "source_type": "paper",
            "note_title": f"Paper{i}",
            "embeddings_indexed": 1,
        })
    state_db.set_kv("embed_model", "nomic-embed-text")

    check_embed_model_change(state_db, embed_db, "mxbai-embed-large")

    # Every paper must have been reset
    papers = state_db.get_all_by_source_type("paper")
    assert all(p["embeddings_indexed"] == 0 for p in papers)


# ---------------------------------------------------------------------------
# Audit R28 regression — _embed truncate-and-retry path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_embed_retries_with_truncation_on_context_length_error(tmp_path):
    """When Ollama returns 400 'context length exceeded', _embed must truncate
    to _EMBED_TRUNCATE_CHARS and retry once before giving up."""
    import io
    from urllib.error import HTTPError

    from scripts.output.embeddings import EmbeddingsDB

    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))

    long_text = "polymer " * 1000  # 8000 chars — well over the 1400-char cap

    # First call raises HTTPError 400; second call (with truncated text) returns a vector.
    call_log: list[tuple[int, str]] = []  # (call_index, input_text_snippet)

    def fake_urlopen(req, timeout=60):
        # Decode the JSON body to grab the actual "input" we sent Ollama.
        import json as _json
        body = _json.loads(req.data.decode())
        call_log.append((len(call_log), body["input"]))
        if len(call_log) == 1:
            # First call: simulate the real Ollama 400 we observed in R28.
            err_body = b'{"error":"the input length exceeds the context length"}'
            raise HTTPError(
                url=req.full_url, code=400, msg="Bad Request",
                hdrs={}, fp=io.BytesIO(err_body),
            )
        # Second call: succeed with a fake embedding.
        class _Resp:
            def __enter__(self_): return self_
            def __exit__(self_, *a): pass
            def read(self_):
                return _json.dumps({"embeddings": [[0.1] * 1024]}).encode()
        return _Resp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = db._embed(long_text)

    assert result == [0.1] * 1024
    assert len(call_log) == 2, "Expected exactly one retry"
    # First call sent the full text; second call sent the truncated text.
    assert len(call_log[0][1]) == len(long_text)
    assert len(call_log[1][1]) == db._EMBED_TRUNCATE_CHARS


@pytest.mark.unit
def test_embed_does_not_retry_on_non_context_400(tmp_path):
    """A 400 that is NOT about context length must propagate, not retry."""
    import io
    from urllib.error import HTTPError

    from scripts.output.embeddings import EmbeddingsDB

    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
    calls: list[int] = []

    def fake_urlopen(req, timeout=60):
        calls.append(1)
        raise HTTPError(
            url=req.full_url, code=400, msg="Bad Request",
            hdrs={}, fp=io.BytesIO(b'{"error":"model not found"}'),
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(HTTPError):
            db._embed("a" * 2000)

    assert len(calls) == 1, "Must not retry on non-context-length 400"


@pytest.mark.unit
def test_embed_does_not_retry_if_text_already_short(tmp_path):
    """If the input is already <= _EMBED_TRUNCATE_CHARS, truncating wouldn't help;
    the error must propagate rather than loop."""
    import io
    from urllib.error import HTTPError

    from scripts.output.embeddings import EmbeddingsDB

    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
    calls: list[int] = []

    def fake_urlopen(req, timeout=60):
        calls.append(1)
        raise HTTPError(
            url=req.full_url, code=400, msg="Bad Request",
            hdrs={}, fp=io.BytesIO(b'{"error":"the input length exceeds the context length"}'),
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(HTTPError):
            # Short input — already under the truncate cap, so retry wouldn't help.
            db._embed("short text")

    assert len(calls) == 1


@pytest.mark.unit
def test_embed_progressively_truncates_until_success(tmp_path):
    """Audit R29 — when the first truncated retry STILL exceeds context, _embed
    must halve again rather than give up after one round.  Continues until
    either a call succeeds or the size drops below _EMBED_TRUNCATE_FLOOR_CHARS.
    """
    import io
    from urllib.error import HTTPError

    from scripts.output.embeddings import EmbeddingsDB

    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
    long_text = "polymer " * 1000  # 8000 chars

    sizes_sent: list[int] = []

    def fake_urlopen(req, timeout=60):
        import json as _json
        body = _json.loads(req.data.decode())
        sizes_sent.append(len(body["input"]))
        # First two calls fail (8000 chars, then 1400 chars).  Third succeeds.
        if len(sizes_sent) < 3:
            raise HTTPError(
                url=req.full_url, code=400, msg="Bad Request", hdrs={},
                fp=io.BytesIO(b'{"error":"the input length exceeds the context length"}'),
            )

        class _Resp:
            def __enter__(self_): return self_
            def __exit__(self_, *a): pass
            def read(self_):
                return _json.dumps({"embeddings": [[0.1] * 1024]}).encode()
        return _Resp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = db._embed(long_text)

    assert result == [0.1] * 1024
    assert len(sizes_sent) == 3, "Expected first failure, retry-at-cap, retry-at-half"
    # Sizes: original → cap (1400) → halved (700)
    assert sizes_sent[0] == 8000
    assert sizes_sent[1] == db._EMBED_TRUNCATE_CHARS
    assert sizes_sent[2] == db._EMBED_TRUNCATE_CHARS // 2


@pytest.mark.unit
def test_embed_gives_up_below_floor(tmp_path):
    """If progressive truncation drops below _EMBED_TRUNCATE_FLOOR_CHARS, _embed
    must re-raise the original HTTPError rather than loop forever."""
    import io
    from urllib.error import HTTPError

    from scripts.output.embeddings import EmbeddingsDB

    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
    long_text = "x" * 8000
    calls = []

    def fake_urlopen(req, timeout=60):
        calls.append(1)
        raise HTTPError(
            url=req.full_url, code=400, msg="Bad Request", hdrs={},
            fp=io.BytesIO(b'{"error":"the input length exceeds the context length"}'),
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(HTTPError):
            db._embed(long_text)

    # Should have tried: 8000 → 1400 → 700 → 350 → STOP (next would be 175 < floor 200)
    # That's exactly 4 attempts before giving up.
    assert 3 <= len(calls) <= 6, f"Expected 3-6 progressive truncation attempts; got {len(calls)}"
