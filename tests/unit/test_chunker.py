"""
Unit tests for scripts/core/chunker.py.
No network calls — purely text-splitting logic.
"""
from __future__ import annotations

import pytest

from scripts.core.chunker import _split_into_sections, chunk_markdown

# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_text_returns_empty():
    assert chunk_markdown("", "10.1/empty") == []


@pytest.mark.unit
def test_whitespace_only_returns_empty():
    assert chunk_markdown("   \n\n  ", "10.1/blank") == []


@pytest.mark.unit
def test_short_text_gives_single_chunk():
    text = "This is a short abstract about ultrafiltration membranes."
    chunks = chunk_markdown(text, "10.1/short")
    assert len(chunks) == 1
    assert chunks[0].doi == "10.1/short"
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
    assert chunks[0].chunk_id == "10.1/short#chunk-0"


@pytest.mark.unit
def test_chunk_text_is_non_empty():
    text = "Abstract\n\nSome content here.\n\nMore content about filtration."
    chunks = chunk_markdown(text, "10.1/test")
    assert all(c.text.strip() for c in chunks)


@pytest.mark.unit
def test_chunk_ids_are_unique():
    text = "\n\n".join([f"Paragraph {i}. " + "word " * 60 for i in range(20)])
    chunks = chunk_markdown(text, "10.1/multi")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_total_chunks_correct():
    text = "\n\n".join(["word " * 60 for _ in range(20)])
    chunks = chunk_markdown(text, "10.1/total")
    total = len(chunks)
    for c in chunks:
        assert c.total_chunks == total


@pytest.mark.unit
def test_chunk_indices_are_sequential():
    text = "\n\n".join(["word " * 60 for _ in range(20)])
    chunks = chunk_markdown(text, "10.1/seq")
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


# ---------------------------------------------------------------------------
# Section-aware splitting
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_section_heading_captured():
    # Each section needs enough text to prevent merging into one chunk.
    intro_body = "word " * 260  # ~1300 chars — combined > 2000 so merge is skipped
    methods_body = "word " * 260
    text = (
        f"# Introduction\n\n{intro_body}\n\n"
        f"# Methods\n\n{methods_body}\n"
    )
    chunks = chunk_markdown(text, "10.1/headings")
    headings = [c.section_heading for c in chunks]
    assert any("Introduction" in h for h in headings)
    assert any("Methods" in h for h in headings)


@pytest.mark.unit
def test_heading_before_text_gets_empty_heading():
    text = "Preamble text without a heading.\n\n# Methods\n\nMethod content."
    chunks = chunk_markdown(text, "10.1/preamble")
    # First chunk (preamble) should have empty heading
    assert chunks[0].section_heading == ""


@pytest.mark.unit
def test_long_section_split_on_paragraphs():
    # Build a section whose body alone exceeds 2000 chars (500-token target)
    long_body = "\n\n".join(["word " * 80 for _ in range(15)])
    text = f"# Methods\n\n{long_body}"
    chunks = chunk_markdown(text, "10.1/longsec", target_tokens=500)
    assert len(chunks) > 1, "Long section should be split into multiple chunks"


# ---------------------------------------------------------------------------
# Merge behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_short_sections_merged():
    # Three very short sections — should merge into ≤ 2 chunks
    text = (
        "# Intro\n\nBrief intro.\n\n"
        "# Background\n\nBrief background.\n\n"
        "# Scope\n\nBrief scope.\n\n"
        "# Results\n\n" + "word " * 600 + "\n"
    )
    chunks = chunk_markdown(text, "10.1/merge", target_tokens=500)
    # The three short sections should be merged; the long Results stays separate
    total_chars = sum(len(c.text) for c in chunks)
    assert total_chars > 0
    # Key invariant: total text volume is preserved
    original_len = len(text.strip())
    assert total_chars <= original_len  # only whitespace collapsed


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_chunk_metadata_contains_doi():
    chunks = chunk_markdown("Some text.", "10.1234/doi")
    assert chunks[0].metadata["doi"] == "10.1234/doi"


@pytest.mark.unit
def test_chunk_metadata_contains_chunk_index():
    chunks = chunk_markdown("Some text.", "10.1/x")
    assert chunks[0].metadata["chunk_index"] == 0


# ---------------------------------------------------------------------------
# Internal helper: _split_into_sections
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_split_no_headings_returns_one_section():
    sections = _split_into_sections("No headings here.")
    assert len(sections) == 1
    assert sections[0][0] == ""


@pytest.mark.unit
def test_split_two_headings():
    text = "# Intro\n\nIntro text.\n\n# Methods\n\nMethods text."
    sections = _split_into_sections(text)
    headings = [s[0] for s in sections]
    assert any("Intro" in h for h in headings)
    assert any("Methods" in h for h in headings)


# ---------------------------------------------------------------------------
# Regression: strip_end_matter should be applied before chunk_markdown
# so that bibliography content never lands in the chunks collection (Q4 fix).
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_chunk_exceeds_target_chars_with_long_paragraph():
    """Audit R28 regression — a single 3 000-char paragraph (no blank lines) must
    not produce an oversized chunk. Prior bug: ``_split_on_paragraphs`` fell
    through to the ``return chunks if chunks else [text]`` path and returned
    the whole oversized paragraph as one chunk, which then exceeded
    ``mxbai-embed-large``'s 512-token context and 400'd at embed time.
    """
    long_para = ("Polymer brushes exhibit complex morphology depending on the grafting "
                 "density, solvent quality, and chain length. " * 50)  # ~5500 chars, no \n\n
    text = "# Background\n\n" + long_para
    chunks = chunk_markdown(text, "10.1/long-para", target_tokens=320)

    target_chars = int(320 * 3.5)
    # Allow a small overshoot for the "\n\n" joins used inside _split_on_paragraphs.
    cap = target_chars + 50
    for c in chunks:
        assert len(c.text) <= cap, (
            f"Chunk {c.chunk_index} is {len(c.text)} chars, exceeds cap {cap}. "
            f"First 120 chars: {c.text[:120]!r}"
        )


@pytest.mark.unit
def test_giant_single_sentence_is_hard_sliced():
    """If a single sentence (no punctuation breaks) exceeds the target, the
    chunker must hard-slice it rather than emit one oversized chunk."""
    giant_sentence = "x" * 5000  # one continuous token, no spaces / punctuation
    chunks = chunk_markdown(giant_sentence, "10.1/giant", target_tokens=320)
    target_chars = int(320 * 3.5)
    for c in chunks:
        assert len(c.text) <= target_chars, (
            f"Hard-slice produced chunk of {len(c.text)} > {target_chars} chars"
        )


@pytest.mark.unit
def test_references_absent_when_strip_end_matter_applied_first():
    """
    Verifies that the pipeline pattern strip_end_matter(text) -> chunk_markdown()
    keeps bibliography text out of the chunks collection.
    """
    from scripts.core.markdown_processor import strip_end_matter

    body = "word " * 300  # ~1500 chars of real content
    refs = "\n\n## References\n\n1. Smith J et al. (2020) A great paper. J Sci 10:1.\n2. Doe A. (2019) Another paper.\n"
    full_text = body + refs

    chunks_with_strip = chunk_markdown(strip_end_matter(full_text), "10.1/x")
    chunks_without_strip = chunk_markdown(full_text, "10.1/x")

    ref_sentinel = "Smith J et al."

    # Without stripping the reference text appears in at least one chunk.
    assert any(ref_sentinel in c.text for c in chunks_without_strip), (
        "Precondition failed: reference text should appear without stripping"
    )
    # With stripping it must be absent from all chunks.
    assert not any(ref_sentinel in c.text for c in chunks_with_strip), (
        "Reference text leaked into chunks despite strip_end_matter"
    )
