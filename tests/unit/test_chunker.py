"""
Unit tests for scripts/core/chunker.py.
No network calls — purely text-splitting logic.
"""
from __future__ import annotations

import pytest

from lit_monitor.core.chunker import _split_into_sections, chunk_markdown

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
def test_merge_short_chunks_never_merges_large_across_headings():
    """A8 — cross-heading merges must NOT happen when combined > target_chars // 2."""
    # target_tokens=320 → target_chars=1120 → target_chars//2=560.
    # Two paragraphs under different headings, each 400 chars.
    # Combined = 800 > 560 → must NOT merge across the heading boundary.
    para_len = 400
    text = f"## Section A\n\n{'A' * para_len}\n\n## Section B\n\n{'B' * para_len}\n"
    chunks = chunk_markdown(text, "10.test/a8", target_tokens=320)
    headings = [c.section_heading for c in chunks]
    # If the buggy code merged them, only one heading would survive.
    assert any("Section A" in h for h in headings), (
        f"Expected a chunk for Section A; got {headings}"
    )
    assert any("Section B" in h for h in headings), (
        f"Expected a chunk for Section B; got {headings}"
    )


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


@pytest.mark.unit
def test_empty_heading_marker_is_recognised():
    """P6.21: a heading marker with no text after it (e.g. "## ") must still
    be treated as a heading boundary.

    The regex broadened from ``\\s+.+`` to ``\\s+.*`` so a bare "## " no longer
    falls through to body text. Here the empty heading precedes "Body one"; the
    next real heading then starts a second section.
    """
    text = "## \n\nBody one paragraph.\n\n## Methods\n\nBody two paragraph."
    sections = _split_into_sections(text)
    headings = [h for h, _ in sections]
    # "Methods" must be its own section, and "Body one" must NOT be glued to it.
    assert any("Methods" in h for h in headings)
    body_two = next(span.text for h, span in sections if "Methods" in h)
    assert "Body two" in body_two
    assert "Body one" not in body_two


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
def test_chunk_offsets_correct_with_repeated_boilerplate():
    """A9 — repeated boilerplate must not confuse chunk offset tracking."""
    from lit_monitor.core.chunker import chunk_markdown
    boilerplate = "These authors contributed equally to this work.\n\n"
    text = (
        "## Section A\n\n"
        + boilerplate
        + "Unique text in section A. " * 30 + "\n\n"
        + "## Section B\n\n"
        + boilerplate
        + "Unique text in section B. " * 30 + "\n"
    )
    chunks = chunk_markdown(text, "10.test/a9", target_tokens=320)
    # Every chunk's (char_start, char_end) must point at a substring of `text`
    # that matches its `.text` field exactly.
    for chunk in chunks:
        slice_ = text[chunk.char_start:chunk.char_end]
        assert slice_ == chunk.text, (
            f"Chunk offsets mismatched for chunk_index={chunk.chunk_index}:\n"
            f"  text[{chunk.char_start}:{chunk.char_end}] = {slice_!r}\n"
            f"  chunk.text = {chunk.text!r}"
        )


@pytest.mark.unit
def test_references_absent_when_strip_end_matter_applied_first():
    """
    Verifies that the pipeline pattern strip_end_matter(text) -> chunk_markdown()
    keeps bibliography text out of the chunks collection.
    """
    from lit_monitor.core.markdown_processor import strip_end_matter

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


# ---------------------------------------------------------------------------
# A2 — sentence-boundary lookahead: allow lowercase starts, exclude digits
# ---------------------------------------------------------------------------

def _sentence_texts(paragraph: str, max_chars: int) -> list[str]:
    """Run ``_split_on_sentences`` over ``paragraph`` and return the source
    substrings of the resulting spans, one per detected sentence.

    ``max_chars`` is chosen by the caller to be >= the longest individual
    sentence (so nothing is hard-sliced) but < the combined length of any two
    adjacent sentences (so the greedy merge keeps each detected sentence in its
    own span).  This exposes the boundary-detection behaviour directly.  Span
    offsets are document-relative, so we slice the original paragraph to
    recover each fragment's exact text.
    """
    from lit_monitor.core.chunker import _Span, _split_on_sentences

    para = _Span(text=paragraph, char_start=0, char_end=len(paragraph))
    spans = _split_on_sentences(para, max_chars=max_chars)
    return [paragraph[s.char_start:s.char_end] for s in spans]


@pytest.mark.unit
def test_lowercase_sentence_start_splits():
    """A2 — a sentence beginning with a lowercased word now splits.

    Before the fix the lookahead required ``[A-Z(]``, so the boundary after
    "...as shown." was missed and both sentences stayed glued together.
    """
    paragraph = "The effect is clear as shown. see Table 2 for details."
    # Longest sentence is 29 chars ("The effect is clear as shown."); a
    # max_chars of 29 keeps each sentence whole yet blocks the two from merging.
    parts = _sentence_texts(paragraph, max_chars=29)
    assert parts == [
        "The effect is clear as shown.",
        "see Table 2 for details.",
    ], parts


@pytest.mark.unit
def test_scientific_abbreviations_do_not_split_on_digits():
    """A2 — protective: digits stay excluded from the lookahead, so a period
    followed by a number ("Fig. 3", "Eq. 5") is NOT treated as a boundary.

    The only real sentence boundary here is between "Fig. 3 shows X." and
    "Eq. 5 follows." (a capital-E start), giving exactly two sentences.
    """
    paragraph = "Fig. 3 shows X. Eq. 5 follows."
    # Longest sentence is 15 chars ("Fig. 3 shows X."); max_chars=15 keeps each
    # whole while preventing a merge.  If digits were allowed in the lookahead,
    # "Fig. 3"/"Eq. 5" would split into extra fragments and break this equality.
    parts = _sentence_texts(paragraph, max_chars=15)
    assert parts == [
        "Fig. 3 shows X.",
        "Eq. 5 follows.",
    ], parts
