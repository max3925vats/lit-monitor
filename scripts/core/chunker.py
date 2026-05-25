"""
Markdown text chunker for chunk-level RAG indexing.

``chunk_markdown(text, target_tokens=320)`` splits a paper's markdown body
into overlapping, section-aligned chunks suitable for ChromaDB indexing.

Sizing: 1 token ≈ 3.5 chars for scientific text (technical terms fragment into
1.5–1.8 BERT subword tokens per word, denser than general English).  The
default ``target_tokens=320`` keeps every chunk safely under
``mxbai-embed-large``'s hard 512-token input ceiling — empirically verified
via Audit R28 (2026-05-18) after observing ``"input length exceeds the
context length"`` 400 errors on 500-token chunks of polymer-chemistry text.

No external tokeniser is used; character counts keep the dep footprint zero.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Conservative for scientific / technical text — BERT subword splits push the
# average up versus the ~4-chars-per-token figure that holds for general prose.
_CHARS_PER_TOKEN = 3.5
_HEADING_RE = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)


@dataclass
class Chunk:
    """A single content chunk from a paper's markdown body."""

    chunk_id: str          # e.g. "10.1234/foo#chunk-3"
    doi: str
    text: str
    section_heading: str   # nearest preceding heading, or "" if none
    chunk_index: int       # 0-based index within the paper
    total_chunks: int      # total number of chunks for this paper (filled in after splitting)
    char_start: int        # character offset in original text
    char_end: int          # character offset in original text

    metadata: dict = field(default_factory=dict)


def chunk_markdown(
    text: str,
    doi: str,
    target_tokens: int = 320,
) -> list[Chunk]:
    """
    Split markdown text into section-aligned chunks for ChromaDB indexing.

    Strategy:
    1. Try to split on heading boundaries first.  Any section that fits within
       target_tokens chars is kept whole.  Sections longer than the target are
       split further on paragraph breaks.
    2. If the whole text has no headings, fall back to paragraph-break splits.
    3. Adjacent short sections are merged until the merged block would exceed
       the target.

    Parameters
    ----------
    text:
        Full markdown content of the paper.
    doi:
        Paper DOI — embedded in chunk_id and metadata.
    target_tokens:
        Approximate token budget per chunk (1 token ≈ 4 chars).
        Default 500 ≈ 2000 chars.

    Returns
    -------
    list[Chunk]
        Ordered list of chunks; empty list if text is blank.
    """
    if not text or not text.strip():
        return []

    target_chars = int(target_tokens * _CHARS_PER_TOKEN)  # default 320 * 3.5 = 1120

    # --- Step 1: split into (heading, body) sections ---
    sections = _split_into_sections(text)

    # --- Step 2: subdivide over-long sections on paragraph breaks ---
    raw_chunks: list[tuple[str, str]] = []  # (heading, text)
    for heading, body in sections:
        if len(body) <= target_chars:
            raw_chunks.append((heading, body))
        else:
            for para_chunk in _split_on_paragraphs(body, target_chars):
                raw_chunks.append((heading, para_chunk))

    # --- Step 3: merge short adjacent sections with the same heading prefix ---
    merged = _merge_short_chunks(raw_chunks, target_chars)

    # --- Step 4: build Chunk objects ---
    total = len(merged)
    chunks: list[Chunk] = []
    offset = 0
    for i, (heading, chunk_text) in enumerate(merged):
        start = text.find(chunk_text, offset)
        if start == -1:
            # fallback: approximate position
            start = offset
        end = start + len(chunk_text)
        chunk_id = f"{doi}#chunk-{i}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doi=doi,
            text=chunk_text.strip(),
            section_heading=heading,
            chunk_index=i,
            total_chunks=total,
            char_start=start,
            char_end=end,
            metadata={
                "doi": doi,
                "section_heading": heading,
                "chunk_index": i,
            },
        ))
        offset = max(offset, start)

    # Back-fill total_chunks (may have changed during merge)
    for c in chunks:
        c.total_chunks = total

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split markdown text into (heading, body) pairs using heading boundaries.
    The text before the first heading is returned with heading="".
    """
    heading_spans = [(m.start(), m.group()) for m in _HEADING_RE.finditer(text)]
    if not heading_spans:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    # text before the first heading
    pre = text[: heading_spans[0][0]].strip()
    if pre:
        sections.append(("", pre))

    for idx, (start, heading_line) in enumerate(heading_spans):
        end = heading_spans[idx + 1][0] if idx + 1 < len(heading_spans) else len(text)
        body = text[start + len(heading_line) : end].strip()
        heading = heading_line.strip()
        if body:
            sections.append((heading, body))

    return sections if sections else [("", text)]


def _split_on_paragraphs(text: str, max_chars: int) -> list[str]:
    """
    Split text on blank lines into chunks of at most max_chars each.
    Consecutive short paragraphs are greedily merged.

    Any single paragraph longer than max_chars is sub-split on sentence
    boundaries (``. `` / ``? `` / ``! ``).  A residual sentence longer than
    max_chars is hard-sliced — last-resort fallback so the embed call never
    sees a payload exceeding the model's context.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Oversized paragraph — flush whatever has accumulated, then sub-split it.
        if len(para) > max_chars:
            _flush()
            chunks.extend(_split_on_sentences(para, max_chars))
            continue
        if current_len + len(para) > max_chars and current_parts:
            _flush()
        current_parts.append(para)
        current_len += len(para) + (2 if current_parts else 0)  # \n\n join cost

    _flush()
    return chunks if chunks else [text]


def _split_on_sentences(text: str, max_chars: int) -> list[str]:
    """
    Split a single over-long paragraph at sentence boundaries.

    A sentence that itself exceeds ``max_chars`` is hard-sliced into
    ``max_chars``-sized windows.  Sentence boundaries are detected with a
    conservative regex that requires ``[.?!]`` followed by whitespace + a
    capital letter — avoids splitting on abbreviations like "Fig. 3" or
    decimal numbers ("0.5").
    """
    # Greedy split on sentence-final punctuation followed by a space + capital.
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z(])", text)
    out: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in parts:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            # Flush accumulated then hard-slice the giant sentence.
            if current:
                out.append(" ".join(current))
                current, current_len = [], 0
            for i in range(0, len(sent), max_chars):
                out.append(sent[i : i + max_chars])
            continue
        if current_len + len(sent) > max_chars and current:
            out.append(" ".join(current))
            current, current_len = [sent], len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1  # space separator

    if current:
        out.append(" ".join(current))
    return out


def _merge_short_chunks(
    raw: list[tuple[str, str]],
    target_chars: int,
) -> list[tuple[str, str]]:
    """
    Greedily merge adjacent chunks that are below target size.
    Adjacent chunks with the same heading are preferred merge candidates.
    Never merges across headings when the combined result exceeds target_chars.
    """
    if not raw:
        return []

    merged: list[tuple[str, str]] = []
    acc_heading, acc_text = raw[0]

    for heading, text in raw[1:]:
        combined_len = len(acc_text) + len(text)
        same_heading = heading == acc_heading
        if combined_len <= target_chars and (same_heading or len(acc_text) < target_chars // 2):
            # merge: keep the section heading of whichever part is larger
            if len(text) > len(acc_text):
                acc_heading = heading
            acc_text = acc_text + "\n\n" + text
        else:
            merged.append((acc_heading, acc_text))
            acc_heading, acc_text = heading, text

    merged.append((acc_heading, acc_text))
    return merged
