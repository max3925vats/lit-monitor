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

Offset tracking (A9, 2026-05-26): every split pass produces ``Span`` records
that carry the fragment's ``(char_start, char_end)`` in the *original* source.
``chunk_markdown`` builds ``Chunk`` objects from contiguous source slices so
that ``text[chunk.char_start:chunk.char_end] == chunk.text`` always holds —
even when repeated boilerplate (e.g. "These authors contributed equally")
would have confused the old ``text.find()`` lookup.
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


@dataclass(frozen=True)
class _Span:
    """A text fragment paired with its (char_start, char_end) in the source.

    ``source[char_start:char_end] == text`` is a hard invariant; downstream
    code relies on it for offset tracking.
    """

    text: str
    char_start: int
    char_end: int


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

    # --- Step 1: split into (heading, body-span) sections ---
    sections = _split_into_sections(text)

    # --- Step 2: subdivide over-long sections on paragraph breaks ---
    raw_chunks: list[tuple[str, _Span]] = []
    for heading, body_span in sections:
        if len(body_span.text) <= target_chars:
            raw_chunks.append((heading, body_span))
        else:
            for para_span in _split_on_paragraphs(body_span, target_chars):
                raw_chunks.append((heading, para_span))

    # --- Step 3: merge short adjacent sections with the same heading prefix ---
    merged = _merge_short_chunks(text, raw_chunks, target_chars)

    # --- Step 4: build Chunk objects from contiguous source slices ---
    total = len(merged)
    chunks: list[Chunk] = []
    for i, (heading, span) in enumerate(merged):
        chunk_id = f"{doi}#chunk-{i}"
        # Slice straight from source so text[char_start:char_end] == chunk.text.
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doi=doi,
            text=text[span.char_start:span.char_end],
            section_heading=heading,
            chunk_index=i,
            total_chunks=total,
            char_start=span.char_start,
            char_end=span.char_end,
            metadata={
                "doi": doi,
                "section_heading": heading,
                "chunk_index": i,
            },
        ))

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_span(text: str, start: int, end: int) -> _Span:
    """Trim leading/trailing whitespace from ``text[start:end]`` and return a
    ``_Span`` whose offsets bracket the trimmed slice exactly.

    Returns an empty-text span (start == end) if the slice is all whitespace.
    """
    raw = text[start:end]
    lstripped = raw.lstrip()
    new_start = start + (len(raw) - len(lstripped))
    rstripped = lstripped.rstrip()
    new_end = new_start + len(rstripped)
    return _Span(text=rstripped, char_start=new_start, char_end=new_end)


def _split_into_sections(text: str) -> list[tuple[str, _Span]]:
    """
    Split markdown text into (heading, body-span) pairs using heading boundaries.
    The text before the first heading is returned with heading="".

    Body spans carry the *trimmed* source offsets so downstream slices like
    ``text[span.char_start:span.char_end]`` return exactly ``span.text``.
    """
    heading_spans = [(m.start(), m.group()) for m in _HEADING_RE.finditer(text)]
    if not heading_spans:
        whole = _strip_span(text, 0, len(text))
        return [("", whole)]

    sections: list[tuple[str, _Span]] = []
    # text before the first heading
    pre = _strip_span(text, 0, heading_spans[0][0])
    if pre.text:
        sections.append(("", pre))

    for idx, (start, heading_line) in enumerate(heading_spans):
        section_end = heading_spans[idx + 1][0] if idx + 1 < len(heading_spans) else len(text)
        body_start = start + len(heading_line)
        body = _strip_span(text, body_start, section_end)
        heading = heading_line.strip()
        if body.text:
            sections.append((heading, body))

    if sections:
        return sections
    whole = _strip_span(text, 0, len(text))
    return [("", whole)]


def _split_on_paragraphs(body: _Span, max_chars: int) -> list[_Span]:
    """
    Split a body span on blank lines into chunks of at most max_chars each.
    Consecutive short paragraphs are greedily merged.

    Any single paragraph longer than max_chars is sub-split on sentence
    boundaries (``. `` / ``? `` / ``! ``).  A residual sentence longer than
    max_chars is hard-sliced — last-resort fallback so the embed call never
    sees a payload exceeding the model's context.

    All returned spans carry true source offsets (offsets relative to the
    *whole document*, not relative to ``body``).
    """
    # Walk the body once, recording (para_start, para_end) for every non-blank
    # paragraph.  Offsets start out relative to body.text, then get lifted to
    # document coordinates by adding body.char_start.
    text = body.text
    base = body.char_start
    raw_ranges: list[tuple[int, int]] = []
    pos = 0
    for match in re.finditer(r"\n\s*\n", text):
        raw_ranges.append((pos, match.start()))
        pos = match.end()
    raw_ranges.append((pos, len(text)))

    para_spans: list[_Span] = []
    for s, e in raw_ranges:
        # Trim whitespace within the local slice; lift to document coords.
        raw = text[s:e]
        lstripped = raw.lstrip()
        local_s = s + (len(raw) - len(lstripped))
        rstripped = lstripped.rstrip()
        local_e = local_s + len(rstripped)
        if rstripped:
            para_spans.append(_Span(
                text=rstripped,
                char_start=base + local_s,
                char_end=base + local_e,
            ))

    return _group_paragraphs(para_spans, max_chars)


def _group_paragraphs(paras: list[_Span], max_chars: int) -> list[_Span]:
    """Greedily merge adjacent paragraph spans up to ``max_chars``.

    Over-long paragraphs are sub-split on sentence boundaries first.
    Returned spans are contiguous source-text slices: a merged span covers
    ``[first.char_start, last.char_end]`` so callers can slice the original
    document and get exactly the original whitespace between paragraphs.
    """
    chunks: list[_Span] = []
    current: list[_Span] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append(_Span(
                text="",  # placeholder; caller re-derives from source
                char_start=current[0].char_start,
                char_end=current[-1].char_end,
            ))
            current = []
            current_len = 0

    for para in paras:
        if len(para.text) > max_chars:
            _flush()
            # Sub-split a giant paragraph at sentence boundaries.
            chunks.extend(_split_on_sentences(para, max_chars))
            continue
        if current_len + len(para.text) > max_chars and current:
            _flush()
        current.append(para)
        # Joining cost: every additional paragraph in the same chunk adds the
        # whitespace between it and the previous one.  Track conservatively as
        # 2 chars (one "\n\n" separator) per join.
        current_len += len(para.text) + (2 if len(current) > 1 else 0)

    _flush()
    return chunks


def _split_on_sentences(para: _Span, max_chars: int) -> list[_Span]:
    """
    Split a single over-long paragraph at sentence boundaries.

    A sentence that itself exceeds ``max_chars`` is hard-sliced into
    ``max_chars``-sized windows.  Sentence boundaries are detected with a
    conservative regex that requires ``[.?!]`` followed by whitespace + a
    capital letter — avoids splitting on abbreviations like "Fig. 3" or
    decimal numbers ("0.5").

    Returned spans carry document-coordinate offsets.
    """
    # Walk para.text and record sentence (start, end) offsets relative to the
    # paragraph; then shift by para.char_start to get document coordinates.
    text = para.text
    base = para.char_start
    boundaries = [0]
    for m in re.finditer(r"(?<=[.?!])\s+(?=[A-Z(])", text):
        boundaries.append(m.end())  # next sentence starts after the whitespace
    boundaries.append(len(text))

    sent_spans: list[_Span] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        # Trim leading/trailing whitespace offset-accurately.
        raw = text[s:e]
        lstripped = raw.lstrip()
        new_s = s + (len(raw) - len(lstripped))
        rstripped = lstripped.rstrip()
        new_e = new_s + len(rstripped)
        if rstripped:
            sent_spans.append(_Span(text=rstripped, char_start=base + new_s, char_end=base + new_e))

    out: list[_Span] = []
    current: list[_Span] = []
    current_len = 0

    def _emit_current() -> None:
        nonlocal current, current_len
        if current:
            out.append(_Span(
                text="",
                char_start=current[0].char_start,
                char_end=current[-1].char_end,
            ))
            current = []
            current_len = 0

    for sent in sent_spans:
        if len(sent.text) > max_chars:
            _emit_current()
            # Hard-slice the giant sentence into max_chars windows.
            for i in range(0, len(sent.text), max_chars):
                slice_start = sent.char_start + i
                slice_end = min(sent.char_start + i + max_chars, sent.char_end)
                out.append(_Span(text="", char_start=slice_start, char_end=slice_end))
            continue
        if current_len + len(sent.text) > max_chars and current:
            _emit_current()
        current.append(sent)
        current_len += len(sent.text) + (1 if len(current) > 1 else 0)  # space join

    _emit_current()
    return out


def _merge_short_chunks(
    source: str,
    raw: list[tuple[str, _Span]],
    target_chars: int,
) -> list[tuple[str, _Span]]:
    """
    Greedily merge adjacent chunks that are below target size.
    Adjacent chunks with the same heading are preferred merge candidates.
    Same-heading chunks may merge up to ``target_chars``. Cross-heading
    chunks are merged only when the combined length is at most
    ``target_chars // 2`` — small enough that they genuinely belong together.

    Merging is offset-preserving: the merged span covers
    ``[first.char_start, last.char_end]`` of the source document.
    """
    if not raw:
        return []

    merged: list[tuple[str, _Span]] = []
    acc_heading, acc_span = raw[0]
    # Track effective length using the source slice so cross-heading
    # merge decisions account for the real whitespace between fragments.
    acc_len = acc_span.char_end - acc_span.char_start

    for heading, span in raw[1:]:
        span_len = span.char_end - span.char_start
        combined_len = acc_len + span_len
        same_heading = heading == acc_heading
        if same_heading:
            can_merge = combined_len <= target_chars
        else:
            can_merge = combined_len <= target_chars // 2
        if can_merge:
            # Keep the section heading of whichever part is larger.
            if span_len > acc_len:
                acc_heading = heading
            acc_span = _Span(
                text="",  # caller re-derives from source
                char_start=acc_span.char_start,
                char_end=span.char_end,
            )
            acc_len = acc_span.char_end - acc_span.char_start
        else:
            merged.append((acc_heading, acc_span))
            acc_heading, acc_span = heading, span
            acc_len = span_len

    merged.append((acc_heading, acc_span))
    return merged
