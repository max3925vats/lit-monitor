"""Sanitise external text before it enters an LLM prompt.

External text means anything sourced from outside the codebase — paper
titles, abstracts, DOIs, user-supplied topic strings, markdown attachments
from Zotero, note content. The helper is intentionally conservative; it
does NOT try to detect prompt-injection attacks, only to remove
mechanically-confusing markers that could change how a downstream model
interprets the prompt boundary.
"""
from __future__ import annotations

import re

# Run-of-3-or-more newlines collapse to two; visually separates without
# letting a payload push the original prompt out of context.
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Role markers used by chat-formatted models (Llama, ChatML, etc.).
_ROLE_MARKERS = re.compile(
    r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>",
    flags=re.IGNORECASE,
)


def sanitize_for_prompt(text: str | None, strip_code_fences: bool = True) -> str:
    """Return a copy of *text* safe to f-string into an LLM prompt.

    Removes ChatML / Llama role markers (the actual prompt-injection breakout
    vector) and collapses runs of newlines. Optionally removes triple-backtick
    fences. Returns "" for None or empty input. Does NOT strip leading/trailing
    whitespace from normal content.

    Parameters
    ----------
    strip_code_fences:
        When True (default), also removes ALL triple-backticks, including
        legitimate code fences. Correct for short metadata (titles, keywords,
        domain_context) which should not carry real code blocks.

        When False, PRESERVES ` ``` ` fences while still stripping role markers.
        Use for paper FULLTEXT, where fence-stripping would destroy legitimate
        methods/algorithm code blocks and degrade extraction quality. The
        high-value defence (role-marker removal) still applies.
    """
    if not text:
        return ""
    s = str(text)
    if strip_code_fences:
        s = s.replace("```", "")
    s = _ROLE_MARKERS.sub("", s)
    s = _EXCESS_NEWLINES.sub("\n\n", s)
    return s
