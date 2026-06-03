"""Shared token-budgeting constants for generative LLM context sizing.

These two constants are genuinely shared (same value AND same intent) by the
extractor (scripts/llm/extractor.py) and the vocabulary clusterer
(scripts/vocabulary/clusterer.py), which both budget input characters against
an Ollama-style chat model's num_ctx window.

NOT shared here on purpose:
  - _SAFETY_FACTOR  — same numeric value (0.75) today, but different *meaning*
    per caller (extractor reserves output separately via an output-reserve
    constant; the clusterer's factor covers input+output combined), so each
    module keeps its own.
  - _FALLBACK_CTX   — diverges by caller (extractor 16384, clusterer 8192).
  - scripts/core/chunker.py uses 3.5 chars/token for a DIFFERENT (embedding)
    tokenizer and must not consume these values.
"""
from __future__ import annotations

# Rule of thumb for English scientific prose under chat-model tokenizers.
CHARS_PER_TOKEN: int = 4

# Per-call system-prompt overhead estimate (tokens) reserved before budgeting
# the user/input text.
SYSTEM_PROMPT_RESERVE_TOKENS: int = 500

__all__ = ["CHARS_PER_TOKEN", "SYSTEM_PROMPT_RESERVE_TOKENS"]
