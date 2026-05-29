"""G2: entity normalizer for the lit-monitor knowledge graph.

5-step pipeline (in order):
  1. lowercase + ASCII fold
  2. singularize (via inflect)
  3. punctuation normalize (strip trailing punct, collapse whitespace)
  4. alias lookup (O(1), per-type)
  5. fuzzy collapse (rapidfuzz, per-type scope — never crosses types)

Reused by G3 (entity extractor), G4 (relationship extractor),
G7 (query API), G11 (propose-aliases).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Literal

import inflect
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# Single shared inflect engine. Our pipeline is single-threaded at ingest time,
# so one global instance is safe and avoids the ~10 ms init cost per call.
_INFLECT = inflect.engine()

# Trailing punctuation to strip, and whitespace collapser.
_TRAILING_PUNCT = re.compile(r"[.,;:!?]+$")
_WHITESPACE = re.compile(r"\s+")

# Transliteration table for scientific symbols that NFKD/ASCII-encode drops
# entirely (e.g. Greek letters common in bioscience nomenclature).
# Lowercase only — applied AFTER .lower() in _basic_normalize.
#
# NOTE: μ → "u" is the Greek-letter rule (matches "mu"); it does NOT handle
# the SI prefix μ (micro-). Inputs like "μg", "μM", "μL" become "ug", "um",
# "ul" — acceptable for entity names but wrong for quantity strings. Strip
# units before normalizing if quantity accuracy matters.
_TRANSLITERATE: dict[str, str] = {
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "e",
    "θ": "th",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "u",
    "ν": "n",
    "ξ": "x",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "τ": "t",
    "υ": "u",
    "φ": "ph",
    "χ": "ch",
    "ψ": "ps",
    "ω": "o",
}
_TRANSLITERATE_TABLE = str.maketrans(_TRANSLITERATE)

MatchVia = Literal["identity", "alias", "fuzzy"]


def _basic_normalize(s: str) -> str:
    """Apply steps 1–3: lowercase + ASCII fold → punct strip → singularize.

    Order matters: inflect.singular_noun() returns False on plurals with
    trailing punctuation (e.g. 'antibodies.' → False; 'antibodies' → 'antibody').
    Strip punct BEFORE singularizing so real-text input normalizes correctly.

    This is a module-level function (not a method) so it is testable in
    isolation and reusable by other modules without instantiating the full
    EntityNormalizer class.
    """
    # Step 1: lowercase, then transliterate known scientific symbols (Greek
    # letters etc.) that NFKD decomposition cannot map to ASCII, then decompose
    # accented Latin characters (é → e) and strip any remaining non-ASCII bytes.
    s = s.lower()
    s = s.translate(_TRANSLITERATE_TABLE)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    # Step 2: strip trailing punctuation then collapse internal whitespace.
    # Must come BEFORE singularize — inflect does not recognize plurals with
    # trailing punctuation ('antibodies.' → False instead of 'antibody').
    s = _TRAILING_PUNCT.sub("", s)
    s = _WHITESPACE.sub(" ", s).strip()

    # Step 3: singularize.  singular_noun() returns the singular if the word
    # IS plural, or False when it is already singular — so we only reassign
    # when a plural form was detected.
    singular = _INFLECT.singular_noun(s)
    if singular:
        s = singular

    return s


class EntityNormalizer:
    """Deterministic entity normalization with per-type vocab scope.

    Parameters
    ----------
    aliases:
        Pre-loaded alias dict ``{type_name: {lower(surface): canonical}}``.
        Obtain via :func:`scripts.graph.aliases.load_aliases`.
    vocab_by_type:
        Optional pre-populated per-type vocabulary.  More commonly built up
        incrementally via :meth:`add_to_vocab` as entities are ingested.
    """

    # G2: ingest is stricter than query — ingest collapses on near-identical
    # surfaces only; query lets a typo or fuzzier user input still match.
    DEFAULT_INGEST_RATIO: int = 90
    DEFAULT_QUERY_RATIO: int = 85

    def __init__(
        self,
        aliases: dict[str, dict[str, str]],
        vocab_by_type: dict[str, set[str]] | None = None,
    ) -> None:
        self._aliases = aliases
        # Each type owns its own set; fuzzy collapse only ever searches within
        # the set for the requested type — NEVER across types.
        self._vocab: dict[str, set[str]] = vocab_by_type or {}

    def add_to_vocab(self, type_: str, canonical: str) -> None:
        """Register a canonical entity in the per-type vocab index.

        Call this for every entity that has been successfully committed to
        KuzuDB so the fuzzy collapse can match future near-duplicates.
        """
        self._vocab.setdefault(type_, set()).add(canonical)

    def normalize(
        self,
        surface: str,
        type_: str,
        min_ratio: int = 90,
    ) -> tuple[str, MatchVia]:
        """Normalize a surface form to its canonical representation.

        Parameters
        ----------
        surface:
            Raw entity string as extracted from text.
        type_:
            Entity type key (e.g. ``"material"``, ``"method"``). Used to
            scope both alias lookup and fuzzy collapse — never crosses types.
        min_ratio:
            Minimum rapidfuzz ``fuzz.ratio`` (0–100) required for a fuzzy
            match to be accepted.  90 at ingest, 85 at query time per spec.

        Returns
        -------
        (canonical, match_via)
            ``match_via`` is ``"identity"`` (basic pipeline only),
            ``"alias"`` (alias dict hit), or ``"fuzzy"`` (vocab collapse).
        """
        normed = _basic_normalize(surface)

        # Step 4: alias lookup.
        # Try the original surface (lowercased) first, then the post-normalized
        # form (handles cases like "mAb." → basic_normalize → "mab").
        alias_table = self._aliases.get(type_, {})
        if surface.lower() in alias_table:
            return alias_table[surface.lower()], "alias"
        if normed in alias_table:
            return alias_table[normed], "alias"

        # Step 5: fuzzy collapse — strictly within the requested type's vocab.
        vocab = self._vocab.get(type_, set())
        if vocab:
            match = process.extractOne(normed, list(vocab), scorer=fuzz.ratio)
            if match is not None and match[1] >= min_ratio:
                return match[0], "fuzzy"

        return normed, "identity"
