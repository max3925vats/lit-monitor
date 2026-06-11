"""
Keyword normalisation and vocabulary theme assignment.

normalise_keywords: groups raw keyword strings into canonical clusters using
  rapidfuzz near-duplicate deduplication (threshold 88).

assign_themes: maps a list of keyword strings to theme names from the vocabulary
  file (config/concepts.yaml, fallback config/concepts_draft.yaml) using fuzzy
  matching. Returns [] when the vocab file is absent — callers should not crash.
"""
from __future__ import annotations

import logging
from pathlib import Path

from rapidfuzz import fuzz

from lit_monitor.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)

# Candidate vocabulary files in preference order (user-reviewed copy first)
_VOCAB_CANDIDATES = [
    Path("config/concepts.yaml"),
    Path("config/concepts_draft.yaml"),
]

# One-shot flag so the draft-fallback warning fires only once per process.
_warned_concepts_fallback: bool = False
# One-shot flag so a corrupted vocab file warns once, not once per paper.
_warned_vocab_corruption: bool = False
# Similarity threshold for near-duplicate deduplication (0-100 scale).
# 88 collapses obvious spelling variants (acronym vs. lowercase, capitalised
# vs. uncapitalised) while preserving truly distinct terms.
_THRESHOLD = 88
def normalise_keywords(raw: list[str]) -> dict[str, list[str]]:
    """
    Group near-duplicate keyword strings into canonical clusters.
    Parameters
    ----------
    raw:
        Flat list of keyword strings (may include duplicates, mixed case,
        extra whitespace).
    Returns
    -------
    dict[str, list[str]]
        Mapping from canonical form → list of original variant strings.
        Canonical form is the lexicographically first variant in each cluster,
        giving deterministic output across runs.
    """
    if not raw:
        return {}
    # Normalise before deduplication: strip + lower-case, remove empty
    normalised = sorted({k.strip().lower() for k in raw if k.strip()})
    if not normalised:
        return {}
    # Union-Find (greedy clustering): for each string, merge into the
    # earliest representative whose similarity score >= threshold.
    # Result is deterministic because `normalised` is sorted.
    representative: dict[str, str] = {}  # term → canonical
    for term in normalised:
        # Find an existing representative that is near-duplicate of term
        matched = None
        for rep in representative.values():
            if fuzz.ratio(term, rep) >= _THRESHOLD:
                matched = rep
                break
        if matched is None:
            matched = term  # become its own representative
        representative[term] = matched

    # Invert: canonical → [all variants]
    clusters: dict[str, list[str]] = {}
    for term, rep in representative.items():
        clusters.setdefault(rep, []).append(term)
    result: dict[str, list[str]] = {rep: sorted(variants) for rep, variants in
    sorted(clusters.items())}
    logger.info(
        "Normalised %d raw keywords → %d canonical clusters (threshold=%d)",
        len(normalised), len(result), _THRESHOLD,
    )
    return result


def assign_themes(
    keywords: list[str],
    vocab_path: Path | str | None = None,
    threshold: int = 85,
) -> list[str]:
    """Map keyword strings to vocabulary theme names via fuzzy matching.

    Loads the vocabulary file (config/concepts.yaml first, then
    config/concepts_draft.yaml) and returns the names of all themes where at
    least one paper keyword fuzzy-matches a theme keyword above the threshold.

    Returns [] — without raising — when the vocabulary file is absent or empty,
    so ingestion pipelines work before ``build-vocabulary`` has been run.

    Parameters
    ----------
    keywords:
        Keyword strings from the paper (typically Zotero tags).
    vocab_path:
        Override path to a concepts YAML file. Useful in tests.
    threshold:
        rapidfuzz ratio score (0–100) required to count as a match.
        Default 85 is looser than the dedup threshold (88) to allow for
        minor phrasing differences between Zotero tags and vocab entries.
    """
    import yaml  # import here — yaml not needed in the hot module-load path

    if not keywords:
        return []

    # Resolve vocabulary file path
    global _warned_concepts_fallback
    resolved: Path | None = None
    if vocab_path is not None:
        resolved = Path(vocab_path)
        if not resolved.exists():
            logger.debug("Specified vocab file not found: %s — skipping theme assignment", resolved)
            return []
    else:
        for i, candidate in enumerate(_VOCAB_CANDIDATES):
            if candidate.exists():
                resolved = candidate
                # N8b: warn once when falling back to the draft file — the user
                # hasn't yet promoted concepts_draft.yaml → concepts.yaml.
                if i > 0 and not _warned_concepts_fallback:
                    _warned_concepts_fallback = True
                    logger.warning(
                        "assign_themes: config/concepts.yaml not found — "
                        "falling back to %s. "
                        "Promote when ready: "
                        "cp config/concepts_draft.yaml config/concepts.yaml",
                        candidate,
                    )
                break

    if resolved is None:
        logger.debug("No vocabulary file found — theme assignment skipped")
        return []

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            vocab = yaml.safe_load(fh)
    except Exception as exc:
        strict_fallback(
            logger,
            f"Failed to load vocabulary file {resolved}: {exc}. "
            "Theme assignment will be skipped for all papers in this run.",
            exc,
        )
        return []

    themes_data = (vocab or {}).get("themes", [])
    if not themes_data:
        return []

    # Normalise paper keywords once for all comparisons
    paper_kwds = [k.strip().lower() for k in keywords if k.strip()]
    if not paper_kwds:
        return []

    matched: list[str] = []
    malformed_themes: list[str] = []
    skipped_kw_counts: dict[str, int] = {}
    for idx, theme in enumerate(themes_data):
        if not isinstance(theme, dict):
            malformed_themes.append(f"index {idx} (not a mapping: {type(theme).__name__})")
            continue
        theme_name = theme.get("name", "")
        raw_kwds = theme.get("keywords", []) or []
        # Defensive: a corrupted topic-merger append (Audit R28) can produce
        # dict-shaped keyword entries instead of strings.  Count and report
        # non-string skips so the user sees the corruption rather than
        # silently mis-classifying papers.
        valid_kwds: list[str] = []
        non_string_count = 0
        for tk in raw_kwds:
            if isinstance(tk, str) and tk.strip():
                valid_kwds.append(tk.strip().lower())
            elif tk is not None:
                non_string_count += 1
        if non_string_count:
            skipped_kw_counts[theme_name or f"<unnamed#{idx}>"] = non_string_count
        if not theme_name or not valid_kwds:
            if not theme_name:
                malformed_themes.append(f"index {idx} (missing 'name' field)")
            continue
        theme_kwds = valid_kwds

        # A theme matches if ANY paper keyword is a near-duplicate of ANY theme keyword
        hit = any(
            fuzz.ratio(pk, tk) >= threshold
            for pk in paper_kwds
            for tk in theme_kwds
        )
        if hit:
            matched.append(theme_name)

    if matched:
        logger.debug("Assigned %d theme(s): %s", len(matched), matched)

    # One-shot WARNING per process if the vocab file has structural issues.
    # Without this, an M4 indentation bug (Audit R26) or a hand-edited corruption
    # would silently degrade theme assignment for every paper.
    global _warned_vocab_corruption
    if (malformed_themes or skipped_kw_counts) and not _warned_vocab_corruption:
        _warned_vocab_corruption = True
        if malformed_themes:
            logger.warning(
                "assign_themes: %s has %d malformed theme entr%s — %s. "
                "Affected themes are skipped silently for subsequent papers.",
                resolved, len(malformed_themes),
                "y" if len(malformed_themes) == 1 else "ies",
                "; ".join(malformed_themes[:5]),
            )
        if skipped_kw_counts:
            sample = ", ".join(
                f"{name!r}: {n} non-string"
                for name, n in list(skipped_kw_counts.items())[:5]
            )
            logger.warning(
                "assign_themes: %s has themes with non-string keyword entries: %s. "
                "Run `python -c 'import yaml; print(yaml.safe_load(open(\"%s\")))' | head` "
                "to inspect; the M4 topic-merger indentation bug typically causes this.",
                resolved, sample, resolved,
            )
    return matched
