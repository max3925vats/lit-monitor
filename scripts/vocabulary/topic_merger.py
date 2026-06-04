"""
M4 — discovered_topics auto-append to topics.yaml and concepts.yaml.

At end-of-run, brain-build and discovery call merge_discovered_topics()
with the list of DOI→topics pairs collected during extraction.  Novel topics
(rapidfuzz ratio < 85 against all existing entries) are appended to:

  config/topics.yaml   — new search entry (name + simple query)
  config/concepts.yaml — new theme entry (name + seed keyword)

If either target file is absent it is silently skipped.  Only one append per
run even if multiple papers share the same novel topic.

Comments in target files are NOT preserved when python-yaml re-serialises the
whole document.  To avoid destroying existing hand-written comments, this
module splices raw YAML text directly into the list block of each file rather
than parsing + dumping the entire file.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from scripts.core.atomic_write import atomic_write_text
from scripts.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)

# Default paths — override in tests via the *_path parameters.
_TOPICS_PATH = Path("config/topics.yaml")
_CONCEPTS_PATH = Path("config/concepts.yaml")

# Minimum rapidfuzz ratio to consider a topic "already present".
_NOVELTY_THRESHOLD: float = 85.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_search_names(topics_path: Path) -> list[str]:
    """Return the list of 'name' strings from topics.yaml searches."""
    if not topics_path.exists():
        return []
    try:
        data: dict[str, Any] = yaml.safe_load(topics_path.read_text(encoding="utf-8")) or {}
        return [s["name"] for s in data.get("searches", []) if isinstance(s, dict) and "name" in s]
    except Exception as exc:
        logger.warning("merge_discovered_topics: could not read %s: %s", topics_path, exc)
        return []


def _load_theme_names(concepts_path: Path) -> list[str]:
    """Return the list of theme 'name' strings from concepts.yaml."""
    if not concepts_path.exists():
        return []
    try:
        data: dict[str, Any] = yaml.safe_load(concepts_path.read_text(encoding="utf-8")) or {}
        return [t["name"] for t in data.get("themes", []) if isinstance(t, dict) and "name" in t]
    except Exception as exc:
        logger.warning("merge_discovered_topics: could not read %s: %s", concepts_path, exc)
        return []


def _is_novel(topic: str, existing_names: list[str], already_appended: list[str]) -> bool:
    """Return True when *topic* has no fuzzy match above threshold."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Fall back to exact match if rapidfuzz is somehow unavailable.
        combined = [n.lower() for n in existing_names + already_appended]
        return topic.lower() not in combined

    needle = topic.lower()
    for name in existing_names + already_appended:
        if fuzz.ratio(needle, name.lower()) >= _NOVELTY_THRESHOLD:
            return False
    return True


def _yaml_scalar(value: str) -> str:
    """Safe inline YAML scalar without the document-end marker added by yaml.dump().

    yaml.dump() on a plain string emits 'value\\n...\\n' in some PyYAML versions.
    This helper returns the value as-is or single-quoted when YAML-special
    characters are present, so the result is valid as an inline YAML value.
    """
    if not value:
        return "''"
    needs_quoting = (
        any(c in value for c in ':#{}[]|>&*!,?')
        or value[0] in ('"', "'", '-', '?', ':', '@', '`', '%', '!')
        or value.startswith('- ')
    )
    if needs_quoting:
        return "'" + value.replace("'", "''") + "'"
    return value


def _splice_into_list_block(text: str, block_key: str, new_entry: str) -> str:
    """Insert *new_entry* after the last list item in the *block_key* sequence.

    Preserves all existing content including hand-written comments.  Does NOT
    touch any top-level scalar keys that follow the sequence (e.g.,
    ``date_window_days`` after ``searches:``).

    The item indentation is auto-detected from the first list item in the
    block so the function handles both 0-indent and 2-indent layouts.
    """
    lines = text.splitlines(keepends=True)

    # Locate the block_key line.
    block_line = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s == block_key or s.startswith(block_key + " "):
            block_line = i
            break

    if block_line == -1:
        # Key not found — fall back to plain EOF append.
        return text.rstrip("\n") + "\n" + new_entry

    # Auto-detect item indentation from the first real list item.
    item_indent = 0
    for j in range(block_line + 1, min(block_line + 20, len(lines))):
        s = lines[j].rstrip()
        if s.strip() and not s.strip().startswith("#"):
            item_indent = len(s) - len(s.lstrip())
            break

    # Scan forward to find where the sequence ends: the first non-blank,
    # non-comment line that sits at indent <= item_indent AND is NOT a list
    # item (does not start with '- ' after stripping to item_indent).
    end_line = len(lines)
    for i in range(block_line + 1, len(lines)):
        stripped = lines[i].rstrip("\n").rstrip("\r")
        if not stripped.strip():
            continue  # blank line — skip, keep scanning
        lstripped = stripped.lstrip()
        if lstripped.startswith("#"):
            continue  # comment — not a structural terminator
        effective_indent = len(stripped) - len(lstripped)
        if effective_indent > item_indent:
            pass  # sub-key of a list item — still inside the block
        elif lstripped.startswith("- "):
            pass  # another list item at the expected indent level
        else:
            # Non-list, non-comment, non-blank line at same or shallower
            # indent — this is a sibling mapping key, block has ended.
            end_line = i
            break

    return "".join(lines[:end_line]) + new_entry + "".join(lines[end_line:])


def _append_topics_entry(topics_path: Path, topic: str, doi: str, today: str) -> None:
    """Splice one search entry inside the searches: block of topics.yaml.

    Uses auto-detected indentation to match the existing list style.
    Inserts before any trailing top-level scalar keys (e.g. date_window_days)
    so the searches: list remains structurally intact.
    """
    words = [w.strip() for w in topic.lower().split() if w.strip()]
    query = " AND ".join(words) if words else topic.lower()
    # Indentation is resolved at splice time; build at 0-indent here and let
    # _splice_into_list_block detect whether the file uses 0 or 2 spaces.
    # We embed the indent after reading the file below.
    text = topics_path.read_text(encoding="utf-8")
    # Detect existing item indent to match the file's style.
    _item_indent = 0
    for line in text.splitlines():
        s = line.rstrip()
        if s.strip() and not s.strip().startswith("#") and "searches:" not in s:
            _item_indent = len(s) - len(s.lstrip())
            break
    ind = " " * _item_indent
    sub = " " * (_item_indent + 2)
    new_entry = (
        f"\n{ind}# auto-added {today} from {doi}\n"
        f"{ind}- name: {_yaml_scalar(topic)}\n"
        f"{sub}query: {_yaml_scalar(query)}\n"
        f"{sub}databases: [pubmed, arxiv]\n"
    )
    new_text = _splice_into_list_block(text, "searches:", new_entry)
    # A3-3: atomic write (temp file + fsync + os.replace) so a crash mid-write
    # can never corrupt the user's config/topics.yaml — they end up with either
    # the old file or the fully-written new one, never a truncated splice.
    atomic_write_text(topics_path, new_text)


def _append_concepts_entry(concepts_path: Path, topic: str, doi: str, today: str) -> None:
    """Splice one theme entry inside the themes: block of concepts.yaml.

    Uses auto-detected indentation to match the file's existing list style.
    Audit R28 fix: prior implementation hard-coded 2-space indent, which made
    new themes land as dict-shaped *keywords* of the last theme when the
    existing file used 0-space indent under ``themes:``.  The detection logic
    here mirrors ``_append_topics_entry`` so both vocabulary files behave
    consistently regardless of how the existing YAML was authored.
    """
    seed_keyword = topic.lower()
    text = concepts_path.read_text(encoding="utf-8")
    if text.strip().startswith("themes:"):
        # Detect existing item indent (0 or 2) so new entry matches the style.
        _item_indent = 0
        for line in text.splitlines():
            s = line.rstrip()
            if s.strip() and not s.strip().startswith("#") and "themes:" not in s:
                _item_indent = len(s) - len(s.lstrip())
                break
        ind = " " * _item_indent
        sub = " " * (_item_indent + 2)
        new_entry = (
            f"\n{ind}# auto-added {today} from {doi}\n"
            f"{ind}- name: {_yaml_scalar(topic)}\n"
            f"{sub}keywords:\n"
            f"{sub}- {_yaml_scalar(seed_keyword)}\n"
        )
        new_text = _splice_into_list_block(text, "themes:", new_entry)
    else:
        # Bare list — append at end (no trailing scalar keys expected here).
        new_entry = (
            f"\n# auto-added {today} from {doi}\n"
            f"- name: {_yaml_scalar(topic)}\n"
            f"  keywords:\n"
            f"  - {_yaml_scalar(seed_keyword)}\n"
        )
        new_text = text.rstrip("\n") + "\n" + new_entry
    # A3-3: atomic write — see _append_topics_entry. Guards config/concepts.yaml
    # against corruption from a crash partway through the write.
    atomic_write_text(concepts_path, new_text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_discovered_topics(
    doi_topics: list[tuple[str, list[str]]],
    *,
    topics_path: Path = _TOPICS_PATH,
    concepts_path: Path = _CONCEPTS_PATH,
) -> list[str]:
    """Append novel discovered_topics from a completed run to vocabulary files.

    Parameters
    ----------
    doi_topics:
        List of (doi, discovered_topics) pairs collected during the run.
        Each discovered_topics entry is the list of 3–7 noun phrases returned
        by the complex-phase LLM extraction for that paper.
    topics_path:
        Path to topics.yaml.  Defaults to config/topics.yaml.
    concepts_path:
        Path to concepts.yaml.  Defaults to config/concepts.yaml.

    Returns
    -------
    list[str]
        Topics that were actually appended (novel across all pairs and
        existing file content).
    """
    if not doi_topics:
        return []

    today = str(date.today())

    # Load existing names once — we append as we go and track what we've added.
    existing_search_names = _load_search_names(topics_path)
    existing_theme_names = _load_theme_names(concepts_path)
    already_appended: list[str] = []

    for doi, topics in doi_topics:
        if not topics:
            continue
        for topic in topics:
            if not topic or not isinstance(topic, str):
                continue
            topic = topic.strip()
            if not topic:
                continue

            # Novelty check against both file sets and already-appended topics.
            all_existing = existing_search_names + existing_theme_names
            if not _is_novel(topic, all_existing, already_appended):
                logger.debug(
                    "merge_discovered_topics: skipping %r — too similar to existing entry", topic
                )
                continue

            # P4.3: track write outcomes so we never report a topic as
            # "appended" when every attempted write actually failed. A write is
            # "attempted" only for a file that exists; if neither file exists
            # there is nothing to write to and the topic still counts as merged
            # (preserves prior behaviour for the no-files case).
            attempted = 0
            wrote = 0

            # Append to topics.yaml if it exists.
            if topics_path.exists():
                attempted += 1
                try:
                    _append_topics_entry(topics_path, topic, doi, today)
                    wrote += 1
                    logger.info(
                        "merge_discovered_topics: appended %r to %s (from %s)",
                        topic, topics_path, doi,
                    )
                except Exception as exc:
                    logger.warning(
                        "merge_discovered_topics: failed to append %r to %s: %s",
                        topic, topics_path, exc,
                    )

            # Append to concepts.yaml if it exists.
            if concepts_path.exists():
                attempted += 1
                try:
                    _append_concepts_entry(concepts_path, topic, doi, today)
                    wrote += 1
                    logger.info(
                        "merge_discovered_topics: appended %r to %s (from %s)",
                        topic, concepts_path, doi,
                    )
                except Exception as exc:
                    logger.warning(
                        "merge_discovered_topics: failed to append %r to %s: %s",
                        topic, concepts_path, exc,
                    )

            # If we attempted at least one write and every one failed, the topic
            # was NOT persisted — escalate in --strict (raises) and, in
            # non-strict, skip reporting it as appended so callers don't act on
            # a topic that never reached disk.
            if attempted > 0 and wrote == 0:
                strict_fallback(
                    logger,
                    f"merge_discovered_topics: all {attempted} write(s) failed "
                    f"for topic {topic!r} (from {doi}); not reporting as appended",
                )
                continue

            already_appended.append(topic)

    if already_appended:
        logger.info(
            "merge_discovered_topics: %d novel topic(s) appended: %s",
            len(already_appended), already_appended,
        )
    return already_appended
