"""
Obsidian note writer — renders Jinja2 templates and writes to vault.
Responsibilities:
- Generate stable note titles (filename = note_title + '.md')
- Handle filename collisions (append b/c/... suffix)
- Write paper, book, and chapter notes using templates
- Preserve persist zones when updating existing notes
Persist zone pattern from mgmeyers/obsidian-zotero-integration:
    {% persist "zone_name" %}
    ...user content...
    {% endpersist %}
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, nodes
from jinja2.ext import Extension

from scripts.llm.extractor import compute_confidence_score


class _PersistExtension(Extension):
    """
    Jinja2 extension for the Obsidian persist zone tag.
    Renders {% persist "name" %}...{% endpersist %} as the literal
    marker text in the output file, so the Obsidian plugin can process
    them. The body content is rendered normally by Jinja2.
    """
    tags = {"persist"}
    def parse(self, parser):
        lineno = next(parser.stream).lineno
        zone_name_node = parser.parse_expression()
        body = parser.parse_statements(["name:endpersist"], drop_needle=True)
        zone_name = zone_name_node.as_const()
        open_node = nodes.Output(
            [nodes.Const(f'{{% persist "{zone_name}" %}}')],
            lineno=lineno,
        )
        close_node = nodes.Output(
            [nodes.Const("{% endpersist %}")],
            lineno=lineno,
        )
        return [open_node] + body + [close_node]
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Persist zone regex — matches {% persist "name" %}...{% endpersist %}
# ---------------------------------------------------------------------------
_PERSIST_PATTERN = re.compile(
    r'(\{%\s*persist\s+"([^"]+)"\s*%\})(.*?)(\{%\s*endpersist\s*%\})',
    re.DOTALL,
)

# Detects ANYTHING that looks like the start of a persist marker, even
# malformed (missing quote, missing endpersist, unquoted name, empty name).
# Used by _find_malformed_persist_markers to spot user-content destruction
# risk BEFORE we overwrite a note.
_PERSIST_OPEN_LOOSE = re.compile(r'\{%\s*persist\b[^%]*%\}', re.DOTALL)
# Allowed characters in Dataview-compatible tag slugs
_TAG_STRIP = re.compile(r"[^a-z0-9/\-]")

# Jinja2 environment (lazy singleton per template_dir)
_JINJA_ENV: dict[Path, Environment] = {}
def _get_env(template_dir: Path) -> Environment:
    if template_dir not in _JINJA_ENV:
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            extensions=[_PersistExtension],
        )
        import json
        env.filters["tojson"] = lambda v: json.dumps(v)
        # regex_replace(value, pattern, replacement) — used in paper_note.md.j2
        # e.g. {{ themes | map('regex_replace', '^(.*)$', '[[\\1]]') | join(' · ') }}
        env.filters["regex_replace"] = lambda val, pat, repl: re.sub(pat, repl, str(val))
        _JINJA_ENV[template_dir] = env
    return _JINJA_ENV[template_dir]
# ---------------------------------------------------------------------------
# Note title / filename logic
# ---------------------------------------------------------------------------
def note_title_for(
    authors: list[str],
    year: int,
    short_title: str,
    suffix: str | None = None,
) -> str:
    """
    Build a stable note title string.
    Format: {FirstAuthorLastName}{Year}_{ShortTitle}
    - ALL-CAPS surnames are Title-cased: SMITH → Smith
    - Non-ASCII characters normalised to ASCII equivalents
    - Suffix for chapters: '_Ch{N:02d}'
    The return value is the note title (without .md).
    Collision handling (b/c/...) is done by write_*_note functions.
    """
    first_author = authors[0] if authors else "Unknown"
    # Strip commas before splitting — Zotero stores authors as "LastName, FirstName"
    last_name = first_author.replace(",", " ").split()[0] if first_author.strip() else "Unknown"
    # Normalise ALL-CAPS surnames
    if last_name.isupper():
        last_name = last_name.capitalize()
    # NFKD normalise to ASCII
    last_name = unicodedata.normalize("NFKD", last_name).encode("ascii", "ignore").decode()
    # Sanitise short_title: keep alphanum + space + hyphen, collapse spaces
    safe_title = re.sub(r"[^\w\s-]", "", short_title).strip()
    safe_title = re.sub(r"\s+", "_", safe_title)
    # Limit to 40 chars
    safe_title = safe_title[:40]
    title = f"{last_name}{year}_{safe_title}"
    if suffix:
        title = title + suffix
    return title
def _collision_suffix(base: str, existing: set[str]) -> str:
    """
    Return base + numeric suffix (_2, _3, ...) so result is not in existing.
    If base itself is not in existing, return base unchanged.
    """

    if base not in existing:
        return base
    counter = 2
    while counter <= 1000:
        candidate = f"{base}_{counter}"
        if candidate not in existing:
            return candidate
        counter += 1
    raise RuntimeError(f"Too many collision suffixes for note title: {base!r}")
def _existing_titles(folder: Path) -> set[str]:
    """Return stems of all .md files in folder (no extension)."""
    if not folder.exists():
        return set()
    return {p.stem for p in folder.glob("*.md")}
# ---------------------------------------------------------------------------
# Persist zone helpers
# ---------------------------------------------------------------------------
def _extract_persist_zones(content: str) -> dict[str, str]:
    """Return {zone_name: zone_content} from an existing note."""
    zones: dict[str, str] = {}
    for match in _PERSIST_PATTERN.finditer(content):
        zones[match.group(2)] = match.group(3)
    return zones


def _find_malformed_persist_markers(content: str) -> list[str]:
    """Return human-readable descriptions of every persist marker that looks
    syntactically broken (would be silently dropped on rerender).

    Detects: opening tokens that don't match the strict pattern (missing
    closing quote, missing closing %}, empty name, unquoted name) AND
    opening tokens that match but never reach a {% endpersist %} (orphan
    open). Returns [] when the note is clean — fast path for the common case.

    NB: nested persist zones are NOT flagged here. The outer zone matches
    the strict pattern (greedy non-greedy stops at the first endpersist),
    and detecting orphan inner markers requires walking each extracted
    zone's body. Audit-29 #4 deferred nested-zone hardening as out of scope.
    """
    if "{% persist" not in content and "{%persist" not in content:
        return []

    # Span-set of well-formed zones (so we can skip past them).
    well_formed_spans = [m.span() for m in _PERSIST_PATTERN.finditer(content)]

    def in_well_formed(pos: int) -> bool:
        return any(start <= pos < end for start, end in well_formed_spans)

    issues: list[str] = []
    for loose in _PERSIST_OPEN_LOOSE.finditer(content):
        if in_well_formed(loose.start()):
            continue  # this open is part of a well-formed block
        # Anything left is a malformed open. Compute the line number for context.
        line_no = content.count("\n", 0, loose.start()) + 1
        snippet = loose.group(0).strip()
        if len(snippet) > 60:
            snippet = snippet[:57] + "..."
        issues.append(f'line {line_no}: {snippet!r}')

    # Also flag opens that have no matching close anywhere after them.
    # The loose regex catches the open; we look for an "{% endpersist %}"
    # token after the loose match position. If absent → orphan open.
    for loose in _PERSIST_OPEN_LOOSE.finditer(content):
        if in_well_formed(loose.start()):
            continue
        if "{% endpersist" not in content[loose.end():]:
            line_no = content.count("\n", 0, loose.start()) + 1
            already_flagged = any(f"line {line_no}" in iss for iss in issues)
            if not already_flagged:
                issues.append(
                    f"line {line_no}: orphan opening — no {{% endpersist %}} reached"
                )

    return issues


def _apply_persist_zones(new_content: str, zones: dict[str, str]) -> str:
    """Restore saved persist zone content into a freshly rendered note."""
    def replacer(match: re.Match) -> str:
        name = match.group(2)
        if name in zones:
            return match.group(1) + zones[name] + match.group(4)
        return match.group(0)
    return _PERSIST_PATTERN.sub(replacer, new_content)


def update_note_preserve_persist_zones(path: str | Path, new_content: str) -> None:
    """Write new_content to path, preserving any existing persist zone content.

    If the existing note contains malformed persist markers (a marker that
    looks like ``{% persist ... %}`` but doesn't match the strict syntax),
    the rerender is ABORTED and a WARNING is logged listing the offending
    lines — the original file is left untouched so user content is never
    silently destroyed. Under ``--strict`` / ``LIT_MONITOR_STRICT=1``,
    the same condition raises ``RuntimeError``.

    Creates parent directories if needed.
    """
    from scripts.core.strict_mode import strict_fallback

    path = Path(path)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        malformed = _find_malformed_persist_markers(existing)
        if malformed:
            strict_fallback(
                logger,
                f"Persist marker(s) in {path} are malformed; rerender aborted "
                f"so your edits are not lost. Fix the marker syntax and retry. "
                f"Offending lines:\n  - " + "\n  - ".join(malformed),
            )
            return  # default-mode fallthrough: do NOT overwrite the file
        zones = _extract_persist_zones(existing)
        new_content = _apply_persist_zones(new_content, zones)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
# ---------------------------------------------------------------------------
# Note writers
# ---------------------------------------------------------------------------
def write_paper_note(
    paper: dict[str, Any],
    extraction: dict[str, Any],
    config: Any,
    template_dir: Path | None = None,
    schema_fields: list[dict] | None = None,
    note_path_override: str | Path | None = None,
) -> str:
    """
    Write a paper note to the vault. Returns the note path (str).
    Parameters
    ----------
    paper:
        Dict with keys: doi, title, authors (list), year, journal,
        zotero_key, first_seen_date, keywords (list), themes (list),
        tracked_author (bool), fulltext_analyzed (bool).
    extraction:
        Extraction JSON blob from state DB.
    config:
        Config object with obsidian.vault_path, obsidian.papers_folder,
        extraction_provider, extraction_model.
    template_dir:
        Override template directory (for tests).
    schema_fields:
        Override schema field list (for tests).
    note_path_override:
        Write to this exact path instead of computing a collision-safe
        filename. Used by rerender to update a note in place.
    """
    if template_dir is None:
        template_dir = Path("config/templates")
    if schema_fields is None:
        schema_fields = _DEFAULT_PAPER_SCHEMA_FIELDS
    vault_path = Path(config.obsidian.vault_path)
    folder = vault_path / config.obsidian.papers_folder
    folder.mkdir(parents=True, exist_ok=True)
    if note_path_override is not None:
        note_path = Path(note_path_override)
        note_title = note_path.stem
    else:
        short_title = _make_short_title(paper.get("title", ""))
        base_title = note_title_for(paper.get("authors", []), paper.get("year", 0), short_title)
        existing = _existing_titles(folder)
        note_title = _collision_suffix(base_title, existing)
        note_path = folder / f"{note_title}.md"
    env = _get_env(template_dir)
    template = env.get_template("paper_note.md.j2")
    keywords = paper.get("keywords", [])
    themes = paper.get("themes", [])
    tags = _make_tags(keywords, themes)
    # N8c: backfill for notes rendered from old DB rows that predate the C1 fix
    if "_overall_confidence" not in extraction:
        extraction = dict(extraction)
        extraction["_overall_confidence"] = compute_confidence_score(extraction)
    # I3: flag degraded extraction quality in tags
    if extraction.get("_extraction_quality") == "degraded_high_chunking":
        if "extraction-quality-degraded" not in tags:
            tags = list(tags) + ["extraction-quality-degraded"]
    content = template.render(
        note_title=note_title,
        doi=paper.get("doi", ""),
        title=paper.get("title", ""),
        year=_coerce_year(paper.get("year")),
        authors=paper.get("authors", []),
        journal=paper.get("journal", ""),
        zotero_key=paper.get("zotero_key", ""),
        first_seen_date=paper.get("first_seen_date", str(date.today())),
        keywords=keywords,
        themes=themes,
        tags=tags,
        tracked_author=paper.get("tracked_author", False),
        fulltext_analyzed=paper.get("fulltext_analyzed", False),
        extraction=extraction,
        # N8d: read from paper dict (populated by brain_build / discovery
        # with the actual runtime provider) — fall back to config only for
        # legacy callers that pre-date N8d (e.g. old rerender paths).
        extraction_provider=paper.get("extraction_provider") or getattr(config, "extraction_provider", ""),
        extraction_model=paper.get("extraction_model") or getattr(config, "extraction_model", ""),
        run_date=str(date.today()),
        schema_fields=schema_fields,
    )
    # I3: prepend chunk-quality callout above persist zones when quality is degraded
    if extraction.get("_extraction_quality") == "degraded_high_chunking":
        _callout = (
            "\n> [!warning] Extraction Quality Degraded\n"
            "> This paper required more than 3 chunks per extraction pass. "
            "Field completeness may be reduced — consider re-extracting with a "
            "model with a larger context window.\n\n"
        )
        if content.startswith("---"):
            _fm_end = content.find("\n---", 3)
            if _fm_end != -1:
                _insert = _fm_end + 4
                content = content[:_insert] + _callout + content[_insert:]
    update_note_preserve_persist_zones(note_path, content)
    logger.info("Wrote paper note: %s", note_path)
    return str(note_path)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_year(raw: Any) -> int:
    """Coerce year to int for Dataview-compatible YAML. Returns 0 when missing or unparseable.

    Uses explicit None/"" check rather than truthiness so the integer 0 stays consistent
    (year 0 isn't real, but the pattern avoids surprises).
    """
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


def _make_tags(keywords: list[str], themes: list[str]) -> list[str]:
    """Build sanitized Dataview tag slugs from keywords and themes.

    Rules: lowercase, spaces→hyphens, strip chars outside [a-z0-9/-],
    deduplicate (keywords first, then themes).
    Non-string items (None, int) are coerced to str to avoid AttributeError.
    """
    tags: list[str] = []
    seen: set[str] = set()
    for raw in list(keywords) + list(themes):
        # str() first — extraction can occasionally yield None or non-string values
        tag = _TAG_STRIP.sub("", str(raw).lower().strip().replace(" ", "-")).strip("-")
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def _make_short_title(title: str, max_words: int = 5) -> str:
    """Extract first max_words meaningful words from a title."""
    # Strip common leading stopwords
    stop = {"a", "an", "the", "on", "of", "in", "for", "to", "and", "or"}
    words = title.split()
    kept = [w for w in words if w.lower() not in stop][:max_words]
    return "_".join(kept) if kept else title[:30]
# Default schema field definitions (used when no config override provided)

_DEFAULT_PAPER_SCHEMA_FIELDS = [
    {"id": "core_finding",          "label": "Core Finding",         "extract": True},
    {"id": "methods_summary",       "label": "Methods Summary",      "extract": True},
    {"id": "results_summary",       "label": "Results Summary",      "extract": True},
    {"id": "conclusions",           "label": "Conclusions",          "extract": True},
    {"id": "background_motivation", "label": "Background",           "extract": True},
    {"id": "experimental_conditions","label": "Experimental Conditions","extract": True},
    {"id": "materials_systems",     "label": "Materials & Systems",  "extract": True},
    {"id": "key_parameters",        "label": "Key Parameters",       "extract": True},
    {"id": "statistical_methods",   "label": "Statistical Methods",  "extract": True},
    {"id": "limitations",           "label": "Limitations",          "extract": True},
    {"id": "assumptions",           "label": "Assumptions",          "extract": True},
    {"id": "novelty_statement",     "label": "Novelty Statement",    "extract": True},
    {"id": "actionable_insights",   "label": "Actionable Insights",  "extract": True},
    {"id": "relevance_to_domain",   "label": "Relevance to Domain",  "extract": True},
    {"id": "future_work",           "label": "Future Work",          "extract": True},
    {"id": "open_questions",        "label": "Open Questions",       "extract": True},
    {"id": "software_code",         "label": "Software / Code",      "extract": True},
    {"id": "funding_conflicts",     "label": "Funding & Conflicts",  "extract": True},
    {"id": "data_availability",     "label": "Data Availability",    "extract": True},
]
