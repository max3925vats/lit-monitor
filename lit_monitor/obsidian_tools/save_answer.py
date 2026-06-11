"""WA3: persist an /ask Q&A to the Obsidian Connections folder.

Self-contained writer for the web `/ask` page's "Save to vault" affordance.
It derives its own markdown (front-matter + question + prose + results table +
Cypher) from the raw ``rows`` so it stays decoupled from the ask pipeline — it
does NOT thread ``AskResult.rendered`` through.

The write path mirrors
``scripts/obsidian_tools/synthesize.py::_write_synthesis_note`` exactly: resolve
``vault_path / connections_folder``, ``mkdir(parents=True, exist_ok=True)``, slug
the question, and ``atomic_write_text`` to ``Ask_{slug}.md`` — preserving any
user-edited ``## Notes`` section on re-save. Path-contained to the vault.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from lit_monitor.core.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert a question string to a safe filename slug.

    Replicates synthesize.py::_slugify so the writer stays self-contained.
    """
    slug = re.sub(r"[^\w\s-]", "", text)
    slug = re.sub(r"[\s_]+", "_", slug.strip())
    slug = slug[:60].strip("_")
    # A punctuation-only question slugifies to "" → fall back so distinct such
    # questions don't all collide on a single "Ask_.md".
    return slug or "untitled"


def _extract_user_section(content: str) -> str:
    """Return the user's hand-edited text below the last ``## Notes`` header.

    The auto-generated block (front-matter + Ask/Results/Cypher + the
    ``*Last updated:*`` divider line) never contains a ``## Notes`` heading, so
    splitting on the LAST ``## Notes`` reliably isolates the user's notes — and
    re-prefixing a single ``## Notes`` on save is then idempotent (no doubled
    header across repeated saves, unlike a ``---``-divider split which leaves the
    heading inside the extracted text when YAML front-matter is present).
    """
    marker = "\n## Notes\n"
    idx = content.rfind(marker)
    if idx == -1:
        return ""
    return content[idx + len(marker):].strip()


def _render_table(rows: list[dict]) -> str:
    """Render rows as a markdown table (union-of-keys header).

    Empty rows → an italic ``_no rows_`` placeholder so the section is never
    blank.
    """
    if not rows:
        return "_no rows_"
    # Header = union of all row keys in stable first-seen order.
    keys: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    header = "| " + " | ".join(str(k) for k in keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body_lines = []
    for row in rows:
        cells = []
        for k in keys:
            val = row.get(k)
            cells.append("" if val is None else str(val))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_lines])


def save_ask_answer(
    question: str,
    prose: str | None,
    rows: list[dict],
    cypher: str | None,
    config,
) -> str:
    """Write an /ask Q&A as ``Ask_{slug}.md`` in the Connections folder.

    Args:
        question: The natural-language question the user asked.
        prose: The LLM prose answer, or None (a placeholder is written instead).
        rows: The raw Cypher result rows (arbitrary dicts).
        cypher: The Cypher that produced the rows, or None (section omitted).
        config: Config object exposing ``obsidian.vault_path`` and
            (optionally) ``obsidian.connections_folder``.

    Returns:
        The absolute path to the written note, as a string.
    """
    vault_path = Path(config.obsidian.vault_path)
    folder_rel = getattr(
        config.obsidian, "connections_folder", "Literature/Connections"
    )
    folder = vault_path / folder_rel
    folder.mkdir(parents=True, exist_ok=True)

    slug = _slugify(question)
    note_path = folder / f"Ask_{slug}.md"

    run_date = str(date.today())
    prose_block = prose.strip() if prose else "*No summary was generated.*"
    table = _render_table(rows)

    parts = [
        "---",
        "tags: [ask]",
        f'generated: "{run_date}"',
        "---",
        f"# Ask: {question}",
        "",
        f"> *Saved from the `/ask` page · {run_date}*",
        "",
        prose_block,
        "",
        "## Results",
        table,
    ]
    if cypher:
        parts += ["", "## Cypher", "```cypher", cypher, "```"]
    parts += ["", "---", f"*Last updated: {run_date}*", ""]
    content = "\n".join(parts)

    # Preserve any user-added ## Notes section below the generated block.
    if note_path.exists():
        existing = note_path.read_text(encoding="utf-8")
        user_section = _extract_user_section(existing)
        if user_section:
            content = content.rstrip("\n") + "\n\n## Notes\n" + user_section

    atomic_write_text(note_path, content)
    return str(note_path)


__all__ = ["save_ask_answer"]
