"""
Obsidian retheme — bulk rename themes/wikilinks across all vault notes.
Usage:
    retheme(vault_path, old_theme, new_theme)
Renames the theme page (if it exists) and rewrites all [[old_theme]] wikilinks
in the vault to [[new_theme]]. Only modifies files inside vault_path.
Wikilink patterns handled:
  [[ThemeName]]
  [[ThemeName|Display Text]]
  [[ThemeName#heading]]
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from lit_monitor.core.atomic_write import atomic_write_text
from lit_monitor.core.strict_mode import strict_fallback

logger = logging.getLogger(__name__)
def retheme(vault_path: str | Path, old_theme: str, new_theme: str) -> dict[str, int]:
    """
    Rename a theme in the Obsidian vault.
    1. Rename the theme page file (if it exists): [[OldTheme]].md → [[NewTheme]].md
    2. Rewrite all wikilinks in all .md files from [[OldTheme]] to [[NewTheme]]
    Returns
    -------
    dict with keys: files_modified, wikilinks_rewritten, page_renamed (0 or 1)
    """
    vault = Path(vault_path)
    stats = {"files_modified": 0, "wikilinks_rewritten": 0, "page_renamed": 0}
    # 1. Rename the theme page itself (anywhere in vault)
    old_page = _find_page(vault, old_theme)
    if old_page:
        new_page = old_page.parent / f"{new_theme}.md"
        if new_page.exists():
            logger.warning(
                "Theme page %s already exists; not renaming %s over it.",
                new_page, old_page,
            )
        else:
            old_page.rename(new_page)
            stats["page_renamed"] = 1
            logger.info("Renamed theme page: %s → %s", old_page, new_page)
    # 2. Rewrite wikilinks in all .md files.
    # Each file is written atomically (temp + fsync + os.replace) so no
    # individual note is ever truncated by a mid-write crash. NOTE: this loop
    # is still non-transactional ACROSS files — a crash partway through leaves
    # some notes rewritten and others not. A full vault-wide transaction is a
    # larger, separate concern and is intentionally out of scope here.
    for md_file in vault.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            new_content, n = _rewrite_wikilinks(content, old_theme, new_theme)
            if n > 0:
                atomic_write_text(md_file, new_content)
                stats["files_modified"] += 1
                stats["wikilinks_rewritten"] += n
                logger.debug("Rewrote %d wikilink(s) in %s", n, md_file)
        except Exception as exc:
            # P4.4: per-file best-effort. Escalate in --strict; non-strict logs
            # and continues to the next file.
            strict_fallback(logger, f"Could not process {md_file}: {exc}", exc)
    logger.info(

        "Retheme %r → %r: %d files, %d links, page renamed=%d",
        old_theme, new_theme,
        stats["files_modified"], stats["wikilinks_rewritten"], stats["page_renamed"],
    )
    return stats
def _rewrite_wikilinks(content: str, old_theme: str, new_theme: str) -> tuple[str, int]:
    """
    Replace [[old_theme]] (and variants) with [[new_theme]] in content.
    Returns (new_content, count_of_replacements).
    """
    count = 0
    # Build pattern that matches exactly old_theme as the link target
    # (case-insensitive, ignoring trailing spaces)
    escaped = re.escape(old_theme)
    pattern = re.compile(
        r"(\[\[)" + escaped + r"(\s*(?:#[^\]|]*)?\s*(?:\|[^\]]*)?\]\])",
        re.IGNORECASE,
    )
    def replacer(m: re.Match) -> str:
        nonlocal count
        count += 1
        suffix = m.group(2)  # e.g. "#heading|alias]]" or "]]"
        return f"[[{new_theme}{suffix}"
    new_content = pattern.sub(replacer, content)
    return new_content, count
def _find_page(vault: Path, theme_name: str) -> Path | None:
    """Find a .md file named theme_name anywhere in the vault."""
    candidates = list(vault.rglob(f"{theme_name}.md"))
    if candidates:
        return candidates[0]
    return None
