"""P7: shared render of a discovery run as a Markdown digest.

Single source of truth for the CLI export surface:
- `lit-monitor discovery export-md` writes via this module.

The auto-write path in scripts/pipelines/discovery.py uses its own
_DIGEST_TEMPLATE + _format_paper_list functions to preserve the existing
pipeline-generated digest format byte-for-byte.  render_digest produces
a human-readable export variant intended for on-demand CLI use.

Input: run dict (id, started_at, total_ingested) + papers list
(each: doi, title, score, rationale).
Output: markdown string ending in a newline.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def render_digest(run: dict[str, Any], papers: list[dict[str, Any]]) -> str:
    """Render a discovery run's results as Markdown.

    Args:
        run:    Discovery run row dict.  Expected keys: id, started_at,
                total_ingested.  Missing keys fall back gracefully.
        papers: List of paper dicts.  Each may contain: doi, title, score,
                rationale.  Missing keys are replaced with safe defaults.

    Returns:
        Markdown string.  Always ends with a trailing newline.
    """
    started = (run.get("started_at") or "")[:10] or datetime.now().strftime("%Y-%m-%d")
    total = run.get("total_ingested", len(papers))
    lines = [
        f"# Discovery Digest — {started}",
        "",
        f"> **Run #{run.get('id', '?')}** · {total} papers · Generated {started}",
        "",
    ]
    if not papers:
        lines.append("_No new papers found in this run._")
        return "\n".join(lines) + "\n"

    for i, p in enumerate(papers, 1):
        title = p.get("title") or "Untitled"
        doi = p.get("doi") or ""
        score = p.get("score") or 0.0
        rationale = p.get("rationale") or ""
        doi_link = f"[{doi}](https://doi.org/{doi})" if doi else "_(no DOI)_"
        lines += [
            f"## {i}. {title}",
            "",
            f"**DOI:** {doi_link}",
            f"**Relevance score:** {score:.3f}",
            "",
        ]
        if rationale:
            lines += [rationale, ""]

    return "\n".join(lines) + "\n"
