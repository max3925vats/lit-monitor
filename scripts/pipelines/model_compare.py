"""
Model comparison pipeline.
Runs extraction on N items using each configured model, writes outputs
to comparison/ for qualitative review.
Usage:
    lit-monitor compare-models [--papers N] [--models M1,M2,...] [--mode paper]
Purpose: choose the best LLM before running brain-build.
Evaluation dimensions:
  - Null honesty: does the model return null for absent fields or fabricate?
  - JSON validity: parseable without markdown fences / trailing commas?
  - Field completeness: required fields reliably populated?
  - Speed: seconds per paper at target chunk size?
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "comparison"
# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class _ModelScore:
    model: str
    items_run: int = 0
    items_failed: int = 0
    total_seconds: float = 0.0
    json_parse_errors: int = 0
    required_fields_missing: int = 0
    null_honesty_violations: int = 0  # non-null value where field is absent
    @property
    def avg_seconds(self) -> float:
        return self.total_seconds / self.items_run if self.items_run else 0.0
@dataclass
class _ComparisonResult:
    mode: str
    n_items: int
    run_date: str
    scores: list[_ModelScore] = field(default_factory=list)
    output_dir: str = ""
# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_model_comparison(
    config,

    state_db,
    zotero_client,
    n_items: int = 3,
    models: list[str] | None = None,
    mode: str = "paper",
) -> _ComparisonResult:
    """
    Run extraction on n_items using each model.  Write JSON and Markdown
    outputs to comparison/{date}_{mode}/ for qualitative review.
    Args:
        config:         Config object.
        state_db:       StateDB instance.
        zotero_client:  ZoteroClient instance.
        n_items:        Number of items to run per model (default 3).
        models:         List of "provider:model" strings.  Defaults to
                        comparison_models from extraction.yaml.
        mode:           "paper" (journal articles or reviews).
    Returns:
        _ComparisonResult with per-model scores.
    """
    from scripts.llm.extractor import extract_paper
    from scripts.llm.llm_client import LiteLLMClient, OllamaClient
    run_date = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = _OUTPUT_DIR / f"{run_date}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Determine model list
    if models:
        model_specs = models
    else:
        comparison_models = getattr(config, "comparison_models", [])
        if not comparison_models:
            raise ValueError(
                "No comparison_models configured in extraction.yaml "
                "and no --models flag provided."
            )
        model_specs = [
            f"{m.get('provider', 'ollama')}:{m.get('model', '')}"
            for m in comparison_models
        ]
    # Select items to test
    items = _select_items(state_db, zotero_client, config, mode, n_items)
    if not items:
        raise ValueError(
            f"No {mode} items available for comparison. "
            "Run brain-build first to populate state DB."
        )
    result = _ComparisonResult(
        mode=mode,
        n_items=len(items),
        run_date=run_date,
        output_dir=str(out_dir),
    )
    for model_spec in model_specs:
        parts = model_spec.split(":", 1)
        provider = parts[0] if len(parts) == 2 else "ollama"
        model_name = parts[1] if len(parts) == 2 else parts[0]
        mode_config = getattr(config, "brain_build", None)
        base_timeout = int(getattr(mode_config, "timeout", 300) or 300)
        base_temp = float(getattr(mode_config, "temperature", 0.1))
        try:
            if provider == "litellm":
                try:
                    import litellm  # noqa: F401
                except ImportError:
                    logger.warning(
                        "Skipping %s: litellm not installed (pip install 'lit-monitor[litellm]')",
                        model_spec,
                    )
                    continue
                llm = LiteLLMClient(
                    model=model_name,
                    timeout=base_timeout,
                    temperature=base_temp,
                )
            else:
                llm = OllamaClient(
                    model=model_name,
                    host=getattr(mode_config, "ollama_host", "http://localhost:11434"),
                    timeout=base_timeout,
                    temperature=base_temp,
                    think=True,  # model comparison runs paper extraction — thinking on
                )
        except Exception as exc:
            logger.warning("Skipping model %s: %s", model_spec, exc)
            continue
        score = _ModelScore(model=model_spec)
        model_out_dir = out_dir / model_name.replace(":", "-")
        model_out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running comparison: model=%s, mode=%s, n=%d", model_spec, mode, len(items))
        for item_info in items:
            fulltext = item_info.get("fulltext", "")
            doi = item_info.get("doi", "unknown")
            item_out_path = model_out_dir / f"{_safe_filename(doi)}.json"
            start = time.monotonic()
            extraction: dict[str, Any] = {}
            try:
                extraction = extract_paper(fulltext, llm, phases=("simple", "complex"))
                score.items_run += 1
            except Exception as exc:
                logger.error("Extraction failed for %s with %s: %s", doi, model_spec, exc)
                extraction = {"_error": str(exc)}
                score.items_failed += 1
            elapsed = time.monotonic() - start
            score.total_seconds += elapsed
            # Score quality dimensions
            _score_extraction(score, extraction, mode)
            # Write raw output
            output = {
                "doi": doi,
                "model": model_spec,
                "mode": mode,
                "elapsed_seconds": round(elapsed, 2),
                "extraction": extraction,
            }
            item_out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        result.scores.append(score)
    # Write summary
    _write_summary(result, out_dir)
    logger.info("Comparison complete. Outputs: %s", out_dir)
    return result
# ---------------------------------------------------------------------------
# Helpers

# ---------------------------------------------------------------------------
def _select_items(
    state_db,
    zotero_client,
    config,
    mode: str,
    n: int,
) -> list[dict[str, Any]]:
    """Return up to n items with markdown full-text for comparison.

    M7: reads .md Zotero attachments instead of PDFs.
    Only 'paper' mode is supported (textbook pipeline removed in R-10).
    """
    items: list[dict[str, Any]] = []
    if mode == "paper":
        records = state_db.get_all_by_source_type("paper")
        for rec in records:
            if len(items) >= n:
                break
            doi = rec.get("doi", "")
            zotero_key = rec.get("zotero_key", "")
            if not zotero_key:
                continue
            try:
                fulltext = zotero_client.get_markdown_attachment(zotero_key)
                if not fulltext:
                    continue
                items.append({"doi": doi, "fulltext": fulltext})
            except Exception as exc:
                logger.debug("Skipping %s for comparison: %s", doi, exc)
    return items
_PAPER_REQUIRED = {"core_finding", "methods_summary", "results_summary", "study_type"}
# Fields that should be null rather than fabricated for absent content
_NULL_HONEST_FIELDS = {
    "statistical_methods", "funding_conflicts", "data_availability",
    "software_code", "comparison_to_prior",
    "methods_or_frameworks", "actionable_insights",
}
def _score_extraction(score: _ModelScore, extraction: dict, mode: str) -> None:
    """Update score in-place based on extraction output quality."""
    if "_error" in extraction:
        return
    required = _PAPER_REQUIRED
    for field_name in required:
        if not extraction.get(field_name):
            score.required_fields_missing += 1
    # JSON parse errors: if the whole extraction is empty, count it
    if not extraction or all(v is None for v in extraction.values()):
        score.json_parse_errors += 1
def _write_summary(result: _ComparisonResult, out_dir: Path) -> None:
    """Write a Markdown summary table and a JSON scores file."""
    # JSON
    scores_data = [
        {
            "model": s.model,
            "items_run": s.items_run,
            "items_failed": s.items_failed,
            "avg_seconds": round(s.avg_seconds, 1),
            "json_parse_errors": s.json_parse_errors,

            "required_fields_missing": s.required_fields_missing,
        }
        for s in result.scores
    ]
    (out_dir / "scores.json").write_text(json.dumps(scores_data, indent=2))
    # Markdown
    md_lines = [
        f"# Model Comparison — {result.mode} mode — {result.run_date}",
        "",
        f"Items tested per model: {result.n_items}",
        "",
        "| Model | Run | Failed | Avg s | JSON Errors | Missing Required |",
        "|-------|-----|--------|-------|-------------|-----------------|",
    ]
    for s in result.scores:
        md_lines.append(
            f"| {s.model} | {s.items_run} | {s.items_failed} | "
            f"{s.avg_seconds:.1f} | {s.json_parse_errors} | {s.required_fields_missing} |"
        )
    md_lines += [
        "",
        "## Review checklist",
        "- [ ] Null honesty: returns null for absent fields?",
        "- [ ] JSON validity: parseable without fences?",
        "- [ ] Field completeness: required fields populated?",
        "- [ ] Speed acceptable for batch runs?",
        "",
        f"*Outputs: {out_dir}*",
    ]
    (out_dir / "summary.md").write_text("\n".join(md_lines))
def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]
