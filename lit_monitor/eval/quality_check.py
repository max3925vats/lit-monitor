"""Idea 4: pure scoring helpers for Tier-4 extraction quality regression.

Compares a paper's live extraction (from state.db) against the hand-written
expected values in ``docs/internal/reference_papers.yaml`` and labels each
field PASS / WARN / FAIL. Two kinds of check:

1. Text-cosine fields (``core_finding``, ``methods_summary``,
   ``relevance_to_domain``): embed both actual and expected text, cosine-
   compare, threshold the similarity.
2. ``expected_null`` fields: the paper genuinely lacks these, so a faithful
   model abstains (null/empty). Any non-null value is a null-honesty
   violation → FAIL.

Everything here is pure and side-effect free except for logging. The embedding
function is injected (``embed_fn``) so unit tests can pass a deterministic fake
and the live Tier-4 path can pass a real Ollama embed call. The thin CLI at the
bottom (``python -m scripts.eval.quality_check``) wires in the real embedder and
prints an augmented status table; it is ADVISORY and never exits non-zero on a
regression.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Status labels (module constants so callers don't hardcode strings).
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# Cosine-similarity cutoffs for text-field scoring. These are EMPIRICAL and
# TUNABLE: 0.80 is "close paraphrase of the expected text", 0.60 is "related
# but drifting". They were chosen by eyeballing mxbai-embed-large similarities
# on paraphrase vs. off-topic pairs and should be revisited if the embedding
# model changes. >= PASS_THRESHOLD → PASS; >= WARN_THRESHOLD → WARN; else FAIL.
PASS_THRESHOLD = 0.80
WARN_THRESHOLD = 0.60

# Embedding function signature: str -> 1-D vector (np.ndarray or list[float]).
EmbedFn = Callable[[str], "np.ndarray | list[float]"]

# Text fields compared by cosine similarity. Mirrors the three fields the
# Tier-4 table in scripts/test_everything.sh already prints.
TEXT_FIELDS: tuple[str, ...] = ("core_finding", "methods_summary", "relevance_to_domain")


@dataclass(frozen=True)
class FieldResult:
    """One row of the quality table.

    Attributes:
        field: Extraction field name (e.g. ``core_finding``).
        kind: ``"text"`` (cosine-compared) or ``"null"`` (null-honesty).
        status: One of PASS / WARN / FAIL.
        score: Cosine similarity for text fields; ``None`` for null fields.
        detail: Short human-readable explanation of the status.
    """

    field: str
    kind: str
    status: str
    score: float | None
    detail: str


def _is_empty(value: str | None) -> bool:
    """Return True when a field value counts as 'absent' (None or blank)."""
    return value is None or (isinstance(value, str) and value.strip() == "")


def cosine(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Cosine similarity between two vectors.

    A standalone helper because the existing cosine logic in
    ``scripts/llm/ranker.py`` is inlined inside a scoring loop (computed as
    ``np.dot(...) / (norm * norm)`` against ChromaDB candidate arrays) rather
    than exposed as a reusable function. Degenerate (zero-norm / non-finite)
    vectors yield 0.0 rather than NaN, matching ranker.py's guard behaviour.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]; 0.0 for degenerate inputs.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a < 1e-9 or norm_b < 1e-9 or not np.isfinite(norm_a) or not np.isfinite(norm_b):
        logger.warning("cosine: degenerate vector (norm≈0 or non-finite); returning 0.0")
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def score_field(
    actual: str | None,
    expected: str | None,
    embed_fn: EmbedFn,
) -> tuple[float, str]:
    """Score one text field by cosine similarity of its embeddings.

    Args:
        actual: Live extraction value for the field (may be None/empty).
        expected: Hand-written expected value (may be None/empty).
        embed_fn: Callable embedding a string to a vector.

    Returns:
        ``(score, status)`` where status is PASS / WARN / FAIL. The score is
        the cosine similarity; for the no-text-to-compare edge cases it is a
        sentinel (1.0 for the both-empty PASS, 0.0 for the one-empty FAIL).

    Edge cases:
        - Both empty/None → ``(1.0, PASS)`` — nothing to compare, not a regression.
        - Exactly one empty → ``(0.0, FAIL)`` — model and reference disagree on
          whether the field exists at all.
    """
    actual_empty = _is_empty(actual)
    expected_empty = _is_empty(expected)

    if actual_empty and expected_empty:
        return 1.0, PASS
    if actual_empty != expected_empty:
        # One side has text, the other doesn't — a clear presence mismatch.
        return 0.0, FAIL

    # Both non-empty: real cosine comparison. mypy: narrowed to str above.
    score = cosine(embed_fn(actual), embed_fn(expected))  # type: ignore[arg-type]
    if score >= PASS_THRESHOLD:
        status = PASS
    elif score >= WARN_THRESHOLD:
        status = WARN
    else:
        status = FAIL
    return score, status


def check_null_honesty(actual: str | None, field: str) -> tuple[bool, str]:
    """Check that the model abstained on a field the paper genuinely lacks.

    Args:
        actual: Live extraction value for the field.
        field: Field name (for the explanatory message).

    Returns:
        ``(passed, detail)``. ``passed`` is True when ``actual`` is null/empty
        (the model correctly abstained); False when the model emitted content
        the paper does not contain (a fabrication / null-honesty violation).
    """
    if _is_empty(actual):
        return True, f"{field}: correctly null (model abstained)"
    return False, f"{field}: fabricated value where null expected"


def evaluate_paper(
    actual_extraction: dict,
    reference: dict,
    embed_fn: EmbedFn,
) -> list[FieldResult]:
    """Evaluate one paper's extraction against its reference entry.

    Produces one :class:`FieldResult` per text field present in ``reference``
    plus one per field listed in ``reference['expected_null']``.

    Args:
        actual_extraction: The paper's live extraction dict (parsed from the
            state-DB ``extraction_json`` column).
        reference: The per-paper reference dict from reference_papers.yaml
            (expected text values + an ``expected_null`` list).
        embed_fn: Callable embedding a string to a vector.

    Returns:
        List of FieldResult rows, text fields first then null-honesty fields.
    """
    rows: list[FieldResult] = []

    # --- Text-cosine fields: only those the reference actually specifies. ---
    for field in TEXT_FIELDS:
        if field not in reference:
            # Reference didn't pin this field → nothing to regress against.
            continue
        actual_val = actual_extraction.get(field)
        expected_val = reference.get(field)
        score, status = score_field(actual_val, expected_val, embed_fn)
        if status == PASS:
            detail = "matches expected" if score < 1.0 else "no text to compare"
        elif status == WARN:
            detail = "drifting from expected"
        else:
            detail = "diverged from expected" if score > 0.0 else "presence mismatch"
        rows.append(
            FieldResult(field=field, kind="text", status=status, score=score, detail=detail)
        )

    # --- Null-honesty fields: paper genuinely lacks these. ---
    for field in reference.get("expected_null") or []:
        actual_val = actual_extraction.get(field)
        passed, detail = check_null_honesty(actual_val, field)
        rows.append(
            FieldResult(
                field=field,
                kind="null",
                status=PASS if passed else FAIL,
                score=None,
                detail=detail,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Thin CLI — wired into Tier-4 of scripts/test_everything.sh. ADVISORY ONLY:
# prints the augmented status table and a summary line, but ALWAYS exits 0 so
# the release gate's exit code is never changed by a quality regression.
# ---------------------------------------------------------------------------

# Status glyphs for the printed table.
_GLYPH = {PASS: "✔", WARN: "⚠", FAIL: "✗"}


def _real_embed_fn() -> EmbedFn:
    """Build a real embedding function from the configured provider/model.

    Routes through the same Ollama embed call the rest of the pipeline uses
    (``scripts.llm.embedding_client.embed_via_ollama``) with the model/host
    from config. Only constructed for the live Tier-4 path — unit tests inject
    a fake instead.
    """
    from scripts.core.config import get_config
    from scripts.llm.embedding_client import embed_via_ollama

    cfg = get_config()
    model = cfg.embedding.ollama.model
    host = cfg.embedding.ollama.host

    def _embed(text: str) -> np.ndarray:
        return embed_via_ollama(text, model=model, host=host)

    return _embed


def _load_references(ref_yaml: str) -> dict:
    """Load the papers mapping from reference_papers.yaml."""
    import yaml

    with open(ref_yaml) as fh:
        refs = yaml.safe_load(fh) or {}
    return refs.get("papers") or {}


def _load_actual(doi: str, db_path: str) -> dict | None:
    """Fetch and parse a paper's extraction_json from state.db (None if absent)."""
    import json
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT extraction_json FROM papers WHERE doi = ?", (doi,)
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row[0]:
        return None
    return json.loads(row[0])


def _emit_report(ref_yaml: str) -> None:
    """Render the advisory PASS/WARN/FAIL table to stdout.

    Reuses the Tier-4 data-loading shape (reference_papers.yaml + state.db
    ``extraction_json``), then augments each field row with a status from the
    pure scorers above. May raise on unrecoverable I/O (Ollama down, missing
    state.db, malformed YAML); ``main`` catches and swallows so the gate is
    never failed by the advisory check.
    """
    from scripts.core.config import get_config

    db_path = str(get_config().state_db.path)
    papers = _load_references(ref_yaml)
    if not papers:
        print(f"_no reference DOIs configured in {ref_yaml} — add some_")
        return

    embed_fn = _real_embed_fn()
    below_threshold = 0

    for doi, reference in papers.items():
        print(f"### {doi}")
        print()
        actual = _load_actual(doi, db_path)
        if actual is None:
            print("_not in state DB — re-run brain-build_")
            print()
            continue
        rows = evaluate_paper(actual, reference or {}, embed_fn)
        # Column header for the eyeball table.
        print("| field | kind | status | score | detail |")
        print("|---|---|---|---|---|")
        for r in rows:
            score_txt = "—" if r.score is None else f"{r.score:.2f}"
            glyph = _GLYPH.get(r.status, "?")
            print(
                f"| {r.field} | {r.kind} | {glyph} {r.status} | {score_txt} | {r.detail} |"
            )
            if r.status in (WARN, FAIL):
                below_threshold += 1
        print()

    if below_threshold:
        print(f"⚠ {below_threshold} field(s) below threshold — review")
    else:
        print("✔ all reference fields within threshold")


def main(argv: list[str] | None = None) -> int:
    """Advisory entrypoint for Tier-4. ALWAYS returns 0 — a regression OR a
    crash in the quality check must never fail the release gate (the shell side
    also guards with ``|| true``; this is the belt to that suspender)."""
    logging.basicConfig(level=logging.WARNING)
    ref_yaml = argv[0] if argv else "docs/internal/reference_papers.yaml"
    try:
        _emit_report(ref_yaml)
    except Exception:  # noqa: BLE001 — advisory: never propagate a failure to the gate
        logger.warning(
            "reference-paper quality check crashed; skipping advisory report.",
            exc_info=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
