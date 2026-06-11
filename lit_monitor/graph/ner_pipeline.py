"""N4: composable NER pipeline.

Schema (N3 extract_entities) + BioBERT (N1) + cloud-Ollama (N2) →
merged MentionEdge list ready for graph_db.add_paper.

Defensive perimeter: any failure in BioBERT or cloud-LLM falls back to
schema-only path with a WARN log. Phase 1 vector-only ingest is never
blocked.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level once-flag so we log the [nlp]-missing notice once per process.
_NLP_MISSING_LOGGED: bool = False

# BioBERT spans below this confidence threshold are forwarded to the N2
# cloud-LLM for validation — they may be real but uncertain.
_BIOBERT_LOW_CONF_THRESHOLD: float = 0.65


def _maybe_log_nlp_missing(exc: Exception) -> None:
    """Log a single WARN when [nlp] extra import fails; subsequent calls skip."""
    global _NLP_MISSING_LOGGED  # noqa: PLW0603
    if not _NLP_MISSING_LOGGED:
        logger.warning(
            "N4: [nlp] extra not available (%s). "
            "Continuing schema-only. Install with: uv sync --extra nlp",
            exc,
        )
        _NLP_MISSING_LOGGED = True


def build_mention_edges(
    *,
    paper_id: str,
    text: str | None,
    extraction_json: dict[str, Any],
    paper_metadata: dict[str, Any],
    normalizer: Any,
    use_biobert: bool = True,
    use_cloud_llm: bool = True,
    biobert_factory: Any = None,
    llm_long_tail_fn: Any = None,
) -> list[Any]:
    """N4: build the merged MentionEdge list for one paper.

    Pipeline:
      1. Schema entities via N3 extract_entities (always runs).
      2. BioBERT NER (N1) on body text — gated by use_biobert + [nlp] availability.
      3. Cloud-Ollama long-tail (N2) — gated by use_cloud_llm + config + key.
      4. Merge via N3 merge_mentions.

    ``biobert_factory`` and ``llm_long_tail_fn`` are dependency injection points
    for testing. In production both are ``None`` and resolved internally.

    Args:
        paper_id: DOI or other stable identifier for the paper.
        text: Full body text to run BioBERT and LLM on. May be None/empty —
            both NER steps are skipped when no text is available.
        extraction_json: LLM extraction output dict (from G3 / Phase 1).
        paper_metadata: Paper-level metadata dict (authors, journal, keywords).
        normalizer: ``EntityNormalizer`` instance from G2 for canonical form lookup.
        use_biobert: Set False to skip the BioBERT step (e.g. in tests or when
            the [nlp] extra is known to be absent).
        use_cloud_llm: Set False to skip the cloud-Ollama long-tail step
            (matches ``graph.ner.cloud_long_tail_enabled=false`` or absent key).
        biobert_factory: Optional callable that returns a BiobertNER instance;
            injected in tests to avoid loading model weights.
        llm_long_tail_fn: Optional callable with signature
            ``(text, low_conf_entities) -> dict``; injected in tests.

    Returns:
        List of ``MentionEdge`` objects ready for ``graph_db.add_paper``.
        Empty list is valid — ``graph_db.add_paper`` handles this gracefully.
    """
    from lit_monitor.graph.entity_extractor import (  # noqa: PLC0415
        EntityTuple,
        MentionEdge,
        extract_entities,
        from_biobert,
        from_llm_cloud,
        merge_mentions,
    )

    # ------------------------------------------------------------------
    # 1. Schema entities (always runs — the extraction_json is always present)
    # ------------------------------------------------------------------
    schema_tuples: list[EntityTuple] = []
    try:
        schema_tuples = extract_entities(
            extraction_json or {},
            paper_metadata or {},
            normalizer,
        )
    except Exception as exc:
        logger.warning("N4: schema extraction failed for %s: %s", paper_id, exc)

    # Convert EntityTuples → MentionEdge with source='schema', confidence=1.0.
    schema_edges: list[MentionEdge] = [
        MentionEdge(
            paper_id=paper_id,
            entity_key=t.canonical_id,
            type=t.type,
            source="schema",
            surface=t.surface,
            field=t.field,
            confidence=1.0,
            span_start=t.span_start,
            span_end=t.span_end,
        )
        for t in schema_tuples
    ]

    # ------------------------------------------------------------------
    # 2. BioBERT NER (optional — gated by use_biobert flag and text availability)
    # ------------------------------------------------------------------
    biobert_edges: list[MentionEdge] = []
    if use_biobert and text:
        try:
            if biobert_factory is None:
                from lit_monitor.graph.ner import BiobertNER  # noqa: PLC0415
                biobert = BiobertNER()
            else:
                biobert = biobert_factory()
            spans = biobert.extract(text)
            biobert_edges = from_biobert(spans, paper_id, normalizer)
        except ImportError as exc:
            # [nlp] extra not installed — warn once and fall back to schema-only.
            _maybe_log_nlp_missing(exc)
        except Exception as exc:
            logger.warning("N4: BioBERT extraction failed for %s: %s", paper_id, exc)

    # ------------------------------------------------------------------
    # 3. Cloud-Ollama long-tail validation (optional)
    # ------------------------------------------------------------------
    llm_edges: list[MentionEdge] = []
    if use_cloud_llm and text:
        try:
            # Pass BioBERT low-confidence spans for LLM validation; N2 handles
            # the case where low_conf is empty (just does new-entity discovery).
            low_conf = [
                {"surface": e.surface, "confidence": e.confidence}
                for e in biobert_edges
                if e.confidence < _BIOBERT_LOW_CONF_THRESHOLD
            ]
            if llm_long_tail_fn is None:
                from lit_monitor.graph.long_tail import (
                    extract_long_tail_and_validate,  # noqa: PLC0415
                )
                llm_long_tail_fn = extract_long_tail_and_validate
            payload = llm_long_tail_fn(text, low_conf)
            llm_edges = from_llm_cloud(payload, paper_id, normalizer)
        except Exception as exc:
            logger.warning("N4: LLM long-tail failed for %s: %s", paper_id, exc)

    # ------------------------------------------------------------------
    # 4. Merge with provenance-aware deduplication (N3 merge_mentions)
    # ------------------------------------------------------------------
    return merge_mentions(schema_edges, biobert_edges, llm_edges)
