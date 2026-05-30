"""N1: BioBERT NER wrapper.

Lazy-loads transformers + torch on first .extract() call so the module is
importable without the [nlp] extra. Device auto-selected (MPS > CUDA > CPU).
Model id overridable via extraction.yaml::graph.ner.model.

Phase 2 entity-density backbone — feeds entity_extractor and the cloud-Ollama
long-tail validator (N2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NerSpan:
    """One NER-extracted span. Offsets slice the source string (text[start:end] == text)."""

    text: str
    label: str
    start: int
    end: int
    confidence: float


def _mps_available() -> bool:
    """N1: probe MPS availability lazily so the module imports without torch."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _cuda_available() -> bool:
    """N1: probe CUDA availability lazily so the module imports without torch."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _select_device() -> str:
    """N1: MPS > CUDA > CPU device-selection priority."""
    if _mps_available():
        return "mps"
    if _cuda_available():
        return "cuda"
    return "cpu"


_DEFAULT_MODEL_ID = "alvaroalon2/biobert_genetic_ner"


class BiobertNER:
    """Lazy BioBERT NER wrapper. Construction is cheap; .extract() triggers model load.

    Use::

        ner = BiobertNER(model_id="alvaroalon2/biobert_genetic_ner")
        spans = ner.extract("EGFR mutation in lung cancer")

    Tests should use ``BiobertNERStub`` from ``tests/unit/conftest.py`` instead.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id or _DEFAULT_MODEL_ID
        # if None, auto-selected on first _load()
        self.device = device
        self._pipeline: Any = None
        self._loaded_logged: bool = False

    def _load(self) -> Any:
        """N1: lazy-load transformers pipeline. Raises ImportError if [nlp] missing."""
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "BiobertNER requires the [nlp] extra. "
                "Install with: uv sync --extra nlp"
            ) from exc

        device = self.device or _select_device()
        if not self._loaded_logged:
            logger.info("BiobertNER: loading %s on device=%s", self.model_id, device)
            self._loaded_logged = True

        # device parameter for transformers pipeline:
        #   "mps" / "cuda" / "cpu" or integer index
        self._pipeline = pipeline(
            "token-classification",
            model=self.model_id,
            aggregation_strategy="simple",
            device=device,
        )
        return self._pipeline

    def extract(self, text: str) -> list[NerSpan]:
        """N1: extract NER spans from a single string.

        Returns a list of NerSpan with offsets that slice the input text.
        Empty list if text is empty or NER produces no entities.

        Args:
            text: The source string to run NER over.

        Returns:
            Sorted list of NerSpan objects (by start offset).

        Raises:
            ImportError: When the [nlp] extra is not installed.
        """
        if not text or not text.strip():
            return []
        pipe = self._load()
        raw = pipe(text)
        out: list[NerSpan] = []
        for ent in raw:
            try:
                start = int(ent["start"])
                end = int(ent["end"])
                # Re-slice from source text so the offset contract is always met.
                span_text = text[start:end]
                out.append(
                    NerSpan(
                        text=span_text,
                        label=str(ent.get("entity_group", ent.get("entity", "UNKNOWN"))),
                        start=start,
                        end=end,
                        confidence=float(ent.get("score", 0.0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.debug("BiobertNER: skipping malformed span %r", ent)
                continue
        return out
