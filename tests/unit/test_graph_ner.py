"""N1: BioBERT NER wrapper tests (stub-based; no model download in CI)."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from scripts.graph.ner import BiobertNER, NerSpan


class TestNerSpan:
    def test_is_frozen_dataclass(self):
        """N1: NerSpan is immutable for safe sharing."""
        span = NerSpan(text="EGFR", label="GENE", start=0, end=4, confidence=0.95)
        with pytest.raises(FrozenInstanceError):
            span.text = "ALTERED"

    def test_offset_round_trip(self):
        """N1: span.start/end correctly slice the original text."""
        text = "EGFR mutation in lung cancer"
        span = NerSpan(text="EGFR", label="GENE", start=0, end=4, confidence=0.99)
        assert text[span.start:span.end] == span.text


class TestBiobertNERImport:
    def test_module_imports_without_nlp_extra(self):
        """N1: importing scripts.graph.ner does NOT trigger transformers import."""
        # Importing the module should succeed even when transformers is missing.
        # The lazy import happens inside _load(), so this should be safe.
        import scripts.graph.ner  # noqa: F401

    def test_extract_raises_clear_error_without_nlp_extra(self, monkeypatch):
        """N1: when transformers is unavailable, extract() raises a clear ImportError."""
        import sys
        # Block the transformers import
        monkeypatch.setitem(sys.modules, "transformers", None)
        # Reset cached _pipeline so the next .extract() re-triggers _load
        ner = BiobertNER(model_id="alvaroalon2/biobert_genetic_ner")
        ner._pipeline = None
        with pytest.raises(ImportError, match=r"\[nlp\]|uv sync --extra nlp"):
            ner.extract("EGFR mutation")


class TestBiobertNERWithStub:
    """Default unit tests use the stub fixture — no model download."""

    def test_stub_returns_ner_spans(self, biobert_stub):
        """N1: the stub fixture returns NerSpan objects (interface parity)."""
        spans = biobert_stub.extract("EGFR mutation in lung cancer")
        assert isinstance(spans, list)
        assert all(isinstance(s, NerSpan) for s in spans)

    def test_stub_offsets_slice_text_correctly(self, biobert_stub):
        """N1: stub-produced offsets slice the source text correctly (interface contract)."""
        text = "EGFR mutation in lung cancer"
        spans = biobert_stub.extract(text)
        for span in spans:
            assert text[span.start:span.end] == span.text, (
                f"offset mismatch: {span.start}:{span.end} "
                f"-> {text[span.start:span.end]!r} != {span.text!r}"
            )


class TestDeviceSelect:
    def test_device_select_prefers_mps_when_available(self, monkeypatch):
        """N1: _select_device returns 'mps' when MPS is available."""
        import scripts.graph.ner as ner_mod
        monkeypatch.setattr(ner_mod, "_mps_available", lambda: True)
        monkeypatch.setattr(ner_mod, "_cuda_available", lambda: True)
        assert ner_mod._select_device() == "mps"

    def test_device_select_falls_back_to_cuda(self, monkeypatch):
        """N1: _select_device returns 'cuda' when MPS is absent but CUDA is present."""
        import scripts.graph.ner as ner_mod
        monkeypatch.setattr(ner_mod, "_mps_available", lambda: False)
        monkeypatch.setattr(ner_mod, "_cuda_available", lambda: True)
        assert ner_mod._select_device() == "cuda"

    def test_device_select_falls_back_to_cpu(self, monkeypatch):
        """N1: _select_device returns 'cpu' when neither MPS nor CUDA is available."""
        import scripts.graph.ner as ner_mod
        monkeypatch.setattr(ner_mod, "_mps_available", lambda: False)
        monkeypatch.setattr(ner_mod, "_cuda_available", lambda: False)
        assert ner_mod._select_device() == "cpu"
