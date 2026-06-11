"""
Shared pytest fixtures for unit tests.

The main job here is to keep ChromaDB's process-wide singletons from leaking
across test modules.  ChromaDB ``PersistentClient`` caches a ``System`` object
per persist_path in ``SharedSystemClient._identifier_to_system``; when an
earlier test's ``tmp_path`` has been cleaned up, a later test that asks for a
*different* persist_path can still hit a stale handle through that cache and
fail with ``code: 14 unable to open database file`` from the Rust bindings.

Clearing the cache before every test is cheap (it is a no-op when empty) and
makes ChromaDB-backed tests order-independent.

N1 (Phase 2): BiobertNERStub and biobert_stub fixture are also defined here so
any unit test can reference them without importing transformers or torch.
"""
from __future__ import annotations

import re

import pytest


@pytest.fixture(autouse=True)
def _reset_strict_mode():
    """Reset the strict-mode global state before and after every unit test.

    ``set_strict()`` writes to a module-level variable.  Without this fixture,
    tests that call ``set_strict(True)`` (e.g. TestStrictModeFlag and
    TestLoadSecretsStrictMode) would leak into subsequent tests and turn
    expected-warning sites into unexpected RuntimeErrors.
    """
    import lit_monitor.core.strict_mode as _sm
    _sm._strict_override = None
    yield
    _sm._strict_override = None


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Reset the module-level config cache before and after every unit test.

    ``scripts.core.config.get_config()`` memoises the resolved ``Config`` in a
    module-level ``_config_cache``. Without this fixture, the first test to call
    ``get_config()`` would pin a Config built from whatever paths/env were active
    at that moment, and every later test that relies on different config (or that
    expects a fresh build) would silently see the stale cached object. Verified
    safe: no unit test populates the cache except via monkeypatch.
    """
    import lit_monitor.core.config as _cfg
    _cfg._config_cache = None
    yield
    _cfg._config_cache = None


@pytest.fixture(autouse=True)
def _reset_chromadb_shared_system_cache():
    """Drop ChromaDB's cached System objects before every unit test."""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except Exception:
        # chromadb not importable in this env (defensive — should always be present)
        yield
        return

    # Wipe cache before the test runs so any prior pollution is gone.
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        # On older chromadb the helper may not exist; fall back to the dict.
        cache = getattr(SharedSystemClient, "_identifier_to_system", None)
        if cache is not None:
            cache.clear()
    yield
    # And again after, so the next test in the file isn't tripped up by THIS one.
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        cache = getattr(SharedSystemClient, "_identifier_to_system", None)
        if cache is not None:
            cache.clear()


# ---------------------------------------------------------------------------
# N1 (Phase 2): BioBERT NER stub — no transformers / torch dependency.
# ---------------------------------------------------------------------------

from lit_monitor.graph.ner import NerSpan  # noqa: E402  (after imports block is fine)


class BiobertNERStub:
    """N1 test fixture: deterministic stub of BiobertNER.

    Returns NerSpan results by regex over text (no transformers/torch needed).
    Recognized patterns (case-sensitive):
    - gene-like names (2+ uppercase letters, optional trailing digits) → GENE
    - 'mutation', 'cancer', 'tumor', 'tumour' → MEDICAL_CONDITION
    """

    # Two-or-more uppercase letters, optional trailing digits (e.g. EGFR, BRCA1, TP53)
    _GENE_RE = re.compile(r"\b[A-Z]{2,}[0-9]*\b")
    # Common medical condition tokens (case-sensitive lowercase)
    _CONDITION_RE = re.compile(r"\b(?:mutation|cancer|tumor|tumour)\b")

    def extract(self, text: str) -> list[NerSpan]:
        """Return deterministic NerSpan list; offsets always slice text correctly."""
        spans: list[NerSpan] = []
        for m in self._GENE_RE.finditer(text):
            spans.append(
                NerSpan(
                    text=m.group(0),
                    label="GENE",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95,
                )
            )
        for m in self._CONDITION_RE.finditer(text):
            spans.append(
                NerSpan(
                    text=m.group(0),
                    label="MEDICAL_CONDITION",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.90,
                )
            )
        # Deterministic order by start offset
        return sorted(spans, key=lambda s: s.start)


@pytest.fixture
def biobert_stub() -> BiobertNERStub:
    """N1: deterministic BioBERT stub for unit tests (no model download)."""
    return BiobertNERStub()
