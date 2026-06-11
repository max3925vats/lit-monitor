"""
Unit tests for scripts/output/reranker.py (N19 cross-encoder reranking).

These tests mock the CrossEncoder model to avoid downloading ~670MB during CI
and to allow running before `sentence-transformers` has been installed.

Strategy: inject a mock `sentence_transformers` module into sys.modules so
that the local ``from sentence_transformers import CrossEncoder`` inside
``Reranker.__init__`` resolves without the real package being present.

We verify:
  - Reranker instantiation is lazy (no I/O at import)
  - rerank() correctly reorders candidates based on model scores
  - disabled reranker config skips the reranker path in embeddings.py
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidates(texts: list[str]) -> list[dict]:
    """Build minimal result dicts as returned by ChromaDB query."""
    return [
        {"id": f"10.1/p{i}", "score": 0.9 - i * 0.1, "document": t, "metadata": {}}
        for i, t in enumerate(texts)
    ]


def _make_reranker_config(enabled: bool = True, model: str = "mock-model"):
    return SimpleNamespace(
        enabled=enabled,
        model=model,
        candidate_multiplier=3,
        device=None,
    )


# ---------------------------------------------------------------------------
# Fixture: inject a fake sentence_transformers module
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_st():
    """
    Insert a minimal fake sentence_transformers module into sys.modules.

    The fake's CrossEncoder attribute is a plain MagicMock — tests that
    need specific predict() return values should configure it themselves.
    Also resets the reranker singleton so each test gets a fresh state.
    """
    import lit_monitor.output.reranker as reranker_mod

    fake_mod = types.ModuleType("sentence_transformers")
    mock_ce_cls = MagicMock(name="CrossEncoder")
    fake_mod.CrossEncoder = mock_ce_cls

    # Inject without disturbing a real installation if present.
    was_present = "sentence_transformers" in sys.modules
    prev = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_mod

    # Reset singleton
    reranker_mod._reranker_instance = None
    reranker_mod._reranker_model_loaded = None

    yield fake_mod

    # Restore previous state
    if was_present and prev is not None:
        sys.modules["sentence_transformers"] = prev
    elif not was_present:
        sys.modules.pop("sentence_transformers", None)

    reranker_mod._reranker_instance = None
    reranker_mod._reranker_model_loaded = None


# ---------------------------------------------------------------------------
# Test 1: Reranker module is importable without side effects
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reranker_module_imports_without_side_effects():
    """Importing reranker.py must not download or load any model."""
    import importlib
    mod = importlib.import_module("lit_monitor.output.reranker")
    # Module-level singleton should be None until get_reranker() is called.
    assert mod._reranker_instance is None


# ---------------------------------------------------------------------------
# Test 2: rerank() reorders candidates by model scores
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rerank_reorders_by_cross_encoder_score(fake_st):
    """rerank() returns candidates sorted by cross-encoder score, not input order."""
    from lit_monitor.output.reranker import Reranker

    candidates = _make_candidates(["worst match", "medium match", "best match"])
    # Scores assigned in reverse order: 'best match' gets highest score.
    mock_scores = [0.1, 0.5, 0.95]

    mock_ce_instance = MagicMock()
    mock_ce_instance.predict.return_value = mock_scores
    fake_st.CrossEncoder.return_value = mock_ce_instance

    reranker = Reranker(model_name="mock-model")
    result = reranker.rerank("my query", candidates, top_k=3)

    assert result[0]["document"] == "best match"
    assert result[1]["document"] == "medium match"
    assert result[2]["document"] == "worst match"


@pytest.mark.unit
def test_rerank_truncates_to_top_k(fake_st):
    """top_k truncation works after reranking."""
    from lit_monitor.output.reranker import Reranker

    candidates = _make_candidates(["a", "b", "c", "d"])
    mock_scores = [0.8, 0.6, 0.4, 0.2]

    mock_ce_instance = MagicMock()
    mock_ce_instance.predict.return_value = mock_scores
    fake_st.CrossEncoder.return_value = mock_ce_instance

    reranker = Reranker(model_name="mock-model")
    result = reranker.rerank("query", candidates, top_k=2)

    assert len(result) == 2
    assert result[0]["document"] == "a"  # highest score


@pytest.mark.unit
def test_rerank_empty_candidates_returns_empty(fake_st):
    """rerank() with empty input returns []."""
    from lit_monitor.output.reranker import Reranker

    fake_st.CrossEncoder.return_value = MagicMock()
    reranker = Reranker(model_name="mock-model")
    result = reranker.rerank("query", [])

    assert result == []


@pytest.mark.unit
def test_rerank_result_has_rerank_score_key(fake_st):
    """Each output dict must carry a rerank_score field."""
    from lit_monitor.output.reranker import Reranker

    candidates = _make_candidates(["doc"])
    mock_ce_instance = MagicMock()
    mock_ce_instance.predict.return_value = [0.7]
    fake_st.CrossEncoder.return_value = mock_ce_instance

    reranker = Reranker(model_name="mock-model")
    result = reranker.rerank("q", candidates)

    assert "rerank_score" in result[0]
    assert isinstance(result[0]["rerank_score"], float)


# ---------------------------------------------------------------------------
# Test 3: disabled config skips reranker in embeddings.py helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reranker_disabled_config_skips_reranking():
    """`_reranker_enabled` returns False when config.enabled is False."""
    from lit_monitor.output.embeddings import _reranker_enabled

    cfg = _make_reranker_config(enabled=False)
    assert _reranker_enabled(cfg) is False


@pytest.mark.unit
def test_reranker_none_config_skips_reranking():
    """`_reranker_enabled` returns False when config is None."""
    from lit_monitor.output.embeddings import _reranker_enabled

    assert _reranker_enabled(None) is False


@pytest.mark.unit
def test_reranker_enabled_config_returns_true():
    """`_reranker_enabled` returns True when config.enabled is True."""
    from lit_monitor.output.embeddings import _reranker_enabled

    cfg = _make_reranker_config(enabled=True)
    assert _reranker_enabled(cfg) is True


@pytest.mark.unit
def test_reranker_multiplier_with_disabled_config_returns_1():
    """`_reranker_multiplier` returns 1 (no extra candidates) when reranker disabled."""
    from lit_monitor.output.embeddings import _reranker_multiplier

    cfg = _make_reranker_config(enabled=False)
    assert _reranker_multiplier(cfg, "query") == 1


@pytest.mark.unit
def test_reranker_multiplier_with_enabled_config():
    """`_reranker_multiplier` returns configured multiplier when reranker enabled."""
    from lit_monitor.output.embeddings import _reranker_multiplier

    cfg = _make_reranker_config(enabled=True)
    cfg.candidate_multiplier = 3
    assert _reranker_multiplier(cfg, "query") == 3


@pytest.mark.unit
def test_find_similar_to_text_calls_reranker_when_enabled(tmp_path):
    """find_similar_to_text passes candidates to the reranker when config enables it."""
    from lit_monitor.output.embeddings import EmbeddingsDB

    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
    db._embed = MagicMock(return_value=[0.1] * 768)

    db.add_paper("doi1", "text1", {"source_type": "paper", "note_title": "A"})
    db.add_paper("doi2", "text2", {"source_type": "paper", "note_title": "B"})

    cfg = _make_reranker_config(enabled=True)

    with patch("lit_monitor.output.embeddings._apply_reranker") as mock_apply:
        mock_apply.return_value = [{"id": "doi1", "score": 0.9, "document": "text1", "metadata": {}}]
        result = db.find_similar_to_text(
            "query", top_k=1, rerank_with_query="query", reranker_config=cfg
        )

    mock_apply.assert_called_once()
    assert len(result) == 1


@pytest.mark.unit
def test_find_similar_to_text_skips_reranker_when_disabled(tmp_path):
    """find_similar_to_text does NOT call reranker when config.enabled is False."""
    from lit_monitor.output.embeddings import EmbeddingsDB

    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
    db._embed = MagicMock(return_value=[0.1] * 768)
    db.add_paper("doi1", "text1", {"source_type": "paper", "note_title": "A"})

    cfg = _make_reranker_config(enabled=False)

    with patch("lit_monitor.output.embeddings._apply_reranker") as mock_apply:
        db.find_similar_to_text("query", top_k=1, rerank_with_query="query", reranker_config=cfg)

    mock_apply.assert_not_called()
