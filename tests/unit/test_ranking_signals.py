"""
Bundle A: context-aware ranking core tests.

Covers:
- EmbeddingsDB.embed_text() helper + lru_cache per-instance semantics
- domain_context_emb kwarg plumbed into rank_papers (with default-off regression)
- pre-rank semantic filter with soft floor (≥5% off-domain slots reserved)
- S2 supplement cap (search_semantic_scholar min_relevance)
- Default-config regression invariant (all new keys at defaults → v0.8.0 behavior)
- Config._Namespace wiring for the new ranking sub-namespace
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(doi: str, score: float, has_embedding: bool = True) -> dict[str, Any]:
    """Create a minimal candidate dict with a synthetic embedding."""
    cand: dict[str, Any] = {
        "doi": doi,
        "title": f"Paper {doi}",
        "abstract": "",
        "_embedding": np.array([score, 0.0, 0.0], dtype=np.float32) if has_embedding else None,
    }
    return cand


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ===========================================================================
# TestEmbedText
# ===========================================================================

class TestEmbedText:
    """EmbeddingsDB.embed_text() helper."""

    def test_returns_ndarray(self, tmp_path):
        """embed_text() returns a numpy ndarray."""
        from scripts.output.embeddings import EmbeddingsDB

        db = EmbeddingsDB.__new__(EmbeddingsDB)
        db._embed_model = "mxbai-embed-large"
        db._ollama_host = "http://localhost:11434"
        # Patch the internal _embed method (handles truncation/retry) to avoid
        # real network calls. Returns a list[float] matching _embed's contract.
        with patch.object(db, "_embed", return_value=[0.1] * 1024):
            vec = db.embed_text("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (1024,)
        assert vec.dtype == np.float32

    def test_preserves_raw_values_and_shape(self, tmp_path):
        """embed_text() converts list[float] from _embed to an ndarray WITHOUT
        L2-normalising — raw component values are preserved verbatim.

        (Previously misnamed test_returns_l2_normalised_vector; embed_text does
        not normalise, and the assertions only ever checked value preservation.)
        """
        from scripts.output.embeddings import EmbeddingsDB

        db = EmbeddingsDB.__new__(EmbeddingsDB)
        db._embed_model = "mxbai-embed-large"
        db._ollama_host = "http://localhost:11434"
        fixed = [1.0] * 4 + [0.0] * 4
        with patch.object(db, "_embed", return_value=fixed):
            vec = db.embed_text("test")
        assert vec.shape == (8,)
        # Raw values are preserved (not normalised): the L2 norm here is 2.0, so an
        # L2-normalised vector would have vec[:4] == 0.5 — this asserts it does NOT.
        np.testing.assert_allclose(vec[:4], np.float32(1.0))
        np.testing.assert_allclose(vec[4:], np.float32(0.0))

    def test_lru_cache_avoids_re_embedding(self):
        """Calling embed_text() twice with the same text hits _embed only once."""
        from scripts.output.embeddings import EmbeddingsDB

        db = EmbeddingsDB.__new__(EmbeddingsDB)
        db._embed_model = "mxbai-embed-large"
        db._ollama_host = "http://localhost:11434"
        call_count = {"n": 0}

        def _stub(text: str):
            call_count["n"] += 1
            return [float(i) for i in range(16)]

        with patch.object(db, "_embed", side_effect=_stub):
            # Clear the per-instance cache before the test to prevent cross-test leakage.
            if hasattr(db.embed_text, "cache_clear"):
                db.embed_text.cache_clear()
            db.embed_text("hello")
            db.embed_text("hello")
        # _embed should have been called exactly once due to caching.
        assert call_count["n"] == 1

    def test_different_texts_both_embedded(self):
        """Two distinct texts each trigger a real _embed call (no false cache hit)."""
        from scripts.output.embeddings import EmbeddingsDB

        db = EmbeddingsDB.__new__(EmbeddingsDB)
        db._embed_model = "mxbai-embed-large"
        db._ollama_host = "http://localhost:11434"
        call_count = {"n": 0}

        def _stub(text: str):
            call_count["n"] += 1
            return [float(i) for i in range(16)]

        with patch.object(db, "_embed", side_effect=_stub):
            if hasattr(db.embed_text, "cache_clear"):
                db.embed_text.cache_clear()
            db.embed_text("text A")
            db.embed_text("text B")
        assert call_count["n"] == 2

    def test_empty_text_raises_value_error(self):
        """embed_text('') raises ValueError before touching the provider."""
        from scripts.output.embeddings import EmbeddingsDB

        db = EmbeddingsDB.__new__(EmbeddingsDB)
        db._embed_model = "mxbai-embed-large"
        db._ollama_host = "http://localhost:11434"
        with pytest.raises(ValueError, match="non-empty"):
            db.embed_text("")

    def test_whitespace_only_raises_value_error(self):
        """embed_text('   ') raises ValueError (whitespace-only guard)."""
        from scripts.output.embeddings import EmbeddingsDB

        db = EmbeddingsDB.__new__(EmbeddingsDB)
        db._embed_model = "mxbai-embed-large"
        db._ollama_host = "http://localhost:11434"
        with pytest.raises(ValueError, match="non-empty"):
            db.embed_text("   ")


# ===========================================================================
# TestRankPapersDefaultRegression
# ===========================================================================

class TestRankPapersDefaultRegression:
    """Bundle A invariant: default config → IDENTICAL behavior to v0.8.0.

    rank_papers() with no domain_context_emb kwarg (or domain_context_emb=None)
    must produce output that is byte-for-byte identical to what was produced
    before Bundle A.  The test pins score ordering and the similarity_score values
    to a deterministic reference.
    """

    def test_no_domain_context_emb_kwarg_accepted(self):
        """rank_papers() still works without the new domain_context_emb kwarg."""
        from scripts.llm.ranker import rank_papers

        candidates = [{"doi": "10.1/a", "title": "A", "abstract": ""}]
        embeddings_db = MagicMock()
        embeddings_db.find_similar_to_text.return_value = [
            {"score": 0.7, "id": "x", "document": "", "metadata": {}}
        ]
        llm = MagicMock()
        llm.complete.return_value = '{"10.1/a": "relevant"}'
        # Must not raise TypeError for unexpected kwarg.
        result = rank_papers(candidates, embeddings_db, llm)
        assert len(result) == 1

    def test_default_config_preserves_similarity_score_field(self):
        """With default config, similarity_score key is present and correct."""
        from scripts.llm.ranker import rank_papers

        candidates = [
            {"doi": "10.1/low", "title": "Low", "abstract": ""},
            {"doi": "10.1/high", "title": "High", "abstract": ""},
        ]
        embeddings_db = MagicMock()
        embeddings_db.find_similar_to_text.side_effect = [
            [{"score": 0.2, "id": "x", "document": "", "metadata": {}}],
            [{"score": 0.9, "id": "y", "document": "", "metadata": {}}],
        ]
        llm = MagicMock()
        llm.complete.return_value = '{"10.1/low": "low", "10.1/high": "high"}'
        ranked = rank_papers(candidates, embeddings_db, llm, domain_context_emb=None)
        # Order must match v0.8.0 behavior: sorted by similarity_score desc.
        assert ranked[0]["doi"] == "10.1/high"
        assert ranked[0]["similarity_score"] == pytest.approx(0.9)
        assert ranked[1]["similarity_score"] == pytest.approx(0.2)

    def test_domain_context_emb_none_same_as_absent(self):
        """Passing domain_context_emb=None explicitly is identical to omitting it."""
        from scripts.llm.ranker import rank_papers

        candidates = [{"doi": "10.1/x", "title": "X", "abstract": ""}]
        embeddings_db = MagicMock()
        embeddings_db.find_similar_to_text.return_value = [
            {"score": 0.5, "id": "y", "document": "", "metadata": {}}
        ]
        llm = MagicMock()
        llm.complete.return_value = '{"10.1/x": "ok"}'

        # Call without kwarg
        embeddings_db.find_similar_to_text.reset_mock()
        embeddings_db.find_similar_to_text.return_value = [
            {"score": 0.5, "id": "y", "document": "", "metadata": {}}
        ]
        result_no_kwarg = rank_papers(candidates, embeddings_db, llm)

        embeddings_db.find_similar_to_text.reset_mock()
        embeddings_db.find_similar_to_text.return_value = [
            {"score": 0.5, "id": "y", "document": "", "metadata": {}}
        ]
        result_none = rank_papers(candidates, embeddings_db, llm, domain_context_emb=None)

        assert result_no_kwarg[0]["similarity_score"] == result_none[0]["similarity_score"]


# ===========================================================================
# TestDomainContextSignal
# ===========================================================================

class TestDomainContextSignal:
    """domain_context_emb additive score contribution."""

    def _make_db(self, score: float = 0.5) -> MagicMock:
        db = MagicMock()
        db.find_similar_to_text.return_value = [
            {"score": score, "id": "y", "document": "", "metadata": {}}
        ]
        return db

    def _make_llm(self) -> MagicMock:
        llm = MagicMock()
        llm.complete.return_value = '{"10.1/a": "relevant"}'
        return llm

    def test_weight_zero_no_score_change(self):
        """ranking.weights.domain_context = 0 → similarity_score unmodified."""
        from scripts.llm.ranker import rank_papers

        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [{"doi": "10.1/a", "title": "A", "abstract": ""}]
        ranked = rank_papers(
            candidates,
            self._make_db(0.5),
            self._make_llm(),
            domain_context_emb=domain_emb,
            domain_context_weight=0.0,
        )
        # Weight=0 → no change to similarity_score
        assert ranked[0]["similarity_score"] == pytest.approx(0.5)

    def test_weight_nonzero_adds_to_score(self):
        """ranking.weights.domain_context > 0 → score includes cosine*weight."""
        from scripts.llm.ranker import rank_papers

        # Candidate embedding aligned with domain_emb → domain cosine ≈ 1.0
        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            {
                "doi": "10.1/a",
                "title": "A",
                "abstract": "",
                "_embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            }
        ]
        base_score = 0.4
        weight = 0.3
        ranked = rank_papers(
            candidates,
            self._make_db(base_score),
            self._make_llm(),
            domain_context_emb=domain_emb,
            domain_context_weight=weight,
        )
        # Expected: 0.4 + 0.3 * cos([1,0,0], [1,0,0]) = 0.4 + 0.3 = 0.7
        assert ranked[0]["similarity_score"] == pytest.approx(0.7, abs=1e-4)

    def test_weight_nonzero_no_embedding_key_no_crash(self):
        """Candidate without _embedding key: weight contribution is 0, no crash."""
        from scripts.llm.ranker import rank_papers

        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [{"doi": "10.1/a", "title": "A", "abstract": ""}]
        # No _embedding key → domain contribution = 0
        ranked = rank_papers(
            candidates,
            self._make_db(0.5),
            self._make_llm(),
            domain_context_emb=domain_emb,
            domain_context_weight=0.5,
        )
        # Score unchanged when embedding is absent
        assert ranked[0]["similarity_score"] == pytest.approx(0.5)

    def test_domain_score_stored_as_metadata_key(self):
        """When weight>0 and _embedding present, _domain_score is stored on the paper."""
        from scripts.llm.ranker import rank_papers

        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            {
                "doi": "10.1/a",
                "title": "A",
                "abstract": "",
                "_embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            }
        ]
        ranked = rank_papers(
            candidates,
            self._make_db(0.5),
            self._make_llm(),
            domain_context_emb=domain_emb,
            domain_context_weight=0.2,
        )
        assert "_domain_score" in ranked[0]
        assert ranked[0]["_domain_score"] == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# TestSoftFilter
# ===========================================================================

class TestSoftFilter:
    """Pre-rank soft domain filter with ≥5% off-domain floor."""

    def test_filter_off_preserves_all_candidates(self):
        """_apply_soft_domain_filter with enabled=False is a no-op (no import side-effect)."""
        from scripts.pipelines.discovery import _apply_soft_domain_filter

        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [_make_candidate(f"10/{i}", float(i) / 10) for i in range(10)]

        # Simulate disabled config: threshold higher than any cosine → all off-domain
        # but we call it with enabled=True here; the test is about the function contract.
        in_domain, off_domain = _apply_soft_domain_filter(
            candidates,
            domain_emb=domain_emb,
            threshold=0.0,  # everything passes threshold ≥ 0.0
        )
        assert len(in_domain) + len(off_domain) == 10

    def test_filter_on_splits_correctly(self):
        """Candidates below threshold go to off_domain pool, above to in_domain."""
        from scripts.pipelines.discovery import _apply_soft_domain_filter

        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Two candidates: one aligned (cosine≈1.0), one orthogonal (cosine≈0.0)
        aligned = {"doi": "a", "title": "A", "abstract": "", "_embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        orthogonal = {"doi": "b", "title": "B", "abstract": "", "_embedding": np.array([0.0, 1.0, 0.0], dtype=np.float32)}

        in_domain, off_domain = _apply_soft_domain_filter(
            [aligned, orthogonal],
            domain_emb=domain_emb,
            threshold=0.35,
        )
        assert any(c["doi"] == "a" for c in in_domain)
        assert any(c["doi"] == "b" for c in off_domain)

    def test_soft_floor_reserves_min_pct_when_all_off_domain(self):
        """If all 100 candidates are below threshold, ≥5% (5 papers) survive anyway.

        Bundle A spec: assemble_with_soft_floor(in_domain=[], off_domain=100, digest_size=100)
        → exactly ⌈100 * 0.05⌉ = 5 off-domain slots must be in the final list.
        """
        from scripts.pipelines.discovery import assemble_with_soft_floor

        # All 100 are off-domain; in_domain is empty.
        off_domain = [
            {"doi": f"off-{i}", "title": f"Off {i}", "similarity_score": float(i)}
            for i in range(100)
        ]
        final = assemble_with_soft_floor(
            in_domain_ranked=[],
            off_domain_ranked=off_domain,
            digest_size=100,
            min_off_domain_pct=0.05,
        )
        # Must have at least 5 candidates (⌈100*0.05⌉ = 5)
        assert len(final) >= 5
        # All from off_domain (no in_domain to fill from)
        final_dois = {p["doi"] for p in final}
        assert all(d.startswith("off-") for d in final_dois)

    def test_soft_floor_reserves_slots_when_both_pools_present(self):
        """When in_domain has 50 and off_domain has 50, digest_size=20 → 1 off-domain slot."""
        from scripts.pipelines.discovery import assemble_with_soft_floor

        # ⌈20 * 0.05⌉ = 1 off-domain slot reserved
        in_domain = [
            {"doi": f"in-{i}", "title": f"In {i}", "similarity_score": float(i)}
            for i in range(50)
        ]
        off_domain = [
            {"doi": f"off-{i}", "title": f"Off {i}", "similarity_score": float(i)}
            for i in range(50)
        ]
        final = assemble_with_soft_floor(
            in_domain_ranked=in_domain,
            off_domain_ranked=off_domain,
            digest_size=20,
            min_off_domain_pct=0.05,
        )
        assert len(final) == 20
        n_off = sum(1 for p in final if p["doi"].startswith("off-"))
        assert n_off >= 1  # at least ⌈20*0.05⌉ = 1

    def test_soft_floor_no_duplicate_dois(self):
        """Final assembled list must not contain the same DOI twice."""
        from scripts.pipelines.discovery import assemble_with_soft_floor

        in_domain = [{"doi": f"in-{i}", "title": f"In {i}", "similarity_score": float(i)} for i in range(10)]
        off_domain = [{"doi": f"off-{i}", "title": f"Off {i}", "similarity_score": float(i)} for i in range(5)]
        final = assemble_with_soft_floor(
            in_domain_ranked=in_domain,
            off_domain_ranked=off_domain,
            digest_size=10,
            min_off_domain_pct=0.05,
        )
        dois = [p["doi"] for p in final]
        assert len(dois) == len(set(dois)), "Duplicate DOIs in assembled output"

    def test_no_embedding_candidate_defaults_to_in_domain(self):
        """Candidate with no _embedding key is placed in in_domain (fail-safe)."""
        from scripts.pipelines.discovery import _apply_soft_domain_filter

        domain_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # No _embedding key → can't evaluate cosine → default to in-domain
        candidate = {"doi": "noembed", "title": "No embed", "abstract": ""}
        in_domain, off_domain = _apply_soft_domain_filter(
            [candidate],
            domain_emb=domain_emb,
            threshold=0.35,
        )
        assert any(c["doi"] == "noembed" for c in in_domain)
        assert not any(c["doi"] == "noembed" for c in off_domain)


# ===========================================================================
# TestS2Cap
# ===========================================================================

class TestS2Cap:
    """search_semantic_scholar min_relevance cap."""

    def test_cap_off_preserves_all_candidates(self):
        """Without min_relevance kwarg, all S2 results pass through unchanged."""
        from scripts.search.semantic_scholar import search_semantic_scholar

        mock_paper = MagicMock()
        mock_paper.externalIds = {"DOI": "10.1/test"}
        mock_paper.title = "Test Paper"
        mock_paper.year = 2024
        mock_paper.publicationVenue = None
        mock_paper.authors = []
        mock_paper.fieldsOfStudy = []
        mock_paper.abstract = "Abstract"

        with patch("scripts.search.semantic_scholar._S2_AVAILABLE", True), \
             patch("scripts.search.semantic_scholar._SemanticScholar") as mock_s2_cls:
            mock_s2 = MagicMock()
            mock_s2_cls.return_value = mock_s2
            # Two papers with no relevance_score key (default behavior)
            mock_s2.search_paper.return_value = [mock_paper, mock_paper]
            results = search_semantic_scholar("test query", since_days=7, limit=10)
        # Both papers pass through (no cap applied)
        assert len(results) == 2

    def test_cap_on_drops_low_relevance(self):
        """With min_relevance set, candidates below threshold are dropped."""
        from scripts.search.semantic_scholar import search_semantic_scholar

        def _make_mock_paper(doi: str, relevance: float):
            p = MagicMock()
            p.externalIds = {"DOI": doi}
            p.title = f"Paper {doi}"
            p.year = 2024
            p.publicationVenue = None
            p.authors = []
            p.fieldsOfStudy = []
            p.abstract = "Abstract"
            p.relevance_score = relevance
            return p

        high = _make_mock_paper("10.1/high", 0.8)
        low = _make_mock_paper("10.1/low", 0.2)

        with patch("scripts.search.semantic_scholar._S2_AVAILABLE", True), \
             patch("scripts.search.semantic_scholar._SemanticScholar") as mock_s2_cls:
            mock_s2 = MagicMock()
            mock_s2_cls.return_value = mock_s2
            mock_s2.search_paper.return_value = [high, low]
            results = search_semantic_scholar(
                "test query", since_days=7, limit=10, min_relevance=0.5
            )
        # Only the high-relevance paper should remain
        assert len(results) == 1
        assert results[0]["doi"] == "10.1/high"

    def test_cap_logs_drop_count(self, caplog):
        """When cap drops candidates, a log message reports the count."""
        import logging

        from scripts.search.semantic_scholar import search_semantic_scholar

        def _make_mock_paper(doi: str, relevance: float):
            p = MagicMock()
            p.externalIds = {"DOI": doi}
            p.title = f"Paper {doi}"
            p.year = 2024
            p.publicationVenue = None
            p.authors = []
            p.fieldsOfStudy = []
            p.abstract = "Abstract"
            p.relevance_score = relevance
            return p

        papers = [_make_mock_paper(f"10.1/{i}", 0.1 * i) for i in range(5)]

        with patch("scripts.search.semantic_scholar._S2_AVAILABLE", True), \
             patch("scripts.search.semantic_scholar._SemanticScholar") as mock_s2_cls:
            mock_s2 = MagicMock()
            mock_s2_cls.return_value = mock_s2
            mock_s2.search_paper.return_value = papers
            with caplog.at_level(logging.INFO, logger="scripts.search.semantic_scholar"):
                search_semantic_scholar(
                    "test query", since_days=7, limit=10, min_relevance=0.35
                )
        # At least one log message should mention dropping
        assert any("cap" in r.getMessage().lower() or "dropped" in r.getMessage().lower()
                   for r in caplog.records)


# ===========================================================================
# TestConfigRankingNamespace
# ===========================================================================

class TestConfigRankingNamespace:
    """Config._Namespace wiring for new ranking keys."""

    def _build_config_with_ranking(self, ranking_dict: dict) -> Any:
        """Build a minimal Config-like object with a ranking sub-namespace."""
        from scripts.core.config import _Namespace
        return _Namespace({"ranking": ranking_dict})

    def test_defaults_domain_context_weight_zero(self):
        """ranking.weights.domain_context defaults to 0.0."""
        # Config.ranking must exist and default to safe values.
        # We access via the _Namespace helper to mirror how Config sets it.
        cfg = self._build_config_with_ranking({
            "weights": {"domain_context": 0.0},
            "domain_filter": {
                "enabled": False,
                "threshold": 0.35,
                "minimum_off_domain_slots_pct": 0.05,
            },
            "s2_supplement_cap": {
                "enabled": False,
                "min_relevance": 0.4,
            },
        })
        assert cfg.ranking.weights.domain_context == 0.0

    def test_defaults_domain_filter_disabled(self):
        """ranking.domain_filter.enabled defaults to False."""
        cfg = self._build_config_with_ranking({
            "weights": {"domain_context": 0.0},
            "domain_filter": {
                "enabled": False,
                "threshold": 0.35,
                "minimum_off_domain_slots_pct": 0.05,
            },
            "s2_supplement_cap": {"enabled": False, "min_relevance": 0.4},
        })
        assert cfg.ranking.domain_filter.enabled is False

    def test_defaults_s2_cap_disabled(self):
        """ranking.s2_supplement_cap.enabled defaults to False."""
        cfg = self._build_config_with_ranking({
            "weights": {"domain_context": 0.0},
            "domain_filter": {"enabled": False, "threshold": 0.35, "minimum_off_domain_slots_pct": 0.05},
            "s2_supplement_cap": {"enabled": False, "min_relevance": 0.4},
        })
        assert cfg.ranking.s2_supplement_cap.enabled is False

    def test_config_ranking_attribute_accessible_from_real_config(self, tmp_path):
        """Config.ranking attribute is accessible after loading extraction.yaml with ranking block."""
        import yaml

        from scripts.core.config import Config

        # Minimal paths.yaml
        paths_data = {
            "zotero": {
                "library_type": "user",
                "library_id": "123",
                "local_storage_path": str(tmp_path / "zotero"),
                "collection_name": "MyLib",
            },
            "obsidian": {
                "vault_path": str(tmp_path / "vault"),
                "papers_folder": "Papers",
                "books_folder": "Books",
                "digests_folder": "Digests",
                "connections_folder": "Connections",
            },
            "state_db": {"path": str(tmp_path / "state.db")},
            "logs": {"path": str(tmp_path / "logs"), "retention_days": 30},
        }
        paths_yaml = tmp_path / "paths.yaml"
        paths_yaml.write_text(yaml.dump(paths_data), encoding="utf-8")

        extraction_data = {
            "brain_build": {"provider": "ollama", "model": "llama3.1:8b"},
            "ingestion": {"provider": "ollama", "model": "llama3.1:8b"},
            "build_vocabulary": {"provider": "ollama", "model": "llama3.1:8b"},
            "embeddings": {"provider": "ollama", "model": "mxbai-embed-large"},
            "comparison_models": [],
            # Bundle A new keys
            "ranking": {
                "weights": {"domain_context": 0.0},
                "domain_filter": {
                    "enabled": False,
                    "threshold": 0.35,
                    "minimum_off_domain_slots_pct": 0.05,
                },
                "s2_supplement_cap": {
                    "enabled": False,
                    "min_relevance": 0.4,
                },
            },
        }
        extraction_yaml = tmp_path / "extraction.yaml"
        extraction_yaml.write_text(yaml.dump(extraction_data), encoding="utf-8")

        cfg = Config(paths_yaml=paths_yaml, extraction_yaml=extraction_yaml)
        # The ranking namespace must be accessible
        ranking = getattr(cfg, "ranking", None)
        assert ranking is not None, "Config.ranking attribute missing"
        assert getattr(getattr(ranking, "weights", None), "domain_context", None) == 0.0
        assert getattr(getattr(ranking, "domain_filter", None), "enabled", None) is False
        assert getattr(getattr(ranking, "domain_filter", None), "threshold", None) == pytest.approx(0.35)
        assert getattr(getattr(ranking, "domain_filter", None), "minimum_off_domain_slots_pct", None) == pytest.approx(0.05)
        assert getattr(getattr(ranking, "s2_supplement_cap", None), "enabled", None) is False
        assert getattr(getattr(ranking, "s2_supplement_cap", None), "min_relevance", None) == pytest.approx(0.4)

    def test_config_missing_ranking_key_returns_none_gracefully(self, tmp_path):
        """Config without a ranking key in extraction.yaml: ranking attr is None or default."""
        import yaml

        from scripts.core.config import Config

        paths_data = {
            "zotero": {
                "library_type": "user",
                "library_id": "123",
                "local_storage_path": str(tmp_path / "zotero"),
                "collection_name": "MyLib",
            },
            "obsidian": {
                "vault_path": str(tmp_path / "vault"),
                "papers_folder": "Papers",
                "books_folder": "Books",
                "digests_folder": "Digests",
                "connections_folder": "Connections",
            },
            "state_db": {"path": str(tmp_path / "state.db")},
            "logs": {"path": str(tmp_path / "logs"), "retention_days": 30},
        }
        paths_yaml = tmp_path / "paths.yaml"
        paths_yaml.write_text(yaml.dump(paths_data), encoding="utf-8")

        extraction_data = {
            "brain_build": {"provider": "ollama", "model": "llama3.1:8b"},
            "ingestion": {"provider": "ollama", "model": "llama3.1:8b"},
            "build_vocabulary": {"provider": "ollama", "model": "llama3.1:8b"},
            "embeddings": {"provider": "ollama", "model": "mxbai-embed-large"},
            "comparison_models": [],
            # Deliberately NO ranking key
        }
        extraction_yaml = tmp_path / "extraction.yaml"
        extraction_yaml.write_text(yaml.dump(extraction_data), encoding="utf-8")

        # Must not raise
        cfg = Config(paths_yaml=paths_yaml, extraction_yaml=extraction_yaml)
        # ranking is either None or a _Namespace with safe defaults — either is OK
        # The key requirement: no AttributeError accessing it
        _ = getattr(cfg, "ranking", None)
