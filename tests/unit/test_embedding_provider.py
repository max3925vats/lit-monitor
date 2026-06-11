"""Bundle F: embedding provider abstraction tests.

Coverage:
  - embedding_provenance table CRUD
  - EmbeddingsDB provider dispatch (ollama + litellm stub)
  - invalid provider rejection
  - provenance overrides constructor args (dim-mismatch guard)
  - Bundle A regression: embed_text still cached + dispatched through provider
  - extraction.example.yaml has the embedding block
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Provenance table
# ---------------------------------------------------------------------------
class TestProvenanceTable:
    def test_table_exists(self, tmp_path):
        """embedding_provenance table is created on first StateDB init."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_provenance'"
            ).fetchall()
        assert rows, "embedding_provenance table should exist after StateDB init"

    def test_record_provenance(self, tmp_path):
        """record_embedding_provenance writes a row readable via raw SQL."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        db.record_embedding_provenance(
            collection_name="papers",
            provider="ollama",
            model="mxbai-embed-large",
            dim=1024,
        )
        with db._connect() as conn:
            row = conn.execute(
                "SELECT provider, model, dim, is_current "
                "FROM embedding_provenance WHERE collection_name='papers'"
            ).fetchone()
        assert row[0] == "ollama"
        assert row[1] == "mxbai-embed-large"
        assert row[2] == 1024
        assert row[3] == 1  # newly recorded → is_current

    def test_get_provenance_returns_dict(self, tmp_path):
        """get_embedding_provenance returns a proper dict, not None."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        db.record_embedding_provenance("papers", "ollama", "mxbai-embed-large", 1024)
        result = db.get_embedding_provenance("papers")
        assert result is not None
        assert result["provider"] == "ollama"
        assert result["model"] == "mxbai-embed-large"
        assert result["dim"] == 1024
        assert result["is_current"] is True

    def test_get_provenance_missing_returns_none(self, tmp_path):
        """get_embedding_provenance returns None for unknown collection."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        assert db.get_embedding_provenance("nonexistent") is None

    def test_set_current_clears_old(self, tmp_path):
        """set_current_embedding_collection marks only one row current."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        db.record_embedding_provenance("papers_v1", "ollama", "mxbai-embed-large", 1024)
        db.record_embedding_provenance("papers_v2_openai", "litellm", "text-embedding-3-large", 3072)
        db.set_current_embedding_collection("papers_v2_openai")
        with db._connect() as conn:
            rows = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT collection_name, is_current FROM embedding_provenance"
                ).fetchall()
            }
        assert rows["papers_v1"] == 0
        assert rows["papers_v2_openai"] == 1

    def test_list_provenance_returns_all_rows(self, tmp_path):
        """list_embedding_provenance returns all rows in created_at DESC order."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        db.record_embedding_provenance("papers_v1", "ollama", "mxbai-embed-large", 1024)
        db.record_embedding_provenance("papers_v2_litellm", "litellm", "text-embedding-3-large", 3072)
        rows = db.list_embedding_provenance()
        assert len(rows) == 2
        names = [r["collection_name"] for r in rows]
        assert "papers_v1" in names
        assert "papers_v2_litellm" in names

    def test_record_provenance_upsert(self, tmp_path):
        """record_embedding_provenance is idempotent — second call overwrites."""
        from lit_monitor.core.state_db import StateDB
        db = StateDB(tmp_path / "state.db")
        db.record_embedding_provenance("papers", "ollama", "mxbai-embed-large", 1024)
        db.record_embedding_provenance("papers", "ollama", "nomic-embed-text", 768)
        result = db.get_embedding_provenance("papers")
        assert result["model"] == "nomic-embed-text"
        assert result["dim"] == 768


# ---------------------------------------------------------------------------
# EmbeddingsDB provider dispatch
# ---------------------------------------------------------------------------
class TestEmbeddingsDBProviderDispatch:
    """Bundle F: EmbeddingsDB dispatches to Ollama or LiteLLM based on provider."""

    def test_ollama_default_dispatch(self, tmp_path):
        """Default constructor (no provider kwarg) routes through Ollama path.

        The Ollama path goes through _embed (truncation/retry) → _embed_call,
        NOT through embed_via_ollama directly.  We patch _embed_call to avoid
        real network calls while still confirming the dispatch path.
        """
        from lit_monitor.output.embeddings import EmbeddingsDB
        fake_vec = [1.0] * 1024  # _embed_call returns list[float]
        db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
        assert db.provider == "ollama"
        with patch.object(db, "_embed_call", return_value=fake_vec) as mock:
            vec = db.embed_text("hello world")
        mock.assert_called_once()
        assert isinstance(vec, np.ndarray)

    def test_litellm_dispatch_when_configured(self, tmp_path):
        """provider='litellm' routes through LiteLLM path (litellm import optional)."""
        pytest.importorskip("litellm")
        from lit_monitor.output.embeddings import EmbeddingsDB
        fake_vec = np.ones(3072, dtype=np.float32)
        with patch("scripts.llm.embedding_client.embed_via_litellm", return_value=fake_vec) as mock:
            db = EmbeddingsDB(
                persist_dir=str(tmp_path / "chroma"),
                provider="litellm",
                model="text-embedding-3-large",
                dim=3072,
            )
            vec = db.embed_text("hello world")
        mock.assert_called_once()
        assert isinstance(vec, np.ndarray)

    def test_invalid_provider_raises(self, tmp_path):
        """provider='bogus' raises ValueError with 'provider' in the message."""
        from lit_monitor.output.embeddings import EmbeddingsDB
        with pytest.raises(ValueError, match="provider"):
            EmbeddingsDB(persist_dir=str(tmp_path / "chroma"), provider="bogus")

    def test_add_paper_uses_provider(self, tmp_path):
        """add_paper() goes through the provider abstraction, not hardcoded Ollama."""
        from lit_monitor.output.embeddings import EmbeddingsDB
        db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
        # _embed_call is the final Ollama HTTP call; patch it to avoid network I/O.
        with patch.object(db, "_embed_call", return_value=[0.5] * 1024):
            db.add_paper(
                doi="10.1234/test",
                text="Test title. Test abstract.",
                metadata={"source_type": "paper", "title": "Test", "year": 2024},
            )
        # If no error raised, the dispatch worked correctly
        assert db.count() == 1


# ---------------------------------------------------------------------------
# Provenance overrides constructor args
# ---------------------------------------------------------------------------
class TestProvenanceOverridesConstructor:
    """Bundle F: if collection recorded in provenance, those values win."""

    def test_existing_collection_provenance_wins(self, tmp_path):
        """Re-opening a collection uses stored provenance, not constructor args.

        Scenario: user recorded ollama/mxbai/1024 previously. Then they construct
        EmbeddingsDB with litellm/3072. The provenance record should win.
        """

        # Manually plant a provenance record in state.db
        state_db_path = tmp_path / "state.db"
        from lit_monitor.core.state_db import StateDB
        sdb = StateDB(state_db_path)
        sdb.record_embedding_provenance(
            collection_name="lit_monitor_v1",  # _COLLECTION_NAME default
            provider="ollama",
            model="mxbai-embed-large",
            dim=1024,
        )

        fake_ollama_vec = np.ones(1024, dtype=np.float32)
        with patch("scripts.llm.embedding_client.embed_via_ollama", return_value=fake_ollama_vec):
            # Construct with litellm — provenance should override
            from lit_monitor.output.embeddings import EmbeddingsDB
            db = EmbeddingsDB(
                persist_dir=str(tmp_path / "chroma"),
                provider="litellm",
                model="text-embedding-3-large",
                dim=3072,
                state_db_path=str(state_db_path),
            )
            assert db.provider == "ollama"  # provenance wins
            assert db.dim == 1024

    def test_no_provenance_uses_constructor_args(self, tmp_path):
        """With no recorded provenance, constructor args are used as-is."""
        from lit_monitor.output.embeddings import EmbeddingsDB

        # Import here to avoid circular import issues
        db = EmbeddingsDB(
            persist_dir=str(tmp_path / "chroma"),
            provider="ollama",
            model="mxbai-embed-large",
            dim=1024,
        )
        assert db.provider == "ollama"
        assert db.model == "mxbai-embed-large"
        assert db.dim == 1024


# ---------------------------------------------------------------------------
# Bundle A regression: embed_text still works through the abstraction
# ---------------------------------------------------------------------------
class TestBundleAEmbedTextRegression:
    """Bundle F: embed_text() helper must still work after the provider abstraction."""

    def test_embed_text_cache_still_active(self, tmp_path):
        """Same text embedded twice → second call returns cached result (no second dispatch).

        For the Ollama path, _embed_call is the underlying call.  Patch it to
        count calls without real network I/O.
        """
        from lit_monitor.output.embeddings import EmbeddingsDB
        call_count = 0

        def counting_embed_call(text: str) -> list:
            nonlocal call_count
            call_count += 1
            return [1.0] * 1024

        db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
        with patch.object(db, "_embed_call", side_effect=counting_embed_call):
            _ = db.embed_text("hello cache test")
            _ = db.embed_text("hello cache test")  # second call — should hit cache

        assert call_count == 1, "_embed_call should be called only once (cache hit on second)"

    def test_embed_text_cache_is_per_instance(self, tmp_path):
        """The embed_text cache lives on the instance, not in a shared class-level
        store. This is the regression guard for the lru_cache memory leak: the old
        ``@lru_cache`` on a bound method kept every EmbeddingsDB instance alive in a
        module-global cache. A per-instance dict means instance B does NOT see
        instance A's cached vectors (so A is free to be garbage-collected).
        """
        from lit_monitor.output.embeddings import EmbeddingsDB

        db_a = EmbeddingsDB(persist_dir=str(tmp_path / "chroma_a"))
        db_b = EmbeddingsDB(persist_dir=str(tmp_path / "chroma_b"))

        with patch.object(db_a, "_embed_call", return_value=[1.0] * 1024):
            db_a.embed_text("shared text")
        # db_a now has the entry; db_b must NOT — separate caches.
        assert "shared text" in db_a._embed_text_cache
        assert "shared text" not in getattr(db_b, "_embed_text_cache", {})

        # db_b embedding the same text dispatches its own call (no cross-instance hit).
        with patch.object(db_b, "_embed_call", return_value=[2.0] * 1024) as mock_b:
            db_b.embed_text("shared text")
        mock_b.assert_called_once()

    def test_embed_text_cache_evicts_lru(self, tmp_path):
        """Cache is bounded at _EMBED_TEXT_CACHE_MAXSIZE; the least-recently-used
        entry is evicted once the cap is exceeded."""
        from lit_monitor.output.embeddings import EmbeddingsDB

        db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
        cap = db._EMBED_TEXT_CACHE_MAXSIZE
        with patch.object(db, "_embed_call", return_value=[1.0] * 1024):
            for i in range(cap + 5):
                db.embed_text(f"text {i}")
        assert len(db._embed_text_cache) == cap
        # The earliest inserts were evicted; the most recent are retained.
        assert "text 0" not in db._embed_text_cache
        assert f"text {cap + 4}" in db._embed_text_cache

    def test_embed_text_returns_ndarray(self, tmp_path):
        """embed_text returns np.ndarray, not a raw list.

        The Ollama path routes through EmbeddingsDB._embed (not
        embed_via_ollama directly), so we patch the bound method. This
        keeps the test offline — without it, CI runners with no Ollama
        connection fail with URLError.
        """
        from lit_monitor.output.embeddings import EmbeddingsDB
        db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
        # _embed returns list[float]; embed_text wraps it via np.array(..., float32).
        with patch.object(db, "_embed", return_value=[1.0] * 1024):
            result = db.embed_text("some text")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1024,)

    def test_embed_text_empty_raises(self, tmp_path):
        """embed_text with empty string still raises ValueError."""
        from lit_monitor.output.embeddings import EmbeddingsDB
        db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
        with pytest.raises(ValueError):
            db.embed_text("")

    def test_embed_text_dispatches_through_provider(self, tmp_path):
        """embed_text routes through _embed_via_provider (which for Ollama calls _embed).

        Verify the provider attribute is set correctly and the call reaches
        _embed_call (the final Ollama HTTP call) rather than bypassing it.
        """
        from lit_monitor.output.embeddings import EmbeddingsDB
        db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))
        assert db.provider == "ollama"
        assert db.model == "mxbai-embed-large"
        with patch.object(db, "_embed_call", return_value=[0.5] * 1024) as mock:
            db.embed_text("dispatch test")
        # The Ollama path: embed_text → _embed_via_provider → _embed → _embed_call
        mock.assert_called_once_with("dispatch test")


# ---------------------------------------------------------------------------
# extraction.example.yaml has the embedding block
# ---------------------------------------------------------------------------
class TestExtractionExampleYaml:
    def test_embedding_block_present(self):
        """config/extraction.example.yaml must contain the full embedding: block."""
        from pathlib import Path
        raw = (Path(__file__).parents[2] / "config" / "extraction.example.yaml").read_text()
        assert "embedding:" in raw, "embedding: top-level key missing"
        assert "provider:" in raw, "embedding.provider key missing"
        assert "ollama:" in raw, "embedding.ollama sub-block missing"
        assert "litellm:" in raw, "embedding.litellm sub-block missing"
