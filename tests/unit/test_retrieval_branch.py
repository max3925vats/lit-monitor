"""G9: unit tests for scripts/retrieval/branch.py and related helpers.

Tests the retrieve_doi_candidates dispatcher, synthesize._expand_query fallback,
and synthesize rag-mode branching — all without discovery-pipeline imports
(avoids habanero dependency that blocks test_pipelines_and_tools.py collection).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_state_db(tmp_path: Path):
    from scripts.core.state_db import StateDB
    return StateDB(tmp_path / "state.db")


def _make_config(tmp_path: Path) -> SimpleNamespace:
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    (vault / "Literature" / "Connections").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        obsidian=SimpleNamespace(
            vault_path=str(vault),
            connections_folder="Literature/Connections",
        ),
        reranker=None,
        retrieval=SimpleNamespace(default_mode="vector"),
    )


# ===========================================================================
# retrieve_doi_candidates dispatch
# ===========================================================================

@pytest.mark.unit
def test_retrieve_doi_candidates_vector_returns_embeddings():
    """G9 branch helper: vector mode calls embeddings_db.find_similar_to_text."""
    from scripts.retrieval.branch import retrieve_doi_candidates

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = [
        {"id": "10.0/a", "score": 0.9},
        {"id": "10.0/b", "score": 0.7},
    ]
    graph_db = MagicMock()

    results = retrieve_doi_candidates(
        "vector",
        query_text="filtration protein",
        embeddings_db=embeddings_db,
        graph_db=graph_db,
        k=5,
    )

    embeddings_db.find_similar_to_text.assert_called_once()
    graph_db.find_similar_papers.assert_not_called()
    assert results[0][0] == "10.0/a"
    assert results[0][1] == pytest.approx(0.9)


@pytest.mark.unit
def test_retrieve_doi_candidates_graph_returns_graph_results():
    """G9 branch helper: graph mode calls graph_db.find_similar_papers."""
    from scripts.retrieval.branch import retrieve_doi_candidates

    embeddings_db = MagicMock()
    graph_db = MagicMock()
    graph_db.find_similar_papers.return_value = [("10.0/g1", 5.0), ("10.0/g2", 3.0)]

    results = retrieve_doi_candidates(
        "graph",
        seed_doi="10.0/seed",
        embeddings_db=embeddings_db,
        graph_db=graph_db,
        k=5,
    )

    graph_db.find_similar_papers.assert_called_with("10.0/seed", k=5)
    embeddings_db.find_similar_to_text.assert_not_called()
    assert results[0] == ("10.0/g1", 5.0)


@pytest.mark.unit
def test_retrieve_doi_candidates_hybrid_fuses_both():
    """G9 branch helper: hybrid mode calls both and returns fused DOIs."""
    from scripts.retrieval.branch import retrieve_doi_candidates

    embeddings_db = MagicMock()
    embeddings_db.find_similar_to_text.return_value = [
        {"id": "10.0/vec", "score": 0.9},
    ]
    graph_db = MagicMock()
    graph_db.find_similar_papers.return_value = [("10.0/gra", 5.0)]

    results = retrieve_doi_candidates(
        "hybrid",
        seed_doi="10.0/seed",
        query_text="some topic",
        embeddings_db=embeddings_db,
        graph_db=graph_db,
        k=10,
    )

    embeddings_db.find_similar_to_text.assert_called()
    graph_db.find_similar_papers.assert_called()
    dois = [d for d, _ in results]
    assert "10.0/vec" in dois
    assert "10.0/gra" in dois


@pytest.mark.unit
def test_retrieve_doi_candidates_vector_empty_when_no_embeddings_db():
    """G9 branch helper: vector mode returns [] when embeddings_db is None."""
    from scripts.retrieval.branch import retrieve_doi_candidates

    results = retrieve_doi_candidates(
        "vector",
        query_text="test",
        embeddings_db=None,
        graph_db=None,
        k=5,
    )
    assert results == []


@pytest.mark.unit
def test_retrieve_doi_candidates_graph_empty_when_no_graph_db():
    """G9 branch helper: graph mode returns [] when graph_db is None."""
    from scripts.retrieval.branch import retrieve_doi_candidates

    results = retrieve_doi_candidates(
        "graph",
        seed_doi="10.0/x",
        embeddings_db=None,
        graph_db=None,
        k=5,
    )
    assert results == []


@pytest.mark.unit
def test_branch_helper_unknown_mode_raises():
    """G9: retrieve_doi_candidates raises ValueError for unknown rag_mode."""
    from scripts.retrieval.branch import retrieve_doi_candidates

    with pytest.raises(ValueError, match="Unknown rag_mode"):
        retrieve_doi_candidates("turbo", query_text="test")


@pytest.mark.unit
def test_retrieve_doi_candidates_graph_failure_returns_empty():
    """G9 branch helper: graph retrieval failure is caught and returns []."""
    from scripts.retrieval.branch import retrieve_doi_candidates

    graph_db = MagicMock()
    graph_db.find_similar_papers.side_effect = RuntimeError("graph DB down")

    results = retrieve_doi_candidates(
        "graph",
        seed_doi="10.0/x",
        graph_db=graph_db,
        k=5,
    )
    assert results == []


# ===========================================================================
# _expand_query helpers (synthesize)
# ===========================================================================

@pytest.mark.unit
def test_expand_query_falls_back_to_topic_on_llm_failure():
    """G9: _expand_query returns [topic] and logs WARN when LLM raises."""
    from scripts.obsidian_tools.synthesize import _expand_query

    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("LLM offline")

    # Patch load_prompt to avoid file-system dependency in unit tests.
    with patch("scripts.obsidian_tools.synthesize.load_prompt") as mock_lp:
        mock_lp.return_value.system = "You are a helper."
        mock_lp.return_value.render_user.return_value = "Expand: topic"
        mock_lp.return_value.max_tokens = 256

        result = _expand_query("membrane fouling", llm)

    assert result == ["membrane fouling"], (
        f"Expected topic-only fallback, got {result}"
    )


@pytest.mark.unit
def test_expand_query_raises_under_strict():
    """P4.4: in --strict, a query-expansion LLM failure escalates to a raise."""
    import scripts.core.strict_mode as _sm
    from scripts.obsidian_tools.synthesize import _expand_query

    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("LLM offline")

    _sm.set_strict(True)
    with patch("scripts.obsidian_tools.synthesize.load_prompt") as mock_lp:
        mock_lp.return_value.system = "You are a helper."
        mock_lp.return_value.render_user.return_value = "Expand: topic"
        mock_lp.return_value.max_tokens = 256
        with pytest.raises(RuntimeError, match="Query expansion failed"):
            _expand_query("membrane fouling", llm)


@pytest.mark.unit
def test_expand_query_returns_list_on_success():
    """G9: _expand_query returns LLM-parsed list on success."""
    from scripts.obsidian_tools.synthesize import _expand_query

    llm = MagicMock()
    expanded_json = '["membrane fouling", "biofouling", "flux decline"]'
    llm.complete.return_value = expanded_json

    with patch("scripts.obsidian_tools.synthesize.load_prompt") as mock_lp:
        mock_lp.return_value.system = "system text"
        mock_lp.return_value.render_user.return_value = "user text"
        mock_lp.return_value.max_tokens = 256

        result = _expand_query("membrane fouling", llm)

    assert result == ["membrane fouling", "biofouling", "flux decline"]


@pytest.mark.unit
def test_expand_query_none_llm_returns_topic():
    """G9: _expand_query returns [topic] when llm is None."""
    from scripts.obsidian_tools.synthesize import _expand_query

    result = _expand_query("bioreactor fouling", None)
    assert result == ["bioreactor fouling"]


@pytest.mark.unit
def test_expand_query_bad_json_falls_back():
    """G9: _expand_query returns [topic] when LLM returns non-JSON."""
    from scripts.obsidian_tools.synthesize import _expand_query

    llm = MagicMock()
    llm.complete.return_value = "Sorry, I cannot help with that."

    with patch("scripts.obsidian_tools.synthesize.load_prompt") as mock_lp:
        mock_lp.return_value.system = "system text"
        mock_lp.return_value.render_user.return_value = "user text"
        mock_lp.return_value.max_tokens = 256

        result = _expand_query("some topic", llm)

    assert result == ["some topic"]


# ===========================================================================
# Config.retrieval.default_mode
# ===========================================================================

@pytest.mark.unit
def test_config_retrieval_default_mode_defaults_to_vector(tmp_path):
    """G9: Config.retrieval.default_mode defaults to 'vector' when absent from YAML."""
    # Write minimal valid configs.
    import yaml

    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(
        yaml.dump({
            "obsidian": {"vault_path": str(tmp_path / "vault")},
        }),
        encoding="utf-8",
    )
    (tmp_path / "vault").mkdir()

    extraction_yaml = tmp_path / "extraction.yaml"
    extraction_yaml.write_text(
        yaml.dump({
            "brain_build": {"provider": "ollama"},
            "ingestion": {"provider": "ollama"},
            "build_vocabulary": {"provider": "ollama"},
            # No "retrieval" key.
        }),
        encoding="utf-8",
    )

    from scripts.core.config import Config
    cfg = Config(paths_yaml=paths_yaml, extraction_yaml=extraction_yaml)
    assert cfg.retrieval.default_mode == "vector"


@pytest.mark.unit
def test_config_retrieval_default_mode_reads_from_yaml(tmp_path):
    """G9: Config.retrieval.default_mode reads 'graph' from extraction.yaml."""
    import yaml

    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(
        yaml.dump({
            "obsidian": {"vault_path": str(tmp_path / "vault")},
        }),
        encoding="utf-8",
    )
    (tmp_path / "vault").mkdir()

    extraction_yaml = tmp_path / "extraction.yaml"
    extraction_yaml.write_text(
        yaml.dump({
            "brain_build": {"provider": "ollama"},
            "ingestion": {"provider": "ollama"},
            "build_vocabulary": {"provider": "ollama"},
            "retrieval": {"default_mode": "graph"},
        }),
        encoding="utf-8",
    )

    from scripts.core.config import Config
    cfg = Config(paths_yaml=paths_yaml, extraction_yaml=extraction_yaml)
    assert cfg.retrieval.default_mode == "graph"


# ===========================================================================
# synthesize rag-mode integration (no discovery/habanero dependency)
# ===========================================================================

@pytest.mark.unit
def test_synthesize_graph_mode_calls_expand_query(tmp_path):
    """G9: synthesize with rag_mode='graph' triggers _expand_query."""
    from scripts.obsidian_tools.synthesize import synthesize

    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    embeddings_db.find_similar_chunks.return_value = []
    embeddings_db.find_similar_to_text.return_value = [
        {"id": "10.1/related", "score": 0.8, "document": "", "metadata": {}},
    ]
    llm = MagicMock()
    llm.complete.return_value = "Synthesis text about the topic."

    state_db.upsert_paper({
        "doi": "10.1/related", "title": "Related Paper", "source_type": "paper",
        "note_title": "Related_Paper",
        "extraction_json": json.dumps({"core_finding": "test"}),
        "status": "extraction_complete",
    })

    expand_calls: list = []

    def fake_expand(topic, llm_):
        expand_calls.append(topic)
        return [topic]

    with patch("scripts.obsidian_tools.synthesize._expand_query", side_effect=fake_expand), \
         patch("scripts.retrieval.branch.retrieve_doi_candidates",
               return_value=[("10.1/related", 0.8)]):
        synthesize(
            "protein filtration", config, state_db, embeddings_db, llm,
            rag_mode="graph",
        )

    assert expand_calls, "_expand_query must be called for graph mode"
    assert expand_calls[0] == "protein filtration"


@pytest.mark.unit
def test_synthesize_vector_mode_unchanged(tmp_path):
    """G9: synthesize with rag_mode='vector' uses the existing chunk/paper path unchanged."""
    from scripts.obsidian_tools.synthesize import synthesize

    config = _make_config(tmp_path)
    state_db = _make_state_db(tmp_path)
    embeddings_db = MagicMock()
    llm = MagicMock()
    llm.complete.return_value = "Synthesis paragraph."

    state_db.upsert_paper({
        "doi": "10.1/r", "title": "R Paper", "source_type": "paper",
        "note_title": "R_Paper",
        "extraction_json": json.dumps({"core_finding": "finding"}),
        "status": "extraction_complete",
    })

    # N18 fallback: no chunks, paper-level similarity used.
    embeddings_db.find_similar_chunks.return_value = []
    embeddings_db.find_similar_to_text.return_value = [
        {"id": "10.1/r", "score": 0.88, "document": "", "metadata": {}}
    ]

    with patch("scripts.obsidian_tools.synthesize._expand_query") as mock_expand:
        note_path = synthesize(
            "ultrafiltration", config, state_db, embeddings_db, llm,
            rag_mode="vector",
        )
        mock_expand.assert_not_called()

    assert note_path != ""
