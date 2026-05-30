"""G11: propose-aliases tests."""
from __future__ import annotations

import yaml

from scripts.graph import GraphDB
from scripts.graph.entity_extractor import EntityTuple
from scripts.graph.propose_aliases import propose_aliases


def _seed(db: GraphDB):
    """Populate graph with surfaces that should cluster (3 mAb variants + 1 isolated method)."""
    entities = [
        # Three near-identical surfaces — should cluster
        EntityTuple(canonical_id="monoclonal antibody", type="material",
                    surface="monoclonal antibody", field="materials_systems",
                    span_start=0, span_end=20),
        EntityTuple(canonical_id="monoclonal antibodies", type="material",
                    surface="monoclonal antibodies", field="materials_systems",
                    span_start=0, span_end=22),
        EntityTuple(canonical_id="monoclonal antbody", type="material",  # typo
                    surface="monoclonal antbody", field="materials_systems",
                    span_start=0, span_end=18),
        # Isolated entity, should NOT be in any cluster
        EntityTuple(canonical_id="ultrafiltration", type="method",
                    surface="ultrafiltration", field="methods_summary",
                    span_start=0, span_end=15),
    ]
    db.add_paper(doi="10.0/a", entities=entities, relationships=[],
                 paper_metadata={"title": "A", "year": 2024, "journal": "X"})


class TestProposeAliases:
    def test_clusters_similar_surfaces_within_type(self, tmp_path):
        """G11: 3 similar surfaces in same type → 1 cluster in proposal."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _seed(db)
        proposals = propose_aliases(db, min_ratio=80)
        # Expect a 'material' key with at least one cluster
        assert "material" in proposals
        material_proposals = proposals["material"]
        # The 3 mAb variants should be aliased to a common canonical
        # (the most frequent surface in the cluster wins)
        canonicals = set(material_proposals.values())
        assert len(canonicals) == 1, f"expected 1 canonical, got {canonicals}"
        # All 3 surfaces appear (one is the canonical itself, two are aliases)
        # Some may be in the alias map; depends on cluster size
        assert len(material_proposals) >= 2  # at least the 2 non-canonical surfaces

    def test_singletons_are_not_proposed(self, tmp_path):
        """G11: an isolated surface (no nearby cluster) is NOT proposed."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _seed(db)
        proposals = propose_aliases(db, min_ratio=80)
        # 'ultrafiltration' is the only method entity → no cluster → not proposed
        method_proposals = proposals.get("method", {})
        assert "ultrafiltration" not in method_proposals

    def test_existing_aliases_are_skipped(self, tmp_path, monkeypatch):
        """G11: surfaces already aliased in entity_aliases.yaml are not re-proposed."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _seed(db)

        # Mock load_aliases to claim 'monoclonal antibodies' is already aliased
        monkeypatch.setattr(
            "scripts.graph.propose_aliases.load_aliases",
            lambda: {"material": {"monoclonal antibodies": "monoclonal antibody"}},
        )
        proposals = propose_aliases(db, min_ratio=80)
        material_proposals = proposals.get("material", {})
        # 'monoclonal antibodies' should be filtered out
        assert "monoclonal antibodies" not in material_proposals

    def test_min_ratio_parameter_affects_clustering(self, tmp_path):
        """G11: at min_ratio=95 (strict), only very-close pairs cluster."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _seed(db)
        # At 80: monoclonal antibody, monoclonal antibodies, monoclonal antbody all cluster
        # At 95: typo 'antbody' may not match 'antibody' at 95 — fewer proposals
        loose = propose_aliases(db, min_ratio=80)
        strict = propose_aliases(db, min_ratio=95)
        loose_count = sum(len(v) for v in loose.values())
        strict_count = sum(len(v) for v in strict.values())
        # Strict produces ≤ loose
        assert strict_count <= loose_count

    def test_empty_graph_returns_empty_proposal(self, tmp_path):
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        # No entities at all
        proposals = propose_aliases(db, min_ratio=80)
        assert proposals == {}

    def test_no_module_level_llm_imports(self):
        """G11 + N6: propose_aliases.py must not import any LLM/network client
        at module load time.

        N6 adds an LLM-augmented path, but ONLY via lazy imports inside
        ``_consult_llm``. The module surface must remain LLM-free so that
        the fuzzy-only path (and every test that doesn't exercise --with-llm)
        loads with no network/LLM dependency.
        """
        import ast
        import inspect

        import scripts.graph.propose_aliases as mod

        tree = ast.parse(inspect.getsource(mod))
        forbidden_modules = {
            "requests", "httpx", "litellm", "openai", "anthropic",
            "scripts.llm.llm_client", "scripts.llm.prompt_registry",
        }
        for node in tree.body:  # only the top-level body
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_modules, (
                        f"forbidden top-level import: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_modules, (
                    f"forbidden top-level from-import: {node.module}"
                )


class TestProposeAliasesWriteFile:
    def test_writes_yaml_with_header(self, tmp_path, monkeypatch):
        """G11: writes config/entity_aliases.suggested.yaml with date header."""
        from scripts.graph.propose_aliases import write_proposal_file
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _seed(db)

        proposals = {"material": {"monoclonal antibodies": "monoclonal antibody"}}
        out_path = tmp_path / "entity_aliases.suggested.yaml"
        write_proposal_file(out_path, proposals)

        assert out_path.exists()
        content = out_path.read_text()
        assert "Proposed by" in content
        assert "lit-monitor graph propose-aliases" in content
        # Roundtrip via YAML loader
        data = yaml.safe_load(content)
        assert data == proposals


# ---------------------------------------------------------------------------
# N6: LLM-augmented consensus pass
# ---------------------------------------------------------------------------
class TestN6LLMAccept:
    def test_llm_accept_keeps_cluster(self, tmp_path, monkeypatch):
        """N6: LLM accept=True keeps the cluster; canonical comes from verdict."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        entities = [
            EntityTuple(canonical_id="monoclonal antibody", type="material",
                        surface="monoclonal antibody", field="materials_systems",
                        span_start=0, span_end=20),
            EntityTuple(canonical_id="monoclonal antibodies", type="material",
                        surface="monoclonal antibodies", field="materials_systems",
                        span_start=0, span_end=22),
            EntityTuple(canonical_id="monoclonal antbody", type="material",
                        surface="monoclonal antbody", field="materials_systems",
                        span_start=0, span_end=18),
        ]
        db.add_paper(doi="10.0/a", entities=entities, relationships=[],
                     paper_metadata={"title": "A", "year": 2024, "journal": "X"})

        # Mock _consult_llm to accept the only cluster, with explicit canonical.
        def fake_consult_llm(clusters_by_type, **kwargs):
            # There should be exactly one material cluster.
            assert "material" in clusters_by_type
            return {
                "material__0": {
                    "accept": True,
                    "canonical": "monoclonal antibody",
                    "rationale": "synonyms",
                }
            }

        monkeypatch.setattr(
            "scripts.graph.propose_aliases._consult_llm", fake_consult_llm,
        )

        proposals = propose_aliases(db, min_ratio=80, with_llm=True)
        assert "material" in proposals
        # All non-canonical members map to the verdict's canonical.
        assert all(c == "monoclonal antibody" for c in proposals["material"].values())
        assert len(proposals["material"]) >= 2


class TestN6LLMReject:
    def test_llm_reject_drops_cluster(self, tmp_path, monkeypatch):
        """N6: LLM accept=False drops the cluster from proposals."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        entities = [
            EntityTuple(canonical_id="monoclonal antibody", type="material",
                        surface="monoclonal antibody", field="materials_systems",
                        span_start=0, span_end=20),
            EntityTuple(canonical_id="monoclonal antibodies", type="material",
                        surface="monoclonal antibodies", field="materials_systems",
                        span_start=0, span_end=22),
        ]
        db.add_paper(doi="10.0/a", entities=entities, relationships=[],
                     paper_metadata={"title": "A", "year": 2024, "journal": "X"})

        def reject_llm(clusters_by_type, **kwargs):
            verdicts = {}
            for type_, clusters in clusters_by_type.items():
                for idx, _members in enumerate(clusters):
                    verdicts[f"{type_}__{idx}"] = {
                        "accept": False,
                        "canonical": "n/a",
                        "rationale": "different concepts",
                    }
            return verdicts

        monkeypatch.setattr(
            "scripts.graph.propose_aliases._consult_llm", reject_llm,
        )

        proposals = propose_aliases(db, min_ratio=80, with_llm=True)
        # All clusters rejected -> no material proposals at all.
        assert proposals.get("material", {}) == {}

    def test_llm_missing_verdict_drops_cluster(self, tmp_path, monkeypatch):
        """N6: a cluster_id absent from verdicts is treated as accept=false."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        entities = [
            EntityTuple(canonical_id="monoclonal antibody", type="material",
                        surface="monoclonal antibody", field="materials_systems",
                        span_start=0, span_end=20),
            EntityTuple(canonical_id="monoclonal antibodies", type="material",
                        surface="monoclonal antibodies", field="materials_systems",
                        span_start=0, span_end=22),
        ]
        db.add_paper(doi="10.0/a", entities=entities, relationships=[],
                     paper_metadata={"title": "A", "year": 2024, "journal": "X"})

        # Empty verdicts — no cluster_ids present.
        monkeypatch.setattr(
            "scripts.graph.propose_aliases._consult_llm",
            lambda clusters_by_type, **kw: {},
        )

        proposals = propose_aliases(db, min_ratio=80, with_llm=True)
        assert proposals.get("material", {}) == {}


class TestN6LLMFallback:
    def test_llm_failure_falls_back_to_fuzzy_byte_identical(
        self, tmp_path, monkeypatch, caplog,
    ):
        """N6: when _consult_llm returns None, output == fuzzy-only output."""
        db = GraphDB(persist_dir=str(tmp_path / "g.kuzu"))
        _seed(db)

        # Fuzzy-only baseline.
        fuzzy_only = propose_aliases(db, min_ratio=80, with_llm=False)

        # Simulate LLM failure.
        monkeypatch.setattr(
            "scripts.graph.propose_aliases._consult_llm",
            lambda clusters_by_type, **kw: None,
        )

        import logging
        with caplog.at_level(logging.WARNING):
            with_llm_failed = propose_aliases(db, min_ratio=80, with_llm=True)

        # Byte-identical to fuzzy-only.
        assert with_llm_failed == fuzzy_only
        # Fallback warning logged.
        assert any(
            "llm" in r.message.lower() or "fall" in r.message.lower()
            for r in caplog.records
        ), f"expected a fallback warning, got: {[r.message for r in caplog.records]}"


class TestN6ConsultLLM:
    def test_consult_llm_empty_clusters_returns_empty_dict(self):
        """N6: zero clusters short-circuits to {} (success, not failure)."""
        from unittest.mock import MagicMock

        from scripts.graph.propose_aliases import _consult_llm
        client = MagicMock()
        result = _consult_llm(
            {},
            client=client,
            prompt={"system": "S", "user_template": "{clusters_json}"},
        )
        assert result == {}
        # No LLM call when there's nothing to validate.
        client.complete.assert_not_called()

    def test_consult_llm_handles_malformed_json(self):
        """N6: malformed JSON response -> None (caller falls back)."""
        from unittest.mock import MagicMock

        from scripts.graph.propose_aliases import _consult_llm
        client = MagicMock()
        client.complete.return_value = "not json at all"
        result = _consult_llm(
            {"material": [["mab", "mabs"]]},
            client=client,
            prompt={"system": "S", "user_template": "{clusters_json}"},
        )
        assert result is None

    def test_consult_llm_no_api_key_returns_none(self, monkeypatch):
        """N6: when client construction needed and no key, returns None."""
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        from scripts.graph.propose_aliases import _consult_llm
        # No client injected -> internal construction path -> no key path.
        result = _consult_llm({"material": [["a", "b"]]})
        assert result is None

    def test_consult_llm_drops_malformed_entries_keeps_rest(self):
        """N6: per-item validation drops bad entries, keeps good ones."""
        import json
        from unittest.mock import MagicMock

        from scripts.graph.propose_aliases import _consult_llm
        client = MagicMock()
        # Mix of good + bad entries.
        client.complete.return_value = json.dumps({
            "material__0": {
                "accept": True,
                "canonical": "monoclonal antibody",
                "rationale": "synonyms",
            },
            "material__1": {"accept": "yes", "canonical": "x"},  # bad accept type
            "material__2": {"accept": True, "canonical": ""},     # empty canonical
            "material__3": "not a dict",                          # wrong type
        })
        result = _consult_llm(
            {"material": [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]},
            client=client,
            prompt={"system": "S", "user_template": "{clusters_json}"},
        )
        assert result is not None
        # Only the valid entry survives.
        assert set(result.keys()) == {"material__0"}
        assert result["material__0"]["accept"] is True

    def test_consult_llm_strips_markdown_fences(self):
        """N6: tolerate ```json ... ``` wrappers."""
        import json
        from unittest.mock import MagicMock

        from scripts.graph.propose_aliases import _consult_llm
        client = MagicMock()
        client.complete.return_value = (
            "```json\n"
            + json.dumps({
                "material__0": {
                    "accept": True,
                    "canonical": "X",
                    "rationale": "ok",
                }
            })
            + "\n```"
        )
        result = _consult_llm(
            {"material": [["a", "b"]]},
            client=client,
            prompt={"system": "S", "user_template": "{clusters_json}"},
        )
        assert result is not None
        assert result["material__0"]["accept"] is True


class TestN6PromptRegistry:
    def test_alias_consensus_prompt_loads(self):
        """N6: the prompt registers and renders the required placeholder."""
        from scripts.llm.prompt_registry import (
            _reset_prompt_cache,
            load_prompt,
            required_placeholders,
        )
        _reset_prompt_cache()
        prompt = load_prompt("alias_consensus")
        # System prompt is non-empty and mentions the closed vocab.
        assert prompt.system
        # required_placeholders is registered.
        assert "clusters_json" in required_placeholders("alias_consensus")
        # Renders without raising when the placeholder is supplied.
        rendered = prompt.render_user(clusters_json="[]")
        assert "[]" in rendered
