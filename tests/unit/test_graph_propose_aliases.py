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

    def test_no_llm_imports_in_module(self):
        """G11: propose_aliases.py must NOT import any LLM/network client."""
        import inspect

        import scripts.graph.propose_aliases as mod
        src = inspect.getsource(mod)
        # Check that none of these LLM-client imports appear
        forbidden = ["import requests", "import httpx", "import litellm",
                     "from scripts.llm.llm_client", "import openai", "import anthropic"]
        for forbid in forbidden:
            assert forbid not in src, f"G11 must not import {forbid}"


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
