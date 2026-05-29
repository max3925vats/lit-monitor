"""G2: tests for the entity normalizer pipeline (scripts.graph.normalizer)."""
from __future__ import annotations

from scripts.graph.normalizer import EntityNormalizer


class TestNormalizerCore:
    def test_singularization(self):
        """G2: plurals are normalized to singular form via inflect."""
        n = EntityNormalizer(aliases={})
        result, via = n.normalize("antibodies", type_="material")
        assert result == "antibody"
        assert via == "identity"

    def test_lowercase(self):
        """G2: input case doesn't matter — output is lowercased."""
        n = EntityNormalizer(aliases={})
        result, via = n.normalize("ANTIBODY", type_="material")
        assert result == "antibody"

    def test_ascii_fold(self):
        """G2: non-ASCII chars are folded to ASCII (β → b)."""
        n = EntityNormalizer(aliases={})
        result, _via = n.normalize("β-lactoglobulin", type_="material")
        assert result == "b-lactoglobulin"

    def test_punctuation_strip(self):
        """G2: trailing punctuation and double-spaces collapse."""
        n = EntityNormalizer(aliases={})
        result, _via = n.normalize("ion exchange.", type_="method")
        assert result == "ion exchange"


class TestNormalizerAlias:
    def test_alias_resolves_acronym(self):
        """G2: mAb resolves to 'monoclonal antibody' via the alias."""
        aliases = {"material": {"mab": "monoclonal antibody"}}
        n = EntityNormalizer(aliases=aliases)
        result, via = n.normalize("mAb", type_="material")
        assert result == "monoclonal antibody"
        assert via == "alias"

    def test_alias_is_case_insensitive(self):
        """G2: aliases lookup ignores case."""
        aliases = {"material": {"mab": "monoclonal antibody"}}
        n = EntityNormalizer(aliases=aliases)
        result, via = n.normalize("MAB", type_="material")
        assert result == "monoclonal antibody"
        assert via == "alias"


class TestNormalizerFuzzy:
    def test_fuzzy_collapse_at_default_threshold(self):
        """G2: typo of an existing canonical entity (≥90 ratio) collapses to it."""
        n = EntityNormalizer(aliases={})
        n.add_to_vocab("method", "ion exchange chromatography")
        result, via = n.normalize("ion exchnage chromatography", type_="method")
        # 'exchnage' is a transposition typo — fuzz.ratio ~96, well above 90
        assert result == "ion exchange chromatography"
        assert via == "fuzzy"

    def test_fuzzy_respects_type_scope(self):
        """G2: a topic entity does NOT match into method vocab."""
        n = EntityNormalizer(aliases={})
        n.add_to_vocab("topic", "ion exchange chromatography process")
        # Query as method type — should not pull the topic vocab in
        result, via = n.normalize("ion exchange chromatography", type_="method")
        # No method vocab populated, so identity passthrough
        assert via == "identity"
        assert result == "ion exchange chromatography"

    def test_fuzzy_min_ratio_parameter(self):
        """G2: min_ratio is parameterized — a near-match (~94) passes at 90 but fails at 95."""
        n = EntityNormalizer(aliases={})
        n.add_to_vocab("method", "ion exchange chromatography")
        # This typo scores ~94.5 → passes default (90) but not strict (95)
        typo = "ion exchnage chromatographyy"  # extra trailing 'y' + transposition
        # Confirm it passes at default threshold
        result_loose, via_loose = n.normalize(typo, type_="method")
        assert via_loose == "fuzzy"
        assert result_loose == "ion exchange chromatography"
        # At min_ratio=95 (stricter), should NOT collapse
        result, via = n.normalize(typo, type_="method", min_ratio=95)
        assert via == "identity"
