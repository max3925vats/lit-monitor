"""G2: tests for the entity normalizer pipeline (scripts.graph.normalizer)."""
from __future__ import annotations

from scripts.graph.normalizer import EntityNormalizer, _basic_normalize


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

    def test_ascii_fold_alpha(self):
        """G2: α (Greek alpha) → 'a'."""
        n = EntityNormalizer(aliases={})
        result, _via = n.normalize("α-lactalbumin", type_="material")
        assert result == "a-lactalbumin"

    def test_ascii_fold_mu(self):
        """G2: μ (Greek mu) → 'u'. Documents the deliberate choice (not 'micro')."""
        n = EntityNormalizer(aliases={})
        result, _via = n.normalize("μg", type_="material")
        assert result == "ug"  # NOT "microg"; see transliteration table comment


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


class TestBasicNormalize:
    """Direct tests for the step-1-3 normalization function."""

    def test_plural_with_trailing_period_singularizes(self):
        """G2 bug fix: 'antibodies.' must singularize despite the period.

        Catches the step-order bug where punct strip ran after singularize.
        """
        assert _basic_normalize("antibodies.") == "antibody"

    def test_plural_with_trailing_comma_singularizes(self):
        assert _basic_normalize("proteins,") == "protein"

    def test_plural_with_trailing_semicolon_singularizes(self):
        assert _basic_normalize("inhibitors;") == "inhibitor"

    def test_acronym_plural_with_period(self):
        """'HCPs.' should normalize to 'hcp' (lowercase, punct-stripped).

        inflect may or may not recognize 'hcps' as a plural — the test
        confirms the function returns something reasonable, not False/raise.
        """
        result = _basic_normalize("HCPs.")
        # Lowercase + punct stripped; singular form may or may not happen
        assert result in ("hcp", "hcps"), f"unexpected: {result}"
        assert "." not in result
        assert "h" in result and "c" in result and "p" in result

    def test_already_singular_passthrough(self):
        assert _basic_normalize("antibody") == "antibody"

    def test_lowercase(self):
        assert _basic_normalize("ANTIBODY") == "antibody"

    def test_ascii_fold_beta(self):
        assert _basic_normalize("β-lactoglobulin") == "b-lactoglobulin"


class TestNormalizerStepOrder:
    """Regression tests for the punct-before-singularize bug."""

    def test_normalize_plural_with_period_via_class(self):
        """Real-text plural input must singularize end-to-end."""
        n = EntityNormalizer(aliases={})
        result, via = n.normalize("antibodies.", type_="material")
        assert result == "antibody"
        assert via == "identity"

    def test_alias_on_punctuated_plural(self):
        """'HCPs.' should normalize through punct strip + singularize, then hit the alias.

        Full pipeline: 'HCPs.' → lowercase 'hcps.' → punct strip 'hcps'
        → singularize 'hcp' → alias lookup hits 'hcp' key.
        The alias key must be the post-normalization form, not the raw plural.
        """
        aliases = {"material": {"hcp": "host cell protein"}}
        n = EntityNormalizer(aliases=aliases)
        result, via = n.normalize("HCPs.", type_="material")
        assert result == "host cell protein"
        assert via == "alias"
