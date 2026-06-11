"""Tests for G15: closed-vocabulary loader with alias + deprecated support."""
import logging
from importlib.resources import files as _pkg_files
from pathlib import Path

import yaml

from lit_monitor.graph.vocab import EntityTypeVocab, PredicateVocab

# The vocab example YAMLs ship as package data (relocated from config/ for the
# PyPI wheel). Resolve via importlib.resources so these guard tests follow them.
_EXAMPLES = _pkg_files("lit_monitor._data.config_examples")
_PREDICATES_EXAMPLE = Path(str(_EXAMPLES / "predicates.example.yaml"))
_ENTITY_TYPES_EXAMPLE = Path(str(_EXAMPLES / "entity_types.example.yaml"))


class TestPredicateVocab:
    def test_canonical_resolves_to_self(self, tmp_path):
        """G15: a canonical predicate resolves to itself."""
        yaml_path = tmp_path / "preds.yaml"
        yaml_path.write_text(
            "EXTENDS:\n"
            "  description: ext\n"
            "  aliases: []\n"
            "  deprecated: false\n"
        )
        vocab = PredicateVocab.load(yaml_path)
        assert vocab.resolve("EXTENDS") == "EXTENDS"

    def test_alias_resolves_to_canonical(self, tmp_path):
        """G15: aliases resolve to their canonical form."""
        yaml_path = tmp_path / "preds.yaml"
        yaml_path.write_text(
            "EXTENDS:\n"
            "  aliases: [BUILDS_ON, IMPROVES_UPON]\n"
            "  deprecated: false\n"
        )
        vocab = PredicateVocab.load(yaml_path)
        assert vocab.resolve("BUILDS_ON") == "EXTENDS"
        assert vocab.resolve("IMPROVES_UPON") == "EXTENDS"

    def test_unknown_resolves_to_none(self, tmp_path):
        """G15: unknown predicates return None."""
        yaml_path = tmp_path / "preds.yaml"
        yaml_path.write_text("EXTENDS:\n  aliases: []\n")
        vocab = PredicateVocab.load(yaml_path)
        assert vocab.resolve("MADE_UP_PREDICATE") is None


class TestPredicateVocabDeprecated:
    def test_deprecated_predicate_emits_warning_once(self, tmp_path, caplog):
        """G15: deprecated predicates resolve but warn ONCE per session."""
        # Reset the module-level cache so this test is deterministic in any run order.
        import lit_monitor.graph.vocab as vocab_mod
        vocab_mod._WARNED.clear()

        yaml_path = tmp_path / "preds.yaml"
        yaml_path.write_text(
            "OLD_PRED:\n"
            "  aliases: []\n"
            "  deprecated: true\n"
        )
        vocab = PredicateVocab.load(yaml_path)
        with caplog.at_level(logging.WARNING):
            vocab.resolve("OLD_PRED")  # first call — should warn
            vocab.resolve("OLD_PRED")  # second call — must NOT re-warn

        warn_count = sum("deprecated" in r.message.lower() for r in caplog.records)
        assert warn_count == 1, f"expected exactly 1 deprecation warning, got {warn_count}"

    def test_deprecated_predicate_still_resolves(self, tmp_path, caplog):
        """G15: deprecated predicates still return the canonical name."""
        import lit_monitor.graph.vocab as vocab_mod
        vocab_mod._WARNED.clear()

        yaml_path = tmp_path / "preds.yaml"
        yaml_path.write_text(
            "OLD_PRED:\n"
            "  aliases: []\n"
            "  deprecated: true\n"
        )
        vocab = PredicateVocab.load(yaml_path)
        with caplog.at_level(logging.WARNING):
            result = vocab.resolve("OLD_PRED")
        assert result == "OLD_PRED"


class TestEntityTypeVocab:
    def test_entity_type_canonical_resolves(self, tmp_path):
        """G15: EntityTypeVocab works identically to PredicateVocab."""
        yaml_path = tmp_path / "types.yaml"
        yaml_path.write_text(
            "method:\n"
            "  aliases: [technique]\n"
            "  deprecated: false\n"
        )
        vocab = EntityTypeVocab.load(yaml_path)
        assert vocab.resolve("method") == "method"
        assert vocab.resolve("technique") == "method"
        assert vocab.resolve("unknown") is None

    def test_entity_type_deprecated_warns_once(self, tmp_path, caplog):
        """G15: EntityTypeVocab deprecated warning uses the same _WARNED cache."""
        import lit_monitor.graph.vocab as vocab_mod
        vocab_mod._WARNED.clear()

        yaml_path = tmp_path / "types.yaml"
        yaml_path.write_text(
            "old_type:\n"
            "  aliases: []\n"
            "  deprecated: true\n"
        )
        vocab = EntityTypeVocab.load(yaml_path)
        with caplog.at_level(logging.WARNING):
            vocab.resolve("old_type")
            vocab.resolve("old_type")

        warn_count = sum("deprecated" in r.message.lower() for r in caplog.records)
        assert warn_count == 1


class TestShippedExampleFiles:
    """Verify the example YAML files ship with the right substrate."""

    def test_predicates_example_has_all_phase1_predicates(self):
        path = _PREDICATES_EXAMPLE
        assert path.exists(), f"{path} does not exist"
        data = yaml.safe_load(path.read_text())
        expected = {
            "MENTIONS", "CITES", "COMPARES_TO", "DEPENDS_ON",
            "PROPOSES", "LIMITED_BY", "INTRODUCES", "RAISES_QUESTION",
        }
        assert expected.issubset(data.keys()), f"missing: {expected - data.keys()}"

    def test_predicates_example_has_at_least_one_alias_declared(self):
        path = _PREDICATES_EXAMPLE
        data = yaml.safe_load(path.read_text())
        # At least one predicate must have a non-empty aliases list.
        any_alias = any(
            entry.get("aliases") for entry in data.values()
            if isinstance(entry, dict)
        )
        assert any_alias, "G15: example file must demo at least one alias"

    def test_entity_types_example_has_phase1_types(self):
        path = _ENTITY_TYPES_EXAMPLE
        assert path.exists(), f"{path} does not exist"
        data = yaml.safe_load(path.read_text())
        expected = {"topic", "method", "material", "author", "journal", "keyword"}
        assert expected.issubset(data.keys()), f"missing: {expected - data.keys()}"

    def test_example_files_load_cleanly_via_vocab_classes(self):
        """G15: example YAMLs are valid inputs for the vocab classes."""
        pred_path = _PREDICATES_EXAMPLE
        type_path = _ENTITY_TYPES_EXAMPLE

        pred_vocab = PredicateVocab.load(pred_path)
        type_vocab = EntityTypeVocab.load(type_path)

        # Canonicals list should not be empty.
        assert len(pred_vocab.canonicals()) >= 8
        assert len(type_vocab.canonicals()) == 6

    def test_predicates_example_compares_to_has_demo_aliases(self):
        """G15: COMPARES_TO carries the demo aliases as spec'd."""
        path = _PREDICATES_EXAMPLE
        data = yaml.safe_load(path.read_text())
        aliases = data["COMPARES_TO"].get("aliases", [])
        assert len(aliases) >= 1, "COMPARES_TO must carry at least one demo alias"
