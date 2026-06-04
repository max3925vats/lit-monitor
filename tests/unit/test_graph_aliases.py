"""G2: tests for the alias loader (scripts.graph.aliases)."""
from __future__ import annotations

from scripts.graph.aliases import load_aliases


def test_load_aliases_returns_flat_surface_to_canonical_dict_per_type(tmp_path):
    """G2: aliases load into {type: {lower(surface): canonical}}."""
    yaml_path = tmp_path / "entity_aliases.yaml"
    yaml_path.write_text(
        "material:\n"
        "  mAb: monoclonal antibody\n"
        "  HCP: host cell protein\n"
        "method:\n"
        "  IEX: ion exchange\n"
    )
    aliases = load_aliases(yaml_path)
    assert aliases["material"]["mab"] == "monoclonal antibody"  # lowercased keys
    assert aliases["material"]["hcp"] == "host cell protein"
    assert aliases["method"]["iex"] == "ion exchange"


def test_load_aliases_missing_file_returns_empty_dict(tmp_path, caplog):
    """G2: missing aliases file → empty dict + WARNING (not raise)."""
    missing = tmp_path / "nope.yaml"
    with caplog.at_level("WARNING"):
        aliases = load_aliases(missing)
    assert aliases == {}
    assert "not found" in caplog.text.lower() or "missing" in caplog.text.lower()


def test_load_aliases_warns_on_non_dict_entry_and_loads_good_ones(tmp_path, caplog):
    """Q4.2: a malformed (non-dict) type entry is skipped WITH a WARNING, while
    the well-formed entries still load."""
    yaml_path = tmp_path / "entity_aliases.yaml"
    yaml_path.write_text(
        "material:\n"
        "  mAb: monoclonal antibody\n"
        "bogus: just a scalar string\n"  # malformed: not a mapping
    )
    with caplog.at_level("WARNING"):
        aliases = load_aliases(yaml_path)
    # Good entry loaded; malformed entry dropped.
    assert aliases["material"]["mab"] == "monoclonal antibody"
    assert "bogus" not in aliases
    # The skip is observable.
    warnings = [r.message for r in caplog.records if r.levelno >= 30]
    assert any("bogus" in m and "malformed" in m.lower() for m in warnings), (
        f"Expected a WARNING naming the skipped entry; got: {warnings}"
    )


def test_load_aliases_uses_example_fallback(tmp_path, monkeypatch):
    """G2: when path arg is None, falls back to packaged .example.yaml."""
    # Implementation note: load_aliases(None) should look at config/entity_aliases.yaml
    # first; if missing, fall back to config/entity_aliases.example.yaml. This test
    # confirms the example fallback works.
    aliases = load_aliases(None)
    # The shipped example should at minimum have material.mab
    assert "material" in aliases
    assert "mab" in aliases["material"]
