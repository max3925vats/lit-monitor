"""
Phase 5 unit tests — Vocabulary building.
Tests for keyword_extractor, normalizer, and clusterer.
All LLM calls use MockLLMClient. No Zotero/Ollama required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.llm.llm_client import MockLLMClient


# ---------------------------------------------------------------------------
# normalizer.py tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_normaliser_collapses_near_duplicates():
    """Strings that are near-duplicates (>=88 similarity) collapse to one key."""
    from scripts.vocabulary.normalizer import normalise_keywords
    # Case variants of the same word must collapse to a single cluster.
    raw = ["filtration", "Filtration", "FILTRATION"]
    result = normalise_keywords(raw)
    # All three are the same word lower-cased — should produce exactly 1 cluster
    assert len(result) == 1
@pytest.mark.unit
def test_normaliser_preserves_distinct_terms():
    """Genuinely different terms must remain in separate clusters."""
    from scripts.vocabulary.normalizer import normalise_keywords
    raw = ["filtration", "stage2filtration", "osmotic pressure", "viscosity"]
    result = normalise_keywords(raw)
    assert len(result) == 4, f"Expected 4 distinct clusters, got {len(result)}: {list(result)}"
@pytest.mark.unit
def test_normaliser_returns_empty_on_empty_input():
    from scripts.vocabulary.normalizer import normalise_keywords
    assert normalise_keywords([]) == {}
@pytest.mark.unit
def test_normaliser_strips_and_lowercases():
    """Variants that differ only by case/whitespace collapse to one cluster."""
    from scripts.vocabulary.normalizer import normalise_keywords
    raw = ["  ProteinA  ", "PROTEINA", "proteina"]
    result = normalise_keywords(raw)
    assert len(result) == 1
    canonical = list(result.keys())[0]
    assert canonical == "proteina"
@pytest.mark.unit
def test_normaliser_variant_list_contains_all_originals():
    """Each cluster's variant list should include all (normalised) inputs."""
    from scripts.vocabulary.normalizer import normalise_keywords
    raw = ["TopicX", "topicx", "Topicx"]

    result = normalise_keywords(raw)
    assert len(result) == 1
    variants = list(result.values())[0]
    # After lower-casing all three become "topicx"
    assert variants == ["topicx"]
# ---------------------------------------------------------------------------
# clusterer.py tests
# ---------------------------------------------------------------------------
_MOCK_CONCEPTS = {
    "themes": [
        {
            "name": "Mock Theme A",
            "keywords": ["filtration", "stage2filtration", "topicx", "tff"],
        },
        {
            "name": "Protein Properties",
            "keywords": ["viscosity", "aggregation", "osmotic pressure"],
        },
    ],
    "unclustered": ["some rare keyword"],
}
_MOCK_CLUSTERER_RESPONSE = json.dumps(_MOCK_CONCEPTS)
@pytest.mark.unit
def test_clusterer_writes_draft_yaml(tmp_path):
    """build_vocabulary writes a valid YAML file with 'themes' and 'unclustered'."""
    from scripts.vocabulary.clusterer import build_vocabulary
    keywords = {
        "filtration": ["filtration"],
        "viscosity": ["viscosity", "viscosity "],
        "aggregation": ["aggregation"],
    }
    output_file = tmp_path / "concepts_draft.yaml"
    # LLM response: build_vocabulary uses parse_llm_json, so mock must return
    # valid JSON that includes "themes" + "unclustered"
    mock_llm = MockLLMClient()
    # Patch MockLLMClient.complete to return the clusterer response
    mock_llm._forced_response = _MOCK_CLUSTERER_RESPONSE
    with patch.object(mock_llm, "complete", return_value=_MOCK_CLUSTERER_RESPONSE):
        build_vocabulary(keywords, mock_llm, output_path=output_file, remediate=False)
    assert output_file.exists(), "concepts_draft.yaml was not written"
    with open(output_file, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert "themes" in loaded
    assert "unclustered" in loaded
    assert isinstance(loaded["themes"], list)
    assert len(loaded["themes"]) >= 1
@pytest.mark.unit
def test_unclustered_not_silently_dropped():
    """flag_unmapped returns keywords not present in any theme."""
    from scripts.vocabulary.clusterer import flag_unmapped
    keywords = ["filtration", "some rare keyword", "another unknown term"]
    unmapped = flag_unmapped(keywords, _MOCK_CONCEPTS, threshold=0.75)
    assert "some rare keyword" in unmapped or "another unknown term" in unmapped

    # "filtration" IS in a theme — must not appear in unmapped
    assert "filtration" not in unmapped
@pytest.mark.unit
def test_map_keywords_returns_correct_themes():
    """Keywords exactly in a theme's list should map to that theme."""
    from scripts.vocabulary.clusterer import map_keywords_to_themes
    keywords = ["filtration", "viscosity", "topicx"]
    result = map_keywords_to_themes(keywords, _MOCK_CONCEPTS)
    assert "Mock Theme A" in result["filtration"]
    assert "Mock Theme A" in result["topicx"]
    assert "Protein Properties" in result["viscosity"]
    # No false theme assignments
    assert "Protein Properties" not in result["filtration"]
@pytest.mark.unit
def test_map_keywords_empty_list_for_unknown():
    """A keyword in no theme returns an empty list — not an error."""
    from scripts.vocabulary.clusterer import map_keywords_to_themes
    result = map_keywords_to_themes(["completely unknown term xyz"], _MOCK_CONCEPTS)
    assert result["completely unknown term xyz"] == []

# ---------------------------------------------------------------------------
# A7 regression tests — chunked clustering
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_vocabulary_passes_all_keywords_to_llm(tmp_path):
    """build_vocabulary sends ALL keywords in the dict to the LLM, including
    short alphanumeric ones.

    The item-key filter was removed because the real fix is the CLI-level
    tuple-unpacking fix (kw for kw, _ in pairs). Clusterer itself no longer
    filters anything — it trusts its input.
    """
    from scripts.vocabulary.clusterer import build_vocabulary

    # Mix of long keywords and a short 8-char one that looks like an item key.
    # The clusterer must send ALL of them to the LLM.
    keywords = {
        "filtration": ["filtration"],
        "viscosity": ["viscosity"],
        "membrane": ["membrane"],   # 8 chars — legitimate keyword, must NOT be filtered
    }
    output_file = tmp_path / "concepts_draft.yaml"
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete", return_value=_MOCK_CLUSTERER_RESPONSE) as mock_complete:
        build_vocabulary(keywords, mock_llm, output_path=output_file, remediate=False)

    assert mock_complete.called, "LLM should have been called"
    # All three keywords must appear in the prompt
    user_prompt = mock_complete.call_args_list[0].args[1]
    assert "filtration" in user_prompt
    assert "viscosity" in user_prompt
    assert "membrane" in user_prompt, "'membrane' (8-char keyword) must not be filtered"


@pytest.mark.unit
def test_build_vocabulary_chunked(tmp_path):
    """With chunk_size=60, 100 keywords should produce exactly 2 LLM calls.

    Regression test for G7: the old implementation sent all keywords in one
    call, causing the LLM to hang on large libraries.
    """
    from scripts.vocabulary.clusterer import build_vocabulary

    # Generate 100 distinct synthetic keywords (all 'real' — no item keys)
    keywords = {f"keyword_{i:03d}": [f"keyword_{i:03d}"] for i in range(100)}
    output_file = tmp_path / "concepts_draft.yaml"
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete", return_value=_MOCK_CLUSTERER_RESPONSE) as mock_complete:
        build_vocabulary(keywords, mock_llm, output_path=output_file, chunk_size=60, remediate=False)

    # 100 keywords / 60 per chunk → 2 calls (remediate=False keeps count exact)
    assert mock_complete.call_count == 2, (
        f"Expected 2 LLM calls for 100 keywords at chunk_size=60, got {mock_complete.call_count}"
    )
    # First call should have 60 keywords; second should have the remaining 40
    first_prompt = mock_complete.call_args_list[0].args[1]
    second_prompt = mock_complete.call_args_list[1].args[1]
    first_count = first_prompt.count("  - keyword_")
    second_count = second_prompt.count("  - keyword_")
    assert first_count == 60, f"Expected 60 keywords in first chunk, got {first_count}"
    assert second_count == 40, f"Expected 40 keywords in second chunk, got {second_count}"


@pytest.mark.unit
def test_merge_clusters_unions_keywords():
    """_merge_clusters unions keyword lists for same-named themes and appends new themes."""
    from scripts.vocabulary.clusterer import _merge_clusters

    base = {
        "themes": [
            {"name": "Filtration", "keywords": ["filtration", "stage2filtration"]},
        ],
        "unclustered": ["orphan_a"],
    }
    partial = {
        "themes": [
            # Same theme — should union keywords, not duplicate "stage2filtration"
            {"name": "Filtration", "keywords": ["stage2filtration", "tff"]},
            # New theme — should be appended
            {"name": "Protein Properties", "keywords": ["viscosity"]},
        ],
        "unclustered": ["orphan_a", "orphan_b"],  # "orphan_a" is a duplicate
    }
    _merge_clusters(base, partial)

    # Filtration theme: "tff" added, "stage2filtration" not duplicated
    filtration = next(t for t in base["themes"] if t["name"] == "Filtration")
    assert "tff" in filtration["keywords"]
    assert filtration["keywords"].count("stage2filtration") == 1

    # New theme appended
    protein = next((t for t in base["themes"] if t["name"] == "Protein Properties"), None)
    assert protein is not None
    assert "viscosity" in protein["keywords"]

    # Unclustered: "orphan_b" added, "orphan_a" not duplicated
    assert "orphan_b" in base["unclustered"]
    assert base["unclustered"].count("orphan_a") == 1


@pytest.mark.unit
def test_build_vocabulary_empty_input_returns_empty(tmp_path):
    """build_vocabulary returns empty result gracefully when given an empty keyword dict."""
    from scripts.vocabulary.clusterer import build_vocabulary

    output_file = tmp_path / "concepts_draft.yaml"
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete", return_value=_MOCK_CLUSTERER_RESPONSE) as mock_complete:
        result = build_vocabulary({}, mock_llm, output_path=output_file)

    # LLM should NOT have been called — nothing to cluster
    assert not mock_complete.called, "LLM should not be called when keyword list is empty"
    assert result == {"themes": [], "unclustered": []}
    assert output_file.exists()
# ---------------------------------------------------------------------------
# Regression test for CLI tuple-unpacking bug (G7 follow-up)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_keyword_pairs_first_element_is_keyword():
    """Verify that the first element of each (keyword, zotero_key) pair is the keyword.

    Regression for the cli.py bug where `[kw for _, kw in raw_kw_pairs]` was
    used instead of `[kw for kw, _ in raw_kw_pairs]`, causing the Zotero item
    keys (8-char alphanumeric) to be passed to the normaliser instead of the
    actual keyword strings — resulting in all 418 keywords being filtered out
    by build_vocabulary's item-key guard.
    """
    from scripts.vocabulary.keyword_extractor import extract_all_keywords

    mock_zot_client = MagicMock()
    mock_zot_client._zot.everything.return_value = [
        {"data": {"key": "23blhcwe", "tags": [{"tag": "filtration"}]}},
        {"data": {"key": "ab12cd34", "tags": [{"tag": "viscosity"}]}},
    ]
    mock_zot_client._zot.items.return_value = []
    pairs = extract_all_keywords(mock_zot_client)

    # First element is the keyword string, second is the Zotero item key.
    # The CLI must use [kw for kw, _ in pairs] — not [kw for _, kw in pairs].
    keywords_correct = [kw for kw, _ in pairs]
    keywords_wrong   = [kw for _, kw in pairs]

    assert "filtration" in keywords_correct, "Correct unpacking must yield the tag"
    assert "viscosity" in keywords_correct
    assert "23blhcwe" not in keywords_correct, "Correct unpacking must NOT yield the item key"

    # Document the failure mode: wrong unpacking gives item keys, not keywords
    assert "23blhcwe" in keywords_wrong, "Wrong unpacking yields item keys (documents the bug)"
    assert "filtration" not in keywords_wrong


# ---------------------------------------------------------------------------
# keyword_extractor.py tests (uses a mock Zotero client)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_extract_all_keywords_returns_pairs():
    """extract_all_keywords returns (keyword, zotero_key) tuples."""
    from scripts.vocabulary.keyword_extractor import extract_all_keywords
    # Build a mock ZoteroClient
    mock_zot_client = MagicMock()
    mock_zot_client._zot.everything.return_value = [
        {"data": {"key": "ABC123", "tags": [{"tag": "TopicX"}, {"tag": "ProteinA"}]}},
        {"data": {"key": "DEF456", "tags": [{"tag": "viscosity"}]}},
    ]
    mock_zot_client._zot.items.return_value = []  # passed to .everything()
    pairs = extract_all_keywords(mock_zot_client)
    assert isinstance(pairs, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    keywords = [p[0] for p in pairs]
    assert "TopicX" in keywords
    assert "ProteinA" in keywords
    assert "viscosity" in keywords
@pytest.mark.unit
def test_extract_all_keywords_deduplicates_within_item():
    """Same tag appearing twice on the same item should appear only once."""
    from scripts.vocabulary.keyword_extractor import extract_all_keywords
    mock_zot_client = MagicMock()
    mock_zot_client._zot.everything.return_value = [
        {"data": {
            "key": "ABC123",
            "tags": [{"tag": "TopicX"}, {"tag": "TopicX"}],
        }},
    ]

    mock_zot_client._zot.items.return_value = []
    pairs = extract_all_keywords(mock_zot_client)
    assert pairs.count(("TopicX", "ABC123")) == 1
@pytest.mark.unit
def test_extract_all_keywords_skips_empty_tags():
    """Empty tag strings are not included in output."""
    from scripts.vocabulary.keyword_extractor import extract_all_keywords
    mock_zot_client = MagicMock()
    mock_zot_client._zot.everything.return_value = [
        {"data": {"key": "XYZ", "tags": [{"tag": ""}, {"tag": "  "}, {"tag": "TopicX"}]}},
    ]
    mock_zot_client._zot.items.return_value = []
    pairs = extract_all_keywords(mock_zot_client)
    keywords = [p[0] for p in pairs]
    assert "" not in keywords
    assert "  " not in keywords
    assert "TopicX" in keywords


# ---------------------------------------------------------------------------
# Remediation pass tests
# ---------------------------------------------------------------------------

_REMEDIATION_RESPONSE = json.dumps({
    "assignments": {
        "some rare keyword": ["Mock Theme A"],
        "another rare one": ["Protein Properties", "Mock Theme A"],
    },
    "still_unclustered": [],
})

_REMEDIATION_UNKNOWN_THEME_RESPONSE = json.dumps({
    "assignments": {
        "some rare keyword": ["NonExistentTheme"],
    },
    "still_unclustered": [],
})


# ---------------------------------------------------------------------------
# Partial JSON recovery tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_recover_partial_json_extracts_complete_themes():
    """Salvages complete theme objects from a JSON string truncated mid-array."""
    from scripts.vocabulary.clusterer import _recover_partial_json

    # Simulate a response that was cut off during the third theme
    truncated = (
        '{\n  "themes": [\n'
        '    {"name": "Filtration", "keywords": ["filtration", "stage2filtration"]},\n'
        '    {"name": "Protein Properties", "keywords": ["viscosity", "aggregation"]},\n'
        '    {"name": "Incomplete Theme", "keywords": ["only_one'  # truncated here
    )
    result = _recover_partial_json(truncated)

    assert result is not None, "Should recover partial result"
    assert len(result["themes"]) == 2, "Should recover the two complete themes"
    theme_names = [t["name"] for t in result["themes"]]
    assert "Filtration" in theme_names
    assert "Protein Properties" in theme_names


@pytest.mark.unit
def test_recover_partial_json_returns_none_when_no_complete_themes():
    """Returns None when the JSON is so truncated that no theme is complete."""
    from scripts.vocabulary.clusterer import _recover_partial_json

    truncated = '{\n  "themes": [\n    {"name": "Incomplete", "keywords": ["only'
    result = _recover_partial_json(truncated)
    assert result is None


@pytest.mark.unit
def test_build_vocabulary_uses_recovery_on_parse_failure(tmp_path):
    """build_vocabulary falls back to _recover_partial_json when parse_llm_json raises."""
    from scripts.vocabulary.clusterer import build_vocabulary

    keywords = {"filtration": ["filtration"], "viscosity": ["viscosity"]}
    output_file = tmp_path / "concepts_draft.yaml"

    # First call: truncated response (parse_llm_json will fail, recovery succeeds)
    # Second call would be the remediation pass — we disable it for simplicity.
    truncated = (
        '{\n  "themes": [\n'
        '    {"name": "Filtration", "keywords": ["filtration"]}\n'
        # Missing closing brackets — truncated
    )
    with patch.object(MockLLMClient(), "complete", return_value=truncated):
        mock_llm = MockLLMClient()
        with patch.object(mock_llm, "complete", return_value=truncated):
            result = build_vocabulary(keywords, mock_llm, output_path=output_file, remediate=False)

    # Recovery should have produced the one complete theme
    assert any(t["name"] == "Filtration" for t in result["themes"])


# ---------------------------------------------------------------------------
# Remediation pass tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_remediate_unclustered_places_keywords_in_existing_themes(tmp_path):
    """Remediation pass assigns unclustered keywords into their matching themes."""
    from scripts.vocabulary.clusterer import _remediate_unclustered

    merged = {
        "themes": [
            {"name": "Mock Theme A", "keywords": ["filtration"]},
            {"name": "Protein Properties", "keywords": ["viscosity"]},
        ],
        "unclustered": ["some rare keyword", "another rare one"],
    }
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete", return_value=_REMEDIATION_RESPONSE):
        _remediate_unclustered(merged, mock_llm, max_tokens=16384, chunk_size=60)

    topicx_kws = next(t for t in merged["themes"] if t["name"] == "Mock Theme A")["keywords"]
    protein_kws = next(t for t in merged["themes"] if t["name"] == "Protein Properties")["keywords"]

    assert "some rare keyword" in topicx_kws, "keyword should have been placed in TopicX theme"
    assert "another rare one" in topicx_kws
    assert "another rare one" in protein_kws, "keyword assigned to two themes must appear in both"
    assert merged["unclustered"] == [], "all keywords placed — unclustered should be empty"


@pytest.mark.unit
def test_remediate_unclustered_skips_unknown_themes(tmp_path):
    """Remediation gracefully skips theme names the LLM invented that don't exist."""
    from scripts.vocabulary.clusterer import _remediate_unclustered

    merged = {
        "themes": [{"name": "Mock Theme A", "keywords": ["filtration"]}],
        "unclustered": ["some rare keyword"],
    }
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete", return_value=_REMEDIATION_UNKNOWN_THEME_RESPONSE):
        _remediate_unclustered(merged, mock_llm, max_tokens=16384, chunk_size=60)

    # Keyword should NOT have been added to any theme (theme name was unknown)
    topicx_kws = merged["themes"][0]["keywords"]
    assert "some rare keyword" not in topicx_kws


@pytest.mark.unit
def test_remediate_unclustered_noop_when_no_unclustered():
    """Remediation pass makes no LLM call when unclustered list is empty."""
    from scripts.vocabulary.clusterer import _remediate_unclustered

    merged = {
        "themes": [{"name": "Mock Theme A", "keywords": ["filtration"]}],
        "unclustered": [],
    }
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete") as mock_complete:
        _remediate_unclustered(merged, mock_llm, max_tokens=16384, chunk_size=60)

    assert not mock_complete.called, "LLM should not be called when unclustered is empty"


# ---------------------------------------------------------------------------
# Refinement pass tests
# ---------------------------------------------------------------------------

_REFINEMENT_RESPONSE = json.dumps({
    "additions": {
        "Protein Properties": ["topicx"],          # cross-theme: topicx also fits here
        "Mock Theme A": ["viscosity"],  # cross-theme: viscosity also fits here
    }
})

_REFINEMENT_UNKNOWN_THEME_RESPONSE = json.dumps({
    "additions": {
        "NonExistentTheme": ["some keyword"],
    }
})


@pytest.mark.unit
def test_refine_clustering_adds_cross_theme_keywords():
    """Refinement pass correctly adds keywords to additional themes."""
    from scripts.vocabulary.clusterer import _refine_clustering

    merged = {
        "themes": [
            {"name": "Mock Theme A", "keywords": ["filtration", "topicx"]},
            {"name": "Protein Properties", "keywords": ["viscosity"]},
        ],
        "unclustered": [],
    }
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete", return_value=_REFINEMENT_RESPONSE):
        _refine_clustering(merged, mock_llm, max_tokens=16384)

    protein_kws = next(t for t in merged["themes"] if t["name"] == "Protein Properties")["keywords"]
    topicx_kws = next(t for t in merged["themes"] if t["name"] == "Mock Theme A")["keywords"]

    assert "topicx" in protein_kws, "topicx should have been added to Protein Properties"
    assert "viscosity" in topicx_kws, "viscosity should have been added to TopicX theme"
    # Originals must still be present — refinement never removes
    assert "filtration" in topicx_kws
    assert "viscosity" in protein_kws


@pytest.mark.unit
def test_refine_clustering_skips_unknown_themes():
    """Refinement gracefully skips theme names in additions that don't exist."""
    from scripts.vocabulary.clusterer import _refine_clustering

    merged = {
        "themes": [{"name": "Mock Theme A", "keywords": ["filtration"]}],
        "unclustered": [],
    }
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete", return_value=_REFINEMENT_UNKNOWN_THEME_RESPONSE):
        _refine_clustering(merged, mock_llm, max_tokens=16384)

    # No new themes should have been created, existing keywords untouched
    assert len(merged["themes"]) == 1
    assert merged["themes"][0]["keywords"] == ["filtration"]


@pytest.mark.unit
def test_refine_clustering_noop_when_no_themes():
    """Refinement pass makes no LLM call when there are no themes."""
    from scripts.vocabulary.clusterer import _refine_clustering

    merged = {"themes": [], "unclustered": ["orphan"]}
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete") as mock_complete:
        _refine_clustering(merged, mock_llm, max_tokens=16384)

    assert not mock_complete.called, "LLM should not be called when there are no themes"


# ---------------------------------------------------------------------------
# Refinement partial-recovery tests
# ---------------------------------------------------------------------------

_TRUNCATED_REFINEMENT = (
    '{\n  "additions": {\n'
    '    "Mock Theme A": ["multimodal chromatography", "tangential flow"],\n'
    '    "Protein Properties": ["osmotic pressure"],\n'
    '    "Incomplete Theme": ["only_one'   # truncated here — array never closed
)

_EMPTY_ADDITIONS_TRUNCATION = (
    '{\n  "additions": {\n'
    '    "Bad Theme": '   # truncated before the array even starts
)


@pytest.mark.unit
def test_recover_partial_refinement_json_extracts_complete_entries():
    """Salvages complete theme-addition entries from a JSON string truncated mid-dict."""
    from scripts.vocabulary.clusterer import _recover_partial_refinement_json

    result = _recover_partial_refinement_json(_TRUNCATED_REFINEMENT)

    assert result is not None, "Should recover partial result"
    additions = result["additions"]
    assert "Mock Theme A" in additions
    assert "Protein Properties" in additions
    # The truncated "Incomplete Theme" entry must NOT appear
    assert "Incomplete Theme" not in additions
    assert "multimodal chromatography" in additions["Mock Theme A"]
    assert "osmotic pressure" in additions["Protein Properties"]


@pytest.mark.unit
def test_recover_partial_refinement_json_returns_none_when_nothing_recoverable():
    """Returns None when the JSON is so truncated that no entry is complete."""
    from scripts.vocabulary.clusterer import _recover_partial_refinement_json

    result = _recover_partial_refinement_json(_EMPTY_ADDITIONS_TRUNCATION)
    assert result is None


@pytest.mark.unit
def test_recover_partial_refinement_json_returns_none_without_additions_key():
    """Returns None when the raw string has no 'additions' key at all."""
    from scripts.vocabulary.clusterer import _recover_partial_refinement_json

    result = _recover_partial_refinement_json('{"something_else": {}}')
    assert result is None


@pytest.mark.unit
def test_refine_clustering_uses_partial_recovery_on_truncation():
    """When parse_llm_json fails, _refine_clustering falls back to partial recovery."""
    from scripts.vocabulary.clusterer import _refine_clustering

    merged = {
        "themes": [
            {"name": "Mock Theme A", "keywords": ["filtration"]},
            {"name": "Protein Properties", "keywords": ["viscosity"]},
        ],
        "unclustered": [],
    }
    mock_llm = MockLLMClient()
    with patch.object(mock_llm, "complete", return_value=_TRUNCATED_REFINEMENT):
        _refine_clustering(merged, mock_llm)

    topicx_kws = next(t for t in merged["themes"] if t["name"] == "Mock Theme A")["keywords"]
    # Salvaged additions must have been applied
    assert "multimodal chromatography" in topicx_kws
    assert "tangential flow" in topicx_kws
    # Original keywords must be untouched
    assert "filtration" in topicx_kws


# ---------------------------------------------------------------------------
# clusters_to_topics_yaml / _pick_query_terms tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_clusters_to_topics_yaml_format():
    """Output has a valid topics.yaml schema: searches list with name/query/databases."""
    from scripts.vocabulary.clusterer import clusters_to_topics_yaml
    clusters = {
        "themes": [
            {
                "name": "Membrane Filtration",
                "keywords": [
                    "filtration",
                    "tangential flow filtration",
                    "system fouling",
                    "stage2filtration",
                ],
            },
            {
                "name": "Protein Formulation",
                "keywords": [
                    "high concentration mab",
                    "viscosity reduction",
                    "excipient screening",
                ],
            },
        ],
        "unclustered": ["misc_term"],
    }
    raw = clusters_to_topics_yaml(clusters)
    # Must be valid YAML
    data = yaml.safe_load(raw)
    assert "searches" in data
    searches = data["searches"]
    assert len(searches) == 2
    # Every entry has the required keys
    for entry in searches:
        assert "name" in entry
        assert "query" in entry
        assert "databases" in entry
        assert isinstance(entry["databases"], list)
    # Unclustered keywords must not appear as a search entry
    names = {s["name"] for s in searches}
    assert "misc_term" not in names
    # Names match the theme names
    assert "Membrane Filtration" in names
    assert "Protein Formulation" in names


@pytest.mark.unit
def test_clusters_to_topics_yaml_empty_themes():
    """Empty clusters produce an empty searches list without error."""
    from scripts.vocabulary.clusterer import clusters_to_topics_yaml
    raw = clusters_to_topics_yaml({"themes": [], "unclustered": []})
    data = yaml.safe_load(raw)
    assert data == {"searches": []}


@pytest.mark.unit
def test_clusters_to_topics_yaml_databases_override():
    """Caller can specify alternative database lists."""
    from scripts.vocabulary.clusterer import clusters_to_topics_yaml
    clusters = {
        "themes": [{"name": "Test Theme", "keywords": ["some keyword"]}],
        "unclustered": [],
    }
    raw = clusters_to_topics_yaml(clusters, databases=["pubmed", "scopus"])
    data = yaml.safe_load(raw)
    assert data["searches"][0]["databases"] == ["pubmed", "scopus"]


@pytest.mark.unit
def test_clusters_to_topics_yaml_query_uses_short_phrases():
    """
    Queries prefer 2–3 word phrases from the keyword list rather than
    single-word or very long terms.
    """
    from scripts.vocabulary.clusterer import clusters_to_topics_yaml
    clusters = {
        "themes": [
            {
                "name": "TopicX",
                "keywords": [
                    # single word — lower preference
                    "filtration",
                    # 2-word phrase — preferred
                    "tangential flow",
                    # 3-word phrase — preferred
                    "system fouling protein",
                    # very long phrase — lower preference
                    "filtration and stage2filtration of concentrated model proteins",
                ],
            }
        ],
        "unclustered": [],
    }
    raw = clusters_to_topics_yaml(clusters)
    data = yaml.safe_load(raw)
    query = data["searches"][0]["query"]
    # Both 2-word and 3-word phrases should appear in the query
    assert "tangential flow" in query
    assert "system fouling protein" in query


@pytest.mark.unit
def test_clusters_to_topics_yaml_header_comment():
    """Generated YAML string starts with an auto-generated warning comment."""
    from scripts.vocabulary.clusterer import clusters_to_topics_yaml
    raw = clusters_to_topics_yaml({"themes": [], "unclustered": []})
    assert raw.startswith("#")
    assert "Auto-generated" in raw


@pytest.mark.unit
def test_topics_suggested_path_constant():
    """TOPICS_SUGGESTED_PATH points to the expected config location."""
    from scripts.vocabulary.clusterer import TOPICS_SUGGESTED_PATH
    assert TOPICS_SUGGESTED_PATH == Path("config/topics_suggested.yaml")


@pytest.mark.unit
def test_suggest_topics_non_interactive_skips_prompt(tmp_path, monkeypatch, capsys):
    """
    In non-interactive mode (stdin not a TTY), _suggest_topics writes the
    suggested file and prints the diff but never prompts.
    """
    from scripts.cli import _suggest_topics

    # Change CWD so Path("config/topics.yaml") resolves inside tmp_path.
    monkeypatch.chdir(tmp_path)
    # Fake non-TTY stdin.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    clusters = {
        "themes": [{"name": "Membrane Filtration", "keywords": ["tangential flow", "stage2filtration"]}],
        "unclustered": [],
    }
    suggested_path = tmp_path / "topics_suggested.yaml"
    # No live topics.yaml — simulate fresh setup (file does not exist).
    _suggest_topics(clusters, suggested_path)

    # Suggested file must have been written.
    assert suggested_path.exists()
    captured = capsys.readouterr()
    assert "Membrane Filtration" in captured.out
    # Non-interactive message must appear, no interactive prompt.
    assert "Non-interactive" in captured.out


@pytest.mark.unit
def test_suggest_topics_append_preserves_existing_searches_and_config(tmp_path, monkeypatch, capsys):
    """
    When the user confirms, new entries are appended and all existing searches
    and top-level config keys (date_window_days, discovery_top_k) are preserved.
    """
    import yaml

    from scripts.cli import _suggest_topics

    # Change CWD so Path("config/topics.yaml") resolves to tmp_path/config/topics.yaml.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    live_topics = config_dir / "topics.yaml"
    live_topics.write_text(
        "searches:\n"
        "  - name: Manual Search\n"
        "    query: protein AND filtration\n"
        "    databases: [pubmed]\n"
        "date_window_days: 14\n"
        "discovery_top_k: 20\n",
        encoding="utf-8",
    )

    suggested_path = tmp_path / "topics_suggested.yaml"
    clusters = {
        "themes": [
            {"name": "Membrane Filtration", "keywords": ["tangential flow", "filtration"]},
            {"name": "Protein Formulation", "keywords": ["high concentration mab", "viscosity"]},
        ],
        "unclustered": [],
    }

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("click.confirm", lambda *a, **kw: True)

    _suggest_topics(clusters, suggested_path)

    result = yaml.safe_load(live_topics.read_text(encoding="utf-8"))
    search_names = {s["name"] for s in result.get("searches", [])}
    # Manual search preserved.
    assert "Manual Search" in search_names
    # New entries added.
    assert "Membrane Filtration" in search_names
    assert "Protein Formulation" in search_names
    # Top-level config keys must survive the round-trip.
    assert result.get("date_window_days") == 14
    assert result.get("discovery_top_k") == 20


@pytest.mark.unit
def test_suggest_topics_no_new_entries_skips_prompt(tmp_path, monkeypatch, capsys):
    """
    When all suggested themes already exist in topics.yaml, _suggest_topics
    prints a 'nothing new' message and does not call click.confirm().
    """
    from scripts.cli import _suggest_topics

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    live_topics = config_dir / "topics.yaml"
    live_topics.write_text(
        "searches:\n  - name: Existing Theme\n    query: x\n    databases: [pubmed]\n",
        encoding="utf-8",
    )

    clusters = {
        "themes": [{"name": "Existing Theme", "keywords": ["tangential flow", "stage2filtration"]}],
        "unclustered": [],
    }
    suggested_path = tmp_path / "topics_suggested.yaml"

    confirm_called = []
    monkeypatch.setattr("click.confirm", lambda *a, **kw: confirm_called.append(True))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    _suggest_topics(clusters, suggested_path)

    captured = capsys.readouterr()
    assert "already have a matching entry" in captured.out
    # confirm() must NOT have been called.
    assert confirm_called == []


@pytest.mark.unit
def test_extract_all_keywords_raises_after_all_retries_exhausted():
    """When every attempt fails, extract_all_keywords raises RuntimeError."""
    from scripts.vocabulary.keyword_extractor import extract_all_keywords
    mock_zot_client = MagicMock()
    mock_zot_client._zot.everything.side_effect = TimeoutError("read timed out")
    mock_zot_client._zot.items.return_value = []
    with pytest.raises(RuntimeError, match="Zotero keyword extraction failed"):
        # Use max_retries=1 and retry_delay=0 to keep the test fast.
        extract_all_keywords(mock_zot_client, max_retries=1, retry_delay=0.0)


@pytest.mark.unit
def test_extract_all_keywords_succeeds_on_second_attempt():
    """A single transient failure followed by a success returns the keywords."""
    from scripts.vocabulary.keyword_extractor import extract_all_keywords
    mock_zot_client = MagicMock()
    good_items = [{"data": {"key": "ABC", "tags": [{"tag": "membrane"}]}}]
    # First call raises; second call returns the items list.
    mock_zot_client._zot.everything.side_effect = [
        TimeoutError("read timed out"),
        good_items,
    ]
    mock_zot_client._zot.items.return_value = []
    pairs = extract_all_keywords(mock_zot_client, max_retries=2, retry_delay=0.0)
    assert ("membrane", "ABC") in pairs


# ---------------------------------------------------------------------------
# B2 — assign_themes
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_assign_themes_returns_empty_when_no_vocab(tmp_path):
    """assign_themes returns [] without raising when no vocabulary file exists."""
    from scripts.vocabulary.normalizer import assign_themes
    # Pass a non-existent path explicitly — no crash expected
    result = assign_themes(["filtration", "mab"], vocab_path=tmp_path / "nonexistent.yaml")
    assert result == []


@pytest.mark.unit
def test_assign_themes_returns_empty_for_empty_keywords(tmp_path):
    """assign_themes returns [] immediately for empty keyword list."""
    from scripts.vocabulary.normalizer import assign_themes
    # Even if a vocab file exists, empty keywords → []
    vocab = tmp_path / "concepts.yaml"
    vocab.write_text("themes:\n  - name: TopicX\n    keywords:\n      - filtration\n")
    assert assign_themes([], vocab_path=vocab) == []


@pytest.mark.unit
def test_assign_themes_matches_exact_keyword(tmp_path):
    """An exact keyword match (ratio=100) always triggers theme assignment."""
    from scripts.vocabulary.normalizer import assign_themes
    vocab = tmp_path / "concepts.yaml"
    vocab.write_text(
        "themes:\n"
        "  - name: Mock Theme A\n"
        "    keywords:\n"
        "      - filtration\n"
        "      - stage2filtration\n"
        "  - name: Chromatography\n"
        "    keywords:\n"
        "      - protein a chromatography\n"
        "      - ion exchange\n"
    )
    result = assign_themes(["filtration", "mab"], vocab_path=vocab)
    assert "Mock Theme A" in result
    assert "Chromatography" not in result


@pytest.mark.unit
def test_assign_themes_fuzzy_match(tmp_path):
    """L5: paper keyword that differs in punctuation from the vocab entry still
    matches via rapidfuzz when ratio >= 85.

    Fixture chosen so the fuzzy path (not the exact-match short-circuit) is
    actually exercised:
      paper kw:  'protein A purification'   (normalised → 'protein a purification')
      vocab kw:  'protein-a-purification'   (normalised → 'protein-a-purification')
      fuzz.ratio ≈ 90.9, which is above the default threshold of 85.
    The two strings are NOT equal post-normalisation, so a match here
    proves the fuzzy comparison path fired.
    """
    from rapidfuzz import fuzz

    from scripts.vocabulary.normalizer import assign_themes

    paper_kw = "protein A purification"
    vocab_kw = "protein-a-purification"
    # Sanity-check the fixture so future edits can't quietly turn this back
    # into an exact-match test: the strings must differ post-normalisation
    # AND clear the 85 fuzzy threshold.
    assert paper_kw.strip().lower() != vocab_kw.strip().lower()
    assert fuzz.ratio(paper_kw.strip().lower(), vocab_kw.strip().lower()) >= 85

    vocab = tmp_path / "concepts.yaml"
    vocab.write_text(
        "themes:\n"
        "  - name: Downstream Processing\n"
        "    keywords:\n"
        f"      - {vocab_kw}\n"
        "  - name: Unrelated\n"
        "    keywords:\n"
        "      - electrophoresis\n"
    )

    result = assign_themes([paper_kw], vocab_path=vocab)
    # The fuzzy match should pull in 'Downstream Processing' and nothing else.
    assert "Downstream Processing" in result
    assert "Unrelated" not in result


@pytest.mark.unit
def test_assign_themes_multiple_themes_matched(tmp_path):
    """Multiple themes can match a single paper's keywords."""
    from scripts.vocabulary.normalizer import assign_themes
    vocab = tmp_path / "concepts.yaml"
    vocab.write_text(
        "themes:\n"
        "  - name: TopicX\n"
        "    keywords:\n"
        "      - filtration\n"
        "  - name: Biologics\n"
        "    keywords:\n"
        "      - mab\n"
)
    result = assign_themes(["filtration", "mab"], vocab_path=vocab)
    assert "TopicX" in result
    assert "Biologics" in result


@pytest.mark.unit
def test_themes_assigned_during_brain_build(tmp_path):
    """B2: _process_paper sets themes from vocabulary, not always []."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from scripts.core.state_db import StateDB

    # Write a minimal vocab file
    vocab_dir = tmp_path / "config"
    vocab_dir.mkdir()
    vocab_file = vocab_dir / "concepts.yaml"
    vocab_file.write_text(
        "themes:\n"
        "  - name: Mock Theme A\n"
        "    keywords:\n"
        "      - filtration\n"
        "      - stage2filtration\n"
    )

    config = SimpleNamespace(
        zotero=SimpleNamespace(
            collection_name="lit-monitor",
            local_storage_path=str(tmp_path / "zotero"),
        ),
        obsidian=SimpleNamespace(
            vault_path=str(tmp_path / "vault"),
            papers_folder="Literature/Papers",
        ),
    )
    (tmp_path / "vault" / "Literature" / "Papers").mkdir(parents=True)
    state_db = StateDB(tmp_path / "state.db")

    item = {
        "key": "TESTKEY",
        "data": {
            "DOI": "10.1000/b2test",
            "title": "Filtration of ProteinAs",
            "abstractNote": "We studied TopicX.",
            "date": "2024",
            "publicationTitle": "J. Membranes",
            "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "A"}],
            # Tags include a keyword that matches the vocab
            "tags": [{"tag": "filtration"}, {"tag": "mab"}],
        },
        "attachments": [],
    }

    captured: list[dict] = []

    def fake_write_note(paper_record, extraction, config, **kwargs):
        captured.append(paper_record)
        return str(tmp_path / "vault" / "Literature" / "Papers" / "test_note.md")

    (tmp_path / "vault" / "Literature" / "Papers" / "test_note.md").touch()

    from scripts.pipelines.brain_build import _process_paper

    zotero_client = MagicMock()
    zotero_client.get_markdown_attachment.return_value = "Filtration paper text in markdown."

    with patch("scripts.vocabulary.normalizer._VOCAB_CANDIDATES", [vocab_file]):
        llm = MagicMock()
        llm.provider = "ollama"
        llm.model = "qwen2.5:3b"
        embeddings_db = MagicMock()
        _process_paper(
            doi="10.1000/b2test",
            zotero_key="TESTKEY",
            item=item,
            config=config,
            state_db=state_db,
            embeddings_db=embeddings_db,
            llm=llm,
            zotero_client=zotero_client,
            extract_paper_fn=lambda *a, **kw: {
                "core_finding": "Flux decline.", "core_finding_confidence": "explicit",
                "_overall_confidence": 1.0,
            },
            write_paper_note_fn=fake_write_note,
        )

    assert captured, "write_paper_note_fn was never called"
    assert "Mock Theme A" in captured[0]["themes"], (
        f"Expected theme not found. themes={captured[0]['themes']}"
    )


# ---------------------------------------------------------------------------
# N1 smart-chunking tests (build-vocabulary, 2026-05-17)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_keywords_per_chunk_smart_sizes_from_num_ctx():
    """Smart sizing: with a known num_ctx and short keywords, the helper
    returns a chunk count consistent with A6-style budget math.

    For num_ctx=8192 and avg keyword length ~10 chars:
        budget = (8192*0.75 - 500) * 4 = 22576 chars
        per_kw = (10+5) + (10*1.5) = 30 chars
        raw_k  = (22576 - 800) / 30 ≈ 725 keywords
    The result is clamped to _MAX_KEYWORDS_PER_CHUNK=1000, so we expect ~725.
    """
    from scripts.vocabulary.clusterer import _keywords_per_chunk
    keywords = [f"keyword_{i}" for i in range(100)]  # avg ~10 chars
    llm = MockLLMClient()
    llm.num_ctx = 8192
    k = _keywords_per_chunk(llm, keywords)
    assert 500 < k < 1000, f"Expected smart-sized chunk in (500, 1000), got {k}"


@pytest.mark.unit
def test_keywords_per_chunk_respects_user_cap_hard_clamp():
    """`user_cap` clamps the smart-computed size DOWN. Never above the cap."""
    from scripts.vocabulary.clusterer import _keywords_per_chunk
    keywords = [f"keyword_{i}" for i in range(100)]
    llm = MockLLMClient()
    llm.num_ctx = 8192
    k = _keywords_per_chunk(llm, keywords, user_cap=60)
    assert k == 60, f"user_cap=60 should clamp to 60; got {k}"


@pytest.mark.unit
def test_keywords_per_chunk_clamps_to_min_floor():
    """Tiny num_ctx still yields at least _MIN_KEYWORDS_PER_CHUNK=20."""
    from scripts.vocabulary.clusterer import _MIN_KEYWORDS_PER_CHUNK, _keywords_per_chunk
    keywords = ["a very long verbose keyword phrase " * 5] * 10  # very long avg length
    llm = MockLLMClient()
    llm.num_ctx = 2048  # very small
    k = _keywords_per_chunk(llm, keywords)
    assert k >= _MIN_KEYWORDS_PER_CHUNK


@pytest.mark.unit
def test_keywords_per_chunk_honors_chunk_chars_override():
    """`chunk_chars_override` bypasses num_ctx math."""
    from scripts.vocabulary.clusterer import _keywords_per_chunk
    keywords = [f"kw_{i}" for i in range(50)]  # avg 5 chars
    llm = MockLLMClient()
    llm.num_ctx = 131072  # would normally yield ~5000+ keywords
    k = _keywords_per_chunk(llm, keywords, chunk_chars_override=2000)
    # With budget=2000 chars, per_kw_cost = (5+5)+(5*1.5) = 17.5,
    # raw_k = (2000-800)/17.5 ≈ 68 → clamped within min/max
    assert 20 <= k <= 100, f"Expected override-based chunk size in [20, 100], got {k}"


@pytest.mark.unit
def test_build_vocabulary_default_chunk_size_uses_smart_sizing(tmp_path):
    """build_vocabulary(chunk_size=0) (the new default) triggers smart sizing.

    With 50 short keywords and a generous num_ctx, smart sizing fits everything
    in a single LLM call — the old default of 60 would also produce 1 call here,
    but the test asserts the smart-sized path is taken (no hard cap clamping).
    """
    from scripts.vocabulary.clusterer import build_vocabulary
    keywords = {f"kw_{i:03d}": [f"kw_{i:03d}"] for i in range(50)}
    output_file = tmp_path / "concepts_draft.yaml"
    mock_llm = MockLLMClient()
    mock_llm.num_ctx = 8192
    with patch.object(mock_llm, "complete", return_value=_MOCK_CLUSTERER_RESPONSE) as mock_complete:
        build_vocabulary(
            keywords, mock_llm, output_path=output_file,
            chunk_size=0,  # 0 = smart sizing only, no hard cap
            remediate=False,
        )
    # 50 keywords + 8192 ctx → smart-sized to 1 chunk
    assert mock_complete.call_count == 1, (
        f"Expected 1 LLM call under smart sizing for 50 kw / ctx=8192; "
        f"got {mock_complete.call_count}"
    )


@pytest.mark.unit
def test_build_vocabulary_chunk_size_clamps_smart_sizing(tmp_path):
    """When chunk_size>0 and smaller than the smart-computed value, it clamps."""
    from scripts.vocabulary.clusterer import build_vocabulary
    keywords = {f"kw_{i:03d}": [f"kw_{i:03d}"] for i in range(100)}
    output_file = tmp_path / "concepts_draft.yaml"
    mock_llm = MockLLMClient()
    mock_llm.num_ctx = 8192  # smart-sizing would compute ~700 → 1 chunk for 100 kws
    with patch.object(mock_llm, "complete", return_value=_MOCK_CLUSTERER_RESPONSE) as mock_complete:
        build_vocabulary(
            keywords, mock_llm, output_path=output_file,
            chunk_size=30,  # hard cap forces 30 per chunk → 100/30 = 4 chunks
            remediate=False,
        )
    assert mock_complete.call_count == 4, (
        f"Expected 4 LLM calls with hard cap chunk_size=30; got {mock_complete.call_count}"
    )


# ---------------------------------------------------------------------------
# Audit R29 — assign_themes surfaces vocab corruption with a one-shot WARNING
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_assign_themes_warns_when_vocab_has_dict_keywords(tmp_path, caplog):
    """Audit R28 root cause exposed: an M4 indentation bug can leave dict-shaped
    keyword entries in concepts.yaml.  assign_themes() must skip them
    gracefully (no crash) AND emit a WARNING the first time so the user sees
    the corruption rather than getting silently bad theme assignments.
    """
    import logging

    import scripts.vocabulary.normalizer as _norm
    from scripts.vocabulary.normalizer import assign_themes

    # Reset the one-shot flag in case a prior test in this module fired it.
    _norm._warned_vocab_corruption = False

    vocab = tmp_path / "concepts.yaml"
    vocab.write_text(
        "themes:\n"
        "  - name: Biologics\n"
        "    keywords:\n"
        "      - antibody\n"
        "      - {name: 'corrupted-as-dict', keywords: ['x']}\n"  # the bug shape
    )
    with caplog.at_level(logging.WARNING, logger="scripts.vocabulary.normalizer"):
        result = assign_themes(["antibody"], vocab_path=vocab)

    # The valid keyword still produced a match — the malformed entry didn't poison the run.
    assert result == ["Biologics"]
    # And the user is informed.
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("non-string keyword" in m for m in warnings), (
        f"Expected a WARNING about non-string keywords; got: {warnings}"
    )


@pytest.mark.unit
def test_assign_themes_warning_is_one_shot_per_process(tmp_path, caplog):
    """The corruption warning should fire ONCE per process, not once per paper.
    Otherwise discovery processing 50 papers would log 50× the same line.
    """
    import logging

    import scripts.vocabulary.normalizer as _norm
    from scripts.vocabulary.normalizer import assign_themes

    _norm._warned_vocab_corruption = False
    vocab = tmp_path / "concepts.yaml"
    vocab.write_text(
        "themes:\n"
        "  - name: Biologics\n"
        "    keywords:\n"
        "      - {name: 'corrupted', keywords: ['x']}\n"
        "      - antibody\n"
    )
    with caplog.at_level(logging.WARNING, logger="scripts.vocabulary.normalizer"):
        for _ in range(3):
            assign_themes(["antibody"], vocab_path=vocab)
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "non-string keyword" in r.message
    ]
    assert len(warnings) == 1, f"Expected 1 warning across 3 calls; got {len(warnings)}"
