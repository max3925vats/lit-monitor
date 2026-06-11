"""Guard tests for the shipped example configs under config/examples/.

Every domain example (bioprocessing / ml-research / climate-science) must:
  1. parse as YAML, and
  2. validate against the SAME loaders/validators the running app uses for
     the live config/*.yaml equivalents.

For topics / domain_context / researchers we reuse the setup wizard's own
validators (scripts.server.routes.setup) — these are exactly the checks a
user's hand-written config goes through. For concepts.yaml we feed the file
through scripts.vocabulary.normalizer.assign_themes (the real vocabulary
loader, which accepts a vocab_path override).

A negative control (test_validators_reject_malformed) proves the validators
actually fail on broken input, so a passing example test is meaningful.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from lit_monitor.server.routes.setup import (
    _validate_domain,
    _validate_researchers,
    _validate_topics,
)
from lit_monitor.vocabulary.normalizer import assign_themes

# config/examples/ lives two levels up from this test file's repo (tests/unit/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "config" / "examples"
_DOMAINS = ["bioprocessing", "ml-research", "climate-science"]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


@pytest.mark.unit
@pytest.mark.parametrize("domain", _DOMAINS)
def test_example_dir_has_all_files(domain: str) -> None:
    """Each domain ships the four user-facing domain-flavored configs."""
    d = _EXAMPLES_DIR / domain
    for fname in ("topics.yaml", "domain_context.yaml", "concepts.yaml", "researchers.yaml"):
        assert (d / fname).is_file(), f"missing {domain}/{fname}"


@pytest.mark.unit
@pytest.mark.parametrize("domain", _DOMAINS)
def test_example_topics_valid(domain: str) -> None:
    """topics.yaml parses and passes the wizard's topic validator."""
    data = _load_yaml(_EXAMPLES_DIR / domain / "topics.yaml")
    searches = data.get("searches")
    assert isinstance(searches, list) and searches, f"{domain}: no searches list"
    errors = _validate_topics(searches)
    assert not errors, f"{domain}/topics.yaml validation errors: {errors}"


@pytest.mark.unit
@pytest.mark.parametrize("domain", _DOMAINS)
def test_example_domain_context_valid(domain: str) -> None:
    """domain_context.yaml parses and passes the wizard's domain validator."""
    data = _load_yaml(_EXAMPLES_DIR / domain / "domain_context.yaml")
    focus = data.get("domain_focus", "")
    assert isinstance(focus, str)
    errors = _validate_domain(focus)
    assert not errors, f"{domain}/domain_context.yaml validation errors: {errors}"


@pytest.mark.unit
@pytest.mark.parametrize("domain", _DOMAINS)
def test_example_researchers_valid(domain: str) -> None:
    """researchers.yaml parses and passes the wizard's researcher validator.

    The validator flags rows that have a name but no orcid/scopus_id, so the
    examples must supply a (placeholder) ORCID for each tracked author.
    """
    data = _load_yaml(_EXAMPLES_DIR / domain / "researchers.yaml")
    researchers = data.get("researchers", [])
    assert isinstance(researchers, list) and researchers, f"{domain}: no researchers"
    errors = _validate_researchers(researchers)
    assert not errors, f"{domain}/researchers.yaml validation errors: {errors}"


@pytest.mark.unit
@pytest.mark.parametrize("domain", _DOMAINS)
def test_example_concepts_loads_via_normalizer(domain: str) -> None:
    """concepts.yaml parses and is consumable by the real vocabulary loader.

    assign_themes() returns [] on a malformed/empty vocab without raising, so
    we assert a positive match: a keyword pulled from the file's own first
    theme must round-trip back to that theme name. This proves the themes/
    keywords structure is actually usable, not just syntactically valid.
    """
    path = _EXAMPLES_DIR / domain / "concepts.yaml"
    data = _load_yaml(path)
    themes = data.get("themes")
    assert isinstance(themes, list) and themes, f"{domain}: no themes"

    first = themes[0]
    assert isinstance(first, dict) and first.get("name"), f"{domain}: malformed first theme"
    kws = first.get("keywords") or []
    assert kws and isinstance(kws[0], str), f"{domain}: first theme has no string keywords"

    matched = assign_themes([kws[0]], vocab_path=path)
    assert first["name"] in matched, (
        f"{domain}/concepts.yaml: keyword {kws[0]!r} did not map back to "
        f"theme {first['name']!r} (got {matched})"
    )


@pytest.mark.unit
def test_validators_reject_malformed() -> None:
    """Negative control: the validators we rely on actually fail on bad input.

    Without this, the positive tests above could pass vacuously if a validator
    were a no-op. We feed deliberately broken stubs and require errors.
    """
    # topics: missing query + bad database
    assert _validate_topics([{"name": "x", "query": "", "databases": ["bogus"]}])
    # topics: empty list
    assert _validate_topics([])
    # domain: empty string
    assert _validate_domain("   ")
    # researchers: name present but no orcid/scopus_id
    assert _validate_researchers([{"name": "Nobody", "orcid": "", "scopus_id": ""}])
    # concepts: a broken vocab file produces no theme match
    broken = _EXAMPLES_DIR  # any dir path that is not a vocab file
    assert assign_themes(["anything"], vocab_path=broken / "does_not_exist.yaml") == []


@pytest.mark.unit
def test_examples_readme_exists() -> None:
    assert (_EXAMPLES_DIR / "README.md").is_file()


@pytest.mark.unit
def test_pyproject_metadata_present() -> None:
    """pyproject.toml parses and carries the PyPI-ready [project] metadata."""
    pyproject = _REPO_ROOT / "pyproject.toml"
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    project = data["project"]
    assert project["description"].strip()
    assert project["readme"] == "README.md"
    assert project["license"]  # { file = "LICENSE" }
    assert project["keywords"]
    classifiers = project["classifiers"]
    assert "Development Status :: 3 - Alpha" in classifiers
    assert "License :: OSI Approved :: MIT License" in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    urls = project["urls"]
    for key in ("Homepage", "Repository", "Issues"):
        assert urls[key].startswith("https://github.com/max3925vats/lit-monitor")
