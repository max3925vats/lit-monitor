"""
Unit tests for H1 — item routing (config/item_routing.yaml + item_router.py).
Covers:
  - load_item_routes() returns correct ItemRoute for every known item type
  - Pydantic validation rejects bad entries
  - get_route() raises KeyError for unknown types
  - Cache is isolated per config_path (test YAML doesn't pollute production cache)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def _write_routing_yaml(tmp_path: Path, content: str) -> Path:
    """Write a temporary item_routing.yaml and return its path."""
    p = tmp_path / "item_routing.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Production YAML — happy-path route lookups
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_item_routes_returns_dict():
    """load_item_routes() returns a non-empty dict of ItemRoute objects."""
    from scripts.core.item_router import ItemRoute, load_item_routes
    routes = load_item_routes()
    assert isinstance(routes, dict)
    assert len(routes) > 0
    for key, val in routes.items():
        assert isinstance(key, str)
        assert isinstance(val, ItemRoute)


@pytest.mark.unit
def test_journal_article_routes_to_brain_build():
    """journalArticle → brain_build pipeline, paper schema, source_type='paper'."""
    from scripts.core.item_router import get_route
    r = get_route("journalArticle")
    assert r.pipeline == "brain_build"
    assert r.default_schema == "paper"
    assert r.source_type == "paper"


@pytest.mark.unit
def test_conference_paper_routes_to_brain_build():
    """conferencePaper routes identical to journalArticle."""
    from scripts.core.item_router import get_route
    r = get_route("conferencePaper")
    assert r.pipeline == "brain_build"
    assert r.default_schema == "paper"


@pytest.mark.unit
def test_preprint_routes_to_brain_build():
    """preprint → brain_build, paper schema."""
    from scripts.core.item_router import get_route
    r = get_route("preprint")
    assert r.pipeline == "brain_build"
    assert r.default_schema == "paper"


@pytest.mark.unit
def test_book_section_routes_to_skip():
    """bookSection → skip (R-10: textbook pipeline removed)."""
    from scripts.core.item_router import get_route
    r = get_route("bookSection")
    assert r.pipeline == "skip"
    assert r.source_type == "skipped_book_section"


@pytest.mark.unit
def test_book_routes_to_skip():
    """book → skip (R-10: textbook pipeline removed)."""
    from scripts.core.item_router import get_route
    r = get_route("book")
    assert r.pipeline == "skip"
    assert r.source_type == "skipped_book"


@pytest.mark.unit
def test_thesis_routes_to_skip():
    """thesis → skip (R-10: textbook pipeline removed)."""
    from scripts.core.item_router import get_route
    r = get_route("thesis")
    assert r.pipeline == "skip"
    assert r.source_type == "skipped_thesis"


@pytest.mark.unit
def test_report_routes_to_skip():
    """report → skip pipeline (logged to run report, not extracted)."""
    from scripts.core.item_router import get_route
    r = get_route("report")
    assert r.pipeline == "skip"


# ---------------------------------------------------------------------------
# get_route() — unknown type raises KeyError
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_route_raises_for_unknown_type():
    """get_route() raises KeyError for item types not in the routing table."""
    from scripts.core.item_router import get_route
    with pytest.raises(KeyError, match="unknown_type"):
        get_route("unknown_type")


# ---------------------------------------------------------------------------
# Pydantic validation — bad YAML entries are rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_invalid_pipeline_value_raises(tmp_path):
    """ItemRoute raises ValidationError when pipeline is not a known literal."""
    yaml_content = """
        routes:
          journalArticle:
            pipeline: invalid_pipeline
            default_schema: paper
            source_type: paper
    """
    p = _write_routing_yaml(tmp_path, yaml_content)
    from scripts.core.item_router import load_item_routes
    # Clear cache so the test YAML is loaded fresh.
    load_item_routes.cache_clear()
    with pytest.raises((ValueError, Exception)):
        load_item_routes(p)
    load_item_routes.cache_clear()


@pytest.mark.unit
def test_empty_source_type_raises(tmp_path):
    """ItemRoute rejects an empty source_type string."""
    yaml_content = """
        routes:
          journalArticle:
            pipeline: brain_build
            default_schema: paper
            source_type: ""
    """
    p = _write_routing_yaml(tmp_path, yaml_content)
    from scripts.core.item_router import load_item_routes
    load_item_routes.cache_clear()
    with pytest.raises((ValueError, Exception)):
        load_item_routes(p)
    load_item_routes.cache_clear()


# ---------------------------------------------------------------------------
# load_item_routes() with custom YAML — cache isolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_item_routes_custom_yaml(tmp_path):
    """load_item_routes() accepts a custom config_path for test injection."""
    yaml_content = """
        routes:
          customType:
            pipeline: brain_build
            default_schema: paper
            source_type: custom
    """
    p = _write_routing_yaml(tmp_path, yaml_content)
    from scripts.core.item_router import load_item_routes
    load_item_routes.cache_clear()
    routes = load_item_routes(p)
    assert "customType" in routes
    assert routes["customType"].source_type == "custom"
    load_item_routes.cache_clear()


@pytest.mark.unit
def test_missing_routes_key_raises(tmp_path):
    """load_item_routes() raises ValueError when YAML has no 'routes' key."""
    p = _write_routing_yaml(tmp_path, "other_key:\n  foo: bar\n")
    from scripts.core.item_router import load_item_routes
    load_item_routes.cache_clear()
    with pytest.raises(ValueError, match="routes"):
        load_item_routes(p)
    load_item_routes.cache_clear()
