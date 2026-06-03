"""B11: per-tool MCP inputSchema accuracy + drift-guard tests.

Acceptance criteria:
- Every one of the 12 tools advertises a non-empty inputSchema (parameterless
  tools are allowed to have empty `properties` but the schema is still closed,
  i.e. additionalProperties is False — never the old permissive empty schema).
- For a representative subset, the schema's `properties` and `required` match
  the actual handler signature in scripts/mcp/tools.py.  This guards against
  schema/handler drift: if someone changes a handler param without updating
  the schema (or vice versa), these tests fail.
"""
from __future__ import annotations

import inspect

from scripts.mcp import tools
from scripts.mcp.graph_server import TOOL_NAMES, _build_tool_schemas

# Handlers that genuinely take no parameters — their `properties` are
# legitimately empty, but the schema must still be closed (no extra props).
_PARAMETERLESS = {"get_corpus_stats", "get_schema"}


def _handler(name: str):
    """Return the handler callable for a tool name."""
    return getattr(tools, name)


def _required_params(func) -> set[str]:
    """Params with no default = required."""
    sig = inspect.signature(func)
    return {
        p.name
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def _all_params(func) -> set[str]:
    """All named params (excludes *args/**kwargs)."""
    sig = inspect.signature(func)
    return {
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def test_all_twelve_have_schema() -> None:
    """Every tool name has a schema entry."""
    schemas = _build_tool_schemas()
    assert set(schemas) == set(TOOL_NAMES)
    assert len(schemas) == 12


def test_no_permissive_empty_schema() -> None:
    """No tool keeps the old permissive {properties:{}, additionalProperties:True}.

    Parameterless tools may have empty properties, but must still be closed.
    """
    schemas = _build_tool_schemas()
    for name, spec in schemas.items():
        schema = spec["inputSchema"]
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False, (
            f"{name} must advertise additionalProperties=False"
        )
        if name not in _PARAMETERLESS:
            assert schema["properties"], (
                f"{name} has parameters but advertises empty properties"
            )


def test_schema_matches_handler_signature() -> None:
    """DRIFT GUARD: schema properties/required must match each handler signature.

    Runs over all 12 tools so any param rename/add/remove that isn't mirrored
    in the schema (or vice versa) fails loudly.
    """
    schemas = _build_tool_schemas()
    for name in TOOL_NAMES:
        func = _handler(name)
        schema = schemas[name]["inputSchema"]
        schema_props = set(schema["properties"])
        schema_required = set(schema.get("required", []))

        assert schema_props == _all_params(func), (
            f"{name}: schema properties {sorted(schema_props)} != "
            f"handler params {sorted(_all_params(func))}"
        )
        assert schema_required == _required_params(func), (
            f"{name}: schema required {sorted(schema_required)} != "
            f"handler required {sorted(_required_params(func))}"
        )


def test_representative_required_arg_tool() -> None:
    """get_paper_details: single required `doi`, no optionals."""
    schema = _build_tool_schemas()["get_paper_details"]["inputSchema"]
    assert schema["required"] == ["doi"]
    assert set(schema["properties"]) == {"doi"}
    assert schema["properties"]["doi"]["type"] == "string"


def test_representative_optional_arg_tool() -> None:
    """find_papers_by_query: required `query`, optional `k` with default."""
    schema = _build_tool_schemas()["find_papers_by_query"]["inputSchema"]
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "k"}
    assert schema["properties"]["k"]["type"] == "integer"
    assert schema["properties"]["k"]["default"] == 20


def test_run_cypher_schema() -> None:
    """run_cypher: required `query` string, optional `limit` int — closed set."""
    schema = _build_tool_schemas()["run_cypher"]["inputSchema"]
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "limit"}
    assert schema["additionalProperties"] is False


def test_enum_params_match_vocabularies() -> None:
    """Enum-bearing params advertise the live closed vocabularies."""
    from scripts.graph.relationship_validator import VALID_PREDICATES

    schemas = _build_tool_schemas()
    pred = schemas["find_papers_by_relationship"]["inputSchema"]["properties"][
        "predicate"
    ]
    assert set(pred["enum"]) == set(VALID_PREDICATES)

    etype = schemas["list_entities_by_type"]["inputSchema"]["properties"][
        "entity_type"
    ]
    assert set(etype["enum"]) == set(tools._ENTITY_TYPES)

    gran = schemas["semantic_search"]["inputSchema"]["properties"]["granularity"]
    assert set(gran["enum"]) == {"paper", "chunk"}
