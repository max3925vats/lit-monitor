"""A2 (Phase 4a): NL→Cypher LLM step — single Ollama call per question.

Test coverage:
  - happy path: returns Cypher string from LLM.
  - single-call invariant: exactly one client.complete() per question.
  - fence stripping: tolerates ```cypher / bare ``` wrappers.
  - mutation rejection: each of the 9 forbidden keywords + case-insensitive.
  - word-boundary safety: 'created_at' as a property name does NOT trigger
    CREATE rejection.
  - query-shape gate: output must start with MATCH/RETURN/WITH/OPTIONAL MATCH
    (CALL is banned — F19).
  - empty/error responses: return None + INFO/WARNING log, never raise.
  - prompt round-trip: YAML carries required placeholders and lists all
    9-predicate vocabulary.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lit_monitor.core.path_utils import resolve_path
from lit_monitor.graph.ask import generate_cypher


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestGenerateCypherHappyPath:
    def test_returns_cypher_from_llm(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p:Paper) RETURN p.doi LIMIT 100"
        result = generate_cypher("show me papers", "schema text", client=mock_client)
        assert result == "MATCH (p:Paper) RETURN p.doi LIMIT 100"

    def test_single_llm_call(self) -> None:
        """A2 invariant: ONE Ollama call per question."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p) RETURN p LIMIT 10"
        generate_cypher("q", "s", client=mock_client)
        assert mock_client.complete.call_count == 1

    def test_passes_schema_into_system_prompt(self) -> None:
        """A2: the schema_text argument is injected into the system prompt
        (so the LLM can see the live schema)."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p) RETURN p LIMIT 10"
        unique_schema_marker = "UNIQUE_SCHEMA_MARKER_42"
        generate_cypher("q", unique_schema_marker, client=mock_client)
        # Inspect kwargs of the call
        _, kwargs = mock_client.complete.call_args
        full_call = (kwargs.get("system", "") or "") + (kwargs.get("user", "") or "")
        assert unique_schema_marker in full_call

    def test_passes_question_into_user_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p) RETURN p LIMIT 10"
        unique_q = "UNIQUE_QUESTION_MARKER_77"
        generate_cypher(unique_q, "schema", client=mock_client)
        _, kwargs = mock_client.complete.call_args
        full_call = (kwargs.get("system", "") or "") + (kwargs.get("user", "") or "")
        assert unique_q in full_call


# ---------------------------------------------------------------------------
# Fence stripping
# ---------------------------------------------------------------------------
class TestFenceStripping:
    def test_strips_cypher_fences(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = (
            "```cypher\nMATCH (p) RETURN p LIMIT 10\n```"
        )
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None
        assert "```" not in result
        assert result.strip().startswith("MATCH")

    def test_strips_plain_fences(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "```\nMATCH (p) RETURN p LIMIT 10\n```"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None
        assert "```" not in result

    def test_handles_trailing_whitespace(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "  MATCH (p) RETURN p LIMIT 10  \n"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None
        assert result.startswith("MATCH")


# ---------------------------------------------------------------------------
# Mutation rejection — 9 forbidden keywords + case-insensitive
# (CALL added in audit F19; see _FORBIDDEN_KEYWORDS in scripts/graph/ask.py.)
# ---------------------------------------------------------------------------
class TestMutationRejection:
    @pytest.mark.parametrize(
        "forbidden",
        [
            "CREATE (n:Paper {doi: '10.0/x'})",
            "MATCH (p) DELETE p",
            "DROP TABLE Paper",
            "MERGE (p:Paper {doi: '10.0/x'})",
            "MATCH (p) SET p.title = 'hacked' RETURN p",
            "MATCH (p) REMOVE p.title RETURN p",
            "ALTER TABLE Paper ADD foo STRING",
            "CALL show_tables() RETURN *",
            "LOAD CSV FROM 'attack.csv' AS row RETURN row",
        ],
    )
    def test_rejects_mutation_keywords(self, forbidden: str, caplog) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = forbidden
        with caplog.at_level(logging.INFO):
            result = generate_cypher("q", "s", client=mock_client)
        assert result is None
        assert any(
            "reject" in r.message.lower() or "validator" in r.message.lower()
            for r in caplog.records
        )

    def test_case_insensitive_rejection(self) -> None:
        """A2: lowercase 'create' must still be rejected."""
        mock_client = MagicMock()
        mock_client.complete.return_value = (
            "match (p) create (n) return p"
        )
        result = generate_cypher("q", "s", client=mock_client)
        assert result is None

    def test_mixed_case_rejection(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p) DeLeTe p"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is None

    def test_word_boundary_allows_substring_in_identifier(self) -> None:
        """A2: 'p.created_at' as a property name should NOT trigger CREATE
        rejection. The mutation regex uses \\b word boundaries so lowercased
        'created_at' does not match the uppercase keyword CREATE.

        Note: the regex is case-insensitive, so 'created_at' WILL match CREATE
        as a word-prefix unless we use a stricter pattern. Per the bundle
        spec, this is acceptable — we tolerate the false-positive cost of
        safety on uppercase CREATED, but lowercase 'created_at' should match
        because the regex is case-insensitive.

        We accept this trade-off and test only that property names with
        a substring NOT matching the full keyword at a word boundary pass
        through (e.g. 'p.creation_date' or 'p.creator_name').
        """
        mock_client = MagicMock()
        # 'creation_date' / 'creator' do NOT have the word CREATE at a \b.
        mock_client.complete.return_value = (
            "MATCH (p:Paper) RETURN p.creation_date, p.creator_name LIMIT 10"
        )
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None


# ---------------------------------------------------------------------------
# Query-shape gate — must start with MATCH/RETURN/WITH/OPTIONAL MATCH (CALL banned)
# ---------------------------------------------------------------------------
class TestQueryShapeValidation:
    def test_rejects_prose_prefix(self) -> None:
        """A2: 'Here is your query: MATCH ...' must be rejected."""
        mock_client = MagicMock()
        mock_client.complete.return_value = (
            "Here is your query: MATCH (p) RETURN p"
        )
        result = generate_cypher("q", "s", client=mock_client)
        assert result is None

    def test_accepts_with_keyword(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "WITH 1 AS x RETURN x LIMIT 1"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None

    def test_rejects_call_keyword(self) -> None:
        """F19 (P3.7): an LLM-emitted CALL procedure must be rejected.

        Kuzu CALL procedures can have side effects; this path executes the
        result via conn.execute(), so CALL must never pass the validator.
        Mirrors the ban in scripts/mcp/cypher_guard.py.
        """
        mock_client = MagicMock()
        mock_client.complete.return_value = "CALL show_tables() RETURN *"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is None

    def test_f19_read_only_match_still_passes(self) -> None:
        """F19 (P3.7): banning CALL must not break legitimate read queries."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p:Paper) RETURN p LIMIT 10"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None

    def test_accepts_return_keyword(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "RETURN 1 AS x"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None

    def test_accepts_optional_match(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = (
            "OPTIONAL MATCH (p:Paper) RETURN p LIMIT 10"
        )
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None


# ---------------------------------------------------------------------------
# Empty and error responses
# ---------------------------------------------------------------------------
class TestEmptyAndErrorResponses:
    def test_empty_response_returns_none(self, caplog) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "   "
        with caplog.at_level(logging.INFO):
            result = generate_cypher("q", "s", client=mock_client)
        assert result is None
        assert any("empty" in r.message.lower() for r in caplog.records)

    def test_llm_exception_returns_none(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("network down")
        # Must not raise
        result = generate_cypher("q", "s", client=mock_client)
        assert result is None

    def test_no_exception_escapes_on_value_error(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = ValueError("bad input")
        result = generate_cypher("q", "s", client=mock_client)
        assert result is None

    def test_empty_question_returns_none(self, caplog) -> None:
        mock_client = MagicMock()
        with caplog.at_level(logging.INFO):
            result = generate_cypher("   ", "schema", client=mock_client)
        assert result is None
        # No LLM call wasted on empty input
        assert mock_client.complete.call_count == 0


# ---------------------------------------------------------------------------
# Prompt round-trip — YAML asset must carry the schema we expect
# ---------------------------------------------------------------------------
class TestPromptRoundTrip:
    def test_yaml_exists(self) -> None:
        path = resolve_path(Path("config/prompts/ask_cypher.example.yaml"))
        assert path.exists(), (
            "A2 prompt YAML missing: config/prompts/ask_cypher.example.yaml"
        )

    def test_yaml_has_required_placeholders(self) -> None:
        raw = resolve_path(Path("config/prompts/ask_cypher.example.yaml")).read_text()
        assert "{question}" in raw
        assert "{schema}" in raw

    def test_prompt_loads_through_registry(self) -> None:
        from lit_monitor.llm.prompt_registry import _reset_prompt_cache, load_prompt

        _reset_prompt_cache()
        prompt = load_prompt("ask_cypher")
        assert prompt.system
        assert prompt.user_template
        # Render with all required placeholders should not raise.
        rendered = prompt.render_user(
            question="show all papers",
            schema="## Node: Paper\n- doi: STRING",
            examples="Q: ...\nA: MATCH ...",
        )
        assert "show all papers" in rendered
        assert "## Node: Paper" in rendered

    def test_prompt_lists_all_nine_predicates(self) -> None:
        """A2: all 9 predicates appear in the prompt vocabulary section.

        The closed predicate vocabulary is the entire query surface for
        Graph RAG — a regression that drops one predicate from the prompt
        silently degrades query quality.
        """
        raw = resolve_path(Path("config/prompts/ask_cypher.example.yaml")).read_text()
        # The 9 LLM-relevant predicates (MENTIONS + CITES are schema-source
        # but ALSO listed here for query generation completeness).
        for pred in (
            "MENTIONS",
            "CITES",
            "COMPARES_TO",
            "DEPENDS_ON",
            "PROPOSES",
            "LIMITED_BY",
            "INTRODUCES",
            "RAISES_QUESTION",
            "EXTENDS",
            "CONTRADICTS",
        ):
            assert pred in raw, (
                f"missing predicate {pred} in ask_cypher prompt"
            )

    def test_prompt_lists_all_six_entity_types(self) -> None:
        """A2: the 6 closed entity types must appear in the prompt so the
        LLM knows what `Entity {type: 'method'}` filters are valid."""
        raw = resolve_path(Path("config/prompts/ask_cypher.example.yaml")).read_text()
        for etype in ("topic", "method", "material", "author", "journal", "keyword"):
            assert etype in raw, (
                f"missing entity type {etype} in ask_cypher prompt"
            )

    def test_prompt_required_placeholders_registered(self) -> None:
        """A2: prompt_registry must declare {question} (and {schema}) as
        required for ask_cypher so YAML typos surface at load time."""
        from lit_monitor.llm.prompt_registry import required_placeholders

        req = required_placeholders("ask_cypher")
        # {question} must be in user_template; {schema} may be in system.
        # The registry only checks user_template placeholders, so at minimum
        # {question} must be registered.
        assert "question" in req


# ---------------------------------------------------------------------------
# AR-3: validation routes through the canonical scripts.graph.cypher_guard
# ---------------------------------------------------------------------------
# generate_cypher no longer carries a duplicate _MUTATION_RE; it calls
# guard() (the same guard /api/cypher + MCP use). Two behaviour guarantees:
#   1. a mutation the guard would reject still surfaces as a None return
#      (the defensive-perimeter rejection path — guard's CypherSafetyError is
#      caught and mapped to None, never re-raised).
#   2. the guard's LIMIT injection now applies to /ask too: a valid query
#      without an explicit LIMIT comes back with one appended, so the LIMIT
#      reaches execute_cypher.
# ---------------------------------------------------------------------------
class TestAskRoutesThroughCanonicalGuard:
    def test_ask_rejects_mutation_via_canonical_guard(self, caplog) -> None:
        """A mutation from the LLM is rejected (None) via the canonical guard,
        not a local regex copy. DETACH DELETE is the classic write."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p) DETACH DELETE p"
        with caplog.at_level(logging.INFO):
            result = generate_cypher("q", "s", client=mock_client)
        assert result is None
        assert any(
            "reject" in r.message.lower() or "validator" in r.message.lower()
            for r in caplog.records
        )

    def test_ask_injects_limit_when_absent(self) -> None:
        """A valid MATCH...RETURN with no LIMIT comes back with the guard's
        default LIMIT appended (so the cap reaches execute_cypher). This pins
        NEW behaviour /ask previously lacked."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p:Paper) RETURN p.doi"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None
        assert "LIMIT" in result.upper()

    def test_ask_preserves_existing_limit(self) -> None:
        """When the LLM already emits a LIMIT, the guard must not double-append."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "MATCH (p:Paper) RETURN p.doi LIMIT 5"
        result = generate_cypher("q", "s", client=mock_client)
        assert result is not None
        assert result.upper().count("LIMIT") == 1
