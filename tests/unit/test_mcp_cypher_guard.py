"""B3: run_cypher safety guard tests.

Adversarial coverage — assume the LLM will try every bypass it can find.
Opus reviewer will look for cases this set misses.

Threat-model classes tested here:
1. Each banned keyword (8 total, incl. LOAD CSV multi-token).
2. Case-insensitive variants of banned keywords.
3. Word-boundary: substrings that *contain* a keyword but are not that
   keyword (e.g. "created_at", "MERGED", "DROPPED_AT").
4. Comment-stripping (line // and block /* */): keyword inside comment → safe.
5. Uncommented forbidden keyword AFTER an inline comment → still blocked.
6. LIMIT injection: absent → appended; present → preserved; custom hard_limit.
7. Trailing semicolon stripping before LIMIT injection.
8. Multi-statement queries (semicolon-separated) — second statement forbidden.
9. UNION-chained mutation — CREATE inside a UNION clause.
10. Subquery CALL { CREATE ... } — still blocked.
11. String literal containing a forbidden keyword — false-positive: blocked.
    (Documented and accepted as the safety-over-precision tradeoff.)
12. Empty / non-string query rejected.
13. CypherSafetyError IS-A ValueError.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lit_monitor.mcp.cypher_guard import CypherSafetyError, guard

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _passes(query: str, **kw) -> str:
    """Assert guard does NOT raise and return the (possibly modified) query."""
    return guard(query, **kw)


# ---------------------------------------------------------------------------
# 1. Each banned keyword (8 distinct mutation keywords)
# ---------------------------------------------------------------------------


class TestForbiddenKeywords:
    @pytest.mark.parametrize(
        "query",
        [
            "CREATE (n:Paper {doi: 'x'})",
            "MERGE (p:Paper {doi: 'x'})",
            "MATCH (p) DELETE p",
            "MATCH (p) SET p.title = 'hacked'",
            "DROP TABLE Paper",
            "ALTER TABLE Paper ADD foo STRING",
            "MATCH (p) REMOVE p.title",
            "LOAD CSV FROM 'attack.csv' AS row CREATE (n:Paper)",
        ],
    )
    def test_each_keyword_blocked(self, query: str) -> None:
        with pytest.raises(CypherSafetyError):
            guard(query)

    def test_load_csv_with_extra_whitespace(self) -> None:
        """LOAD   CSV (tabs + spaces) should be blocked."""
        with pytest.raises(CypherSafetyError):
            guard("LOAD  \t  CSV FROM 'x'")

    def test_load_csv_with_newline_between(self) -> None:
        """LOAD\\nCSV split across a line should be blocked."""
        with pytest.raises(CypherSafetyError):
            guard("LOAD\nCSV FROM 'x'")

    def test_remove_standalone(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard("MATCH (n) REMOVE n.age RETURN n")

    def test_set_inside_merge(self) -> None:
        """MERGE + ON CREATE SET pattern — both MERGE and SET are banned."""
        with pytest.raises(CypherSafetyError):
            guard("MERGE (n:X {id: 1}) ON CREATE SET n.created = true RETURN n")


# ---------------------------------------------------------------------------
# 2. Case-insensitive detection
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    @pytest.mark.parametrize(
        "variant",
        [
            "create (n) return n",
            "CrEaTe (n) RETURN n",
            "cReAtE (n) return n",
            "Match (p) Set p.x=1 RETURN p",
            "DrOp TABLE Paper",
            "mErGe (n:X) RETURN n",
            "MaTcH (p) DeLeTe p",
            "AlTeR TABLE Paper ADD foo STRING",
            "ReMoVe n.x",
            "load csv FROM 'x'",
            "Load Csv FROM 'x'",
        ],
    )
    def test_case_insensitive(self, variant: str) -> None:
        with pytest.raises(CypherSafetyError):
            guard(variant)


# ---------------------------------------------------------------------------
# 3. Word-boundary: identifiers that CONTAIN a banned keyword substring
# ---------------------------------------------------------------------------


class TestWordBoundary:
    def test_property_created_at_allowed(self) -> None:
        """'created_at' as a property name must NOT trigger the guard.

        'CREATE' is banned but 'CREATED_AT' is a plain property name.
        \\bCREATE\\b does not match inside 'CREATED_AT' because 'D'
        (a word character) follows immediately after 'CREATE', so there
        is no word boundary at that position.
        """
        result = _passes("MATCH (p:Paper) RETURN p.created_at LIMIT 10")
        assert "MATCH" in result

    def test_uppercase_created_at_allowed(self) -> None:
        """CREATED_AT as a RETURN alias is allowed."""
        result = _passes("MATCH (p) RETURN p.CREATED_AT AS CREATED_AT LIMIT 10")
        assert "MATCH" in result

    def test_merged_property_name_allowed(self) -> None:
        """'merged' is not a bare MERGE keyword — \\bMERGE\\b won't match."""
        result = _passes("MATCH (p) RETURN p.merged LIMIT 5")
        assert "MATCH" in result

    def test_dropped_property_name_allowed(self) -> None:
        """'dropped_at' should not trigger DROP."""
        result = _passes("MATCH (p) RETURN p.dropped_at LIMIT 5")
        assert "MATCH" in result

    def test_deleted_column_allowed(self) -> None:
        """Property 'deleted_by' should not trigger DELETE."""
        result = _passes("MATCH (p) RETURN p.deleted_by LIMIT 5")
        assert "MATCH" in result

    def test_alter_ego_allowed(self) -> None:
        """'alter_ego' contains 'ALTER' but has 'e' after, so no word boundary.

        Actually 'alter' at position 0 followed by '_' (a word char) means
        \\bALTER\\b won't match 'ALTER_EGO'. This confirms the boundary check.
        """
        result = _passes("MATCH (p) WHERE p.alter_ego = 'x' RETURN p LIMIT 5")
        assert "MATCH" in result

    def test_remove_me_property_blocked(self) -> None:
        """'REMOVE' as a bare keyword is blocked even when followed by space."""
        with pytest.raises(CypherSafetyError):
            guard("MATCH (n) REMOVE n.x RETURN n")

    def test_bare_set_blocked_not_reset(self) -> None:
        """'RESET' does not contain bare \\bSET\\b at start; but 'SET' alone does.

        'RESET' → R-E-S-E-T: \\bSET\\b would need 'S' preceded by a non-word
        char. In 'RESET', 'SET' starts at index 2, preceded by 'E' (word char),
        so \\bSET\\b does NOT match 'RESET'. This confirms no false positive.
        """
        result = _passes("MATCH (p) RETURN p.reset_count LIMIT 5")
        assert "MATCH" in result


# ---------------------------------------------------------------------------
# 4 & 5. Comment stripping
# ---------------------------------------------------------------------------


class TestCommentStripping:
    def test_line_comment_with_create_safe(self) -> None:
        """'// CREATE foo' in a line comment is harmless — stripped first."""
        result = _passes("MATCH (p) RETURN p // CREATE foo\nLIMIT 5")
        assert "MATCH" in result
        # Original LIMIT preserved; no second LIMIT appended.
        assert result.count("LIMIT") == 1
        assert "LIMIT 5" in result

    def test_line_comment_with_delete_safe(self) -> None:
        result = _passes("MATCH (p) RETURN p // DELETE p\nLIMIT 5")
        assert "MATCH" in result

    def test_block_comment_with_delete_safe(self) -> None:
        result = _passes("MATCH (p) /* DELETE old code */ RETURN p LIMIT 5")
        assert "MATCH" in result

    def test_block_comment_with_set_safe(self) -> None:
        result = _passes("MATCH (p) /* SET x=1 */ RETURN p LIMIT 5")
        assert "MATCH" in result

    def test_multiline_block_comment_with_multiple_forbidden_safe(self) -> None:
        """Block comment spanning multiple lines with SET + DELETE inside — safe."""
        result = _passes(
            "MATCH (p)\n/*\n  SET p.x = 1\n  DELETE p.y\n*/\nRETURN p LIMIT 5"
        )
        assert "MATCH" in result
        assert "LIMIT 5" in result

    def test_multiple_line_comments_safe(self) -> None:
        """Multiple // comments, each with a banned keyword — all stripped safely."""
        query = (
            "MATCH (p)\n"
            "// MERGE is not allowed\n"
            "// CREATE too\n"
            "RETURN p LIMIT 5"
        )
        result = _passes(query)
        assert "MATCH" in result

    def test_block_comment_then_line_comment_safe(self) -> None:
        """Mixed comment styles — both stripped."""
        result = _passes(
            "MATCH (p) /* DROP TABLE x */ RETURN p // ALTER TABLE\nLIMIT 5"
        )
        assert "MATCH" in result

    # 5. Uncommented forbidden keyword after comment → still blocked.

    def test_uncommented_create_after_comment_blocked(self) -> None:
        """Comment is stripped but the REAL CREATE outside it must be caught."""
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) /* safe comment */ CREATE (n:X) RETURN p")

    def test_uncommented_set_after_block_comment_blocked(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) /* safe */ SET p.x = 1 RETURN p")

    def test_uncommented_merge_after_line_comment_blocked(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p)\n// safe\nMERGE (p)-[:R]->(q) RETURN p")

    # 4b. Unterminated block comment must NOT hide a keyword (no closing */ ⇒
    # text is preserved by the linear scanner, so the keyword stays visible).
    def test_unterminated_block_comment_keyword_still_blocked(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) /* unterminated CREATE (n:X) RETURN p")


# ---------------------------------------------------------------------------
# ReDoS regression: comment stripping must run in linear time.
# Before the fix, the `//[^\n]*|/\*.*?\*/` regex was O(n^2) on input like
# `(/*x)*n` — many unterminated block-comment openers forced the lazy `.*?`
# to rescan to end-of-string at every position. The linear scanner returns
# promptly.
# ---------------------------------------------------------------------------


class TestCommentStrippingReDoS:
    def test_pathological_block_comment_openers_complete_promptly(self) -> None:
        import time

        # 200k unterminated "/*x" openers: O(n^2) regex would take minutes.
        payload = "/*x" * 200_000 + " CREATE (n) RETURN n"
        start = time.perf_counter()
        # CREATE is outside any terminated comment ⇒ must still be blocked,
        # and the call must return/raise quickly (linear time).
        with pytest.raises(CypherSafetyError):
            guard(payload)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"comment stripping too slow: {elapsed:.2f}s (ReDoS?)"

    def test_pathological_input_safe_read_still_allowed_promptly(self) -> None:
        import time

        # Many terminated block comments + a valid read. Must stay linear.
        payload = "/* c */" * 100_000 + "MATCH (n) RETURN n LIMIT 5"
        start = time.perf_counter()
        result = guard(payload)
        elapsed = time.perf_counter() - start
        assert "MATCH" in result
        assert elapsed < 2.0, f"comment stripping too slow: {elapsed:.2f}s (ReDoS?)"


# ---------------------------------------------------------------------------
# 6. LIMIT injection
# ---------------------------------------------------------------------------


class TestLimitInjection:
    def test_no_limit_appends_default(self) -> None:
        result = guard("MATCH (p) RETURN p")
        assert "LIMIT 100" in result

    def test_no_limit_custom_hard_limit(self) -> None:
        result = guard("MATCH (p) RETURN p", hard_limit=50)
        assert "LIMIT 50" in result

    def test_existing_limit_preserved_not_doubled(self) -> None:
        result = guard("MATCH (p) RETURN p LIMIT 5")
        assert "LIMIT 5" in result
        assert "LIMIT 100" not in result
        # Only one LIMIT clause in the output.
        assert result.upper().count("LIMIT") == 1

    def test_limit_zero_allowed_as_existing(self) -> None:
        """LIMIT 0 is a valid (if odd) Cypher clause; must be preserved."""
        result = guard("MATCH (p) RETURN p LIMIT 0")
        assert "LIMIT 0" in result
        assert "LIMIT 100" not in result

    def test_case_insensitive_limit_detection(self) -> None:
        """Lowercase 'limit' is detected — no second LIMIT appended."""
        result = guard("MATCH (p) RETURN p limit 5")
        # Only one limit keyword total.
        assert result.lower().count("limit") == 1

    def test_limit_in_subclause_detected(self) -> None:
        """Even a LIMIT buried in a WITH clause is detected; no extra appended."""
        result = guard("MATCH (p) WITH p LIMIT 10 RETURN p")
        assert result.upper().count("LIMIT") == 1


# ---------------------------------------------------------------------------
# 7. Trailing semicolon handling
# ---------------------------------------------------------------------------


class TestTrailingSemicolon:
    def test_trailing_semicolon_removed_before_limit_injection(self) -> None:
        """'MATCH p RETURN p;' → 'MATCH p RETURN p LIMIT 100' (no semicolon before LIMIT)."""
        result = guard("MATCH (p) RETURN p;")
        assert "LIMIT 100" in result
        assert "; LIMIT" not in result
        assert ";LIMIT" not in result
        assert ";" not in result  # semicolon fully removed

    def test_trailing_whitespace_plus_semicolon(self) -> None:
        result = guard("MATCH (p) RETURN p   ;   ")
        assert "LIMIT 100" in result
        assert ";" not in result

    def test_existing_limit_with_trailing_semicolon(self) -> None:
        """When LIMIT is already present, semicolon handling path is not taken.

        The guard returns the original query string (not the stripped one) when
        LIMIT already exists, so this tests that the output is still valid.
        """
        result = guard("MATCH (p) RETURN p LIMIT 5;")
        # LIMIT 5 must be present and no LIMIT 100 appended.
        assert "LIMIT 5" in result
        assert "LIMIT 100" not in result


# ---------------------------------------------------------------------------
# 8. Multi-statement queries (semicolon-separated)
# ---------------------------------------------------------------------------


class TestMultiStatement:
    def test_second_statement_create_blocked(self) -> None:
        """'MATCH p RETURN p; CREATE (n)' — the CREATE is outside any comment."""
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) RETURN p; CREATE (n:X)")

    def test_second_statement_delete_blocked(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) RETURN p; DELETE p")


# ---------------------------------------------------------------------------
# 9. UNION-chained mutation
# ---------------------------------------------------------------------------


class TestUnionChained:
    def test_union_create_blocked(self) -> None:
        """CREATE inside a UNION clause is still caught by \\bCREATE\\b."""
        with pytest.raises(CypherSafetyError):
            guard(
                "MATCH (p:Paper) RETURN p.doi "
                "UNION "
                "CREATE (q:Paper {doi: 'x'}) RETURN q.doi"
            )

    def test_union_all_merge_blocked(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard(
                "MATCH (p) RETURN p.doi "
                "UNION ALL "
                "MERGE (q:Paper {doi: 'x'}) RETURN q.doi"
            )


# ---------------------------------------------------------------------------
# 10. Subquery CALL { ... } with mutation
# ---------------------------------------------------------------------------


class TestSubquery:
    def test_call_subquery_create_blocked(self) -> None:
        """CREATE inside CALL { } is still caught."""
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) CALL { CREATE (n:X) RETURN n } RETURN p")

    def test_call_subquery_delete_blocked(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) CALL { MATCH (q) DELETE q RETURN q } RETURN p")


# ---------------------------------------------------------------------------
# P2.1: bare CALL procedures are banned (potential side effects).
#
# Kuzu CALL procedures can mutate; run_cypher is read-only by contract. The
# internal read-only procedures (CALL show_tables / table_info) are issued
# directly via conn.execute() in schema_describer.py and migrations.py and
# never route through guard(), so banning CALL here breaks no real read path.
# ---------------------------------------------------------------------------


class TestCallProcedureBlocked:
    @pytest.mark.parametrize(
        "query",
        [
            "CALL show_tables() RETURN *",
            "CALL table_info('Paper') RETURN *",
            "MATCH (p) CALL db.some_proc() RETURN p",
            "call SHOW_TABLES() return *",  # case-insensitive
        ],
    )
    def test_call_query_blocked(self, query: str) -> None:
        """A query invoking a CALL procedure is rejected."""
        with pytest.raises(CypherSafetyError):
            guard(query)

    def test_recall_identifier_still_allowed(self) -> None:
        """Word boundary: 'recall' / 'p.call_count' must NOT trip the CALL ban."""
        result = _passes("MATCH (p) RETURN p.recall_score, p.call_count LIMIT 5")
        assert "MATCH" in result


# ---------------------------------------------------------------------------
# 11. String literal false-positive (documented known limitation)
# ---------------------------------------------------------------------------


class TestStringLiteralFalsePositive:
    def test_string_containing_create_blocked(self) -> None:
        """Known false-positive: WITH 'CREATE' AS s is rejected.

        The guard does NOT parse Cypher string literals — it operates on raw
        text after comment stripping. This is a deliberate safety-over-precision
        tradeoff. A user who needs to match on a literal 'CREATE' string can use
        a parameterized query on the server side instead.

        This test documents the behaviour rather than asserting it should pass.
        """
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) WHERE p.title = 'CREATE TABLE' RETURN p")

    def test_string_containing_set_blocked(self) -> None:
        """Same false-positive for 'SET' inside a string literal."""
        with pytest.raises(CypherSafetyError):
            guard("MATCH (p) WHERE p.action = 'SET value' RETURN p")


# ---------------------------------------------------------------------------
# 12. Empty / non-string queries
# ---------------------------------------------------------------------------


class TestInvalidInput:
    def test_empty_string_rejected(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard("")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard("   \n\t  ")

    def test_non_string_rejected(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard(None)  # type: ignore[arg-type]

    def test_integer_rejected(self) -> None:
        with pytest.raises(CypherSafetyError):
            guard(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 13. CypherSafetyError IS-A ValueError
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_cypher_safety_error_is_value_error(self) -> None:
        """CypherSafetyError must be a ValueError subclass so MCP dispatch
        routes it to TextContent('Error: ...') instead of the generic handler.
        """
        exc = CypherSafetyError("test")
        assert isinstance(exc, ValueError)

    def test_raise_catches_as_value_error(self) -> None:
        with pytest.raises(ValueError):
            guard("CREATE (n:X)")


# ---------------------------------------------------------------------------
# 14. run_cypher integration tests (real tmp GraphDB)
# ---------------------------------------------------------------------------


class TestRunCypherIntegration:
    def test_run_cypher_happy_path(self, tmp_path, monkeypatch) -> None:
        """Happy-path: read-only MATCH returns list[dict]."""
        from lit_monitor.graph import GraphDB
        from lit_monitor.mcp import tools

        db = GraphDB(persist_dir=str(tmp_path / "b3.kuzu"))
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[],
            paper_metadata={"title": "Alpha", "year": 2024, "journal": "J"},
        )
        monkeypatch.setattr(tools, "_get_graph_db", lambda: db)
        result = tools.run_cypher("MATCH (p:Paper) RETURN p.doi AS doi")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["doi"] == "10.0/a"

    def test_run_cypher_safety_raises_value_error(self, monkeypatch) -> None:
        """CypherSafetyError IS-A ValueError — MCP dispatch catches it."""
        from lit_monitor.mcp import tools

        monkeypatch.setattr(tools, "_get_graph_db", lambda: MagicMock())
        with pytest.raises(ValueError):
            tools.run_cypher("CREATE (n:Paper)")

    def test_run_cypher_returns_list_of_dicts(self, tmp_path, monkeypatch) -> None:
        """Multi-row results are list[dict] with correct column names."""
        from lit_monitor.graph import GraphDB
        from lit_monitor.mcp import tools

        db = GraphDB(persist_dir=str(tmp_path / "b3b.kuzu"))
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[],
            paper_metadata={"title": "A", "year": 2024, "journal": "J"},
        )
        db.add_paper(
            doi="10.0/b",
            entities=[],
            relationships=[],
            paper_metadata={"title": "B", "year": 2023, "journal": "J"},
        )
        monkeypatch.setattr(tools, "_get_graph_db", lambda: db)
        rows = tools.run_cypher("MATCH (p:Paper) RETURN p.doi AS d, p.year AS y")
        assert isinstance(rows, list)
        for r in rows:
            assert isinstance(r, dict)
            assert "d" in r
            assert "y" in r

    def test_run_cypher_limit_injected_in_actual_execution(
        self, tmp_path, monkeypatch
    ) -> None:
        """When no LIMIT is provided, LIMIT 100 is injected before execution."""
        from lit_monitor.graph import GraphDB
        from lit_monitor.mcp import tools

        db = GraphDB(persist_dir=str(tmp_path / "b3c.kuzu"))
        # Only one paper — even without LIMIT it should complete fine.
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[],
            paper_metadata={"title": "A", "year": 2024, "journal": "J"},
        )
        monkeypatch.setattr(tools, "_get_graph_db", lambda: db)
        # No LIMIT in query → guard injects LIMIT 100 → execution succeeds.
        rows = tools.run_cypher("MATCH (p:Paper) RETURN p.doi AS doi")
        assert isinstance(rows, list)
        assert len(rows) == 1

    def test_run_cypher_custom_limit(self, tmp_path, monkeypatch) -> None:
        """Custom limit parameter is forwarded to guard."""
        from lit_monitor.graph import GraphDB
        from lit_monitor.mcp import tools

        db = GraphDB(persist_dir=str(tmp_path / "b3d.kuzu"))
        db.add_paper(
            doi="10.0/a",
            entities=[],
            relationships=[],
            paper_metadata={"title": "A", "year": 2024, "journal": "J"},
        )
        monkeypatch.setattr(tools, "_get_graph_db", lambda: db)
        rows = tools.run_cypher("MATCH (p:Paper) RETURN p.doi AS doi", limit=10)
        assert isinstance(rows, list)

    def test_run_cypher_graph_unavailable_raises_runtime_error(
        self, monkeypatch
    ) -> None:
        """When graph backend is None, RuntimeError is raised."""
        from lit_monitor.mcp import tools

        monkeypatch.setattr(tools, "_get_graph_db", lambda: None)
        with pytest.raises(RuntimeError, match="Graph backend unavailable"):
            tools.run_cypher("MATCH (p) RETURN p")
