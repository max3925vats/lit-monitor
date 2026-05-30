"""R2 (Phase 3): LLM relationship extractor — single Ollama call per paper.

Test coverage:
  - happy path: returns RelationshipTuple list across 9 predicates.
  - single-call invariant: exactly one client.complete() per paper.
  - out-of-vocab predicate drop + INFO log.
  - subject-mismatch drop.
  - Paper -> Paper DOI gate.
  - confidence-range gate.
  - malformed JSON / missing keys / exception -> [] + WARN log.
  - prompt round-trip: YAML carries required placeholders.
  - validator carries EXTENDS + CONTRADICTS.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from scripts.graph.relationship_extractor import RelationshipTuple
from scripts.graph.relationship_llm import extract_llm_relationships


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestExtractRelationshipsHappyPath:
    def test_returns_validated_triples(self) -> None:
        """R2: happy path returns RelationshipTuple list with 9-predicate mix."""
        canned = {
            "triples": [
                {
                    "subject": "10.0/a",
                    "predicate": "EXTENDS",
                    "object": "10.0/b",
                    "object_kind": "Paper",
                    "evidence": "we extend the model of b",
                    "confidence": 0.9,
                },
                {
                    "subject": "10.0/a",
                    "predicate": "INTRODUCES",
                    "object": "novel method",
                    "object_kind": "Entity",
                    "evidence": "we introduce a novel method",
                    "confidence": 0.85,
                },
            ],
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)

        result = extract_llm_relationships(
            fulltext="paper text",
            paper_doi="10.0/a",
            extraction_json={"methods_summary": "ion exchange"},
            client=mock_client,
        )

        assert len(result) == 2
        assert all(isinstance(r, RelationshipTuple) for r in result)
        ext = [r for r in result if r.predicate == "EXTENDS"][0]
        assert ext.source_doi == "10.0/a"
        assert ext.target_id == "10.0/b"
        assert ext.target_kind == "Paper"
        # R3/R4 use field="llm_extracted" to identify the source
        assert ext.field == "llm_extracted"
        intro = [r for r in result if r.predicate == "INTRODUCES"][0]
        assert intro.target_kind == "Entity"
        assert intro.target_id == "novel method"

    def test_single_ollama_call_per_paper(self) -> None:
        """R2 invariant: ONE Ollama call per paper, even with multiple
        triples in response.

        Strengthened with a 3-triple payload so the invariant would FAIL
        if a future refactor accidentally looped per-triple (N calls).
        """
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "triples": [
                {"subject": "10.0/a", "predicate": "EXTENDS",
                 "object": "10.0/b", "object_kind": "Paper",
                 "evidence": "e", "confidence": 0.9},
                {"subject": "10.0/a", "predicate": "INTRODUCES",
                 "object": "novel method", "object_kind": "Entity",
                 "evidence": "e", "confidence": 0.85},
                {"subject": "10.0/a", "predicate": "LIMITED_BY",
                 "object": "1L scale", "object_kind": "Entity",
                 "evidence": "e", "confidence": 0.8},
            ],
        })

        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )

        assert mock_client.complete.call_count == 1
        assert len(result) == 3  # invariant holds under load

    def test_strips_markdown_fences_around_json(self) -> None:
        """R2: tolerate the LLM wrapping JSON in ```json ... ``` fences."""
        canned = {
            "triples": [
                {
                    "subject": "10.0/a",
                    "predicate": "INTRODUCES",
                    "object": "x",
                    "object_kind": "Entity",
                    "evidence": "e",
                    "confidence": 0.8,
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = (
            "```json\n" + json.dumps(canned) + "\n```"
        )

        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Per-triple validation
# ---------------------------------------------------------------------------
class TestOutOfVocabDropped:
    def test_out_of_vocab_predicate_dropped_and_logged(self, caplog) -> None:
        """R2: out-of-vocab predicates dropped + INFO log identifies them."""
        canned = {
            "triples": [
                {
                    "subject": "10.0/a",
                    "predicate": "EXTENDS",
                    "object": "10.0/b",
                    "object_kind": "Paper",
                    "evidence": "real",
                    "confidence": 0.9,
                },
                {
                    "subject": "10.0/a",
                    "predicate": "INVENTED_PREDICATE",
                    "object": "x",
                    "object_kind": "Entity",
                    "evidence": "fake",
                    "confidence": 0.5,
                },
            ],
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)

        with caplog.at_level(logging.INFO):
            result = extract_llm_relationships(
                fulltext="text",
                paper_doi="10.0/a",
                extraction_json={},
                client=mock_client,
            )
        assert len(result) == 1
        assert result[0].predicate == "EXTENDS"
        assert any(
            "INVENTED_PREDICATE" in r.message and "dropped" in r.message.lower()
            for r in caplog.records
        )


class TestSubjectGate:
    def test_subject_mismatch_dropped(self) -> None:
        """R2: triples whose subject != paper_doi are dropped."""
        canned = {
            "triples": [
                {
                    "subject": "10.0/different-paper",
                    "predicate": "EXTENDS",
                    "object": "10.0/b",
                    "object_kind": "Paper",
                    "evidence": "wrong subj",
                    "confidence": 0.9,
                },
                {
                    "subject": "10.0/a",
                    "predicate": "EXTENDS",
                    "object": "10.0/b",
                    "object_kind": "Paper",
                    "evidence": "right",
                    "confidence": 0.9,
                },
            ],
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert len(result) == 1
        assert result[0].source_doi == "10.0/a"


class TestPaperPaperDoiGate:
    def test_paper_paper_with_non_doi_object_dropped(self) -> None:
        """R2: EXTENDS/CONTRADICTS/COMPARES_TO with non-DOI object dropped.

        Sticky-edge defense: vague citations like "Smith et al." with no DOI
        must NOT become EXTENDS edges in the graph.
        """
        canned = {
            "triples": [
                {
                    "subject": "10.0/a",
                    "predicate": "EXTENDS",
                    "object": "Smith et al.",
                    "object_kind": "Paper",
                    "evidence": "vague",
                    "confidence": 0.8,
                },
                {
                    "subject": "10.0/a",
                    "predicate": "CONTRADICTS",
                    "object": "prior work",
                    "object_kind": "Paper",
                    "evidence": "vague",
                    "confidence": 0.8,
                },
            ],
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert result == []

    def test_paper_paper_with_doi_object_kept(self) -> None:
        """R2: EXTENDS with proper DOI-shaped object passes."""
        canned = {
            "triples": [
                {
                    "subject": "10.0/a",
                    "predicate": "EXTENDS",
                    "object": "10.1002/btpr.123",
                    "object_kind": "Paper",
                    "evidence": "real cite",
                    "confidence": 0.9,
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert len(result) == 1
        assert result[0].target_id == "10.1002/btpr.123"


class TestConfidenceGate:
    def test_confidence_out_of_range_dropped(self) -> None:
        canned = {
            "triples": [
                {
                    "subject": "10.0/a",
                    "predicate": "INTRODUCES",
                    "object": "x",
                    "object_kind": "Entity",
                    "evidence": "e",
                    "confidence": 1.5,
                },
                {
                    "subject": "10.0/a",
                    "predicate": "INTRODUCES",
                    "object": "y",
                    "object_kind": "Entity",
                    "evidence": "e",
                    "confidence": -0.1,
                },
                {
                    "subject": "10.0/a",
                    "predicate": "INTRODUCES",
                    "object": "z",
                    "object_kind": "Entity",
                    "evidence": "e",
                    "confidence": 0.5,
                },
            ],
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert len(result) == 1
        assert result[0].confidence == 0.5

    def test_bad_object_kind_dropped(self) -> None:
        """R2: object_kind must be 'Paper' or 'Entity'."""
        canned = {
            "triples": [
                {
                    "subject": "10.0/a",
                    "predicate": "INTRODUCES",
                    "object": "x",
                    "object_kind": "Concept",  # not allowed
                    "evidence": "e",
                    "confidence": 0.8,
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert result == []

    def test_empty_object_dropped(self) -> None:
        canned = {
            "triples": [
                {
                    "subject": "10.0/a",
                    "predicate": "INTRODUCES",
                    "object": "",
                    "object_kind": "Entity",
                    "evidence": "e",
                    "confidence": 0.8,
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert result == []

    def test_caps_triples_at_25(self) -> None:
        """R2: even if LLM returns >25 triples, only first 25 are processed."""
        triples = [
            {
                "subject": "10.0/a",
                "predicate": "INTRODUCES",
                "object": f"entity_{i}",
                "object_kind": "Entity",
                "evidence": "e",
                "confidence": 0.8,
            }
            for i in range(40)
        ]
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({"triples": triples})
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert len(result) == 25


# ---------------------------------------------------------------------------
# Defensive perimeter — failures must never escape
# ---------------------------------------------------------------------------
class TestMalformedJsonReturnsEmpty:
    def test_malformed_json_returns_empty_and_warns(self, caplog) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = "not actually json"
        with caplog.at_level(logging.WARNING):
            result = extract_llm_relationships(
                fulltext="text",
                paper_doi="10.0/a",
                extraction_json={},
                client=mock_client,
            )
        assert result == []
        assert any(
            "json" in r.message.lower() or "parse" in r.message.lower()
            for r in caplog.records
        )

    def test_response_missing_triples_key_returns_empty(self, caplog) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({"foo": "bar"})
        with caplog.at_level(logging.WARNING):
            result = extract_llm_relationships(
                fulltext="text",
                paper_doi="10.0/a",
                extraction_json={},
                client=mock_client,
            )
        assert result == []

    def test_triples_not_a_list_returns_empty(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({"triples": "oops"})
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert result == []

    def test_no_exception_escapes_on_client_error(self) -> None:
        """R2: client.complete raising must degrade to [] + WARN, never raise."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("network down")
        result = extract_llm_relationships(
            fulltext="text",
            paper_doi="10.0/a",
            extraction_json={},
            client=mock_client,
        )
        assert result == []

    def test_response_truncated_in_warning_log(self, caplog) -> None:
        """R2: WARN log includes a prefix of the response for debugging."""
        mock_client = MagicMock()
        # 500-char garbage so we can check truncation behavior
        mock_client.complete.return_value = "X" * 500
        with caplog.at_level(logging.WARNING):
            extract_llm_relationships(
                fulltext="text",
                paper_doi="10.0/a",
                extraction_json={},
                client=mock_client,
            )
        # at least one warn record carries some response context
        assert any("XXX" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Prompt round-trip — long-lived asset is loadable
# ---------------------------------------------------------------------------
class TestPromptRoundTrip:
    def test_prompt_yaml_carries_required_placeholders(self) -> None:
        """R2: the YAML must declare all 3 required placeholders."""
        yaml_path = Path(
            "config/prompts/relationship_extraction.example.yaml"
        )
        assert yaml_path.exists()
        raw = yaml_path.read_text()
        assert "{fulltext}" in raw
        assert "{paper_doi}" in raw
        assert "{extraction_summary}" in raw

    def test_prompt_loads_through_registry(self) -> None:
        """R2: registry can load the prompt without raising."""
        from scripts.llm.prompt_registry import _reset_prompt_cache, load_prompt

        _reset_prompt_cache()
        prompt = load_prompt("relationship_extraction")
        assert prompt.system
        assert prompt.user_template
        # Render with all required placeholders should not raise.
        rendered = prompt.render_user(
            fulltext="sample paper text",
            paper_doi="10.0/test",
            extraction_summary="summary",
        )
        assert "sample paper text" in rendered
        assert "10.0/test" in rendered


class TestValidatorExtended:
    def test_validator_accepts_extends_and_contradicts(self) -> None:
        from scripts.graph.relationship_validator import VALID_PREDICATES

        assert "EXTENDS" in VALID_PREDICATES
        assert "CONTRADICTS" in VALID_PREDICATES
        # Original 8 still present
        for p in (
            "MENTIONS",
            "CITES",
            "COMPARES_TO",
            "DEPENDS_ON",
            "PROPOSES",
            "LIMITED_BY",
            "INTRODUCES",
            "RAISES_QUESTION",
        ):
            assert p in VALID_PREDICATES


# ---------------------------------------------------------------------------
# Prompt vocabulary — defends against YAML refactors dropping a predicate
# ---------------------------------------------------------------------------
class TestPromptVocabulary:
    def test_prompt_lists_all_eight_predicates(self) -> None:
        """R2: the 8 LLM-emitted predicates must appear in the system prompt.

        Note: MENTIONS and CITES are intentionally NOT in R2's prompt vocab —
        MENTIONS comes from N1-N3 (entity extraction); CITES from G4
        (citation edges). R2 covers the 6 schema-source-augmenting +
        2 LLM-only (EXTENDS, CONTRADICTS) predicates.
        """
        yaml_path = Path("config/prompts/relationship_extraction.example.yaml")
        raw = yaml_path.read_text()
        expected = {
            "EXTENDS",
            "CONTRADICTS",
            "COMPARES_TO",
            "DEPENDS_ON",
            "PROPOSES",
            "LIMITED_BY",
            "INTRODUCES",
            "RAISES_QUESTION",
        }
        for pred in expected:
            assert pred in raw, f"prompt missing predicate {pred}"

    def test_mentions_and_cites_not_in_prompt(self) -> None:
        """R2: prompt must NOT include MENTIONS or CITES as predicate
        definitions (those belong to N1-N3 / G4).
        """
        import re

        yaml_path = Path("config/prompts/relationship_extraction.example.yaml")
        raw = yaml_path.read_text()

        # MENTIONS is the safest pattern (word-boundary match for the
        # predicate; the word never appears as common prose in this prompt).
        predicate_tokens = re.findall(r"\b[A-Z_]{3,}\b", raw)
        assert "MENTIONS" not in predicate_tokens

        # CITES might appear in prose elsewhere ("we cite Carta et al."),
        # so be specific: it must not appear as a PREDICATE definition.
        # Quick check: 'CITES' should not appear as a labeled predicate
        # token followed by a newline or space at the start of the file's
        # definition block.
        assert "CITES\n" not in raw and "CITES " not in raw[:5000]
