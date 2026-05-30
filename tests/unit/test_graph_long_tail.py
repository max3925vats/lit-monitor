"""N2: cloud-Ollama long-tail NER + low-confidence validation tests.

Coverage:
  - Happy path: shape of the returned dict (new_entities + validations).
  - Single-LLM-call invariant per paper (cost guard).
  - Disabled flag short-circuits before any LLM construction.
  - Missing OLLAMA_API_KEY short-circuits.
  - Malformed JSON → empty dict + WARNING log.
  - Missing required keys → empty dict + WARNING log.
  - Exceptions from the client never escape.
  - Prompt round-trip: registry loads long_tail_ner and the YAML carries the
    required {text} + {low_conf_json} placeholders.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from scripts.graph.long_tail import extract_long_tail_and_validate


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestExtractLongTailHappyPath:
    def test_returns_shape_with_new_entities_and_validations(self) -> None:
        """N2: happy path returns dict with both keys populated."""
        canned = {
            "new_entities": [
                {
                    "surface": "perfusion bioreactor",
                    "type": "method",
                    "span_start": 0,
                    "span_end": 20,
                },
            ],
            "validations": [
                {"surface": "EGFR", "keep": True, "reason": "real gene"},
            ],
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)

        result = extract_long_tail_and_validate(
            text="perfusion bioreactor used for mAb production",
            low_conf_entities=[{"surface": "EGFR", "confidence": 0.55}],
            client=mock_client,
        )

        assert "new_entities" in result
        assert "validations" in result
        assert len(result["new_entities"]) == 1
        assert result["new_entities"][0]["surface"] == "perfusion bioreactor"
        assert len(result["validations"]) == 1
        assert result["validations"][0]["keep"] is True

    def test_single_ollama_call_per_paper(self) -> None:
        """N2 invariant: exactly ONE Ollama round-trip per paper."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(
            {"new_entities": [], "validations": []}
        )

        extract_long_tail_and_validate(
            text="some paper text",
            low_conf_entities=[],
            client=mock_client,
        )

        assert mock_client.complete.call_count == 1

    def test_strips_markdown_fences_around_json(self) -> None:
        """N2: tolerate the LLM wrapping JSON in ```json ... ``` fences."""
        canned = {"new_entities": [], "validations": []}
        mock_client = MagicMock()
        mock_client.complete.return_value = (
            "```json\n" + json.dumps(canned) + "\n```"
        )

        result = extract_long_tail_and_validate(
            text="paper text",
            low_conf_entities=[],
            client=mock_client,
        )

        assert result == canned


# ---------------------------------------------------------------------------
# Disabled paths (no LLM call at all)
# ---------------------------------------------------------------------------
class TestExtractLongTailDisabled:
    def test_disabled_returns_empty_when_flag_false(self, monkeypatch) -> None:
        """N2: graph.ner.cloud_long_tail_enabled=false → empty, no LLM call."""
        from scripts.graph import long_tail as lt

        fake_config = MagicMock()
        fake_config.graph.ner.cloud_long_tail_enabled = False
        monkeypatch.setattr(lt, "_load_runtime_config", lambda: fake_config)
        # Key present — flag is the gate that fails.
        monkeypatch.setenv("OLLAMA_API_KEY", "fake-key")

        result = extract_long_tail_and_validate(
            text="paper text",
            low_conf_entities=[],
        )
        assert result == {"new_entities": [], "validations": []}

    def test_disabled_when_no_api_key(self, monkeypatch) -> None:
        """N2: OLLAMA_API_KEY absent → empty, no LLM call."""
        from scripts.graph import long_tail as lt

        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        fake_config = MagicMock()
        fake_config.graph.ner.cloud_long_tail_enabled = True
        monkeypatch.setattr(lt, "_load_runtime_config", lambda: fake_config)

        result = extract_long_tail_and_validate(
            text="paper text",
            low_conf_entities=[],
        )
        assert result == {"new_entities": [], "validations": []}


# ---------------------------------------------------------------------------
# Malformed output handling
# ---------------------------------------------------------------------------
class TestExtractLongTailMalformedOutput:
    def test_malformed_json_returns_empty_and_warns(self, caplog) -> None:
        """N2: bad JSON from LLM → empty dict + WARNING (no exception)."""
        mock_client = MagicMock()
        mock_client.complete.return_value = "not actually JSON ```"

        with caplog.at_level(logging.WARNING):
            result = extract_long_tail_and_validate(
                text="paper text",
                low_conf_entities=[],
                client=mock_client,
            )

        assert result == {"new_entities": [], "validations": []}
        assert any(
            "long_tail" in r.message.lower() or "json" in r.message.lower()
            for r in caplog.records
        )

    def test_missing_required_keys_returns_empty(self, caplog) -> None:
        """N2: response missing 'new_entities' or 'validations' → empty."""
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({"foo": "bar"})

        with caplog.at_level(logging.WARNING):
            result = extract_long_tail_and_validate(
                text="paper text",
                low_conf_entities=[],
                client=mock_client,
            )
        assert result == {"new_entities": [], "validations": []}

    def test_no_exception_escapes_when_client_raises(self) -> None:
        """N2: client raises → empty, no propagation to caller."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("network exploded")

        # Must not raise.
        result = extract_long_tail_and_validate(
            text="paper text",
            low_conf_entities=[],
            client=mock_client,
        )
        assert result == {"new_entities": [], "validations": []}


# ---------------------------------------------------------------------------
# Prompt round-trip — the long-lived asset is loadable through the registry
# ---------------------------------------------------------------------------
class TestPromptRoundTrip:
    def test_prompt_yaml_carries_required_placeholders(self) -> None:
        """N2: the long_tail_ner YAML ships {text} and {low_conf_json}."""
        yaml_path = Path("config/prompts/long_tail_ner.example.yaml")
        assert yaml_path.exists(), (
            "long_tail_ner.example.yaml is the long-lived prompt asset — "
            "must exist alongside other config/prompts/*.example.yaml files."
        )
        raw = yaml_path.read_text(encoding="utf-8")
        assert "{text}" in raw
        assert "{low_conf_json}" in raw

    def test_prompt_loads_through_registry(self) -> None:
        """N2: load_prompt('long_tail_ner') returns a Prompt instance.

        The registry's Prompt schema requires `system` + `user_template`, so the
        YAML keys MUST use that naming. This test enforces that contract.
        """
        from scripts.llm.prompt_registry import _reset_prompt_cache, load_prompt

        _reset_prompt_cache()
        prompt = load_prompt("long_tail_ner")
        assert prompt is not None
        # Required placeholders survived the load-time render step.
        assert "{text}" in prompt.user_template
        assert "{low_conf_json}" in prompt.user_template
        # System prompt is non-empty (Pydantic guarantees min_length=1 anyway).
        assert prompt.system.strip()

    def test_prompt_render_with_required_placeholders_does_not_raise(self) -> None:
        """N2: the user_template renders cleanly when both placeholders are supplied."""
        from scripts.llm.prompt_registry import _reset_prompt_cache, load_prompt

        _reset_prompt_cache()
        prompt = load_prompt("long_tail_ner")
        rendered = prompt.render_user(text="SAMPLE TEXT", low_conf_json="[]")
        assert "SAMPLE TEXT" in rendered
        assert "[]" in rendered


# ---------------------------------------------------------------------------
# Offset contract — N2 review fix
# ---------------------------------------------------------------------------
class TestOffsetContract:
    def test_drops_entities_with_bad_offsets(self):
        """N2 fix: when LLM returns wrong offsets, drop the bad entity, keep the good."""
        canned = {
            "new_entities": [
                {"surface": "good", "type": "method", "span_start": 0, "span_end": 4},
                {"surface": "EGFR", "type": "method", "span_start": 0, "span_end": 100},
                {"surface": "missing_key"},  # missing offsets
                {"surface": "wrong_text", "type": "method", "span_start": 0, "span_end": 4},  # text mismatch
            ],
            "validations": [],
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)
        result = extract_long_tail_and_validate(
            text="good text", low_conf_entities=[], client=mock_client,
        )
        # Only "good" survives
        assert len(result["new_entities"]) == 1
        assert result["new_entities"][0]["surface"] == "good"

    def test_drops_validations_with_bad_shape(self):
        canned = {
            "new_entities": [],
            "validations": [
                {"surface": "ok", "keep": True, "reason": "real"},
                {"surface": "bad", "keep": "yes"},  # keep is not bool
                {"keep": False},  # surface missing
                {"surface": "", "keep": True},  # empty surface
            ],
        }
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(canned)
        result = extract_long_tail_and_validate(
            text="t", low_conf_entities=[], client=mock_client,
        )
        assert len(result["validations"]) == 1
        assert result["validations"][0]["surface"] == "ok"


# ---------------------------------------------------------------------------
# Preserve-low-conf on disabled / failure — N2 review fix
# ---------------------------------------------------------------------------
class TestPreserveLowConfOnDisabled:
    def test_disabled_returns_low_conf_as_keep_true(self, monkeypatch):
        """N2 fix: when disabled, preserve low_conf_entities as keep=True (enrichment, not gate)."""
        from scripts.graph import long_tail as lt
        fake_config = MagicMock()
        fake_config.graph.ner.cloud_long_tail_enabled = False
        monkeypatch.setattr(lt, "_load_runtime_config", lambda: fake_config)

        low_conf = [
            {"surface": "EGFR", "confidence": 0.55},
            {"surface": "BRCA1", "confidence": 0.48},
        ]
        result = extract_long_tail_and_validate(
            text="paper", low_conf_entities=low_conf,
        )
        assert result["new_entities"] == []
        assert len(result["validations"]) == 2
        assert all(v["keep"] is True for v in result["validations"])
        assert all("disabled" in v.get("reason", "").lower() for v in result["validations"])

    def test_llm_error_preserves_low_conf(self):
        """N2 fix: when LLM raises, low_conf_entities are kept (not dropped)."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("network exploded")
        result = extract_long_tail_and_validate(
            text="paper",
            low_conf_entities=[{"surface": "EGFR", "confidence": 0.55}],
            client=mock_client,
        )
        assert len(result["validations"]) == 1
        assert result["validations"][0]["keep"] is True
        assert "llm" in result["validations"][0]["reason"].lower() or "call" in result["validations"][0]["reason"].lower()

    def test_malformed_json_preserves_low_conf(self):
        mock_client = MagicMock()
        mock_client.complete.return_value = "not json"
        result = extract_long_tail_and_validate(
            text="paper",
            low_conf_entities=[{"surface": "EGFR"}],
            client=mock_client,
        )
        assert len(result["validations"]) == 1
        assert result["validations"][0]["keep"] is True


# ---------------------------------------------------------------------------
# Single-call invariant under disabled paths
# ---------------------------------------------------------------------------
class TestDisabledPathsSkipLLMConstruction:
    def test_disabled_flag_does_not_invoke_client(self, monkeypatch) -> None:
        """N2: when the flag is off, no client.complete() is ever called.

        Caller passes client=None; we monkeypatch _load_runtime_config so the
        gate fails. No OllamaClient should be constructed (test would explode
        on the import inside the function if we tried).
        """
        from scripts.graph import long_tail as lt

        fake_config = MagicMock()
        fake_config.graph.ner.cloud_long_tail_enabled = False
        monkeypatch.setattr(lt, "_load_runtime_config", lambda: fake_config)
        monkeypatch.setenv("OLLAMA_API_KEY", "fake-key")

        # If construction were attempted, OllamaClient would call /api/show.
        # Patching it to blow up surfaces any accidental client creation.
        called = {"hit": False}

        def _boom(*_args, **_kw):  # pragma: no cover — must NOT run
            called["hit"] = True
            raise RuntimeError("OllamaClient should not be constructed when disabled")

        monkeypatch.setattr("scripts.llm.llm_client.OllamaClient", _boom)

        result = extract_long_tail_and_validate(
            text="paper text",
            low_conf_entities=[],
        )
        assert result == {"new_entities": [], "validations": []}
        assert called["hit"] is False
