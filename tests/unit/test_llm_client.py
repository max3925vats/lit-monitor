"""
Phase 3 unit tests — LLM client abstraction.
All tests use MockLLMClient. No Ollama required.
macOS integration tests (real Ollama) are in tests/integration/.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fence stripping
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_strip_markdown_fences_json_block():
    from scripts.llm.llm_client import strip_markdown_fences
    raw = '```json\n{"key": "value"}\n```'
    assert strip_markdown_fences(raw) == '{"key": "value"}'
@pytest.mark.unit
def test_strip_markdown_fences_plain_block():
    from scripts.llm.llm_client import strip_markdown_fences
    raw = '```\n{"key": "value"}\n```'
    assert strip_markdown_fences(raw) == '{"key": "value"}'
@pytest.mark.unit
def test_strip_markdown_fences_no_fence():
    from scripts.llm.llm_client import strip_markdown_fences
    raw = '{"key": "value"}'
    assert strip_markdown_fences(raw) == '{"key": "value"}'
@pytest.mark.unit
def test_parse_llm_json_with_fences():
    from scripts.llm.llm_client import parse_llm_json
    raw = '```json\n{"core_finding": "test"}\n```'
    result = parse_llm_json(raw)
    assert result["core_finding"] == "test"
@pytest.mark.unit
def test_parse_llm_json_raises_on_invalid():
    from scripts.llm.llm_client import parse_llm_json
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_llm_json("This is not JSON at all.")
# ---------------------------------------------------------------------------
# MockLLMClient behaviour
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_mock_client_returns_string():
    from scripts.llm.llm_client import MockLLMClient
    client = MockLLMClient()
    result = client.complete("system prompt", "user prompt")
    assert isinstance(result, str)
    # Should be valid JSON
    parsed = json.loads(result)

    assert isinstance(parsed, dict)
@pytest.mark.unit
def test_mock_client_tracks_call_count():
    from scripts.llm.llm_client import MockLLMClient
    client = MockLLMClient()
    client.complete("s", "u1")
    client.complete("s", "u2")
    assert client.call_count == 2
@pytest.mark.unit
def test_mock_client_infers_paper_pass1():
    from scripts.llm.llm_client import MockLLMClient
    client = MockLLMClient()
    result = json.loads(
        client.complete("Extract fields.", "Please extract core_finding and methods_summary.")
    )
    assert "core_finding" in result
    assert "methods_summary" in result
@pytest.mark.unit
def test_mock_client_infers_paper_pass2():
    from scripts.llm.llm_client import MockLLMClient
    client = MockLLMClient()
    result = json.loads(
        client.complete("Extract fields.", "Extract background_motivation and limitations.")
    )
    assert "background_motivation" in result
@pytest.mark.unit
def test_mock_client_infers_paper_pass3():
    from scripts.llm.llm_client import MockLLMClient
    client = MockLLMClient()
    result = json.loads(
        client.complete("Extract fields.", "Extract novelty_statement and key_citations.")
    )
    assert "novelty_statement" in result
@pytest.mark.unit
def test_mock_client_explicit_key_override():

    from scripts.llm.llm_client import MockLLMClient
    client = MockLLMClient(mock_response_key="paper_pass3")
    result = json.loads(client.complete("s", "u"))
    assert "novelty_statement" in result
@pytest.mark.unit
def test_mock_client_raises_on_demand():
    from scripts.llm.llm_client import MockLLMClient
    client = MockLLMClient(raise_on_call=RuntimeError("Ollama timeout"))
    with pytest.raises(RuntimeError, match="Ollama timeout"):
        client.complete("s", "u")
@pytest.mark.unit
def test_mock_client_null_fields_present():
    """Null fields (absent values) must be present in mock responses — not omitted."""
    from scripts.llm.llm_client import MockLLMClient
    client = MockLLMClient(mock_response_key="paper_pass2")
    result = json.loads(client.complete("s", "u"))
    # These should be explicitly None, not missing
    assert "experimental_conditions" in result
    assert result["experimental_conditions"] is None
    assert result["experimental_conditions_confidence"] == "absent"
# ---------------------------------------------------------------------------
# get_client factory
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_get_client_returns_ollama_client_for_brain_build():
    from scripts.llm.llm_client import OllamaClient, get_client
    config = MagicMock()
    config.brain_build.provider = "ollama"
    config.brain_build.model = "qwen2.5:7b"
    config.brain_build.ollama_host = "http://localhost:11434"
    config.brain_build.timeout = 300
    config.brain_build.temperature = 0.1
    # MagicMock config has no real num_ctx_override → get_client would let
    # OllamaClient.__init__ fire a real /api/show HTTP call. Patch the boundary
    # so this is network-free in clean CI.
    with patch("requests.post", side_effect=ConnectionError("no network")):
        client = get_client(config, "brain_build")
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:7b"
@pytest.mark.unit
def test_get_client_returns_correct_model_for_ingestion():
    from scripts.llm.llm_client import OllamaClient, get_client
    config = MagicMock()
    config.ingestion.provider = "ollama"
    config.ingestion.model = "qwen2.5:3b"
    config.ingestion.ollama_host = "http://localhost:11434"
    config.ingestion.timeout = 300
    config.ingestion.temperature = 0.1
    # Patch the /api/show boundary — MagicMock config would otherwise let
    # OllamaClient.__init__ make a real HTTP call (fails in clean CI).
    with patch("requests.post", side_effect=ConnectionError("no network")):
        client = get_client(config, "ingestion")
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:3b"
@pytest.mark.unit
def test_get_client_raises_for_unknown_mode():
    from scripts.llm.llm_client import get_client
    config = MagicMock()
    config.nonsense_mode = None
    # getattr will return a Mock for unknown attr, but provider check will fail
    with pytest.raises((ValueError, AttributeError)):
        get_client(config, "nonsense_mode")

@pytest.mark.unit
def test_get_client_rejects_non_ollama_provider():
    from scripts.llm.llm_client import get_client
    config = MagicMock()
    config.ingestion.provider = "anthropic"
    config.ingestion.model = "claude-3-sonnet"
    # match="anthropic" works because the error message echoes {provider!r} which
    # includes the string 'anthropic'.  Keep {provider!r} in the message if refactoring.
    with pytest.raises(ValueError, match="anthropic"):
        get_client(config, "ingestion")
# ---------------------------------------------------------------------------
# OllamaClient timeout configuration
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_ollama_client_stores_timeout():
    from scripts.llm.llm_client import OllamaClient
    # _skip_api_show=True avoids the real /api/show HTTP call in __init__ so this
    # construction is network-free (passes in clean CI / no Ollama running).
    client = OllamaClient(model="phi4-mini", timeout=300, _skip_api_show=True)
    assert client.timeout == 300
@pytest.mark.unit
def test_ollama_client_thinking_mode_is_on():
    """
    Verify that think=True is a top-level field in the Ollama API payload.

    The Ollama /api/chat endpoint expects 'think' at the top level, NOT inside
    the 'options' dict.  This is a regression test for A9: placing it inside
    options caused Ollama to silently ignore the flag.
    """
    from scripts.llm.llm_client import OllamaClient
    # _skip_api_show=True so __init__ makes no real /api/show call; the patch
    # below only covers the complete() POST we actually want to assert on.
    client = OllamaClient(model="qwen2.5:3b", _skip_api_show=True)
    captured_payload = {}
    def mock_post(url, json=None, timeout=None, headers=None):
        captured_payload.update(json or {})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": '{"result": "ok"}'}
        }
        mock_resp.raise_for_status = lambda: None
        return mock_resp
    with patch("requests.post", side_effect=mock_post):
        client.complete("sys", "usr")
    # think must be top-level, not inside options
    assert captured_payload.get("think") is True, (
        "think must be a top-level payload field — not inside options"
    )
    assert "think" not in captured_payload.get("options", {}), (
        "think must NOT be inside options (A9 regression)"
    )


@pytest.mark.unit
def test_get_client_respects_think_false_from_yaml_config():
    """
    When extraction.yaml sets 'think: false' for a mode, get_client() must pass
    think=False to OllamaClient — overriding the default think=True parameter.

    This is the fix for the 400 Bad Request error that occurs when 'think: true'
    is sent to non-reasoning models (phi4-mini, qwen2.5:3b, etc.) on newer
    Ollama versions (0.7+).
    """
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_client

    # Simulate extraction.yaml with 'think: false' (non-reasoning model).
    # Use SimpleNamespace rather than MagicMock so getattr returns None for absent
    # attributes, mirroring the real _Namespace config object.
    mode_config = SimpleNamespace(
        provider="ollama",
        model="phi4-mini",
        ollama_host="http://localhost:11434",
        timeout=300,
        temperature=0.1,
        think=False,  # explicit YAML override — must beat the default think=True
    )
    config = SimpleNamespace(brain_build=mode_config)

    # Patch /api/show boundary — config has no num_ctx_override so __init__
    # would otherwise make a real HTTP call.
    with patch("requests.post", side_effect=ConnectionError("no network")):
        client = get_client(config, "brain_build", think=True)  # caller wants True…
    assert isinstance(client, OllamaClient)
    assert client.think is False, (
        "YAML 'think: false' must override the caller's think=True default"
    )


@pytest.mark.unit
def test_get_client_falls_back_to_caller_think_when_yaml_omits_it():
    """
    When extraction.yaml has no 'think' field for a mode, get_client() must
    use the think parameter supplied by the caller (default True).
    """
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_client

    # No 'think' attribute — simulates a YAML section without the key.
    mode_config = SimpleNamespace(
        provider="ollama",
        model="qwen2.5:3b",
        ollama_host="http://localhost:11434",
        timeout=300,
        temperature=0.1,
        # think is intentionally absent
    )
    config = SimpleNamespace(brain_build=mode_config)

    # Caller passes think=False (e.g. vocabulary clustering) — must be respected.
    with patch("requests.post", side_effect=ConnectionError("no network")):
        client = get_client(config, "brain_build", think=False)
    assert isinstance(client, OllamaClient)
    assert client.think is False, (
        "Caller's think=False must be used when YAML has no 'think' field"
    )
# ---------------------------------------------------------------------------
# Cloud-Ollama (V-4) — api_key / OLLAMA_API_KEY / Bearer auth header
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_ollama_client_explicit_api_key_stored():
    """Explicit api_key parameter is stored on the client."""
    from scripts.llm.llm_client import OllamaClient
    client = OllamaClient(model="gemma4:31b", api_key="explicit-key", _skip_api_show=True)
    assert client.api_key == "explicit-key"


@pytest.mark.unit
def test_ollama_client_reads_api_key_from_env(monkeypatch):
    """When no explicit key is given, OllamaClient falls back to OLLAMA_API_KEY env var."""
    from scripts.llm.llm_client import OllamaClient
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key-abc")
    client = OllamaClient(model="gemma4:31b", _skip_api_show=True)
    assert client.api_key == "env-key-abc"


@pytest.mark.unit
def test_ollama_client_explicit_key_wins_over_env(monkeypatch):
    """Explicit api_key takes precedence over OLLAMA_API_KEY env var."""
    from scripts.llm.llm_client import OllamaClient
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key-should-be-ignored")
    client = OllamaClient(model="gemma4:31b", api_key="explicit-wins", _skip_api_show=True)
    assert client.api_key == "explicit-wins"


@pytest.mark.unit
def test_ollama_client_no_key_when_neither_set(monkeypatch):
    """api_key is None when neither explicit key nor env var is present."""
    from scripts.llm.llm_client import OllamaClient
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    client = OllamaClient(model="qwen2.5:3b", _skip_api_show=True)
    assert client.api_key is None


@pytest.mark.unit
def test_ollama_client_sends_bearer_header_in_complete():
    """complete() includes Authorization: Bearer <key> when api_key is set."""
    from scripts.llm.llm_client import OllamaClient
    client = OllamaClient(model="gemma4:31b", api_key="test-bearer-key", _skip_api_show=True)
    captured = {}

    def mock_post(url, json=None, timeout=None, headers=None):
        captured["headers"] = headers or {}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "{}"}}
        resp.raise_for_status = lambda: None
        return resp

    with patch("requests.post", side_effect=mock_post):
        client.complete("sys", "usr")

    assert captured["headers"].get("Authorization") == "Bearer test-bearer-key", (
        "Authorization header must be sent for cloud-Ollama calls"
    )


@pytest.mark.unit
def test_ollama_client_no_auth_header_without_api_key(monkeypatch):
    """complete() sends no Authorization header when api_key is None (local Ollama)."""
    from scripts.llm.llm_client import OllamaClient
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    client = OllamaClient(model="qwen2.5:3b", _skip_api_show=True)
    captured = {}

    def mock_post(url, json=None, timeout=None, headers=None):
        captured["headers"] = headers or {}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "{}"}}
        resp.raise_for_status = lambda: None
        return resp

    with patch("requests.post", side_effect=mock_post):
        client.complete("sys", "usr")

    assert "Authorization" not in captured["headers"], (
        "No Authorization header should be sent for local Ollama (no api_key)"
    )


@pytest.mark.unit
def test_ollama_client_is_available_sends_bearer_header():
    """is_available() includes Authorization: Bearer <key> for cloud endpoints."""
    from scripts.llm.llm_client import OllamaClient
    client = OllamaClient(
        model="gemma4:31b",
        host="https://ollama.com",
        api_key="cloud-key",
        _skip_api_show=True,
    )
    captured = {}

    def mock_get(url, timeout=None, headers=None):
        captured["headers"] = headers or {}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": [{"name": "gemma4:31b"}]}
        return resp

    with patch("requests.get", side_effect=mock_get):
        result = client.is_available()

    assert result is True
    assert captured["headers"].get("Authorization") == "Bearer cloud-key"


@pytest.mark.unit
def test_get_client_factory_passes_ollama_host():
    """get_client() passes ollama_host from mode config to OllamaClient.host.

    This verifies the factory is fully wired for cloud-Ollama: set
    ``ollama_host: "https://ollama.com"`` in extraction.yaml and export
    ``OLLAMA_API_KEY`` — no factory changes are required.
    """
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_client

    mode_config = SimpleNamespace(
        provider="ollama",
        model="gemma4:31b",
        ollama_host="https://ollama.com",   # cloud endpoint
        timeout=60,
        temperature=0.1,
        # think absent → caller default
    )
    config = SimpleNamespace(brain_build=mode_config)

    with patch("requests.post", side_effect=ConnectionError("no network")):
        client = get_client(config, "brain_build")
    assert isinstance(client, OllamaClient)
    assert client.host == "https://ollama.com", (
        "Factory must pass ollama_host to OllamaClient for cloud-Ollama support"
    )

# ---------------------------------------------------------------------------
# D4 — LiteLLMClient (skipped when litellm not installed)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_litellm_client_complete_calls_litellm():
    """LiteLLMClient.complete() delegates to litellm.completion with correct args."""
    _litellm = pytest.importorskip("litellm")
    if not hasattr(_litellm, "completion"):
        pytest.skip("litellm installed but missing 'completion' attribute (stub/partial install)")
    from scripts.llm.llm_client import LiteLLMClient

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = '{"core_finding": "test"}'

    with patch("litellm.completion", return_value=mock_resp) as mock_comp:
        client = LiteLLMClient(model="anthropic/claude-haiku-4-5", temperature=0.0)
        result = client.complete("system prompt", "user prompt", max_tokens=2048)

    assert result == '{"core_finding": "test"}'
    mock_comp.assert_called_once()
    call_kwargs = mock_comp.call_args[1]
    assert call_kwargs["model"] == "anthropic/claude-haiku-4-5"
    assert call_kwargs["max_tokens"] == 2048
    # JSON mode must be requested — LiteLLM translates this per provider
    assert call_kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# V-5 — Symmetric LiteLLM 429 detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_litellm_client_429_raises_rate_limit_error():
    """LiteLLMClient.complete() re-raises lit-monitor RateLimitError on litellm 429.

    V-5 parity: OllamaClient already raises RateLimitError on HTTP 429; now
    LiteLLMClient does the same so callers can catch it provider-agnostically.
    """
    import sys

    from scripts.llm.llm_client import LiteLLMClient, RateLimitError

    # Build a fake litellm module whose RateLimitError class is the one raised.
    FakeRateLimit = type("RateLimitError", (Exception,), {})
    mock_litellm = MagicMock()
    mock_litellm.exceptions.RateLimitError = FakeRateLimit
    mock_litellm.completion.side_effect = FakeRateLimit("429 Too Many Requests")

    client = LiteLLMClient(model="anthropic/claude-haiku-4-5")

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        with pytest.raises(RateLimitError, match="rate limit"):
            client.complete("sys", "usr")


@pytest.mark.unit
def test_litellm_client_non_429_raises_runtime_error():
    """LiteLLMClient.complete() wraps non-429 errors as RuntimeError, not RateLimitError."""
    import sys

    from scripts.llm.llm_client import LiteLLMClient, RateLimitError

    # RateLimitError class is distinct from the error actually raised.
    FakeRateLimit = type("RateLimitError", (Exception,), {})
    mock_litellm = MagicMock()
    mock_litellm.exceptions.RateLimitError = FakeRateLimit
    mock_litellm.completion.side_effect = ConnectionError("Network unreachable")

    client = LiteLLMClient(model="anthropic/claude-haiku-4-5")

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        with pytest.raises(RuntimeError) as exc_info:
            client.complete("sys", "usr")
    # Must NOT be the subtype RateLimitError — that is reserved for 429 only.
    assert not isinstance(exc_info.value, RateLimitError), (
        "Non-429 connection errors must not be re-raised as RateLimitError"
    )


@pytest.mark.unit
def test_litellm_client_stores_num_ctx():
    """LiteLLMClient.num_ctx is set for extractor._available_input_chars()."""
    _litellm = pytest.importorskip("litellm")
    if not hasattr(_litellm, "completion"):
        pytest.skip("litellm installed but missing 'completion' attribute (stub/partial install)")
    from scripts.llm.llm_client import LiteLLMClient

    client = LiteLLMClient(model="anthropic/claude-sonnet-4-5", num_ctx=200000)
    assert client.num_ctx == 200000


@pytest.mark.unit
def test_get_client_routes_to_litellm():
    """get_client() returns LiteLLMClient when provider='litellm'."""
    _litellm = pytest.importorskip("litellm")
    if not hasattr(_litellm, "completion"):
        pytest.skip("litellm installed but missing 'completion' attribute (stub/partial install)")
    from types import SimpleNamespace

    from scripts.llm.llm_client import LiteLLMClient, get_client

    mode_config = SimpleNamespace(
        provider="litellm",
        litellm_model="anthropic/claude-haiku-4-5",
        temperature=0.1,
        timeout=None,
        context_window=200000,
    )
    config = SimpleNamespace(brain_build=mode_config)

    client = get_client(config, "brain_build")

    assert isinstance(client, LiteLLMClient)
    assert client.model == "anthropic/claude-haiku-4-5"
    assert client.num_ctx == 200000


@pytest.mark.unit
def test_get_client_litellm_raises_without_litellm_model():
    """get_client() raises ValueError when litellm_model is absent from config."""
    _litellm = pytest.importorskip("litellm")
    if not hasattr(_litellm, "completion"):
        pytest.skip("litellm installed but missing 'completion' attribute (stub/partial install)")
    from types import SimpleNamespace

    from scripts.llm.llm_client import get_client

    mode_config = SimpleNamespace(
        provider="litellm",
        temperature=0.1,
        timeout=None,
        # litellm_model intentionally absent
    )
    config = SimpleNamespace(brain_build=mode_config)

    with pytest.raises(ValueError, match="litellm_model"):
        get_client(config, "brain_build")


@pytest.mark.unit
def test_litellm_client_raises_importerror_when_not_installed():
    """LiteLLMClient.complete() raises ImportError with install instructions."""
    import sys

    from scripts.llm.llm_client import LiteLLMClient

    client = LiteLLMClient(model="anthropic/claude-haiku-4-5")
    # Setting sys.modules["litellm"] = None causes `import litellm` to raise ImportError.
    # This is documented CPython behaviour — the canonical way to simulate a missing module.
    with patch.dict(sys.modules, {"litellm": None}):
        with pytest.raises(ImportError, match=r"lit-monitor\[litellm\]"):
            client.complete("system", "user")


@pytest.mark.unit
def test_litellm_client_wraps_provider_errors_as_runtimeerror():
    """Exceptions from litellm.completion() are wrapped in RuntimeError with model name."""
    _litellm = pytest.importorskip("litellm")
    if not hasattr(_litellm, "completion"):
        pytest.skip("litellm installed but missing 'completion' attribute (stub/partial install)")
    from scripts.llm.llm_client import LiteLLMClient

    client = LiteLLMClient(model="anthropic/claude-haiku-4-5")
    with patch("litellm.completion", side_effect=Exception("rate limit exceeded")):
        with pytest.raises(RuntimeError, match="LiteLLM call failed"):
            client.complete("system", "user")

# ---------------------------------------------------------------------------
# F15 — get_clients_for_passes
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_get_clients_for_passes_returns_single_when_no_pass_models():
    """Without passN_model keys, get_clients_for_passes returns a single LLMClient."""
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_clients_for_passes

    mode_config = SimpleNamespace(
        provider="ollama",
        model="phi4-mini",
        ollama_host="http://localhost:11434",
        timeout=300,
        temperature=0.1,
        think=False,
        # no pass1_model / pass2_model / pass3_model
    )
    config = SimpleNamespace(brain_build=mode_config)
    with patch("requests.post", side_effect=ConnectionError("no network")):
        client = get_clients_for_passes(config, "brain_build")
    assert isinstance(client, OllamaClient), "No per-pass keys → must return single client"
    assert client.model == "phi4-mini"


@pytest.mark.unit
def test_get_clients_for_passes_returns_dict_when_pass1_model_set():
    """Setting pass1_model triggers dict-of-clients return."""
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_clients_for_passes

    mode_config = SimpleNamespace(
        provider="ollama",
        model="phi4-mini",
        ollama_host="http://localhost:11434",
        timeout=300,
        temperature=0.1,
        think=False,
        pass1_model="qwen2.5:7b",
        # pass2_model / pass3_model absent → fall back to default "phi4-mini"
    )
    config = SimpleNamespace(brain_build=mode_config)
    with patch("requests.post", side_effect=ConnectionError("no network")):
        result = get_clients_for_passes(config, "brain_build")
    assert isinstance(result, dict), "pass1_model set → must return dict"
    assert set(result.keys()) == {1, 2, 3}
    for client in result.values():
        assert isinstance(client, OllamaClient)


@pytest.mark.unit
def test_get_clients_for_passes_per_pass_model_assigned_correctly():
    """Each pass gets the right model; omitted passes fall back to the mode default."""
    from types import SimpleNamespace

    from scripts.llm.llm_client import get_clients_for_passes

    mode_config = SimpleNamespace(
        provider="ollama",
        model="phi4-mini",          # default
        ollama_host="http://localhost:11434",
        timeout=None,
        temperature=0.1,
        think=False,
        pass1_model="qwen2.5:7b",   # override for pass 1
        # pass2_model absent → phi4-mini
        pass3_model="qwen2.5:3b",   # override for pass 3
    )
    config = SimpleNamespace(brain_build=mode_config)
    with patch("requests.post", side_effect=ConnectionError("no network")):
        clients = get_clients_for_passes(config, "brain_build")
    assert clients[1].model == "qwen2.5:7b"
    assert clients[2].model == "phi4-mini"   # fell back to default
    assert clients[3].model == "qwen2.5:3b"


@pytest.mark.unit
def test_get_clients_for_passes_inherits_shared_settings():
    """Timeout, temperature, and think are inherited from the mode config per-pass."""
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_clients_for_passes

    mode_config = SimpleNamespace(
        provider="ollama",
        model="phi4-mini",
        ollama_host="http://localhost:11434",
        timeout=120,
        temperature=0.2,
        think=False,
        pass1_model="qwen2.5:7b",
    )
    config = SimpleNamespace(brain_build=mode_config)
    with patch("requests.post", side_effect=ConnectionError("no network")):
        clients = get_clients_for_passes(config, "brain_build")
    for client in clients.values():
        assert isinstance(client, OllamaClient)
        assert client.timeout == 120
        assert client.temperature == 0.2
        assert client.think is False


@pytest.mark.unit
def test_get_clients_for_passes_litellm_returns_dict():
    """LiteLLM per-pass model keys produce a dict of LiteLLMClient instances."""
    _litellm = pytest.importorskip("litellm")
    if not hasattr(_litellm, "completion"):
        pytest.skip("litellm installed but missing 'completion' attribute (stub/partial install)")
    from types import SimpleNamespace

    from scripts.llm.llm_client import LiteLLMClient, get_clients_for_passes

    mode_config = SimpleNamespace(
        provider="litellm",
        litellm_model="anthropic/claude-haiku-4-5",
        timeout=60,
        temperature=0.1,
        context_window=200000,
        pass1_litellm_model="anthropic/claude-sonnet-4-5",
        # pass2 / pass3 absent → fall back to claude-haiku-4-5
    )
    config = SimpleNamespace(brain_build=mode_config)
    clients = get_clients_for_passes(config, "brain_build")
    assert isinstance(clients, dict)
    assert clients[1].model == "anthropic/claude-sonnet-4-5"
    assert clients[2].model == "anthropic/claude-haiku-4-5"
    assert clients[3].model == "anthropic/claude-haiku-4-5"
    for client in clients.values():
        assert isinstance(client, LiteLLMClient)


@pytest.mark.unit
def test_get_clients_for_passes_rejects_unknown_provider():
    """Unknown provider raises ValueError up front, before any per-pass work."""
    from types import SimpleNamespace

    from scripts.llm.llm_client import get_clients_for_passes

    mode_config = SimpleNamespace(provider="bogus")
    config = SimpleNamespace(brain_build=mode_config)
    with pytest.raises(ValueError, match="Unknown provider"):
        get_clients_for_passes(config, "brain_build")


# ---------------------------------------------------------------------------
# H5 — /api/show context-window discovery + num_ctx_override escape hatch
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_h5_api_show_model_info_format():
    """OllamaClient discovers num_ctx from /api/show new-format model_info dict."""
    from scripts.llm.llm_client import OllamaClient

    api_show_resp = MagicMock()
    api_show_resp.status_code = 200
    api_show_resp.json.return_value = {
        "model_info": {
            "llama.context_length": 131072,
            "llama.block_count": 32,
        }
    }

    with patch("requests.post", return_value=api_show_resp):
        client = OllamaClient(model="qwen2.5:7b")

    assert client.num_ctx == 131072


@pytest.mark.unit
def test_h5_api_show_details_format():
    """OllamaClient discovers num_ctx from /api/show older details.context_length format."""
    from scripts.llm.llm_client import OllamaClient

    api_show_resp = MagicMock()
    api_show_resp.status_code = 200
    api_show_resp.json.return_value = {
        "details": {"context_length": 32768, "format": "gguf"},
    }

    with patch("requests.post", return_value=api_show_resp):
        client = OllamaClient(model="phi4-mini")

    assert client.num_ctx == 32768


@pytest.mark.unit
def test_h5_num_ctx_override_wins_and_skips_api_show():
    """num_ctx_override wins over /api/show and prevents the HTTP call entirely."""
    from scripts.llm.llm_client import OllamaClient

    with patch("requests.post") as mock_post:
        client = OllamaClient(model="qwen2.5:7b", num_ctx_override=8192)

    mock_post.assert_not_called()
    assert client.num_ctx == 8192


@pytest.mark.unit
def test_h5_api_show_connection_error_falls_back_to_constructor_num_ctx():
    """On /api/show failure, fall back to the constructor num_ctx arg."""
    from scripts.llm.llm_client import OllamaClient

    with patch("requests.post", side_effect=ConnectionError("refused")):
        client = OllamaClient(model="phi4-mini", num_ctx=32768)

    assert client.num_ctx == 32768


@pytest.mark.unit
def test_h5_api_show_connection_error_falls_back_to_16384():
    """/api/show failure with no constructor num_ctx falls back to the 16384 floor."""
    from scripts.llm.llm_client import OllamaClient

    with patch("requests.post", side_effect=ConnectionError("refused")):
        client = OllamaClient(model="phi4-mini")  # no num_ctx arg

    assert client.num_ctx == 16384


@pytest.mark.unit
def test_h5_api_show_bad_status_falls_back_to_num_ctx():
    """/api/show returning a non-200 status falls back to the constructor num_ctx arg."""
    from scripts.llm.llm_client import OllamaClient

    api_show_resp = MagicMock()
    api_show_resp.status_code = 404

    with patch("requests.post", return_value=api_show_resp):
        client = OllamaClient(model="phi4-mini", num_ctx=16384)

    assert client.num_ctx == 16384


@pytest.mark.unit
def test_h5_skip_api_show_uses_constructor_num_ctx():
    """_skip_api_show=True uses constructor num_ctx without any HTTP call."""
    from scripts.llm.llm_client import OllamaClient

    with patch("requests.post") as mock_post:
        client = OllamaClient(model="phi4-mini", num_ctx=32768, _skip_api_show=True)

    mock_post.assert_not_called()
    assert client.num_ctx == 32768


@pytest.mark.unit
def test_h5_skip_api_show_falls_back_to_16384():
    """_skip_api_show=True with no num_ctx arg uses the 16384 floor."""
    from scripts.llm.llm_client import OllamaClient

    with patch("requests.post") as mock_post:
        client = OllamaClient(model="phi4-mini", _skip_api_show=True)

    mock_post.assert_not_called()
    assert client.num_ctx == 16384


@pytest.mark.unit
def test_h5_get_client_reads_num_ctx_override_from_config():
    """get_client() passes num_ctx_override from extraction.yaml to OllamaClient."""
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_client

    mode_config = SimpleNamespace(
        provider="ollama",
        model="qwen2.5:7b",
        ollama_host="http://localhost:11434",
        timeout=None,
        temperature=0.1,
        num_ctx_override=65536,  # explicit YAML override
    )
    config = SimpleNamespace(brain_build=mode_config)

    # override is set → /api/show must NOT be called
    with patch("requests.post") as mock_post:
        client = get_client(config, "brain_build")

    mock_post.assert_not_called()
    assert isinstance(client, OllamaClient)
    assert client.num_ctx == 65536


# ---------------------------------------------------------------------------
# K1 — pass_all_model overrides default model in get_client()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pass_all_model_overrides_mode_model_when_set():
    """
    When pass_strategy='all' and pass_all_model is set in mode config,
    get_client() must build OllamaClient with pass_all_model, not the default model.
    """
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_client

    mode_config = SimpleNamespace(
        provider="ollama",
        model="qwen2.5:3b",          # default per-mode model
        pass_strategy="all",
        pass_all_model="gemma4:31b-cloud",  # should override model when strategy=all
        ollama_host="http://localhost:11434",
        timeout=None,
        temperature=0.1,
        num_ctx_override=65536,  # skip /api/show call in OllamaClient.__init__
    )
    config = SimpleNamespace(brain_build=mode_config)

    with patch("requests.post") as mock_post:
        client = get_client(config, "brain_build")

    mock_post.assert_not_called()
    assert isinstance(client, OllamaClient)
    assert client.model == "gemma4:31b-cloud", (
        f"Expected pass_all_model to override default; got {client.model!r}"
    )


@pytest.mark.unit
def test_pass_all_model_absent_falls_back_to_model():
    """
    When pass_strategy='all' but pass_all_model is NOT set,
    get_client() must fall back to the default model attribute.
    """
    from types import SimpleNamespace

    from scripts.llm.llm_client import OllamaClient, get_client

    mode_config = SimpleNamespace(
        provider="ollama",
        model="qwen2.5:7b",
        pass_strategy="all",
        # pass_all_model intentionally absent
        ollama_host="http://localhost:11434",
        timeout=None,
        temperature=0.1,
        num_ctx_override=65536,  # skip /api/show call in OllamaClient.__init__
    )
    config = SimpleNamespace(brain_build=mode_config)

    with patch("requests.post") as mock_post:
        client = get_client(config, "brain_build")

    mock_post.assert_not_called()
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:7b"


# ---------------------------------------------------------------------------
# N13 — OllamaClient: warn on num_predict=-1 with cloud host
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ollama_client_warns_on_neg1_max_tokens_for_cloud_host(caplog):
    """N13: num_predict=-1 with non-localhost host should log a WARNING before the call."""
    import logging

    from scripts.llm.llm_client import OllamaClient

    client = OllamaClient(
        model="test-model",
        host="https://ollama.com",
        api_key="fake-key",
        num_ctx_override=4096,
    )
    ok_response = {"message": {"content": '{"result": "ok"}'}}

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: ok_response)
        mock_post.return_value.raise_for_status = MagicMock()
        with caplog.at_level(logging.WARNING, logger="scripts.llm.llm_client"):
            client.complete("sys", "usr", max_tokens=-1)

    assert any("num_predict=-1" in r.message for r in caplog.records), (
        "Expected a WARNING about num_predict=-1 for cloud host"
    )


@pytest.mark.unit
def test_ollama_client_no_warn_on_neg1_max_tokens_for_localhost(caplog):
    """N13: num_predict=-1 with localhost should NOT warn (local Ollama handles it fine)."""
    import logging

    from scripts.llm.llm_client import OllamaClient

    client = OllamaClient(
        model="test-model",
        host="http://localhost:11434",
        num_ctx_override=4096,
    )
    ok_response = {"message": {"content": '{"result": "ok"}'}}

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: ok_response)
        mock_post.return_value.raise_for_status = MagicMock()
        with caplog.at_level(logging.WARNING, logger="scripts.llm.llm_client"):
            client.complete("sys", "usr", max_tokens=-1)

    assert not any("num_predict=-1" in r.message for r in caplog.records), (
        "No warning expected for localhost Ollama with num_predict=-1"
    )


@pytest.mark.unit
def test_ollama_client_no_warn_on_neg1_max_tokens_for_custom_local_host(caplog):
    """L4: num_predict=-1 with a custom local hostname (not localhost/127.0.0.1)
    should NOT warn — the previous substring check incorrectly flagged these.
    """
    import logging

    from scripts.llm.llm_client import OllamaClient

    client = OllamaClient(
        model="test-model",
        host="http://my-pi:11434",
        num_ctx_override=4096,
    )
    ok_response = {"message": {"content": '{"result": "ok"}'}}

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: ok_response)
        mock_post.return_value.raise_for_status = MagicMock()
        with caplog.at_level(logging.WARNING, logger="scripts.llm.llm_client"):
            client.complete("sys", "usr", max_tokens=-1)

    assert not any("num_predict=-1" in r.message for r in caplog.records), (
        "No warning expected for a custom local host like http://my-pi:11434"
    )


# ---------------------------------------------------------------------------
# P6.22: mock-response fixtures load lazily, not at import time
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_mock_responses_load_lazily_and_cache() -> None:
    """The fixtures load on first MockLLMClient use, not before, and are cached.

    We do NOT importlib.reload the module here on purpose: reloading would
    rebind RateLimitError to a fresh class object and break other tests'
    ``except RateLimitError`` handlers. Instead we drive the module-level cache
    directly and spy on the loader.
    """
    import scripts.llm.llm_client as _llm

    original_cache = _llm._MOCK_RESPONSES
    try:
        # Simulate the pre-first-use state.
        _llm._MOCK_RESPONSES = None
        load_calls = {"n": 0}
        real_loader = _llm._load_mock_responses

        def _counting_loader():
            load_calls["n"] += 1
            return real_loader()

        with patch.object(_llm, "_load_mock_responses", side_effect=_counting_loader):
            # Cache untouched until first use.
            assert _llm._MOCK_RESPONSES is None
            assert load_calls["n"] == 0

            _llm.MockLLMClient()  # first use → loads once
            assert _llm._MOCK_RESPONSES is not None
            assert load_calls["n"] == 1

            _llm.MockLLMClient()  # second use → served from cache, no reload
            assert load_calls["n"] == 1
    finally:
        _llm._MOCK_RESPONSES = original_cache
