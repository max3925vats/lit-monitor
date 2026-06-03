"""
LLM client abstraction.
All LLM calls in lit-monitor route through this module.
Classes:
    LLMClient       — abstract base
    OllamaClient    — production client (macOS with Ollama running)
    LiteLLMClient   — optional cloud client via LiteLLM (requires pip install lit-monitor[litellm])
    MockLLMClient   — test/dev client (returns canned JSON responses)
Factories:
    get_client(config, mode)            — single client for a pipeline mode
    get_clients_for_passes(config, mode) — single client OR dict[int, LLMClient] when
                                          per-pass model keys (pass1_model, etc.) are set

Provider routing (extraction.yaml `provider:` field):
    'ollama'   → OllamaClient   (default; no extra deps)
    'litellm'  → LiteLLMClient  (requires [litellm] extra; set litellm_model in config)

Per-pass model selection (F15):
    Set pass1_model / pass2_model / pass3_model in any mode section to use different
    Ollama models per extraction pass.  For LiteLLM, use pass1_litellm_model etc.
    Passes without an override fall back to the mode's default model.
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RateLimitError(RuntimeError):
    """Raised when the Ollama API returns HTTP 429 (rate limit / quota exceeded).

    Distinct from RuntimeError so callers can catch it separately and schedule
    a retry rather than treating the run as a permanent failure.
    """


# Known cloud-Ollama host strings.  Inverted from the older
# "not localhost / not 127.0.0.1" test so custom local hostnames
# (e.g. http://my-pi:11434) are correctly treated as local.
_CLOUD_OLLAMA_HOSTS: set[str] = {
    "https://ollama.com",
    "http://ollama.com",
}


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class LLMClient(ABC):
    """Abstract interface for all LLM backends."""
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """
        Send a system + user prompt pair and return the raw response string.
        Raises RuntimeError on connection/timeout failures.
        """
# ---------------------------------------------------------------------------
# Ollama client (macOS production)
# ---------------------------------------------------------------------------
class OllamaClient(LLMClient):
    """Client for a locally-running or Ollama-cloud-hosted instance.

    For cloud models (https://ollama.com/api/chat), pass:
        host="https://ollama.com"
        api_key="<your key>"   # or leave None to read OLLAMA_API_KEY from env
    All other parameters behave identically to the local case.

    Context-window precedence (H5):
        1. ``num_ctx_override`` — explicit value from extraction.yaml (skips /api/show)
        2. ``/api/show`` discovered value — actual model capability from Ollama
        3. ``num_ctx`` constructor arg — mode-config default from _OLLAMA_NUM_CTX_DEFAULTS
        4. 16384 — safe floor for phi4-mini / qwen2.5:3b
    """
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int | None = None,
        temperature: float = 0.1,
        num_ctx: int | None = None,
        think: bool = True,
        api_key: str | None = None,
        num_ctx_override: int | None = None,
        _skip_api_show: bool = False,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.think = think
        # Explicit api_key wins; fall back to the standard env var.
        # None when neither is set → no Authorization header (local Ollama).
        self.api_key: str | None = api_key or os.environ.get("OLLAMA_API_KEY") or None

        # H5: context-window precedence chain
        if num_ctx_override is not None:
            # Level 1: explicit YAML override — most authoritative, skip /api/show
            self.num_ctx: int = num_ctx_override
        elif not _skip_api_show:
            # Level 2: ask Ollama for the model's native context window
            discovered = self._discover_num_ctx()
            self.num_ctx = discovered if discovered is not None else (num_ctx if num_ctx is not None else 16384)
        else:
            # Level 3/4: fall back to constructor arg or absolute floor
            self.num_ctx = num_ctx if num_ctx is not None else 16384

    def _discover_num_ctx(self) -> int | None:
        """POST to /api/show to discover the model's native context window.

        Returns the integer context length on success, None on any failure so
        the caller can continue down the precedence chain.
        """
        try:
            import requests as req
        except ImportError:
            return None
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = req.post(
                f"{self.host}/api/show",
                json={"model": self.model},
                timeout=3,  # short timeout — fail fast if Ollama is not running
                headers=headers,
            )
            if resp.status_code != 200:
                logger.warning(
                    "OllamaClient /api/show returned HTTP %d for model %r — "
                    "falling back to config num_ctx",
                    resp.status_code, self.model,
                )
                return None
            data = resp.json()
            # New Ollama format: model_info with architecture-prefixed keys,
            # e.g. "llama.context_length", "phi3.context_length", "gemma2.context_length"
            model_info = data.get("model_info", {})
            for key, value in model_info.items():
                if key.endswith(".context_length") and isinstance(value, int):
                    logger.debug(
                        "OllamaClient /api/show: model=%r context_length=%d",
                        self.model, value,
                    )
                    return value
            # Older Ollama format: details.context_length
            ctx = data.get("details", {}).get("context_length")
            if isinstance(ctx, int):
                return ctx
            logger.warning(
                "OllamaClient /api/show for model %r: no context_length in response",
                self.model,
            )
            return None
        except Exception as exc:
            logger.warning(
                "OllamaClient /api/show failed for model %r: %s", self.model, exc
            )
            return None

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        url = f"{self.host}/api/chat"
        options: dict = {
            "temperature": self.temperature,
            "num_predict": max_tokens,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
            "stream": False,
            "format": "json",  # force JSON-mode sampling — prevents prose/markdown responses
            # think must be a top-level field in the Ollama /api/chat payload,
            # NOT inside options.  Placing it inside options silently ignores it.
            "think": self.think,
        }
        # N13: cloud Ollama returns 400 for num_predict=-1; warn before the call.
        # L4: invert detection — only warn for known cloud hosts, so custom
        # local hostnames (e.g. http://my-pi:11434) don't trip the warning.
        if options.get("num_predict") == -1 and self.host in _CLOUD_OLLAMA_HOSTS:
            logger.warning(
                "OllamaClient: num_predict=-1 is local-Ollama only; "
                "cloud Ollama (%s) will return 400 — pass a positive max_tokens",
                self.host,
            )
        # Add Bearer token for Ollama cloud; omit header entirely for local Ollama.
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            import requests as req
            resp = req.post(url, json=payload, timeout=self.timeout, headers=headers)
            # Detect rate-limit before raise_for_status so we can surface a
            # specific RateLimitError that callers can catch separately.
            if resp.status_code == 429:
                raise RateLimitError(
                    f"Ollama rate limit / quota exceeded "
                    f"(model={self.model}, host={self.host}). "
                    "Retry after the quota resets."
                )
            # Audit R28: capture the response body BEFORE raise_for_status so
            # the actual Ollama error (e.g. "context length exceeded",
            # "model not found") is visible in logs, not just the HTTP code.
            if resp.status_code != 200:
                body_snippet = ""
                try:
                    body_snippet = resp.text[:500]
                except Exception:
                    pass
                logger.error(
                    "Ollama /api/chat HTTP %d for model=%s host=%s body=%r",
                    resp.status_code, self.model, self.host, body_snippet,
                )
                resp.raise_for_status()
            data = resp.json()
            # Newer Ollama versions put thinking in a separate field; older versions
            # embed it inside content as <think>...</think> — strip_markdown_fences handles both.
            return data["message"].get("content") or ""
        except RateLimitError:
            raise  # preserve the specific type — don't wrap in RuntimeError
        except Exception as exc:
            raise RuntimeError(
                f"Ollama call failed (model={self.model}, host={self.host}): {exc}"
            ) from exc
    def generate(self, prompt: str) -> str:
        """Single-prompt convenience wrapper around complete().

        Note: not part of the LLMClient ABC — Ollama-specific helper only.
        """
        return self.complete(system="", user=prompt)

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the model is available.

        Note: not part of the LLMClient ABC — Ollama-specific helper only.
        Passes auth header for cloud endpoints so the check is not rejected with 401.
        """
        try:
            import requests as req
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = req.get(f"{self.host}/api/tags", timeout=5, headers=headers)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Match exact tag or prefix (e.g. "gemma4:e4b" matches "gemma4:e4b")
            return any(
                m == self.model or m.startswith(self.model.split(":")[0])
                for m in models
            )
        except Exception:
            return False
# ---------------------------------------------------------------------------
# LiteLLM client (cloud providers — optional dependency)
# ---------------------------------------------------------------------------
class LiteLLMClient(LLMClient):
    """
    LLM client that routes calls through LiteLLM, supporting any provider
    that LiteLLM understands (Anthropic, OpenAI, Vertex AI, etc.).

    Requires: ``pip install 'lit-monitor[litellm]'`` (``[cloud]`` also works as a deprecated alias)

    Parameters
    ----------
    model:
        LiteLLM model string, e.g. ``'anthropic/claude-haiku-4-5'``,
        ``'openai/gpt-4o-mini'``, ``'vertex_ai/gemini-pro'``.
    timeout:
        HTTP timeout in seconds, or None for no timeout.
    temperature:
        Sampling temperature (default 0.1 — consistent with OllamaClient).
    num_ctx:
        Approximate context-window size used by ``extractor._available_input_chars()``.
        Set to the model's actual context window (e.g. 200000 for Claude Sonnet).
        None → extractor falls back to 16384 (phi4-mini default).
    """

    def __init__(
        self,
        model: str,
        timeout: int | None = None,
        temperature: float = 0.1,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx  # read by extractor._available_input_chars()

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        # Lazy import so LiteLLMClient can be instantiated without litellm installed.
        # get_client() also guards at factory time for a faster / clearer fail-fast
        # path when routing via config — both guards are intentional.
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is not installed. "
                "Enable cloud provider support with: pip install 'lit-monitor[litellm]'"
            ) from exc
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        logger.debug("LiteLLM call: model=%s max_tokens=%d", self.model, max_tokens)
        try:
            resp = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                # LiteLLM translates response_format to tool-use for Anthropic and to the
                # native OpenAI parameter for compatible providers.  If a provider rejects
                # it, the RuntimeError below will surface the reason.  The system prompts
                # also instruct the model to return JSON, so plain-text fallback is rare.
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except litellm.exceptions.RateLimitError as exc:
            # V-5: re-raise as lit-monitor's RateLimitError so all callers can
            # catch it provider-agnostically (OllamaClient already does this on 429).
            raise RateLimitError(
                f"LiteLLM rate limit exceeded (model={self.model}): {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"LiteLLM call failed (model={self.model}): {exc}"
            ) from exc

# ---------------------------------------------------------------------------
# Mock client (Windows dev / unit tests)
# ---------------------------------------------------------------------------

def _load_mock_responses() -> dict[str, Any]:
    """Load tests/fixtures/mock_responses.yaml.

    Resolved relative to the repository root via walk-up from this file.
    Returns {} if the fixture file is absent (defensive — production code
    that actually uses MockLLMClient should fail loudly elsewhere)."""
    from pathlib import Path

    import yaml

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tests" / "fixtures" / "mock_responses.yaml"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


# P6.22: loaded lazily on first MockLLMClient use rather than at import time,
# so importing llm_client in production never walks the filesystem looking for
# the test fixtures file. Cached after the first call.
_MOCK_RESPONSES: dict[str, Any] | None = None


def _get_mock_responses() -> dict[str, Any]:
    """Return the cached mock-response fixtures, loading them on first call."""
    global _MOCK_RESPONSES
    if _MOCK_RESPONSES is None:
        _MOCK_RESPONSES = _load_mock_responses()
    return _MOCK_RESPONSES


class MockLLMClient(LLMClient):
    """
    Test/dev client that returns canned JSON responses.
    Used on Windows (no Ollama) and in unit tests.
    The response key is inferred from the user prompt content, or can be
    set explicitly via mock_response_key.
    """
    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        mock_response_key: str | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self._responses = responses or _get_mock_responses()
        self._mock_key = mock_response_key
        self._raise = raise_on_call
        self.call_count = 0
        self.last_system: str = ""
        self.last_user: str = ""
    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        if self._raise is not None:
            raise self._raise
        self.call_count += 1
        self.last_system = system
        self.last_user = user
        key = self._mock_key or self._infer_key(system, user)
        payload = self._responses.get(key, {"result": "mock"})
        return json.dumps(payload)
    def _infer_key(self, system: str, user: str) -> str:
        """Infer which mock response to return based on prompt content."""
        combined = (system + user).lower()
        if "core_finding" in combined:
            return "paper_pass1"
        if "background_motivation" in combined:
            return "paper_pass2"
        if "novelty_statement" in combined:
            return "paper_pass3"
        if "rationale" in combined or "relevant" in combined:
            return "ranker"
        if "themes" in combined or "cluster" in combined:
            return "vocabulary"
        return "paper_pass1"
# ---------------------------------------------------------------------------
# Fence stripping utility
# ---------------------------------------------------------------------------
def strip_markdown_fences(raw: str) -> str:
    """
    Remove thinking blocks and markdown code fences from LLM JSON responses.
    Handles:
    - non-string / empty input (returns "" — lets callers treat it as "empty response")
    - <think>...</think> blocks (Ollama thinking mode, older versions embed in content)
    - ```json ... ``` or ``` ... ``` fences

    This is the single canonical fence stripper for the codebase; graph/llm modules
    import it rather than keeping their own copies.
    """
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    # Strip <think>...</think> blocks (may appear before the JSON in thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[: -3].rstrip()
    return text.strip()
def parse_llm_json(raw: str) -> dict:
    """
    Parse an LLM response as JSON, stripping fences first.
    Raises ValueError if parsing fails.
    """
    cleaned = strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON after fence stripping.\n"
            f"Raw: {raw[:300]!r}\n"
            f"Error: {exc}"
        ) from exc
# Per-mode Ollama context-window defaults (tokens).
# brain_build: full paper/review text + thinking + large JSON output.
# build_vocabulary: ~60-keyword chunks — 8192 gives ~7600 output tokens of headroom
# without letting the model run unconstrained (quadratic keyword repetition).
_OLLAMA_NUM_CTX_DEFAULTS: dict[str, int] = {
    "brain_build": 32768,
    "build_vocabulary": 8192,
}
# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def get_client(config, mode: str = "ingestion", think: bool = True) -> LLMClient:
    """
    Return the appropriate LLM client for the given pipeline mode.

    Parameters
    ----------
    config:
        Config object with per-mode attributes (e.g. ``config.brain_build.model``).
    mode:
        Pipeline mode — one of ``'brain_build'``, ``'ingestion'``,
        ``'build_vocabulary'``.
    think:
        Default thinking-mode flag for OllamaClient. Overridden by the
        ``think:`` key in extraction.yaml if present. Ignored by LiteLLMClient.

    Supported providers (set via ``provider:`` in extraction.yaml):
        ``'ollama'``   → OllamaClient (default)
        ``'litellm'``  → LiteLLMClient (requires ``pip install lit-monitor[litellm]``;
                         also set ``litellm_model:`` in the mode's config block)
    """
    mode_config = getattr(config, mode, None)
    if mode_config is None:
        raise ValueError(
            f"Unknown pipeline mode: {mode!r}. "
            "Expected one of: brain_build, ingestion, build_vocabulary"
        )
    provider = getattr(mode_config, "provider", "ollama")
    raw_timeout = getattr(mode_config, "timeout", None)
    timeout = int(raw_timeout) if raw_timeout is not None else None
    temperature = float(getattr(mode_config, "temperature", 0.1))

    # ------------------------------------------------------------------
    # LiteLLM — cloud / multi-provider path
    # ------------------------------------------------------------------
    if provider == "litellm":
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "provider='litellm' requires the [litellm] optional dependency. "
                "Install with: pip install 'lit-monitor[litellm]'"
            ) from exc
        litellm_model = getattr(mode_config, "litellm_model", None)
        if not litellm_model:
            raise ValueError(
                f"provider='litellm' in mode {mode!r} requires 'litellm_model' "
                "to be set in extraction.yaml (e.g. 'anthropic/claude-haiku-4-5')."
            )
        context_window = getattr(mode_config, "context_window", None)
        return LiteLLMClient(
            model=litellm_model,
            timeout=timeout,
            temperature=temperature,
            num_ctx=int(context_window) if context_window is not None else None,
        )

    # ------------------------------------------------------------------
    # Ollama — local / default path
    # ------------------------------------------------------------------
    if provider != "ollama":
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            "Valid values: 'ollama' (default) or 'litellm' (requires [litellm] extra)."
        )
    # Per-mode YAML config can override think.  'think: false' in extraction.yaml for a mode
    # beats the caller's think=True default — used for models that don't support Ollama's
    # thinking mode (phi4-mini, qwen2.5:3b) and return 400 on think: true (Ollama 0.7+).
    mode_think = getattr(mode_config, "think", None)
    effective_think = mode_think if mode_think is not None else think
    # H5: num_ctx_override — explicit context window from extraction.yaml; None → use /api/show
    raw_override = getattr(mode_config, "num_ctx_override", None)
    num_ctx_override = int(raw_override) if isinstance(raw_override, (int, float)) and not isinstance(raw_override, bool) else None
    # K1: when pass_strategy="all" and pass_all_model is set, use it instead of model.
    effective_model = getattr(mode_config, "model", "qwen2.5:3b")
    if getattr(mode_config, "pass_strategy", "individual") == "all":
        _pass_all_model = getattr(mode_config, "pass_all_model", None)
        if _pass_all_model:
            effective_model = _pass_all_model
    return OllamaClient(
        model=effective_model,
        host=getattr(mode_config, "ollama_host", "http://localhost:11434"),
        timeout=timeout,
        temperature=temperature,
        num_ctx=_OLLAMA_NUM_CTX_DEFAULTS.get(mode),
        think=effective_think,
        num_ctx_override=num_ctx_override,
    )


def get_clients_for_passes(
    config, mode: str = "brain_build", think: bool = True
) -> LLMClient | dict[int, LLMClient]:
    """Return LLM client(s) for a pipeline mode, with optional per-pass model selection.

    When **no** per-pass model keys are set in extraction.yaml, this is equivalent
    to ``get_client()`` and returns a single ``LLMClient`` — zero overhead, full
    backward compatibility.

    When any per-pass model key is present, returns ``dict[int, LLMClient]`` mapping
    pass numbers 1–3 to their respective clients.  Passes without an explicit override
    fall back to the mode's default model.  All other settings (host, timeout,
    temperature, think, provider) are inherited from the mode config.

    Configuration keys in extraction.yaml (add under any mode section):

    *Ollama provider* (``provider: "ollama"``)::

        pass1_model: "qwen2.5:7b"   # larger model for core extraction
        pass2_model: "phi4-mini"    # methodology pass (omit = use default model)
        pass3_model: "phi4-mini"    # impact pass    (omit = use default model)

    *LiteLLM provider* (``provider: "litellm"``)::

        pass1_litellm_model: "anthropic/claude-sonnet-4-5"
        pass2_litellm_model: "anthropic/claude-haiku-4-5"
        pass3_litellm_model: "anthropic/claude-haiku-4-5"

    Parameters
    ----------
    config:
        Config object with per-mode attributes.
    mode:
        Pipeline mode — one of ``'brain_build'``, ``'ingestion'``,
        ``'build_vocabulary'``.
    think:
        Default thinking-mode flag for OllamaClient.  Overridden by the
        ``think:`` key in extraction.yaml if present.
    """
    mode_config = getattr(config, mode, None)
    if mode_config is None:
        raise ValueError(
            f"Unknown pipeline mode: {mode!r}. "
            "Expected one of: brain_build, ingestion, build_vocabulary"
        )
    provider = getattr(mode_config, "provider", "ollama")
    if provider not in ("ollama", "litellm"):
        raise ValueError(
            f"Unknown provider {provider!r} in mode {mode!r}. "
            "Expected 'ollama' or 'litellm'."
        )

    # Detect per-pass overrides based on provider.
    if provider == "litellm":
        pass_overrides = {
            n: getattr(mode_config, f"pass{n}_litellm_model", None) for n in (1, 2, 3)
        }
    else:
        pass_overrides = {
            n: getattr(mode_config, f"pass{n}_model", None) for n in (1, 2, 3)
        }

    # No per-pass overrides → single client (backward compatible, zero overhead).
    if not any(pass_overrides.values()):
        return get_client(config, mode=mode, think=think)

    # Shared settings — all per-pass clients inherit these from the mode config.
    raw_timeout = getattr(mode_config, "timeout", None)
    timeout = int(raw_timeout) if raw_timeout is not None else None
    temperature = float(getattr(mode_config, "temperature", 0.1))
    mode_think = getattr(mode_config, "think", None)
    effective_think = mode_think if mode_think is not None else think
    # H5: num_ctx_override applies uniformly across all passes
    raw_override = getattr(mode_config, "num_ctx_override", None)
    num_ctx_override = int(raw_override) if isinstance(raw_override, (int, float)) and not isinstance(raw_override, bool) else None

    clients: dict[int, LLMClient] = {}

    for pass_num in (1, 2, 3):
        if provider == "litellm":
            # Per-pass model, falling back to the mode-level litellm_model.
            pass_model = pass_overrides[pass_num] or getattr(mode_config, "litellm_model", None)
            if not pass_model:
                raise ValueError(
                    f"provider='litellm' in mode {mode!r} requires 'litellm_model' "
                    "(or 'pass{N}_litellm_model') to be set in extraction.yaml."
                )
            try:
                import litellm  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "provider='litellm' requires the [litellm] optional dependency. "
                    "Install with: pip install 'lit-monitor[litellm]'"
                ) from exc
            context_window = getattr(mode_config, "context_window", None)
            clients[pass_num] = LiteLLMClient(
                model=pass_model,
                timeout=timeout,
                temperature=temperature,
                num_ctx=int(context_window) if context_window is not None else None,
            )
        else:  # ollama
            # Per-pass model, falling back to the mode default.
            pass_model = pass_overrides[pass_num] or getattr(mode_config, "model", "qwen2.5:3b")
            clients[pass_num] = OllamaClient(
                model=pass_model,
                host=getattr(mode_config, "ollama_host", "http://localhost:11434"),
                timeout=timeout,
                temperature=temperature,
                num_ctx=_OLLAMA_NUM_CTX_DEFAULTS.get(mode),
                think=effective_think,
                num_ctx_override=num_ctx_override,
            )

    logger.debug(
        "Per-pass clients for mode %r: %s",
        mode,
        {n: getattr(c, "model", "?") for n, c in clients.items()},
    )
    return clients
