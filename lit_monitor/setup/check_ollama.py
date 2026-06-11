"""
Check that Ollama is running and the configured model is available.
Returns a dict of {check_name: (ok: bool, message: str)}.
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

# Literal fallback used only when config is unavailable. The probe normally
# targets the configured Ollama host (see _resolve_ollama_host).
_OLLAMA_BASE = "http://localhost:11434"
_TIMEOUT = 5


def _resolve_ollama_host() -> str:
    """Resolve the configured Ollama host, falling back to localhost.

    Mirrors the host-resolution precedence in ``scripts.cli._make_embeddings_db``:
    ``embedding.ollama.host`` (Bundle F sub-block) → ``embeddings.ollama_host`` →
    ``brain_build.ollama_host`` → the literal localhost default. Config loading is
    lazy and fully guarded so this still works when config is absent.
    """
    try:
        from lit_monitor.core.config import get_config
        cfg = get_config()
    except Exception:  # noqa: BLE001 — config may be absent/unreadable in setup checks
        return _OLLAMA_BASE

    # Bundle F provider sub-block: embedding.ollama.host (preferred).
    try:
        emb_cfg = getattr(cfg, "embedding", None)
        if emb_cfg is not None:
            provider = getattr(emb_cfg, "provider", "ollama")
            provider_sub = getattr(emb_cfg, provider, None)
            if provider == "ollama" and provider_sub is not None:
                host = getattr(provider_sub, "host", None)
                if host:
                    return str(host)
    except Exception:  # noqa: BLE001 — defensive; namespace shape may vary
        pass

    # Legacy embeddings.ollama_host, then brain_build.ollama_host.
    try:
        host = getattr(getattr(cfg, "embeddings", None), "ollama_host", None)
        if host:
            return str(host)
    except Exception:  # noqa: BLE001
        pass
    try:
        host = getattr(getattr(cfg, "brain_build", None), "ollama_host", None)
        if host:
            return str(host)
    except Exception:  # noqa: BLE001
        pass

    return _OLLAMA_BASE


def check_ollama(model: str | None = None) -> dict[str, tuple[bool, str]]:
    """
    Return {check_name: (ok, message)} for each Ollama check.
    Does not raise — always returns results.
    Args:
        model: Optional model name to verify availability (e.g. "mistral:7b").
               If None, only checks that Ollama is reachable.
    """
    results: dict[str, tuple[bool, str]] = {}
    base = _resolve_ollama_host()
    # --- Reachability ---
    try:
        resp = requests.get(f"{base}/api/tags", timeout=_TIMEOUT)
        resp.raise_for_status()
        results["ollama_running"] = (True, f"Ollama reachable at {base}")
        available_models: list[str] = [
            m["name"] for m in resp.json().get("models", [])
        ]
    except requests.exceptions.ConnectionError:
        results["ollama_running"] = (
            False,
            f"Cannot reach Ollama at {base} — is `ollama serve` running?",
        )
        if model:
            results["model_available"] = (False, "Ollama not running")
        return results
    except Exception:
        # Info-leak guard: log the detail; surface a constant message so no
        # exception text reaches the response (py/stack-trace-exposure).
        logger.warning("check_ollama: Ollama check failed", exc_info=True)
        results["ollama_running"] = (False, "Ollama check failed — see server logs.")
        if model:
            results["model_available"] = (False, "Ollama not running")
        return results
    # --- Model availability ---
    if model:
        # Match on exact name, exact ":latest" tag, or the same base name
        # before the tag. Comparing split(":")[0] == model keeps the rule
        # exact: "mistral" matches "mistral:7b" / "mistral:latest" but NOT
        # "mistral-nemo:latest". The old startswith(f"{model}:") check matched
        # too broadly because it compared the whole tagged name as a prefix.
        found = any(
            m == model
            or m == f"{model}:latest"
            or m.split(":")[0] == model
            for m in available_models
        )
        if found:
            results["model_available"] = (True, f"Model '{model}' is available")
        else:
            results["model_available"] = (
                False,
                f"Model '{model}' not found. Available: {available_models or ['(none)']}\n"
                f"  Run: ollama pull {model}",
            )
    return results
