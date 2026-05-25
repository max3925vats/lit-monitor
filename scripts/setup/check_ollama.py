"""
Check that Ollama is running and the configured model is available.
Returns a dict of {check_name: (ok: bool, message: str)}.
"""
from __future__ import annotations

import requests

_OLLAMA_BASE = "http://localhost:11434"
_TIMEOUT = 5
def check_ollama(model: str | None = None) -> dict[str, tuple[bool, str]]:
    """
    Return {check_name: (ok, message)} for each Ollama check.
    Does not raise — always returns results.
    Args:
        model: Optional model name to verify availability (e.g. "mistral:7b").
               If None, only checks that Ollama is reachable.
    """
    results: dict[str, tuple[bool, str]] = {}
    # --- Reachability ---
    try:
        resp = requests.get(f"{_OLLAMA_BASE}/api/tags", timeout=_TIMEOUT)
        resp.raise_for_status()
        results["ollama_running"] = (True, f"Ollama reachable at {_OLLAMA_BASE}")
        available_models: list[str] = [
            m["name"] for m in resp.json().get("models", [])
        ]
    except requests.exceptions.ConnectionError:
        results["ollama_running"] = (
            False,
            f"Cannot reach Ollama at {_OLLAMA_BASE} — is `ollama serve` running?",
        )
        if model:
            results["model_available"] = (False, "Ollama not running")
        return results
    except Exception as exc:
        results["ollama_running"] = (False, f"Ollama check failed: {exc}")
        if model:
            results["model_available"] = (False, "Ollama not running")
        return results
    # --- Model availability ---
    if model:
        # Normalise: "mistral" matches "mistral:latest"
        normalised = model if ":" in model else f"{model}:latest"
        found = any(
            m == model or m == normalised or m.startswith(f"{model}:")
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
