"""Bundle F: cross-provider embedding dispatch.

Provides two pure functions — embed_via_ollama() and embed_via_litellm() —
that EmbeddingsDB calls based on the configured provider.  Each function
handles its own import and error handling; callers never import provider
libraries directly.

Usage
-----
Both functions return a 1-D float32 numpy array.  Dimension is determined
by the model, not hardcoded here.

Ollama (default)
    from lit_monitor.llm.embedding_client import embed_via_ollama
    vec = embed_via_ollama("text", model="mxbai-embed-large")

LiteLLM (any compatible provider)
    from lit_monitor.llm.embedding_client import embed_via_litellm
    vec = embed_via_litellm("text", model="text-embedding-3-large")
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def embed_via_ollama(
    text: str,
    *,
    model: str,
    host: str = "http://localhost:11434",
) -> np.ndarray:
    """Bundle F: embed via local Ollama using the /api/embed endpoint.

    This is the same HTTP call that was previously inlined in
    EmbeddingsDB._embed_call(), extracted here for provider symmetry.
    Reads OLLAMA_API_KEY from the environment and adds a Bearer
    Authorization header when present (required for cloud Ollama
    endpoints; local Ollama ignores the extra header).

    Parameters
    ----------
    text:
        Input string to embed.
    model:
        Ollama model name, e.g. 'mxbai-embed-large'.
    host:
        Ollama API base URL.  Trailing slash is stripped automatically.

    Returns
    -------
    np.ndarray
        1-D float32 embedding vector.

    Raises
    ------
    urllib.error.HTTPError
        On any non-200 response that is not a context-length error.
    ValueError
        If Ollama returns an empty or malformed embedding payload.
    """
    host = host.rstrip("/")
    payload = json.dumps({"model": model, "input": text}).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url=f"{host}/api/embed",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            raw_body = resp.read()
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.error(
            "embed_via_ollama: HTTP %d model=%s len=%d body=%r",
            exc.code, model, len(text), err_body,
        )
        raise

    # A 200 response with a non-JSON body (proxy error page, truncated
    # stream, etc.) must not surface as a bare json.JSONDecodeError — wrap it
    # in a typed RuntimeError matching how the rest of the module raises.
    try:
        body: dict[str, Any] = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError) as exc:
        snippet = raw_body.decode("utf-8", errors="replace")[:500] if raw_body else ""
        raise RuntimeError(
            f"embed_via_ollama: Ollama embedding returned non-JSON response "
            f"(model={model!r}): {exc}. Body={snippet!r}"
        ) from exc

    embeddings = body.get("embeddings") or body.get("embedding")
    if not embeddings:
        raise ValueError(
            f"embed_via_ollama: empty/missing embedding in response "
            f"(keys={list(body.keys())!r})"
        )
    # Ollama /api/embed returns {"embeddings": [[float, ...]]} (batched) or
    # {"embedding": [float, ...]} (older single-input form).
    raw: list[float] = embeddings[0] if isinstance(embeddings[0], list) else embeddings
    return np.array(raw, dtype=np.float32)


def embed_via_litellm(
    text: str,
    *,
    model: str,
) -> np.ndarray:
    """Bundle F: embed via LiteLLM (any compatible provider — OpenAI, Cohere, etc.).

    Requires the [litellm] optional dependency:
        uv sync --extra litellm

    API keys are passed via environment variables per LiteLLM provider docs
    (e.g. OPENAI_API_KEY, COHERE_API_KEY).  LiteLLM reads them automatically.

    Parameters
    ----------
    text:
        Input string to embed.
    model:
        LiteLLM model string, e.g. 'text-embedding-3-large',
        'cohere/embed-english-v3.0', 'voyage/voyage-3'.

    Returns
    -------
    np.ndarray
        1-D float32 embedding vector.

    Raises
    ------
    ImportError
        If litellm is not installed (with install hint).
    RuntimeError
        If the LiteLLM call fails or the response shape is unexpected.
    """
    try:
        import litellm  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "embedding provider 'litellm' requires the [litellm] optional dependency. "
            "Install with: uv sync --extra litellm"
        ) from exc

    if not hasattr(litellm, "embedding"):
        raise ImportError(
            "litellm.embedding is not available in this version of litellm. "
            "Upgrade with: uv sync --extra litellm"
        )

    try:
        response = litellm.embedding(model=model, input=[text])
    except Exception as exc:
        raise RuntimeError(
            f"embed_via_litellm: LiteLLM call failed for model={model!r}: {exc}"
        ) from exc

    # LiteLLM response: {"data": [{"embedding": [float, ...]}, ...], ...}
    try:
        vec: list[float] = response["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"embed_via_litellm: unexpected response shape from model={model!r}: {exc}. "
            f"Keys={list(response.keys()) if hasattr(response, 'keys') else type(response)!r}"
        ) from exc

    return np.array(vec, dtype=np.float32)
