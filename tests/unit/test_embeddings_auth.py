"""Regression tests for OLLAMA_API_KEY → Authorization: Bearer plumbing.

Without these, a future refactor of ``EmbeddingsDB._embed_call`` could
silently drop the Authorization header, which would only surface when a
user routes embeddings through a cloud-authenticated Ollama endpoint
(e.g. ``mxbai-embed-large:cloud``) — at which point the request returns
200 with an empty ``embeddings`` array and triggers a confusing
"Empty or missing embeddings in Ollama response" ValueError.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts.output.embeddings import EmbeddingsDB


def _fake_urlopen_success(captured: list):
    """Return a urlopen replacement that records the Request and returns
    a successful Ollama embed response.
    """

    def _fake(req, timeout=60):
        captured.append(req)

        class _Resp:
            def __enter__(self_):  # noqa: N805
                return self_

            def __exit__(self_, *a):  # noqa: N805
                return False

            def read(self_):  # noqa: N805
                return json.dumps({"embeddings": [[0.1] * 1024]}).encode()

        return _Resp()

    return _fake


@pytest.mark.unit
def test_embed_call_sets_authorization_header_from_env(tmp_path, monkeypatch):
    """When OLLAMA_API_KEY is in the environment, _embed_call must add the
    Authorization: Bearer header. Required for cloud-routed embed models.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key-abc123")
    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))

    captured: list = []
    with patch("urllib.request.urlopen", side_effect=_fake_urlopen_success(captured)):
        result = db._embed_call("any input text")

    assert result == [0.1] * 1024
    assert len(captured) == 1
    headers = captured[0].headers
    # urllib lowercases header names when accessed via .headers
    auth = headers.get("Authorization") or headers.get("authorization")
    assert auth == "Bearer test-key-abc123", (
        f"Expected Bearer token from env, got: {auth!r}. Full headers: {headers!r}"
    )


@pytest.mark.unit
def test_embed_call_omits_authorization_header_when_env_unset(tmp_path, monkeypatch):
    """When OLLAMA_API_KEY is NOT in the environment, _embed_call must not
    add an Authorization header. Local Ollama doesn't need it and stray
    headers can confuse some proxy configurations.
    """
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    db = EmbeddingsDB(persist_dir=str(tmp_path / "chroma"))

    captured: list = []
    with patch("urllib.request.urlopen", side_effect=_fake_urlopen_success(captured)):
        db._embed_call("any input text")

    assert len(captured) == 1
    headers = captured[0].headers
    auth = headers.get("Authorization") or headers.get("authorization")
    assert auth is None, (
        f"Expected no Authorization header when env var unset, got: {auth!r}"
    )
