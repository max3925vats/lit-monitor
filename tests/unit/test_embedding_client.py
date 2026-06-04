"""Unit tests for scripts.llm.embedding_client.embed_via_ollama.

Q3.2: a 200 response with a malformed (non-JSON) body must surface as a
clear, typed RuntimeError — never a bare json.JSONDecodeError leaking out of
the HTTP layer.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from scripts.llm.embedding_client import embed_via_ollama


def _fake_urlopen_returning(body_bytes: bytes):
    """Return a urlopen replacement that yields a 200 response whose .read()
    produces ``body_bytes``.
    """

    def _fake(req, timeout=60):
        class _Resp:
            def __enter__(self_):  # noqa: N805
                return self_

            def __exit__(self_, *a):  # noqa: N805
                return False

            def read(self_):  # noqa: N805
                return body_bytes

        return _Resp()

    return _fake


@pytest.mark.unit
def test_malformed_body_raises_clear_runtime_error():
    """200 + non-JSON body → RuntimeError mentioning the cause, not a bare
    json.JSONDecodeError.
    """
    garbage = b"<html>502 Bad Gateway</html>"
    with patch(
        "urllib.request.urlopen",
        side_effect=_fake_urlopen_returning(garbage),
    ):
        with pytest.raises(RuntimeError, match="non-JSON response"):
            embed_via_ollama("some text", model="mxbai-embed-large")


@pytest.mark.unit
def test_malformed_body_does_not_leak_jsondecodeerror():
    """The raised error must NOT be a bare json.JSONDecodeError."""
    with patch(
        "urllib.request.urlopen",
        side_effect=_fake_urlopen_returning(b"not json at all"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            embed_via_ollama("text", model="m")
    # JSONDecodeError is a ValueError subclass; the public surface must be the
    # typed RuntimeError, not the raw decode error.
    assert not isinstance(exc_info.value, json.JSONDecodeError)


@pytest.mark.unit
def test_valid_body_returns_vector():
    """Regression guard: a well-formed JSON body still parses to a vector."""
    good = json.dumps({"embeddings": [[0.1, 0.2, 0.3]]}).encode()
    with patch(
        "urllib.request.urlopen",
        side_effect=_fake_urlopen_returning(good),
    ):
        vec = embed_via_ollama("text", model="m")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    np.testing.assert_allclose(vec, [0.1, 0.2, 0.3], rtol=1e-6)
