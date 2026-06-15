"""Tests for scripts/core/http_client.py — the shared requests+tenacity GET helper.

These pin the wrapper's OWN behaviour (JSON parse, arg passthrough, the
tenacity retry policy) by mocking only the network boundary (`_SESSION.get`).
Nothing here mocks `get_json` itself, so each test exercises the real code path.

The retry config is `stop_after_attempt(3)` + `reraise=True`, retrying on
`requests.RequestException`. Backoff sleeps are neutralised per-test via the
tenacity `Retrying` object's `.sleep` hook so the suite stays fast.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_monitor.core import http_client


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    """Neutralise tenacity's exponential backoff sleep (min=2s) so retry tests
    don't actually block. `get_json.retry` is the tenacity Retrying instance."""
    original = http_client.get_json.retry.sleep
    http_client.get_json.retry.sleep = lambda *_a, **_k: None
    yield
    http_client.get_json.retry.sleep = original


def _ok_response(payload: dict) -> MagicMock:
    """A fake requests.Response: raise_for_status no-ops, json() returns payload."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


# ---------------------------------------------------------------------------
# Happy path + argument passthrough
# ---------------------------------------------------------------------------
def test_returns_parsed_json_on_success():
    resp = _ok_response({"hello": "world"})
    with patch.object(http_client._SESSION, "get", return_value=resp) as mock_get:
        result = http_client.get_json("https://api.example/x")
    assert result == {"hello": "world"}
    mock_get.assert_called_once()
    # raise_for_status must be checked before .json() is read (4xx/5xx → error path)
    resp.raise_for_status.assert_called_once()
    resp.json.assert_called_once()


def test_passes_params_and_url_through():
    resp = _ok_response({})
    with patch.object(http_client._SESSION, "get", return_value=resp) as mock_get:
        http_client.get_json("https://api.example/y", params={"q": "abc", "n": 3})
    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.example/y"
    assert kwargs["params"] == {"q": "abc", "n": 3}


def test_default_timeout_is_15s():
    """H3 contract: default per-request socket timeout is 15.0s."""
    resp = _ok_response({})
    with patch.object(http_client._SESSION, "get", return_value=resp) as mock_get:
        http_client.get_json("https://api.example/z")
    assert mock_get.call_args.kwargs["timeout"] == 15.0


def test_caller_can_override_timeout():
    """H3 contract: timeout is overridable per call (was previously hardcoded)."""
    resp = _ok_response({})
    with patch.object(http_client._SESSION, "get", return_value=resp) as mock_get:
        http_client.get_json("https://api.example/z", timeout=42.0)
    assert mock_get.call_args.kwargs["timeout"] == 42.0


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
def test_retries_transient_network_error_then_succeeds():
    """A ConnectionError (a RequestException) is retried; a later success wins.
    Proves the retry actually re-invokes — exactly 3 calls (2 failures + 1 ok)."""
    resp = _ok_response({"ok": True})
    side_effects = [
        requests.ConnectionError("boom 1"),
        requests.ConnectionError("boom 2"),
        resp,
    ]
    with patch.object(http_client._SESSION, "get", side_effect=side_effects) as mock_get:
        result = http_client.get_json("https://api.example/retry")
    assert result == {"ok": True}
    assert mock_get.call_count == 3


def test_gives_up_and_reraises_after_three_attempts():
    """Persistent RequestException → reraised (reraise=True) after exactly 3 tries,
    not swallowed and not retried forever."""
    with patch.object(
        http_client._SESSION, "get",
        side_effect=requests.ConnectionError("always down"),
    ) as mock_get:
        with pytest.raises(requests.ConnectionError, match="always down"):
            http_client.get_json("https://api.example/dead")
    assert mock_get.call_count == 3


def test_http_error_from_raise_for_status_is_retried():
    """raise_for_status() raises HTTPError, which IS a RequestException — so a
    persistent 5xx is retried 3× then reraised. Pins this (easy-to-miss) behaviour."""
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    with patch.object(http_client._SESSION, "get", return_value=resp) as mock_get:
        with pytest.raises(requests.HTTPError, match="500"):
            http_client.get_json("https://api.example/500")
    assert mock_get.call_count == 3


def test_non_request_exception_is_not_retried():
    """A JSON-decode failure (ValueError) is NOT a RequestException, so it must
    propagate immediately with NO retry — guards against over-broad retrying."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.side_effect = ValueError("not json")
    with patch.object(http_client._SESSION, "get", return_value=resp) as mock_get:
        with pytest.raises(ValueError, match="not json"):
            http_client.get_json("https://api.example/badjson")
    assert mock_get.call_count == 1


@pytest.mark.unit
def test_user_agent_is_not_hardcoded_legacy_version():
    from lit_monitor.core import http_client
    ua = http_client._SESSION.headers["User-Agent"]
    assert ua.startswith("lit-monitor/")
    assert "0.1.0" not in ua
