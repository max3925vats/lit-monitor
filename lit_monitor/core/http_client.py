"""
HTTP client — thin requests + tenacity wrapper.
Used for direct API calls (Europe PMC, CrossRef, etc.)
that are not handled by findpapers or pyzotero.
"""
from __future__ import annotations

import logging
from importlib.metadata import version as _pkg_version
from typing import Any

try:
    _UA_VERSION = _pkg_version("lit-monitor")
except Exception:
    _UA_VERSION = "dev"

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": f"lit-monitor/{_UA_VERSION} (personal research tool; contact via GitHub)"
})
_RETRY_DECORATOR = retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
@_RETRY_DECORATOR
def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> dict:
    """GET a URL, return parsed JSON.

    ``timeout`` is the per-request socket timeout in seconds and can be
    overridden per call (H3: previously hardcoded).
    """
    resp = _SESSION.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
