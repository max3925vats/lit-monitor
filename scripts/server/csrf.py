"""Origin/Host guard middleware for the localhost-only web server (AR-1).

Threat model
------------
The lit-monitor web server binds to localhost and ships with NO authentication
-- it trusts that only the local user can reach it. That trust is broken by two
classic browser attacks against localhost services:

1. **Cross-origin CSRF.** State-changing endpoints accept ``Form(...)`` bodies.
   Form-encoded POSTs are "simple requests": the browser sends them
   cross-origin WITHOUT a CORS preflight. So any webpage the user visits can
   blind-fire ``fetch("http://127.0.0.1:8765/api/credentials",
   {method: "POST", mode: "no-cors", body: formData})`` and overwrite
   credentials, start a pipeline, or rewrite the schedule. The browser *does*
   attach an ``Origin`` header naming the attacker's site on every such
   cross-origin request -- that header is the tell.

2. **DNS rebinding.** An attacker can point a hostname they control at
   ``127.0.0.1`` and lure the browser into sending requests whose ``Host``
   header is the attacker's domain. Pinning the allowed ``Host`` set defeats
   this.

Design
------
For any non-safe method (anything but GET/HEAD/OPTIONS):

  * Reject when the ``Host`` header is not a known-local host (DNS-rebinding
    defense).
  * Then, *only if* an ``Origin`` header is present, reject when its host is
    not a known-local host (CSRF defense).

Requests with NO ``Origin`` header are ALLOWED. This is deliberate: curl, the
CLI tools, and some same-origin browser POSTs omit Origin, and we must not
break them. Browsers ALWAYS attach Origin on the cross-origin POSTs that are
the actual attack vector, so allowing the absent-Origin case loses no
protection. Safe methods (GET/HEAD/OPTIONS) are never blocked.

This is intentionally simpler than per-form CSRF tokens: a token scheme would
need plumbing through 50+ forms for no extra security on a single-user
localhost tool.
"""
from __future__ import annotations

from urllib.parse import urlparse

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

# Methods that never mutate state -- always allowed through the guard.
_SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})

# Hosts we treat as "local". ``testserver`` is FastAPI's TestClient default
# Host; without it the entire route test suite would 403 (TestClient sends no
# Origin, so only the Host check would fire). ``[::1]`` covers IPv6 loopback in
# both bracketed and bare forms.
_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {"127.0.0.1", "localhost", "::1", "[::1]", "testserver"}
)


def _host_only(raw: str) -> str:
    """Return the lowercased hostname from a Host-header value, port stripped.

    Handles bracketed IPv6 literals (``[::1]:8765`` -> ``[::1]``) as well as
    the common ``host:port`` form. An empty/absent value yields ``""``.
    """
    value = (raw or "").strip().lower()
    if not value:
        return ""
    # IPv6 literal: keep everything up to and including the closing bracket.
    if value.startswith("["):
        end = value.find("]")
        if end != -1:
            return value[: end + 1]
        return value
    # IPv4 / hostname: strip an optional :port suffix.
    return value.split(":", 1)[0]


class LocalOriginMiddleware(BaseHTTPMiddleware):
    """Reject cross-origin / non-local state-changing requests.

    See the module docstring for the full threat model. The middleware must be
    installed BEFORE the routers so it runs on every request.
    """

    def __init__(self, app, allowed_hosts: frozenset[str] | None = None) -> None:
        """Build the guard.

        Args:
            app: The wrapped ASGI application.
            allowed_hosts: Optional override/extension of the local-host set.
                Used when the server is bound to a non-localhost address so the
                configured bind host counts as "local". Defaults to the
                loopback set when ``None``.
        """
        super().__init__(app)
        self._allowed_hosts = allowed_hosts or _ALLOWED_HOSTS

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply the Host/Origin checks, then defer to the next handler."""
        if request.method not in _SAFE_METHODS:
            host = _host_only(request.headers.get("host"))
            if host not in self._allowed_hosts:
                return PlainTextResponse(
                    "forbidden: non-local Host", status_code=403
                )
            origin = request.headers.get("origin")
            if origin:
                origin_host = (urlparse(origin).hostname or "").lower()
                if origin_host not in self._allowed_hosts:
                    return PlainTextResponse(
                        "forbidden: cross-origin request", status_code=403
                    )
        return await call_next(request)


__all__ = ["LocalOriginMiddleware"]
