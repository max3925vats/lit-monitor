"""R1: generalized route-collision regression test.

Catches the class of bug that hit P3 between P8 and P11: a path-PARAM route
(e.g. {run_id} with `run_id: int`, {doi:path} in the path string, or a plain
default-str {slug}) registered BEFORE a more-specific static-segment sibling on
the same parent prefix.

FastAPI matches routes in registration order. If a param route fires first it
shadows the static sibling: a typed param fails coercion and returns 422; a
default-str param matches *any* value and silently swallows the request so the
static handler never runs (P7.6 — the original detector only caught the typed
case). This file ships:

  1. _collect_route_failures(app) — shared walker used by multiple tests.
  2. test_no_typed_param_routes_shadow_static_siblings — live app check.
  3. test_notify_handler_path_specifically — targeted regression for P3/P8.
  4. test_walker_flags_synthetic_broken_app — proves the walker is non-vacuous.
  5. test_walker_flags_default_str_param_shadowing — P7.6 str-param coverage.
  6. test_walker_passes_correct_static_first_order — false-positive guard.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

# Matches explicit type annotations inside path parameters, e.g. {doi:path},
# {id:int}, {uid:uuid}, {score:float}. The plain {name} form (no colon) is
# detected separately via the dependant type-adapter inspection below.
_EXPLICIT_TYPED_PARAM_RE = re.compile(
    r"\{[^}/]+:(?:int|float|path|uuid|str)\}"
)

# Recognise any path-parameter segment (plain or typed).
_ANY_PARAM_RE = re.compile(r"^\{[^}]+\}$")


def _last_segment(path: str) -> str:
    """Return the final slash-delimited segment of a route path."""
    return path.rsplit("/", 1)[-1] if "/" in path else path


def _parent_prefix(path: str) -> str:
    """Return everything before the last segment.

    /discovery/{run_id}       → /discovery
    /discovery/notify-handler → /discovery
    /api/papers/{doi:path}    → /api/papers
    """
    parts = path.rsplit("/", 1)
    return parts[0] if len(parts) > 1 else ""


def _is_param_last_segment(route: object) -> bool:
    """Return True if the route's last path segment is ANY path parameter.

    This covers both shadowing flavours:
    - Typed params (``{id:int}``, ``{run_id}`` with ``run_id: int``): FastAPI
      matches the param route first, then *fails coercion* on a non-matching
      request and returns 422 — the static sibling registered later is
      unreachable.
    - Default-``str`` params (``{slug}`` with ``slug: str``, FastAPI's default):
      this is the MORE dangerous case. A str param matches *any* value
      (including the literal ``notify-handler``), so the param route silently
      swallows every request and the static sibling never 422s — it just never
      runs. The original detector only flagged the typed case and missed this.

    A plain ``{param}`` last segment is detected purely from the path string;
    no dependant introspection is needed because str-vs-int no longer changes
    the verdict (both shadow a later static sibling on the same prefix).
    """
    path: str = getattr(route, "path", "") or ""
    last = _last_segment(path)

    # Explicit colon-typed placeholder, e.g. {doi:path} / {id:int}.
    if _EXPLICIT_TYPED_PARAM_RE.match(last):
        return True

    # Any plain placeholder last segment, e.g. {run_id} or {slug} (default str).
    return bool(_ANY_PARAM_RE.match(last))


def _collect_route_failures(app: FastAPI) -> list[str]:
    """Walk *app.routes* and return descriptions of any shadowing collisions.

    A collision exists when a path-PARAM route at index I (whose last segment is
    a placeholder — typed OR default-str — matching requests to its parent
    prefix) is registered BEFORE a static-segment sibling at index J > I on the
    same (method, parent_prefix) group. FastAPI resolves routes in registration
    order, so the param route wins and the static sibling is unreachable (a
    typed param 422s on coercion; a str param silently swallows the request).

    Returns a list of human-readable failure strings (empty = no collisions).
    """
    # (method, parent_prefix) → list of (index, last_segment, is_param, path)
    by_group: dict[tuple[str, str], list[tuple[int, str, bool, str]]] = (
        defaultdict(list)
    )

    for index, route in enumerate(app.routes):
        path: str = getattr(route, "path", None) or ""
        if not path:
            continue
        methods = getattr(route, "methods", None) or set()
        last = _last_segment(path)
        is_param = _is_param_last_segment(route)
        for method in methods:
            by_group[(method, _parent_prefix(path))].append(
                (index, last, is_param, path)
            )

    failures: list[str] = []
    for (method, prefix), entries in by_group.items():
        param_entries = [e for e in entries if e[2]]  # (idx, last, True, path)
        if not param_entries:
            continue
        for p_idx, _p_last, _, p_path in param_entries:
            for o_idx, _o_last, o_is_param, o_path in entries:
                if o_is_param or o_idx <= p_idx:
                    # Skip: other is also a param route, or registered before
                    # the param one (and therefore reachable).
                    continue
                # Static route at o_idx > p_idx: unreachable — the earlier
                # param route matches first.
                failures.append(
                    f"{method} {o_path!r} (index {o_idx}) is shadowed by "
                    f"{p_path!r} (index {p_idx}) — register the static route first."
                )
    return failures


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_typed_param_routes_shadow_static_siblings() -> None:
    """R1: live app must have no typed-param routes shadowing static siblings.

    If this fails, find the offending include_router pair in
    scripts/server/app.py and ensure the static-segment router is included
    BEFORE the typed-param router.
    """
    from scripts.server.app import create_app

    failures = _collect_route_failures(create_app())
    assert not failures, (
        "Route-order shadowing detected:\n"
        + "\n".join(f"  • {f}" for f in failures)
    )


def test_notify_handler_path_specifically() -> None:
    """R1: targeted regression — the exact P3 ↔ P8 collision path.

    /discovery/notify-handler must be reachable. A 422 response means P8's
    /discovery/{run_id} matched first and failed int coercion — the static
    handler was never reached.
    """
    from fastapi.testclient import TestClient

    from scripts.server.app import create_app

    client = TestClient(create_app(), follow_redirects=False)
    r = client.get("/discovery/notify-handler?run_id=1")
    assert r.status_code != 422, (
        "P3 /discovery/notify-handler is shadowed by P8's "
        "/discovery/{run_id:int}. Check the include_router order in "
        "scripts/server/app.py — discovery_notify_router MUST be included "
        "BEFORE discovery_router."
    )
    # P3 handler returns 200 (chooser HTML), 302/307 (redirect), or 204 (none pref).
    assert r.status_code in (200, 204, 302, 303, 307, 308), (
        f"unexpected status {r.status_code} from /discovery/notify-handler"
    )


def test_walker_flags_synthetic_broken_app() -> None:
    """R1: the walker must flag a known-broken registration order.

    Constructs a minimal FastAPI app with the collision that broke P3: the
    typed-param route /discovery/{run_id} (run_id: int) registered BEFORE
    the static route /discovery/notify-handler. Asserts the walker returns
    at least one failure. If the walker returns no failures here, it would be
    vacuous — it could not catch the real regression.
    """
    from fastapi import APIRouter, FastAPI

    bad_app = FastAPI()

    # Bad order: typed-param sibling registered FIRST — mirrors the P8 bug.
    bad_router = APIRouter()

    @bad_router.get("/discovery/{run_id}")
    def _detail(run_id: int) -> dict:  # type: ignore[return]
        return {}

    bad_app.include_router(bad_router)

    notify_router = APIRouter()

    @notify_router.get("/discovery/notify-handler")
    def _notify() -> dict:  # type: ignore[return]
        return {}

    bad_app.include_router(notify_router)

    failures = _collect_route_failures(bad_app)
    assert failures, (
        "The route-collision walker did NOT flag the known-broken layout "
        "(typed-param before static sibling). The walker is vacuous and "
        "would not catch the P3↔P8 regression."
    )


def test_walker_flags_default_str_param_shadowing() -> None:
    """R1 (P7.6): default-str path params must ALSO be flagged.

    A ``{slug}`` param defaults to ``str`` in FastAPI, which matches *any* value
    — so it silently swallows a static sibling registered afterwards (the static
    route never even 422s; it just never runs). This is the more dangerous case
    the original detector missed because it only inspected for int/float/uuid.
    """
    from fastapi import APIRouter, FastAPI

    bad_app = FastAPI()

    # Default-str param sibling registered FIRST — no explicit type annotation.
    bad_router = APIRouter()

    @bad_router.get("/themes/{slug}")
    def _detail(slug: str) -> dict:  # str matches everything, incl. "export"
        return {}

    bad_app.include_router(bad_router)

    # Static sibling registered AFTER → unreachable.
    static_router = APIRouter()

    @static_router.get("/themes/export")
    def _export() -> dict:
        return {}

    bad_app.include_router(static_router)

    failures = _collect_route_failures(bad_app)
    assert failures, (
        "The walker did NOT flag a default-str {slug} param shadowing a later "
        "static /themes/export sibling — the most dangerous (silent) collision."
    )


def test_walker_passes_correct_static_first_order() -> None:
    """R1 (P7.6): a correctly-ordered app (static BEFORE param) is not flagged.

    Guards against false positives introduced by broadening the detector to
    default-str params: when the static route is registered first it IS
    reachable, so no failure must be reported.
    """
    from fastapi import APIRouter, FastAPI

    good_app = FastAPI()

    static_router = APIRouter()

    @static_router.get("/themes/export")
    def _export() -> dict:
        return {}

    good_app.include_router(static_router)  # static FIRST → reachable

    param_router = APIRouter()

    @param_router.get("/themes/{slug}")
    def _detail(slug: str) -> dict:
        return {}

    good_app.include_router(param_router)

    assert _collect_route_failures(good_app) == []
