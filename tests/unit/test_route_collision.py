"""R1: generalized route-collision regression test.

Catches the class of bug that hit P3 between P8 and P11: a typed-param
route (e.g. {run_id} with `run_id: int` in the signature, or {doi:path}
in the path string) registered BEFORE a more-specific static-segment
sibling on the same parent prefix.

FastAPI matches routes in registration order. If a typed-param route fires
first, it fails its type coercion and returns 422 — without falling through
to the static-segment handler. This file ships:

  1. _collect_route_failures(app) — shared walker used by multiple tests.
  2. test_no_typed_param_routes_shadow_static_siblings — live app check.
  3. test_notify_handler_path_specifically — targeted regression for P3/P8.
  4. test_walker_flags_synthetic_broken_app — proves the walker is non-vacuous.
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


def _is_typed_last_segment(route: object) -> bool:
    """Return True if the route's last path segment is a typed (non-str) param.

    Two detection paths:
    - Explicit: the path string contains a colon-typed placeholder in its last
      segment, e.g. ``{doi:path}`` or ``{id:int}``.
    - Implicit: the last segment is a plain placeholder like ``{run_id}`` but
      the FastAPI dependant reveals its type as int / float / uuid / etc.
    """
    path: str = getattr(route, "path", "") or ""
    last = _last_segment(path)

    # Fast path: explicit type in the path string itself.
    if _EXPLICIT_TYPED_PARAM_RE.match(last):
        return True

    # Only proceed if the last segment looks like any path param at all.
    if not _ANY_PARAM_RE.match(last):
        return False

    # Derive the param name from the segment (strip braces / type suffix).
    raw = last[1:-1]  # strip { }
    param_name = raw.split(":")[0]  # drop any type suffix

    # Walk the dependant's path_params to find a non-str type adapter.
    dep = getattr(route, "dependant", None)
    if dep is None:
        return False
    for p in getattr(dep, "path_params", []):
        if p.name != param_name:
            continue
        # _type_adapter repr contains the python type name.
        ta = repr(getattr(p, "_type_adapter", ""))
        # Flag int, float, and UUID; leave plain str as non-typed.
        if any(tok in ta for tok in ("int", "float", "uuid", "UUID")):
            return True
    return False


def _collect_route_failures(app: FastAPI) -> list[str]:
    """Walk *app.routes* and return descriptions of any shadowing collisions.

    A collision exists when a typed-param route at index I (matching any
    request to its parent prefix) is registered BEFORE a static-segment
    sibling at index J > I on the same (method, parent_prefix) group.
    FastAPI will resolve the typed-param route first, fail coercion, and
    return 422 — the static sibling is unreachable.

    Returns a list of human-readable failure strings (empty = no collisions).
    """
    # (method, parent_prefix) → list of (index, last_segment, is_typed, path)
    by_group: dict[tuple[str, str], list[tuple[int, str, bool, str]]] = (
        defaultdict(list)
    )

    for index, route in enumerate(app.routes):
        path: str = getattr(route, "path", None) or ""
        if not path:
            continue
        methods = getattr(route, "methods", None) or set()
        last = _last_segment(path)
        is_typed = _is_typed_last_segment(route)
        for method in methods:
            by_group[(method, _parent_prefix(path))].append(
                (index, last, is_typed, path)
            )

    failures: list[str] = []
    for (method, prefix), entries in by_group.items():
        typed_entries = [e for e in entries if e[2]]  # (idx, last, True, path)
        if not typed_entries:
            continue
        for t_idx, _t_last, _, t_path in typed_entries:
            for o_idx, _o_last, o_typed, o_path in entries:
                if o_typed or o_idx <= t_idx:
                    # Skip: other is also typed, or registered before the typed one.
                    continue
                # Static route at o_idx > t_idx: unreachable due to earlier typed route.
                failures.append(
                    f"{method} {o_path!r} (index {o_idx}) is shadowed by "
                    f"{t_path!r} (index {t_idx}) — register the static route first."
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
