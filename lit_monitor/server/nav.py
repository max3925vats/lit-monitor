"""Single source of truth for the web-UI sidebar navigation (P0).

The sidebar (templates/_partials/sidebar.html) and server-side active-state both read
NAV_GROUPS, so the IA lives in exactly one place. Grouping per the redesign spec:
Monitor / Semantics / Explore / Tune / Setup.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    label: str
    href: str
    icon: str          # Shoelace icon name (sl-icon name="...")


@dataclass(frozen=True)
class NavGroup:
    label: str
    items: tuple[NavItem, ...]


NAV_GROUPS: tuple[NavGroup, ...] = (
    NavGroup("Monitor", (
        NavItem("Discovery", "/discovery", "broadcast"),
        NavItem("Schedule", "/schedule", "clock"),
    )),
    NavGroup("Semantics", (
        NavItem("Ask", "/ask", "stars"),
        NavItem("Brain-build", "/brain-build", "cpu"),
        NavItem("Corpus Health", "/corpus", "collection"),
    )),
    NavGroup("Explore", (
        NavItem("Knowledge graph", "/graph", "diagram-3"),
        NavItem("Themes", "/themes", "grid"),
        NavItem("Trending", "/trending", "graph-up-arrow"),
    )),
    NavGroup("Tune", (
        NavItem("Domain", "/domain", "bullseye"),
        NavItem("Insights", "/insights", "bar-chart"),
    )),
    NavGroup("Setup", (
        NavItem("Setup", "/setup", "rocket-takeoff"),
    )),
)


_NO_BANNER_PREFIXES = ("/setup", "/dev")


def show_stats_banner(path: str) -> bool:
    """True for pages that should display the corpus stats banner.

    Excluded: Home, Setup (incl. step-9 Tuning), Dev (per the 2026-06-12
    decision). The old standalone /settings page was folded into setup
    step-9, so it no longer needs its own exclusion prefix."""
    if path in ("/", ""):
        return False
    for prefix in _NO_BANNER_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return False
    return True


def active_group_for_path(path: str) -> str | None:
    """Return the label of the group whose any item is a prefix of `path`.

    Longest-prefix match so `/discovery/2` resolves to the `/discovery` item's group.
    """
    best: tuple[int, str] | None = None
    for group in NAV_GROUPS:
        for item in group.items:
            if path == item.href or path.startswith(item.href.rstrip("/") + "/"):
                if best is None or len(item.href) > best[0]:
                    best = (len(item.href), group.label)
    return best[1] if best else None


def breadcrumb_trail(path: str, detail: str | None = None) -> list[tuple[str, str | None]]:
    """Top-bar breadcrumb trail for `path`.

    List page (`/corpus`)  -> [(group, None), (item, None)]
    Detail (`/corpus/x`)   -> [(group, None), (item, item.href), (detail or "Detail", None)]
    Home/unknown           -> []  (no breadcrumbs)

    Special case (Setup): the Setup group's single item is also labelled "Setup"
    (``group.label == item.label``). The naive 3-crumb shape would render a
    redundant "Setup › Setup › Detail". When the group and item labels collide,
    collapse to a single "Setup" crumb linking to /setup, then the detail.
    """
    for group in NAV_GROUPS:
        for item in group.items:
            if path == item.href:
                return [(group.label, None), (item.label, None)]
            if path.startswith(item.href.rstrip("/") + "/"):
                if group.label == item.label:
                    # Collapse the duplicate (e.g. Setup): one linked crumb + detail.
                    return [(item.label, item.href), (detail or "Detail", None)]
                return [(group.label, None), (item.label, item.href), (detail or "Detail", None)]
    return []
