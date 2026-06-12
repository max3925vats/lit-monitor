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
        NavItem("Corpus", "/corpus", "collection"),
        NavItem("Brain-build", "/brain-build", "cpu"),
        NavItem("Ask", "/ask", "stars"),
    )),
    NavGroup("Explore", (
        NavItem("Knowledge graph", "/graph", "diagram-3"),
        NavItem("Themes", "/themes", "grid"),
        NavItem("Trending", "/trending", "graph-up-arrow"),
    )),
    NavGroup("Tune", (
        NavItem("Domain", "/domain", "bullseye"),
        NavItem("Insights", "/insights", "bar-chart"),
        NavItem("Settings", "/settings", "sliders"),
    )),
    NavGroup("Setup", (
        NavItem("Setup", "/setup", "rocket-takeoff"),
    )),
)


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
    """
    for group in NAV_GROUPS:
        for item in group.items:
            if path == item.href:
                return [(group.label, None), (item.label, None)]
            if path.startswith(item.href.rstrip("/") + "/"):
                return [(group.label, None), (item.label, item.href), (detail or "Detail", None)]
    return []
