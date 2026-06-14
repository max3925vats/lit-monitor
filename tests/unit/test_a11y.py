"""Accessibility-pass assertions (P5 — feat/webui-shoelace).

These guard four low-risk, additive a11y fixes that apply to base.html (which
every page extends) plus a couple of per-page surfaces:

1. Every ``<sl-icon-button>`` carries a non-empty ``label=`` (Shoelace renders
   it as the control's ``aria-label``). Asserted over the rendered HTML of a
   page that extends base.html.
2. ``site.css`` defines a ``:focus-visible`` outline so keyboard focus is
   visible.
3. The Setup wizard-strip step links carry an ``aria-label="Go to step N"``
   (the visible text is just the bare number).
4. A ``.status-dot`` renders BOTH ``title=`` and ``aria-label=`` so screen
   readers announce the per-item state (verified on /trending).
5. The Ask textareas (full page + drawer) and the raw-YAML routing textarea
   have a ``label=`` (no associated visible <label>, so the control was
   otherwise nameless).

All tests are TestClient render + served-CSS assertions. NONE click or POST.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app

# Matches a single <sl-icon-button ...> opening tag (no '>' inside attrs).
_ICON_BUTTON_TAG = re.compile(r"<sl-icon-button\b[^>]*>")


def _client() -> TestClient:
    return TestClient(create_app())


# --- 1. icon-only controls have accessible names ---------------------------


@pytest.mark.unit
def test_every_sl_icon_button_has_a_label_on_landing() -> None:
    """Every <sl-icon-button> in the rendered shell carries a non-empty label=.

    base.html holds the theme toggle + the mobile sidebar toggle; both are
    icon-only and need an aria-label, which Shoelace derives from label=.
    """
    html = _client().get("/").text
    tags = _ICON_BUTTON_TAG.findall(html)
    assert tags, "expected at least one <sl-icon-button> in the page shell"
    for tag in tags:
        m = re.search(r'label\s*=\s*"([^"]*)"', tag)
        assert m is not None, f"<sl-icon-button> without a label=: {tag}"
        assert m.group(1).strip(), f"<sl-icon-button> with an empty label=: {tag}"


# --- 2. visible keyboard focus ---------------------------------------------


@pytest.mark.unit
def test_site_css_has_focus_visible_outline() -> None:
    """The served site.css defines a :focus-visible rule with an outline."""
    css = _client().get("/static/site.css").text
    assert ":focus-visible" in css, "site.css must define a :focus-visible rule"
    # A :focus-visible selector block must set an outline.
    assert re.search(
        r":focus-visible[^{]*\{[^}]*outline", css
    ), ":focus-visible rule must declare an outline"


@pytest.mark.unit
def test_site_css_focus_visible_reaches_shoelace_buttons() -> None:
    """Shoelace buttons live in shadow DOM, so the global :focus-visible rule
    can't reach them — a ::part(base):focus-visible rule with an outline is
    needed for sl-button / sl-icon-button keyboard focus to be visible."""
    css = _client().get("/static/site.css").text
    flat = re.sub(r"\s+", " ", css)
    for part in ("sl-button::part(base):focus-visible",
                 "sl-icon-button::part(base):focus-visible"):
        assert part in flat, f"missing {part} rule for keyboard focus"
    assert re.search(
        r"sl-(button|icon-button)::part\(base\):focus-visible[^{]*\{[^}]*outline",
        flat,
    ), "Shoelace ::part(base):focus-visible rules must declare an outline"


# --- 3. wizard-strip step links named -------------------------------------


@pytest.mark.unit
def test_wizard_strip_steps_have_aria_label() -> None:
    """The setup wizard step-strip links announce 'Go to step N'."""
    html = _client().get("/setup/step-1").text
    assert 'aria-label="Go to step' in html, (
        "wizard-strip a.step links must carry aria-label=\"Go to step N\" "
        "(the visible text is only the bare number)"
    )


# --- 4. status dots have names --------------------------------------------


@pytest.mark.unit
def test_status_dot_has_title_and_aria_label_on_trending() -> None:
    """A rendered .status-dot carries BOTH title= and aria-label=.

    /trending renders one status-dot per suggestion row. We seed a row via the
    existing _safe_db seam (no POST, no real DB).
    """
    pending = [
        {
            "id": 1,
            "concept_text": "biorefinery",
            "concept_type": "topic",
            "n_mentions_new": 20,
            "n_mentions_prev": 5,
            "growth_rate": 3.0,
            "suggested_at": "2026-05-30 12:00:00",
            "user_action": "pending",
            "action_at": None,
            "in_existing_topics": False,
        }
    ]
    with patch("lit_monitor.server.routes.trending._safe_db") as mock_db_fn:
        mock_db = MagicMock()
        mock_db.get_pending_trending_suggestions.return_value = pending
        mock_db_fn.return_value = mock_db
        html = _client().get("/trending").text

    # Find each status-dot span and assert both attributes are present on it.
    dots = re.findall(r"<span[^>]*class=\"[^\"]*status-dot[^\"]*\"[^>]*>", html)
    assert dots, "expected at least one .status-dot on /trending"
    for dot in dots:
        assert "title=" in dot, f"status-dot missing title=: {dot}"
        assert "aria-label=" in dot, f"status-dot missing aria-label=: {dot}"


# --- 5. nameless form controls get a label= --------------------------------


@pytest.mark.unit
def test_ask_drawer_textarea_has_label() -> None:
    """The global Ask drawer's question textarea carries a label=."""
    html = _client().get("/").text
    m = re.search(r"<sl-textarea\b[^>]*\bname=\"question\"[^>]*>", html, re.S)
    assert m is not None, "Ask-drawer question textarea not rendered on /"
    assert "label=" in m.group(0), "Ask-drawer question textarea needs a label="


@pytest.mark.unit
def test_setup_routing_raw_yaml_textarea_has_label() -> None:
    """The advanced raw-YAML routing editor textarea carries a label=."""
    html = _client().get("/setup/step-8").text
    m = re.search(r"<sl-textarea\b[^>]*\bname=\"raw_yaml\"[^>]*>", html, re.S)
    assert m is not None, "raw_yaml textarea not rendered on /setup/step-8"
    assert "label=" in m.group(0), "raw_yaml textarea needs a label="
