"""CSS-rule assertions for two consistency fixes on feat/webui-shoelace.

FIX 3 — hover affordance on every <sl-details> header (discovery / brain-build /
        run-history / per-paper cards), matching .step-card / .settings-section.
FIX 4 — run-detail paper cards: 3-up grid, tighter gap, less whitespace above
        the title.

These read the static site.css text and assert the rules exist. They are the
RED→GREEN guard for the CSS-only change (stash site.css → these fail).
"""

import re
from pathlib import Path

CSS = Path("lit_monitor/server/static/site.css").read_text()


def _normalize(text: str) -> str:
    """Collapse whitespace so assertions are insensitive to formatting."""
    return re.sub(r"\s+", " ", text)


CSS_FLAT = _normalize(CSS)


# --- FIX 3: sl-details header hover cue -------------------------------------


def test_sl_details_header_is_clickable():
    """The Shoelace details header part reads as clickable (cursor: pointer)."""
    assert "sl-details::part(header)" in CSS
    # the base header part declares a pointer cursor
    assert re.search(
        r"sl-details::part\(header\)\s*\{[^}]*cursor:\s*pointer", CSS
    ), "sl-details::part(header) must set cursor: pointer"


def test_sl_details_header_has_hover_cue():
    """Hovering the header shows the same --hover-overlay tint as .step-card."""
    assert "sl-details::part(header):hover" in CSS
    assert re.search(
        r"sl-details::part\(header\):hover\s*\{[^}]*var\(--hover-overlay\)", CSS
    ), "sl-details::part(header):hover must use var(--hover-overlay), to match .step-card"


# --- FIX 4: run-detail paper-cards grid -------------------------------------


def test_paper_cards_grid_fits_three_columns():
    """The min column width is reduced so 3 cards fit the run-detail width.

    Old value (minmax(24rem, …)) yielded only 2 columns; reject it and require
    a <= 20rem min that yields 3 across at the dashboard content width.
    """
    rule = re.search(r"\.paper-cards\s*\{[^}]*\}", CSS_FLAT)
    assert rule, ".paper-cards grid rule not found"
    rule_text = rule.group(0)
    assert "minmax(24rem" not in rule_text, "old 2-column minmax(24rem …) still present"
    assert "minmax(20rem" in rule_text, "expected minmax(20rem, 1fr) for a 3-up grid"


def test_paper_cards_gap_reduced():
    """Inter-card gap tightened from 1rem to 0.75rem."""
    rule = re.search(r"\.paper-cards\s*\{[^}]*\}", CSS_FLAT)
    assert rule, ".paper-cards grid rule not found"
    assert "gap: 0.75rem" in rule.group(0), "expected gap: 0.75rem on .paper-cards"


def test_paper_cards_preserve_equal_height():
    """Equal-height parts rule must NOT regress (height:100% flex column)."""
    assert re.search(
        r"\.paper-cards sl-card\.paper-card::part\(base\)\s*\{[^}]*height:\s*100%",
        CSS_FLAT,
    ), "equal-height ::part(base) rule was lost"


def test_paper_card_title_whitespace_trimmed():
    """The dead space above the <h3> title is reduced via a smaller top padding
    on the sl-card body part (Shoelace's own body padding stacked with the old
    .paper-card padding)."""
    assert "sl-card.paper-card::part(body)" in CSS_FLAT, (
        "expected an sl-card.paper-card::part(body) padding override to trim "
        "whitespace above the title"
    )


# --- RR review: .card[hidden] must override .card{display:block} ------------
# A `.card` element with the `hidden` attribute (e.g. the Reset & Rebuild
# "Not set up yet" banner) was always visible: the author rule
# `.card { display: block }` beats the UA `[hidden] { display: none }` (author
# origin wins), so the `hidden` attribute was ignored. An explicit
# `.card[hidden] { display: none !important }` is required — mirrors the
# existing `.live-progress-wrap[hidden]` fix.
def test_card_hidden_attribute_is_honored():
    assert re.search(
        r"\.card\[hidden\]\s*\{[^}]*display:\s*none[^}]*\}", CSS
    ), ".card[hidden] { display: none } rule missing — hidden .card elements won't hide"
