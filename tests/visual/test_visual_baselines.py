"""Opt-in visual-regression baseline suite for the Shoelace web UI.

WHAT THIS IS
------------
Full-page screenshots of every main page, compared pixel-for-pixel against
committed PNG baselines under ``tests/visual/baselines/``. Catches unintended
visual changes (a broken layout, a stray colour, a missing component) that the
render/route unit tests can't see.

OPT-IN — LOCAL ONLY (never runs in CI or the default gate)
----------------------------------------------------------
This whole module is skipped unless ``LIT_MONITOR_VISUAL_BASELINE=1`` is set::

    LIT_MONITOR_VISUAL_BASELINE=1 \\
      uv run --extra dev --extra graph --extra nlp \\
      pytest tests/visual/test_visual_baselines.py -q

Why excluded from CI: pixel screenshots are NOT reproducible across machines.
Font antialiasing / hinting differs between macOS and the Linux CI runners, so
text edges shift by a pixel or two and the diff exceeds tolerance even when the
UI is byte-identical. The baselines committed here are macOS-rendered; the user
runs this suite deliberately on their Mac. Running it on Linux/CI would flake.
The normal ``pytest tests/unit tests/visual`` gate therefore skips it.

REGENERATING BASELINES
----------------------
After an *intended* UI change, regenerate and re-commit the PNGs::

    LIT_MONITOR_VISUAL_BASELINE=1 LIT_MONITOR_VISUAL_BASELINE_UPDATE=1 \\
      uv run --extra dev --extra graph --extra nlp \\
      pytest tests/visual/test_visual_baselines.py -q

With ``LIT_MONITOR_VISUAL_BASELINE_UPDATE=1`` (or when a baseline is missing)
the test WRITES the baseline instead of comparing, then passes. Review the new
PNGs visually before committing them.

SAFETY
------
Navigate + screenshot ONLY. This suite never clicks Run / Start / Dry-run /
Stop or any pipeline- or vault-writing control. ``/discovery`` additionally
monkeypatches ``lit_monitor.server.routes.discovery._vault_root`` to a tmp dir
so its digest auto-load can't read or render the user's real Obsidian vault.
Config/state are already isolated by the ``live_server`` fixture
(``LIT_MONITOR_ROOT`` → temp copy); ``git status --short config/`` stays clean.

MASKED VOLATILE REGIONS
-----------------------
Before each screenshot we hide regions whose value changes run-to-run (so the
baseline is stable). ``_mask_volatile()`` sets ``visibility:hidden`` on:
  * ``.stats-banner .stat-value`` / ``.stats-banner .stat-delta`` — the
    cross-page banner's "Last Run … ago" + live counts/deltas.
  * ``.last-run-line`` — the /discovery + /insights "Started/Finished/Status…"
    summary row (timestamps + run-dependent counts).
  * ``#digest-view`` — the /discovery digest body (HTMX-rendered Markdown,
    date- and content-dependent).
  * ``time``, ``[datetime]`` — any timestamp element.
The page structure around them stays visible, so a layout regression is still
caught; only the changing *text* is blanked.
"""

import io
import os
from pathlib import Path

# numpy backs the pixel diff; Pillow decodes the PNGs (both in the [dev] extra).
import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.skipif(
    os.environ.get("LIT_MONITOR_VISUAL_BASELINE") != "1",
    reason="opt-in visual baselines; run locally with LIT_MONITOR_VISUAL_BASELINE=1",
)

# Also tag as a visual test so it sits with the rest of the Playwright suite.
pytestmark = [
    pytestmark,
    pytest.mark.visual,
]

BASELINE_DIR = Path(__file__).parent / "baselines"

# (url_path, baseline_slug). Navigate-only — none of these mutate state.
PAGES: list[tuple[str, str]] = [
    ("/", "home"),
    ("/discovery", "discovery"),
    ("/brain-build", "brain-build"),
    ("/schedule", "schedule"),
    ("/corpus", "corpus"),
    ("/graph", "graph"),
    ("/themes", "themes"),
    ("/trending", "trending"),
    ("/insights", "insights"),
    ("/setup", "setup"),
    ("/setup/step-9", "setup-step-9"),
    ("/ask", "ask"),
    ("/domain", "domain"),
]

VIEWPORT = {"width": 1280, "height": 900}

# Per-channel intensity delta below which a pixel is considered "same".
_PIXEL_THRESHOLD = 16
# Max fraction of pixels allowed to differ before the baseline is a failure.
_MAX_DIFF_FRACTION = 0.01

# JS run before each screenshot to blank run-to-run-volatile text. Keeping the
# surrounding boxes visible means a layout shift still trips the diff; only the
# changing values are hidden. Selectors are documented in the module docstring.
_MASK_JS = """
() => {
  const sels = [
    '.stats-banner .stat-value',
    '.stats-banner .stat-delta',
    '.last-run-line',
    '#digest-view',
    'time',
    '[datetime]',
  ];
  for (const sel of sels) {
    document.querySelectorAll(sel).forEach((el) => {
      el.style.visibility = 'hidden';
    });
  }
}
"""


def _mask_volatile(page) -> None:
    """Hide run-to-run-volatile regions so the baseline is deterministic."""
    page.evaluate(_MASK_JS)


def _diff_fraction(a_bytes: bytes, b_bytes: bytes) -> tuple[float, tuple, tuple]:
    """Fraction of pixels differing by > threshold on any channel.

    Returns (fraction, size_a, size_b). Sizes are returned so the caller can
    assert they match and produce a clear message if they don't.
    """
    img_a = Image.open(io.BytesIO(a_bytes)).convert("RGB")
    img_b = Image.open(io.BytesIO(b_bytes)).convert("RGB")
    if img_a.size != img_b.size:
        return 1.0, img_a.size, img_b.size
    arr_a = np.asarray(img_a, dtype=np.int16)
    arr_b = np.asarray(img_b, dtype=np.int16)
    # Per-pixel: differs if any channel delta exceeds the threshold.
    per_channel = np.abs(arr_a - arr_b)
    differing = (per_channel > _PIXEL_THRESHOLD).any(axis=-1)
    fraction = float(differing.mean())
    return fraction, img_a.size, img_b.size


def _patch_discovery_vault(tmp_dir: Path):
    """Point /discovery's digest auto-load at a tmp dir, never the real vault.

    Returns a restore callable. Patches the module-level ``_vault_root`` the
    route calls, so the HTMX ``#digest-view`` auto-load reads from an empty tmp
    folder (no digest file → a harmless 'no digest' card) instead of the user's
    Obsidian vault.
    """
    from lit_monitor.server.routes import discovery as discovery_routes

    original = discovery_routes._vault_root
    discovery_routes._vault_root = lambda: tmp_dir  # type: ignore[assignment]

    def restore() -> None:
        discovery_routes._vault_root = original  # type: ignore[assignment]

    return restore


def test_visual_baselines(live_server, tmp_path):
    base = live_server
    update = os.environ.get("LIT_MONITOR_VISUAL_BASELINE_UPDATE") == "1"
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # Vault-safety for /discovery: an empty tmp dir stands in for the vault.
    fake_vault = tmp_path / "fake-vault"
    fake_vault.mkdir(parents=True, exist_ok=True)
    restore_vault = _patch_discovery_vault(fake_vault)

    failures: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # DARK theme is the default — no localStorage change needed.
            context = browser.new_context(viewport=VIEWPORT)
            page = context.new_page()
            for url_path, slug in PAGES:
                # "load" (NOT networkidle): HTMX polling / SSE never settle.
                page.goto(f"{base}{url_path}", wait_until="load")
                # Let Shoelace components hydrate + HTMX fragments swap in.
                page.wait_for_timeout(800)
                _mask_volatile(page)
                page.wait_for_timeout(100)
                fresh = page.screenshot(full_page=True)

                baseline_path = BASELINE_DIR / f"{slug}.png"
                if update or not baseline_path.exists():
                    baseline_path.write_bytes(fresh)
                    continue

                expected = baseline_path.read_bytes()
                fraction, size_fresh, size_base = _diff_fraction(expected, fresh)
                if size_fresh != size_base:
                    failures.append(
                        f"{slug}: size mismatch baseline={size_fresh} "
                        f"fresh={size_base} (regenerate with "
                        f"LIT_MONITOR_VISUAL_BASELINE_UPDATE=1)"
                    )
                elif fraction > _MAX_DIFF_FRACTION:
                    failures.append(
                        f"{slug}: {fraction:.4%} of pixels differ "
                        f"(> {_MAX_DIFF_FRACTION:.0%} tolerance) — visual "
                        f"regression on {url_path}"
                    )
            browser.close()
    finally:
        restore_vault()

    if update:
        # Generation run: nothing to assert beyond "we wrote the files".
        written = sorted(p.name for p in BASELINE_DIR.glob("*.png"))
        assert written, "no baseline PNGs were written"
        return

    assert not failures, "Visual baseline mismatches:\n" + "\n".join(failures)
