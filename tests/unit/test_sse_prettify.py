"""Regression coverage for ``_prettify_jsonl_line``.

The function is consumed by all three SSE log streams (brain-build,
discovery, dev dry-run) via ``_tail`` in scripts/server/routes/sse.py. A
contract regression would surface as the Panel 5 / Discovery / Brain-build
log panels rendering raw JSON in the browser.

Pins:
1. A typical {ts, level, logger, msg} record produces a single-line HTML
   fragment with the expected grid spans.
2. The "scripts." prefix is stripped from the logger name.
3. Malformed JSON falls back to ``<div class="log-line log-raw">…</div>``.
4. All interpolated fields are HTML-escaped (XSS-safe).
"""
from __future__ import annotations

import pytest

from lit_monitor.server.routes.sse import _prettify_jsonl_line


@pytest.mark.unit
def test_prettify_typical_jsonl_record_renders_columns() -> None:
    raw = (
        '{"ts": "2026-05-27T03:00:00+00:00", "level": "INFO", '
        '"logger": "scripts.search.search_runner", '
        '"msg": "Searching topic: \'force field\'"}'
    )
    html = _prettify_jsonl_line(raw)
    # Outer wrapper carries the level for per-row styling.
    assert html.startswith('<div class="log-line log-INFO">')
    assert html.endswith("</div>")
    # All four columns present.
    assert '<span class="log-ts">03:00:00</span>' in html
    assert '<span class="log-level">INFO</span>' in html
    # "scripts." prefix is stripped from the logger.
    assert '<span class="log-logger">search.search_runner</span>' in html
    assert "scripts.search.search_runner" not in html
    # The message field survives the round-trip (apostrophes escaped).
    assert "Searching topic" in html
    # Exactly one line (no embedded newlines — SSE data: can't carry them).
    assert "\n" not in html


@pytest.mark.unit
def test_prettify_strips_scripts_prefix_from_logger() -> None:
    raw = '{"ts": "2026-05-27T03:00:00+00:00", "level": "WARNING", "logger": "scripts.pipelines.discovery", "msg": "rate limit"}'
    html = _prettify_jsonl_line(raw)
    assert "log-WARNING" in html
    assert '<span class="log-logger">pipelines.discovery</span>' in html


@pytest.mark.unit
def test_prettify_non_scripts_logger_passes_through() -> None:
    raw = '{"ts": "2026-05-27T03:00:00+00:00", "level": "INFO", "logger": "root", "msg": "boot"}'
    html = _prettify_jsonl_line(raw)
    assert '<span class="log-logger">root</span>' in html


@pytest.mark.unit
def test_prettify_malformed_json_falls_back_to_raw() -> None:
    html = _prettify_jsonl_line("this is not json {oops")
    assert 'class="log-line log-raw"' in html
    # Original message preserved verbatim (escaped).
    assert "this is not json" in html


@pytest.mark.unit
def test_prettify_escapes_html_in_message() -> None:
    """XSS-safe: an LLM warning or paper title containing `<script>` must NOT
    inject DOM into the page."""
    raw = (
        '{"ts": "2026-05-27T03:00:00+00:00", "level": "ERROR", '
        '"logger": "root", "msg": "<script>alert(1)</script> & other"}'
    )
    html = _prettify_jsonl_line(raw)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp; other" in html


@pytest.mark.unit
def test_prettify_handles_missing_fields_gracefully() -> None:
    """A record missing ``logger`` or ``ts`` (defensively, in case the log
    schema changes) shouldn't crash — falls back to empty strings."""
    raw = '{"level": "INFO", "msg": "no ts no logger"}'
    html = _prettify_jsonl_line(raw)
    assert 'class="log-line log-INFO"' in html
    assert '<span class="log-ts"></span>' in html
    assert '<span class="log-logger"></span>' in html
    assert "no ts no logger" in html
