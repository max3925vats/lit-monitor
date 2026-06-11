"""Unit tests for /api/dev Panel 6 endpoints (Task #71).

Panel 6 wires four Tier-4 surface checks into the /dev page:

- compare-models — runs ``run_model_comparison`` on a single sandbox paper.
  Real LLM calls are mocked in tests.
- safety — synthetic prompt-injection guard against ``sanitize_for_prompt``.
- render-guard — synthetic ``{domain_focus}`` test against ``_render``.
- quality-report — reads ``/tmp/lit-monitor-release-quality-report.md``
  (monkeypatched via the module-level ``QUALITY_REPORT_PATH`` constant).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient against a freshly-created app with /dev mounted."""
    monkeypatch.setenv("LIT_MONITOR_DEV", "1")
    from lit_monitor.server.app import create_app

    return TestClient(create_app())


@pytest.mark.unit
def test_safety_endpoint_strips_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/dev/safety → success pill + cleaned output without role markers."""
    client = _make_dev_client(monkeypatch)
    resp = client.post("/api/dev/safety")
    assert resp.status_code == 200
    body = resp.text
    assert 'class="pill success"' in body or "class='pill success'" in body
    assert "markers stripped" in body
    # The cleaned-output <pre> block must not contain raw ChatML markers. The
    # input block legitimately *does* contain them (HTML-escaped), so we
    # assert on the second <pre> rendering of the cleaned text.
    cleaned_segment = body.split("Output:</strong>", 1)[1]
    assert "im_start" not in cleaned_segment
    assert "im_end" not in cleaned_segment


@pytest.mark.unit
def test_render_guard_endpoint_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/dev/render-guard → success pill (placeholder rendered, JSON kept)."""
    client = _make_dev_client(monkeypatch)
    resp = client.post("/api/dev/render-guard")
    assert resp.status_code == 200
    body = resp.text
    assert 'class="pill success"' in body or "class='pill success'" in body
    assert "biopharma manufacturing" in body
    # The literal JSON snippet must round-trip unmolested (HTML-escaped here).
    # html.escape() renders " as &quot; — assert on that exact form.
    assert "&quot;&lt;doi&gt;&quot;: &quot;rationale&quot;" in body


@pytest.mark.unit
def test_quality_report_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Missing report file → warning pill + pointer to test_everything.sh."""
    client = _make_dev_client(monkeypatch)
    missing = tmp_path / "definitely-not-here.md"
    monkeypatch.setattr(
        "lit_monitor.server.routes.dev.QUALITY_REPORT_PATH", missing
    )
    resp = client.get("/api/dev/quality-report")
    assert resp.status_code == 200
    assert "No report on disk" in resp.text
    assert "test_everything.sh" in resp.text


@pytest.mark.unit
def test_quality_report_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Existing report file → success pill + first 500 chars + web-served link.

    A6: the link must point at the HTTP route, not a dead ``file://`` URL.
    """
    client = _make_dev_client(monkeypatch)
    report = tmp_path / "release-quality-report.md"
    report.write_text(
        "# Quality report\n\nTier 4 ran clean. All checks green.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lit_monitor.server.routes.dev.QUALITY_REPORT_PATH", report
    )
    resp = client.get("/api/dev/quality-report")
    assert resp.status_code == 200
    body = resp.text
    assert "Report found" in body
    assert "Quality report" in body
    assert "Tier 4 ran clean" in body
    # A6: no dead file:// link; the page links to the web-served route instead.
    assert "file://" not in body
    assert "/api/dev/quality-report/raw" in body


@pytest.mark.unit
def test_quality_report_raw_serves_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A6 — GET /api/dev/quality-report/raw returns the full report (200)."""
    client = _make_dev_client(monkeypatch)
    report = tmp_path / "release-quality-report.md"
    body_text = "# Quality report\n\nTier 4 ran clean. All checks green.\n"
    report.write_text(body_text, encoding="utf-8")
    monkeypatch.setattr(
        "lit_monitor.server.routes.dev.QUALITY_REPORT_PATH", report
    )
    resp = client.get("/api/dev/quality-report/raw")
    assert resp.status_code == 200
    assert resp.text == body_text
    assert resp.headers["content-type"].startswith("text/markdown")


@pytest.mark.unit
def test_quality_report_raw_404_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A6 — GET /api/dev/quality-report/raw → 404 when no report exists."""
    client = _make_dev_client(monkeypatch)
    missing = tmp_path / "definitely-not-here.md"
    monkeypatch.setattr(
        "lit_monitor.server.routes.dev.QUALITY_REPORT_PATH", missing
    )
    resp = client.get("/api/dev/quality-report/raw")
    assert resp.status_code == 404


@pytest.mark.unit
def test_quality_report_raw_rejects_path_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A6 security — the route resolves a fixed server-side path and accepts no
    client-supplied filename, so a traversal attempt cannot reach an arbitrary
    file. The report path is pinned inside ``tmp_path``; a crafted URL trying to
    walk out to a secret file must not return that file's contents.
    """
    client = _make_dev_client(monkeypatch)
    report = tmp_path / "release-quality-report.md"
    report.write_text("# Quality report\n", encoding="utf-8")
    monkeypatch.setattr(
        "lit_monitor.server.routes.dev.QUALITY_REPORT_PATH", report
    )

    # Plant a secret outside the report dir and try to reach it via traversal.
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")

    for attack in (
        "/api/dev/quality-report/raw/../../etc/passwd",
        "/api/dev/quality-report/raw?path=/etc/passwd",
        "/api/dev/quality-report/raw/..%2f..%2fsecret.txt",
    ):
        resp = client.get(attack)
        # Either the route ignores the extra path/param and serves the pinned
        # report, or the URL simply doesn't match (404). In no case may the
        # secret leak.
        assert "TOP SECRET" not in resp.text
        assert resp.status_code in (200, 404)


@pytest.mark.unit
def test_compare_models_endpoint_calls_run_model_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/dev/compare-models → run_model_comparison called with sandbox deps."""
    client = _make_dev_client(monkeypatch)

    fake_summary = MagicMock()
    fake_summary.output_dir = "/tmp/comparison/2026-05-26_paper"
    fake_score = MagicMock()
    fake_score.model = "ollama:llama3.2:3b"
    fake_score.items_run = 1
    fake_score.items_failed = 0
    fake_score.avg_seconds = 12.3
    fake_score.json_parse_errors = 0
    fake_score.required_fields_missing = 0
    fake_summary.scores = [fake_score]

    fake_sdb = MagicMock(name="sandbox_state_db")
    fake_cfg = MagicMock(name="config")
    fake_zot = MagicMock(name="zotero_client")

    with patch(
        "lit_monitor.pipelines.model_compare.run_model_comparison",
        return_value=fake_summary,
    ) as mock_run, patch(
        "lit_monitor.server.dev_sandbox.sandbox_state_db", return_value=fake_sdb
    ), patch(
        "lit_monitor.core.config.get_config", return_value=fake_cfg
    ), patch(
        "lit_monitor.server.routes.dev._build_zotero_client_for_dev",
        return_value=fake_zot,
    ):
        resp = client.post("/api/dev/compare-models")

    assert resp.status_code == 200
    assert 'class="pill success"' in resp.text or \
        "class='pill success'" in resp.text
    assert "compare-models complete" in resp.text
    assert "ollama:llama3.2:3b" in resp.text

    mock_run.assert_called_once()
    # run_model_comparison's signature is positional:
    #   (config, state_db, zotero_client, n_items=1, mode="paper")
    args, kwargs = mock_run.call_args
    assert args[0] is fake_cfg
    assert args[1] is fake_sdb
    assert args[2] is fake_zot
    assert kwargs.get("n_items") == 1
    assert kwargs.get("mode") == "paper"
