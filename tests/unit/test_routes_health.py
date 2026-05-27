"""Unit tests for the topnav health badge route (Task #65, Phase C)."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from scripts.server.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.unit
def test_health_badge_returns_html_fragment(client: TestClient) -> None:
    """The badge endpoint must return an HTML fragment with a status-badge class."""
    r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert 'class="status-badge' in r.text


@pytest.mark.unit
def test_health_badge_unconfigured_when_secrets_missing(client: TestClient) -> None:
    """When the secrets file is missing, the badge must roll up to 'unconfigured'."""
    fake_results = {
        "config": {"secrets_file": (False, "missing")},
        "ollama": {"reachable": (True, "ok")},
        "zotero": {"reachable": (True, "ok")},
        "vault": {"vault_exists": (True, "ok")},
    }
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge unconfigured" in r.text


@pytest.mark.unit
def test_health_badge_healthy_when_all_pass(client: TestClient) -> None:
    """All-True probes should produce a healthy badge."""
    ok = {"probe": (True, "ok")}
    fake_results = {"config": ok, "ollama": ok, "zotero": ok, "vault": ok}
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge healthy" in r.text


@pytest.mark.unit
def test_health_badge_misconfigured_when_one_fails(client: TestClient) -> None:
    """Any failing check -> misconfigured (Bug D 3-state model).

    Note: under the new severity model, even a single fail rolls up to
    misconfigured (red). Yellow (degraded) is reserved for warn-only
    states — i.e. optional sources absent.
    """
    ok = {"probe": (True, "ok")}
    bad = {"probe": (False, "down")}
    fake_results = {"config": ok, "ollama": bad, "zotero": ok, "vault": ok}
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge misconfigured" in r.text


@pytest.mark.unit
def test_health_badge_misconfigured_when_two_fail(client: TestClient) -> None:
    """Two or more failing sections -> still misconfigured (red)."""
    ok = {"probe": (True, "ok")}
    bad = {"probe": (False, "down")}
    fake_results = {"config": ok, "ollama": bad, "zotero": bad, "vault": ok}
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge misconfigured" in r.text


@pytest.mark.unit
def test_health_badge_detail_returns_table(client: TestClient) -> None:
    """Detail endpoint must render a per-check table with at least one pill."""
    ok = {"probe": (True, "ok")}
    bad = {"probe": (False, "down")}
    fake_results = {"config": ok, "ollama": bad, "zotero": ok, "vault": ok}
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge/detail")
    assert r.status_code == 200
    assert '<table class="health-detail-table">' in r.text
    assert "pill success" in r.text
    assert "pill danger" in r.text


@pytest.mark.unit
def test_badge_degraded_when_only_warn_severity(client: TestClient) -> None:
    """A section containing a severity='warn' check (no fails) → degraded."""
    from scripts.setup.check_configured import CheckResult

    ok = {"probe": CheckResult(True, "ok", severity="ok")}
    warn = {"scopus.api_key": CheckResult(True, "absent", severity="warn")}
    fake_results = {"config": ok, "ollama": ok, "zotero": ok, "vault": warn}
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge degraded" in r.text


@pytest.mark.unit
def test_badge_misconfigured_when_any_fail(client: TestClient) -> None:
    """severity='fail' wins over 'warn' — even a single fail → misconfigured."""
    from scripts.setup.check_configured import CheckResult

    ok = {"probe": CheckResult(True, "ok", severity="ok")}
    warn = {"scopus.api_key": CheckResult(True, "absent", severity="warn")}
    fail = {"zotero.api_key": CheckResult(False, "missing", severity="fail")}
    fake_results = {"config": ok, "ollama": warn, "zotero": fail, "vault": ok}
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge misconfigured" in r.text


@pytest.mark.unit
def test_badge_detail_renders_warning_pill_for_warn_severity(
    client: TestClient,
) -> None:
    """severity='warn' rows must render with class='pill warning', not success."""
    from scripts.setup.check_configured import CheckResult

    fake_results = {
        "config": {"secrets_file": CheckResult(True, "ok", severity="ok")},
        "ollama": {"reachable": CheckResult(True, "ok", severity="ok")},
        "zotero": {"reachable": CheckResult(True, "ok", severity="ok")},
        "vault": {
            "scopus.api_key": CheckResult(
                True, "scopus.api_key not set — database will be skipped",
                severity="warn",
            )
        },
    }
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge/detail")
    assert r.status_code == 200
    assert 'class="pill warning"' in r.text


@pytest.mark.unit
def test_health_badge_detail_escapes_html_in_messages():
    """If a check returns a message containing <, >, &, the detail panel must escape them."""
    def fake_check():
        return {
            "config": {"secrets_file": (True, "ok")},
            "ollama": {"ollama": (False, "<script>alert('xss')</script>")},
            "zotero": {"zotero": (True, "ok & fine")},
            "vault": {"vault_path_set": (True, "ok")},
        }
    with patch("scripts.setup.health_check.run_health_check", side_effect=fake_check):
        client = TestClient(create_app())
        r = client.get("/api/health/badge/detail")
    assert r.status_code == 200
    assert "<script>" not in r.text  # raw `<` must be escaped
    assert "&lt;script&gt;" in r.text
    assert "ok &amp; fine" in r.text
