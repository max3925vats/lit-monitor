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
def test_health_badge_degraded_when_one_section_fails(client: TestClient) -> None:
    """Exactly one section with a fail → degraded (yellow).

    Under the "fail drives, warn informs" model:
      - 0 fails → healthy (green)
      - 1 fail  → degraded (yellow)
      - ≥2 fails → misconfigured (red)
    """
    ok = {"probe": (True, "ok")}
    bad = {"probe": (False, "down")}
    fake_results = {"config": ok, "ollama": bad, "zotero": ok, "vault": ok}
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge degraded" in r.text


@pytest.mark.unit
def test_health_badge_misconfigured_when_two_fail(client: TestClient) -> None:
    """Two or more failing sections → misconfigured (red)."""
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
def test_badge_healthy_when_only_warn_severity(client: TestClient) -> None:
    """Warns alone never bump the badge above green.

    The detail panel still renders the warn row in yellow (see
    ``test_badge_detail_renders_warning_pill_for_warn_severity``), but the
    overall badge stays healthy. Warn = "look at the detail panel", not
    "something is broken".
    """
    from scripts.setup.check_configured import CheckResult

    ok = {"probe": CheckResult(True, "ok", severity="ok")}
    warn = {"scopus.api_key": CheckResult(True, "absent", severity="warn")}
    fake_results = {"config": ok, "ollama": ok, "zotero": ok, "vault": warn}
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge healthy" in r.text


@pytest.mark.unit
def test_badge_healthy_when_optional_scopus_missing_only(client: TestClient) -> None:
    """A live setup with only ``scopus.api_key`` absent must stay green.

    Regression test for the user-reported "yellow badge with only optional
    scopus missing" bug. The warn row still renders in the detail panel,
    but the rolled-up badge colour reflects "nothing is broken".
    """
    from scripts.setup.check_configured import CheckResult

    fake_results = {
        "config": {"secrets_file": CheckResult(True, "ok", severity="ok")},
        "ollama": {"reachable": CheckResult(True, "ok", severity="ok")},
        "zotero": {"reachable": CheckResult(True, "ok", severity="ok")},
        "vault": {"vault_exists": CheckResult(True, "ok", severity="ok")},
        "sources": {
            "scopus.api_key": CheckResult(
                True,
                "scopus.api_key not set — database will be skipped",
                severity="warn",
            ),
        },
    }
    with patch(
        "scripts.setup.health_check.run_health_check", return_value=fake_results
    ):
        r = client.get("/api/health/badge")
    assert r.status_code == 200
    assert "status-badge healthy" in r.text


@pytest.mark.unit
def test_badge_degraded_when_single_fail_with_warn(client: TestClient) -> None:
    """One fail + one warn → degraded (the fail drives, the warn doesn't add).

    This is the "fail drives, warn informs" model — the warn doesn't escalate
    the badge to misconfigured because misconfigured requires ≥2 *failing*
    sections.
    """
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
    assert "status-badge degraded" in r.text


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
