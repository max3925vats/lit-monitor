"""P3: notify-handler endpoint + chooser tests."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from scripts.server.app import create_app

    return TestClient(create_app(), follow_redirects=False)


class TestPreferredViewerRedirect:
    def test_browser_preference_302_to_discovery_run(self, client):
        with patch(
            "scripts.server.routes.discovery_notify._get_preferred_viewer",
            return_value="browser",
        ):
            r = client.get("/discovery/notify-handler?run_id=42")
        assert r.status_code in (302, 307)
        assert "/discovery/42" in r.headers["location"]

    def test_obsidian_preference_302_to_obsidian_uri(self, client):
        with patch(
            "scripts.server.routes.discovery_notify._get_preferred_viewer",
            return_value="obsidian",
        ):
            r = client.get("/discovery/notify-handler?run_id=42")
        assert r.status_code in (302, 307)
        assert r.headers["location"].startswith("obsidian://")

    def test_none_preference_204(self, client):
        with patch(
            "scripts.server.routes.discovery_notify._get_preferred_viewer",
            return_value="none",
        ):
            r = client.get("/discovery/notify-handler?run_id=42")
        assert r.status_code == 204


class TestChooserPage:
    def test_chooser_rendered_when_unset(self, client):
        with patch(
            "scripts.server.routes.discovery_notify._get_preferred_viewer",
            return_value="",
        ):
            r = client.get("/discovery/notify-handler?run_id=99")
        assert r.status_code == 200
        body = r.text.lower()
        assert "99" in r.text  # run_id rendered
        for opt in ("browser", "obsidian", "none"):
            assert opt in body
        assert "remember" in body


class TestSavePreferencePost:
    def test_remember_true_calls_save(self, client):
        with patch(
            "scripts.server.routes.discovery_notify.safe_save_preference"
        ) as m:
            r = client.post(
                "/discovery/notify-handler/save-preference",
                json={"viewer": "browser", "remember": True},
            )
        assert r.status_code == 200
        m.assert_called_once()
        # First positional or kwarg
        call_args = m.call_args.args
        call_kwargs = m.call_args.kwargs
        assert (call_args and call_args[0] == "browser") or call_kwargs.get(
            "viewer"
        ) == "browser"

    def test_remember_false_does_not_save(self, client):
        with patch(
            "scripts.server.routes.discovery_notify.safe_save_preference"
        ) as m:
            r = client.post(
                "/discovery/notify-handler/save-preference",
                json={"viewer": "browser", "remember": False},
            )
        assert r.status_code == 200
        m.assert_not_called()

    def test_bad_viewer_422(self, client):
        r = client.post(
            "/discovery/notify-handler/save-preference",
            json={"viewer": "bogus", "remember": True},
        )
        assert r.status_code == 422

    def test_save_failure_500_no_leak(self, client, caplog):
        """A3-5: when safe_save_preference raises, the 500 body must NOT contain
        the exception text/path, and the failure must be logged server-side.
        """
        import logging

        secret = "/Users/secret/config/extraction.yaml"
        with caplog.at_level(
            logging.ERROR, logger="scripts.server.routes.discovery_notify"
        ):
            with patch(
                "scripts.server.routes.discovery_notify.safe_save_preference",
                side_effect=OSError(secret),
            ):
                r = client.post(
                    "/discovery/notify-handler/save-preference",
                    json={"viewer": "browser", "remember": True},
                )
        assert r.status_code == 500
        body = r.json()
        assert body == {"ok": False, "error": "Internal error"}
        # No path / exception text in the body.
        assert secret not in str(body)
        # Logged server-side with the real exception.
        assert any(
            secret in rec.getMessage() or rec.exc_info for rec in caplog.records
        )


class TestQueryParam:
    def test_missing_run_id_422(self, client):
        r = client.get("/discovery/notify-handler")
        assert r.status_code == 422
