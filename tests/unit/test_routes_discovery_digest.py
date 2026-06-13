"""Digest viewer: GET /api/discovery/digest + the /discovery digest dropdown.

VAULT-WRITE SAFETY (NON-NEGOTIABLE):
  The create-if-missing path writes a .md into the user's real Obsidian vault.
  LIT_MONITOR_ROOT does NOT redirect the vault. EVERY test here monkeypatches
  ``discovery._vault_root`` to return a ``tmp_path`` so no code path can touch
  ~/...Obsidian.../Literature/Digests. Tests assert on tmp_path only.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app
from lit_monitor.server.runtime import reset_runtime


@pytest.fixture(autouse=True)
def fresh_runtime():
    reset_runtime()
    yield
    reset_runtime()


def _make_fake_runtime(state_db) -> MagicMock:
    rt = MagicMock()
    rt.state_db = state_db
    return rt


@pytest.fixture
def vault(tmp_path):
    """A tmp vault root with the digests folder created. NEVER the real vault.

    Also pins ``_digests_folder_name`` to the literal default so the endpoint's
    path resolution doesn't read a MagicMock runtime config. Tests still patch
    ``_vault_root`` → this tmp_path explicitly.
    """
    folder = tmp_path / "Literature" / "Digests"
    folder.mkdir(parents=True, exist_ok=True)
    with patch(
        "lit_monitor.server.routes.discovery._digests_folder_name",
        return_value="Literature/Digests",
    ):
        yield tmp_path


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# GET /api/discovery/digest — exists path
# ---------------------------------------------------------------------------


class TestDigestEndpointExistsPath:
    def test_renders_existing_digest_as_html(self, client, vault):
        """A pre-written digest is rendered as HTML (headings/lists), file unchanged."""
        digest = vault / "Literature" / "Digests" / "Discovery_2026-06-10.md"
        md = "## Top Results\n\n- Paper A\n- Paper B\n"
        digest.write_text(md, encoding="utf-8")
        before = digest.read_text(encoding="utf-8")

        run = {"id": 7, "started_at": "2026-06-10T08:00:00"}
        with (
            patch(
                "lit_monitor.server.routes.discovery._vault_root",
                return_value=vault,
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_runtime",
                return_value=_make_fake_runtime(MagicMock()),
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_discovery_run",
                return_value=run,
            ),
        ):
            r = client.get("/api/discovery/digest?run_id=7")

        assert r.status_code == 200
        body = r.text
        assert 'class="digest-rendered"' in body
        # Markdown converted to real HTML — heading + list items.
        assert "<h2>" in body
        assert "<li>" in body
        assert "Paper A" in body
        # The file on disk was NOT rewritten by the exists path.
        assert digest.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# GET /api/discovery/digest — create-if-missing path
# ---------------------------------------------------------------------------


class TestDigestEndpointCreatePath:
    def test_creates_digest_when_missing_then_renders(self, client, vault):
        """No file → render_digest is called, file is created under tmp vault, rendered."""
        digest = vault / "Literature" / "Digests" / "Discovery_2026-06-11.md"
        assert not digest.exists()

        run = {"id": 8, "started_at": "2026-06-11T09:00:00"}
        with (
            patch(
                "lit_monitor.server.routes.discovery._vault_root",
                return_value=vault,
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_runtime",
                return_value=_make_fake_runtime(MagicMock()),
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_discovery_run",
                return_value=run,
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_discovery_run_papers",
                return_value=[],
            ),
            patch(
                "lit_monitor.server.routes.discovery.render_digest",
                return_value="## Created Digest\n\n- only item\n",
            ),
        ):
            r = client.get("/api/discovery/digest?run_id=8")

        assert r.status_code == 200
        # File now exists under the TMP vault (never the real vault).
        assert digest.exists()
        assert "Created Digest" in digest.read_text(encoding="utf-8")
        # And the response renders it.
        assert 'class="digest-rendered"' in r.text
        assert "<h2>" in r.text
        assert "Created Digest" in r.text


# ---------------------------------------------------------------------------
# GET /api/discovery/digest — sanitization
# ---------------------------------------------------------------------------


class TestDigestEndpointSanitize:
    def test_script_tags_are_stripped(self, client, vault):
        """Malicious <script> in the digest markdown must not survive to output."""
        digest = vault / "Literature" / "Digests" / "Discovery_2026-06-12.md"
        digest.write_text(
            "## Heading\n\n<script>alert(1)</script>\n\nSafe text.\n",
            encoding="utf-8",
        )
        run = {"id": 9, "started_at": "2026-06-12T10:00:00"}
        with (
            patch(
                "lit_monitor.server.routes.discovery._vault_root",
                return_value=vault,
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_runtime",
                return_value=_make_fake_runtime(MagicMock()),
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_discovery_run",
                return_value=run,
            ),
        ):
            r = client.get("/api/discovery/digest?run_id=9")

        assert r.status_code == 200
        assert "<script>" not in r.text
        assert "alert(1)" not in r.text or "<script>" not in r.text
        # Legitimate content survives.
        assert "Safe text." in r.text


# ---------------------------------------------------------------------------
# GET /api/discovery/digest — no vault / unknown run (no 500)
# ---------------------------------------------------------------------------


class TestDigestEndpointDefensive:
    def test_no_vault_returns_field_note_not_500(self, client):
        run = {"id": 1, "started_at": "2026-06-10T08:00:00"}
        with (
            patch(
                "lit_monitor.server.routes.discovery._vault_root",
                return_value=None,
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_runtime",
                return_value=_make_fake_runtime(MagicMock()),
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_discovery_run",
                return_value=run,
            ),
        ):
            r = client.get("/api/discovery/digest?run_id=1")
        assert r.status_code == 200
        assert "field-note" in r.text
        assert "Vault" in r.text or "vault" in r.text

    def test_unknown_run_returns_swappable_body(self, client, vault):
        with (
            patch(
                "lit_monitor.server.routes.discovery._vault_root",
                return_value=vault,
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_runtime",
                return_value=_make_fake_runtime(MagicMock()),
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_discovery_run",
                return_value=None,
            ),
        ):
            r = client.get("/api/discovery/digest?run_id=99999")
        # An HTMX-swappable body is what matters; 404 status is acceptable.
        assert "Run not found" in r.text


# ---------------------------------------------------------------------------
# GET /discovery — the Digest viewer dropdown (template)
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_db(tmp_path):
    from lit_monitor.core.state_db import StateDB

    db = StateDB(tmp_path / "state.db")
    rid = db.start_discovery_run({"topics": ["x"]})
    db.finish_discovery_run(rid, status="success", total_found=2, total_ingested=1)
    return db, rid


class TestDigestViewerTemplate:
    def test_template_has_sl_select_and_digest_view(self, client, seeded_db):
        db, run_id = seeded_db
        rt = _make_fake_runtime(db)
        with patch(
            "lit_monitor.server.routes.discovery.get_runtime", return_value=rt
        ):
            r = client.get("/discovery")
        assert r.status_code == 200
        body = r.text
        # New Digest viewer: a named select + the htmx target div.
        assert '<sl-select name="run_id"' in body
        assert 'id="digest-view"' in body
        # Option label convention Discovery_{date}.
        assert "Discovery_" in body
        # The old raw <pre> dump is gone.
        assert '<pre class="digest-pre"' not in body

    def test_template_empty_state_when_no_runs(self, client, tmp_path):
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")  # no runs
        rt = _make_fake_runtime(db)
        with patch(
            "lit_monitor.server.routes.discovery.get_runtime", return_value=rt
        ):
            r = client.get("/discovery")
        assert r.status_code == 200
        # No runs → friendly empty state, no select.
        assert "No discovery runs yet" in r.text
