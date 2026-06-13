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
# GET /api/discovery/digest — same-day ordinal disambiguation (the BUG fix)
# ---------------------------------------------------------------------------


class TestDigestEndpointSameDayOrdinal:
    """Two runs on the same date must resolve to DISTINCT files.

    Run #1 (ordinal 1) → bare ``Discovery_{date}.md``; run #2 (ordinal 2) →
    ``Discovery_{date}_2.md``. The 2nd run must read/create ITS own file and
    leave the bare file (run #1's content) untouched.
    """

    def test_second_same_day_run_uses_ordinal_file_not_bare(self, client, vault):
        date_str = "2026-06-10"
        bare = vault / "Literature" / "Digests" / f"Discovery_{date_str}.md"
        ordinal2 = vault / "Literature" / "Digests" / f"Discovery_{date_str}_2.md"
        # Pre-write run #1's content into the bare file; it must stay untouched.
        bare.write_text("## Run One\n\n- 287 papers here\n", encoding="utf-8")
        before_bare = bare.read_text(encoding="utf-8")
        assert not ordinal2.exists()

        run = {"id": 22, "started_at": f"{date_str}T11:00:00"}
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
            # This run is the 2nd of its day.
            patch(
                "lit_monitor.server.routes.discovery.discovery_run_same_day_ordinal",
                return_value=2,
            ),
            patch(
                "lit_monitor.server.routes.discovery.get_discovery_run_papers",
                return_value=[],
            ),
            patch(
                "lit_monitor.server.routes.discovery.render_digest",
                return_value="## Run Two\n\n- 996 papers here\n",
            ),
        ):
            r = client.get("/api/discovery/digest?run_id=22")

        assert r.status_code == 200
        # The 2nd run created its OWN ordinal file...
        assert ordinal2.exists()
        assert "996 papers here" in ordinal2.read_text(encoding="utf-8")
        # ...and the response reflects the 2nd run's own papers.
        assert "Run Two" in r.text
        assert "996 papers here" in r.text
        # The bare file (run #1) was NOT touched or overwritten.
        assert bare.read_text(encoding="utf-8") == before_bare
        assert "Run One" not in r.text

    def test_first_same_day_run_uses_bare_file(self, client, vault):
        date_str = "2026-06-10"
        bare = vault / "Literature" / "Digests" / f"Discovery_{date_str}.md"
        bare.write_text("## Run One\n\n- 287 papers\n", encoding="utf-8")

        run = {"id": 21, "started_at": f"{date_str}T08:00:00"}
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
                "lit_monitor.server.routes.discovery.discovery_run_same_day_ordinal",
                return_value=1,
            ),
        ):
            r = client.get("/api/discovery/digest?run_id=21")

        assert r.status_code == 200
        # Ordinal 1 reads the bare file.
        assert "Run One" in r.text
        assert "287 papers" in r.text


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

    def test_two_same_day_runs_yield_distinct_labels(self, client, tmp_path):
        """Two runs on the same date must render two DISTINCT dropdown labels.

        Run #1 → ``Discovery_{date}``; run #2 → ``Discovery_{date}_2``. Without
        the ordinal suffix both options read ``Discovery_{date}`` and collide.
        """
        from lit_monitor.core.state_db import StateDB

        db = StateDB(tmp_path / "state.db")
        same_day = "2026-06-10T08:00:00"
        rid1 = db.start_discovery_run({"topics": ["a"]})
        rid2 = db.start_discovery_run({"topics": ["b"]})
        with db._connect() as conn:
            conn.execute(
                "UPDATE discovery_runs SET started_at = ? WHERE id IN (?, ?)",
                (same_day, rid1, rid2),
            )
        for rid in (rid1, rid2):
            db.finish_discovery_run(rid, status="success", total_found=1, total_ingested=1)

        rt = _make_fake_runtime(db)
        with patch(
            "lit_monitor.server.routes.discovery.get_runtime", return_value=rt
        ):
            r = client.get("/discovery")
        assert r.status_code == 200
        body = r.text
        # Both distinct labels must be present.
        assert "Discovery_2026-06-10_2" in body
        # The bare label for run #1 also appears (as an exact <sl-option> label,
        # not merely as a prefix of the _2 variant).
        assert ">Discovery_2026-06-10<" in body

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
