"""Unit tests for /api/dev/ingest/* endpoints (Task #68 Panel 3).

Three modes:
- Mode A (markdown paste)  → POST /api/dev/ingest/markdown
- Mode B (Zotero item key) → POST /api/dev/ingest/zotero-key
- Mode C (Zotero by DOI)   → POST /api/dev/ingest/doi

All three share ``_run_sandbox_ingest()``; the tests heavily mock the pipeline
internals so the assertions focus on the *route plumbing*, not on chunker or
LLM behaviour (covered by their own units).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient against a freshly-created app with /dev mounted."""
    monkeypatch.setenv("LIT_MONITOR_DEV", "1")
    from scripts.server.app import create_app

    return TestClient(create_app())


@pytest.mark.unit
def test_ingest_markdown_missing_fields_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mode A with empty doi/markdown short-circuits to a danger pill."""
    client = _make_dev_client(monkeypatch)
    resp = client.post("/api/dev/ingest/markdown", data={"doi": "", "markdown": ""})
    assert resp.status_code == 200
    assert 'class="pill danger"' in resp.text
    assert "required" in resp.text


@pytest.mark.unit
def test_ingest_markdown_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """All five pipeline stages succeed → five success pills in the trace."""
    client = _make_dev_client(monkeypatch)

    # Fake pipeline pieces. ``chunk_markdown`` returns a list with len-able
    # items; ``extract_paper`` returns a fields dict.
    fake_chunks = [MagicMock(), MagicMock()]
    fake_extraction = {
        "title": "Probe paper",
        "authors": ["A", "B"],
        "year": 2025,
        "abstract": "An abstract.",
        "_overall_confidence": 0.9,
    }
    fake_state_db = MagicMock()
    fake_emb_db = MagicMock()
    fake_vault_dir = MagicMock()
    # Make ``vault_dir / fname`` chain return a MagicMock whose
    # ``.relative_to(...)`` returns a printable string.
    fake_note_path = MagicMock()
    fake_note_path.relative_to.return_value = "Literature/_Dev/sandbox_x.md"
    fake_vault_dir.__truediv__.return_value = fake_note_path

    fake_cfg = MagicMock()
    fake_cfg.obsidian.vault_path = "/tmp/vault"

    with patch("scripts.core.chunker.chunk_markdown", return_value=fake_chunks), \
         patch("scripts.llm.extractor.extract_paper", return_value=fake_extraction), \
         patch("scripts.llm.llm_client.get_client", return_value=MagicMock()), \
         patch("scripts.core.config.get_config", return_value=fake_cfg), \
         patch(
            "scripts.server.dev_sandbox.sandbox_state_db",
            return_value=fake_state_db,
         ), \
         patch(
            "scripts.server.dev_sandbox.sandbox_embeddings_db",
            return_value=fake_emb_db,
         ), \
         patch(
            "scripts.server.dev_sandbox.sandbox_vault_subfolder",
            return_value=fake_vault_dir,
         ), \
         patch("scripts.output.obsidian_writer.write_paper_note", return_value="ok"):
        resp = client.post(
            "/api/dev/ingest/markdown",
            data={"doi": "10.0/test", "markdown": "# Hello\n\nBody."},
        )

    assert resp.status_code == 200
    body = resp.text
    # All five stages rendered as success pills (no danger pill anywhere).
    assert body.count('class="pill success"') == 5
    assert 'class="pill danger"' not in body
    assert "chunk_markdown" in body
    assert "extract_paper" in body
    assert "sandbox_state_db.upsert_paper" in body
    assert "sandbox_embeddings_db" in body
    assert "write_paper_note" in body
    # Confirm the sandbox writes actually got called (no silent skip).
    fake_state_db.upsert_paper.assert_called_once()
    fake_emb_db.add_paper.assert_called_once()
    fake_emb_db.add_chunks.assert_called_once()


@pytest.mark.unit
def test_ingest_zotero_key_no_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode B: when the Zotero item has no local .md → danger pill, no ingest."""
    client = _make_dev_client(monkeypatch)

    fake_client = MagicMock()
    fake_client._zot.item.return_value = {"data": {"DOI": "10.0/x"}}
    fake_client.get_markdown_attachment.return_value = None  # no local file

    with patch(
        "scripts.server.routes.dev._build_zotero_client_for_dev",
        return_value=fake_client,
    ):
        resp = client.post("/api/dev/ingest/zotero-key", data={"zotero_key": "ABC123"})

    assert resp.status_code == 200
    assert 'class="pill danger"' in resp.text
    assert "No .md attachment" in resp.text
    assert "ABC123" in resp.text


@pytest.mark.unit
def test_ingest_doi_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode C: no Zotero item with that DOI → single danger pill, no pipeline."""
    client = _make_dev_client(monkeypatch)

    fake_client = MagicMock()
    # New flow uses everything(items(...)) iteration; return an empty iterator.
    fake_client._zot.everything.return_value = iter([])

    with patch(
        "scripts.server.routes.dev._build_zotero_client_for_dev",
        return_value=fake_client,
    ):
        resp = client.post("/api/dev/ingest/doi", data={"doi": "10.0/nope"})

    assert resp.status_code == 200
    assert 'class="pill danger"' in resp.text
    assert "No Zotero item matches" in resp.text
    assert "10.0/nope" in resp.text


@pytest.mark.unit
def test_ingest_zotero_attachment_key_reads_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """Mode B with an attachment key: read .md directly, pull DOI from parent.

    Regression test for Bug B — pyzotero's ``children()`` 400s on markdown
    attachments, so the route must detect ``itemType == 'attachment'`` and
    read the local file directly instead of routing through
    ``get_markdown_attachment``.
    """
    client = _make_dev_client(monkeypatch)

    md_file = tmp_path / "paper.md"
    md_file.write_text("# Hello\n\nBody.")

    fake_client = MagicMock()

    def _fake_item(key):
        if key == "ATT123":
            return {
                "key": "ATT123",
                "data": {
                    "itemType": "attachment",
                    "contentType": "text/markdown",
                    "filename": "paper.md",
                    "parentItem": "PAR456",
                },
            }
        if key == "PAR456":
            return {
                "key": "PAR456",
                "data": {"itemType": "journalArticle", "DOI": "10.0/parent"},
            }
        raise AssertionError(f"unexpected item() call with {key}")

    fake_client._zot.item.side_effect = _fake_item
    fake_client.get_attachment_local_path.return_value = md_file

    captured = {}

    def _capture(*, doi, fulltext):
        captured["doi"] = doi
        captured["fulltext"] = fulltext
        return "<div>ok</div>"

    with patch(
        "scripts.server.routes.dev._build_zotero_client_for_dev",
        return_value=fake_client,
    ), patch(
        "scripts.server.routes.dev._run_sandbox_ingest",
        side_effect=_capture,
    ):
        resp = client.post(
            "/api/dev/ingest/zotero-key", data={"zotero_key": "ATT123"}
        )

    assert resp.status_code == 200
    # children() must NOT be called — that's the whole point of the fix.
    assert not fake_client.get_markdown_attachment.called
    assert captured["doi"] == "10.0/parent"
    assert "Hello" in captured["fulltext"]


@pytest.mark.unit
def test_ingest_zotero_attachment_key_rejects_non_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mode B with a PDF attachment key short-circuits with a clear error."""
    client = _make_dev_client(monkeypatch)

    fake_client = MagicMock()
    fake_client._zot.item.return_value = {
        "key": "ATT999",
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "filename": "paper.pdf",
            "parentItem": "PAR111",
        },
    }

    with patch(
        "scripts.server.routes.dev._build_zotero_client_for_dev",
        return_value=fake_client,
    ):
        resp = client.post(
            "/api/dev/ingest/zotero-key", data={"zotero_key": "ATT999"}
        )

    assert resp.status_code == 200
    assert 'class="pill danger"' in resp.text
    assert "not a markdown file" in resp.text


@pytest.mark.unit
def test_ingest_doi_paginates_full_match_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mode C: the target DOI may be past position 10 — everything() must scan all."""
    client = _make_dev_client(monkeypatch)

    # Build 25 items, only the 22nd has the matching DOI.
    items = []
    for i in range(25):
        if i == 21:
            items.append({"key": f"K{i}", "data": {"DOI": "10.0/target"}})
        else:
            items.append({"key": f"K{i}", "data": {"DOI": f"10.0/other-{i}"}})

    fake_client = MagicMock()
    fake_client._zot.everything.return_value = iter(items)
    fake_client.get_markdown_attachment.return_value = "# md body"

    captured = {}

    def _capture(*, doi, fulltext):
        captured["doi"] = doi
        return "<div>ok</div>"

    with patch(
        "scripts.server.routes.dev._build_zotero_client_for_dev",
        return_value=fake_client,
    ), patch(
        "scripts.server.routes.dev._run_sandbox_ingest",
        side_effect=_capture,
    ):
        resp = client.post("/api/dev/ingest/doi", data={"doi": "10.0/target"})

    assert resp.status_code == 200
    assert captured["doi"] == "10.0/target"


@pytest.mark.unit
def test_ingest_extraction_failure_reports_stage_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When extract_paper raises, the trace shows a <failed:...> danger row."""
    client = _make_dev_client(monkeypatch)

    fake_chunks = [MagicMock()]

    class BoomError(RuntimeError):
        pass

    def _raise(*args, **kwargs):
        raise BoomError("LLM exploded")

    with patch("scripts.core.chunker.chunk_markdown", return_value=fake_chunks), \
         patch("scripts.llm.extractor.extract_paper", side_effect=_raise), \
         patch("scripts.llm.llm_client.get_client", return_value=MagicMock()), \
         patch("scripts.core.config.get_config", return_value=MagicMock()):
        resp = client.post(
            "/api/dev/ingest/markdown",
            data={"doi": "10.0/boom", "markdown": "# x"},
        )

    assert resp.status_code == 200
    body = resp.text
    # Stage 1 (chunk) still succeeded; stage 2 failure shows up.
    assert 'class="pill success"' in body  # chunk_markdown stage
    assert 'class="pill danger"' in body
    assert "BoomError" in body
    assert "LLM exploded" in body
