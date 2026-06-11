"""Tests for the CodeQL-recognized security barriers (fix/codeql-residual).

These cover the behavioural changes made when converting custom sanitizers into
barriers CodeQL's dataflow model recognizes:

- path-injection containment checks now RAISE on traversal (and keep legit
  in-base paths working),
- the Ollama reachability probe enforces a scheme + host allowlist (SSRF),
- the notify redirect is built from a constant + validated int (open-redirect),
- the end-matter regex stays linear (ReDoS) and still strips real headings.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lit_monitor.server.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _vault_config(vault: Path) -> SimpleNamespace:
    (vault / "Literature" / "Connections").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        obsidian=SimpleNamespace(
            vault_path=str(vault),
            papers_folder="Literature/Papers",
            connections_folder="Literature/Connections",
        )
    )


# ---------------------------------------------------------------------------
# Synthesize vault containment
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_synthesis_note_writes_inside_vault(tmp_path: Path) -> None:
    """A normal topic produces a note under the vault (legit path still works)."""
    from lit_monitor.obsidian_tools.synthesize import _write_synthesis_note

    cfg = _vault_config(tmp_path / "vault")
    out = _write_synthesis_note("filtration kinetics", "# body", cfg, None)
    out_path = Path(out).resolve()
    assert out_path.is_relative_to((tmp_path / "vault").resolve())
    assert out_path.exists()


@pytest.mark.unit
def test_synthesis_note_rejects_escape_via_connections_folder(tmp_path: Path) -> None:
    """A connections_folder that escapes the vault is refused by the barrier."""
    from lit_monitor.obsidian_tools.synthesize import _write_synthesis_note

    cfg = _vault_config(tmp_path / "vault")
    # An absolute/escaping folder must not let the write land outside the vault.
    with pytest.raises(ValueError, match="escapes the vault"):
        _write_synthesis_note("topic", "# body", cfg, "../../escape")


# ---------------------------------------------------------------------------
# update_note_preserve_persist_zones containment
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_persist_zone_write_contained_in_base(tmp_path: Path) -> None:
    """With base_dir set, an in-base path writes; an escaping path raises."""
    from lit_monitor.output.obsidian_writer import (
        update_note_preserve_persist_zones,
    )

    base = tmp_path / "vault"
    base.mkdir()
    good = base / "note.md"
    update_note_preserve_persist_zones(good, "hello", base_dir=base)
    assert good.read_text() == "hello"

    outside = tmp_path / "elsewhere" / "evil.md"
    outside.parent.mkdir()
    with pytest.raises(ValueError, match="escapes the vault"):
        update_note_preserve_persist_zones(outside, "x", base_dir=base)
    assert not outside.exists()


# ---------------------------------------------------------------------------
# atomic_write realpath normalization
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_atomic_write_returns_realpath(tmp_path: Path) -> None:
    """atomic_write_text writes the content and returns an absolute real path."""
    from lit_monitor.core.atomic_write import atomic_write_text

    # A path containing a redundant ``.`` segment must normalize cleanly.
    target = tmp_path / "." / "out.md"
    result = atomic_write_text(target, "content")
    assert result.is_absolute()
    assert result == Path(target.resolve())
    assert (tmp_path / "out.md").read_text() == "content"


# ---------------------------------------------------------------------------
# Ollama reachability SSRF allowlist
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_ollama_test_rejects_non_http_scheme(client: TestClient) -> None:
    resp = client.get("/setup/api/ollama-test", params={"host": "file:///etc/passwd"})
    assert resp.status_code == 200
    assert "must start with http" in resp.text


@pytest.mark.unit
def test_ollama_test_rejects_disallowed_host(client: TestClient) -> None:
    """An http URL to an arbitrary host is refused before any request."""
    resp = client.get(
        "/setup/api/ollama-test", params={"host": "http://169.254.169.254"}
    )
    assert resp.status_code == 200
    assert "localhost or ollama.com" in resp.text


@pytest.mark.unit
def test_ollama_test_rejects_userinfo_smuggle(client: TestClient) -> None:
    """``http://localhost@evil.com`` resolves host=evil.com and is refused."""
    resp = client.get(
        "/setup/api/ollama-test", params={"host": "http://localhost@evil.com"}
    )
    assert resp.status_code == 200
    assert "localhost or ollama.com" in resp.text


@pytest.mark.unit
def test_ollama_test_allows_localhost(client: TestClient, monkeypatch) -> None:
    """A localhost URL passes the allowlist and issues the request."""
    import httpx

    captured: dict[str, str] = {}

    class _Resp:
        status_code = 200

        def json(self) -> dict:
            return {"models": [1, 2]}

    def _fake_get(url: str, **kwargs):  # noqa: ANN003
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(httpx, "get", _fake_get)
    resp = client.get(
        "/setup/api/ollama-test", params={"host": "http://localhost:11434"}
    )
    assert resp.status_code == 200
    # URL is rebuilt from validated components, preserving the port.
    assert captured["url"] == "http://localhost:11434/api/tags"
    assert "reachable" in resp.text


# ---------------------------------------------------------------------------
# Dev sandbox note-path containment
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sandbox_note_path_contained(tmp_path: Path, monkeypatch) -> None:
    """Both a normal DOI and a traversal-laden DOI stay inside the sandbox dir.

    The slug regex neutralizes separators/``..`` so the ``is_relative_to``
    barrier confirms (rather than rescues) containment — every returned path is
    provably under the sandbox folder.
    """
    import lit_monitor.server.dev_sandbox as ds
    from lit_monitor.server.routes import dev as dev_routes

    sandbox = (tmp_path / "vault" / "Literature" / "_Dev").resolve()
    sandbox.mkdir(parents=True)
    monkeypatch.setattr(ds, "sandbox_vault_subfolder", lambda *a, **k: sandbox)

    for doi in ("10.1000/abc", "../../../../etc/passwd", "a/b/c"):
        result = dev_routes._sandbox_note_path_for(doi)
        assert result.resolve().is_relative_to(sandbox)


# ---------------------------------------------------------------------------
# Notify redirect (open-redirect barrier)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_notify_browser_redirect_uses_constant_prefix(
    client: TestClient, monkeypatch
) -> None:
    import lit_monitor.server.routes.discovery_notify as dn

    monkeypatch.setattr(dn, "_get_preferred_viewer", lambda: "browser")
    resp = client.get(
        "/discovery/notify-handler", params={"run_id": 7}, follow_redirects=False
    )
    assert resp.status_code == 302
    assert resp.headers["location"] == "/discovery/7"


# ---------------------------------------------------------------------------
# End-matter regex: linear + still strips real headings
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_strip_end_matter_still_strips_heading() -> None:
    from lit_monitor.core.markdown_processor import strip_end_matter

    body = "Intro paragraph.\n" * 50
    text = body + "\n## References\n[1] Foo et al.\n"
    out = strip_end_matter(text)
    assert "References" not in out
    assert "Intro paragraph." in out


@pytest.mark.unit
def test_end_matter_regex_is_linear_on_whitespace_run() -> None:
    """A long trailing-whitespace line must not blow up matching time."""
    from lit_monitor.core.markdown_processor import _END_MATTER_HEADING

    # Craft the classic ReDoS payload: an almost-matching heading followed by a
    # huge whitespace run that never reaches the ``$`` anchor on the same line.
    payload = "references " + (" " * 50_000) + "x"
    start = time.perf_counter()
    assert _END_MATTER_HEADING.search(payload) is None
    elapsed = time.perf_counter() - start
    # Linear matching finishes near-instantly; a backtracking blowup would take
    # seconds. Generous bound to avoid CI flakiness.
    assert elapsed < 1.0
