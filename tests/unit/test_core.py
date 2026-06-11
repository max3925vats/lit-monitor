"""
Phase 2 unit tests — Core infrastructure.
All tests are marked @pytest.mark.unit.
No external services required (Zotero API calls are mocked).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w") as fh:
        yaml.dump(data, fh)
def _make_paths_yaml(tmp_path: Path, vault_path: str | None = None) -> Path:
    p = tmp_path / "paths.yaml"
    _vault = str(tmp_path / "vault") if vault_path is None else vault_path
    _write_yaml(p, {
        "zotero": {
            "library_type": "user",
            "library_id": "12345",
            "local_storage_path": str(tmp_path / "zotero_storage"),
            "collection_name": "lit-monitor",
        },
        "obsidian": {
            "vault_path": _vault,
            "papers_folder": "Literature/Papers",
            "books_folder": "Literature/Books",
            "digests_folder": "Literature/Digests",
            "connections_folder": "Literature/Connections",
        },
        "state_db": {"path": str(tmp_path / "state.db")},
        "logs": {"path": str(tmp_path / "logs"), "retention_days": 30},
    })
    return p
def _make_extraction_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "extraction.yaml"
    _write_yaml(p, {
        "brain_build": {
            "provider": "ollama",
            "model": "qwen2.5:7b",
            "timeout": 300,
        },
        "ingestion": {
            "provider": "ollama",
            "model": "qwen2.5:3b",
            "timeout": 300,
        },
        "comparison_models": [],
    })
    return p
# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_config_loads_successfully(tmp_path):
    from lit_monitor.core.config import Config
    paths = _make_paths_yaml(tmp_path)
    extraction = _make_extraction_yaml(tmp_path)
    cfg = Config(paths_yaml=paths, extraction_yaml=extraction)
    assert cfg.zotero.library_id == "12345"
    assert cfg.brain_build.model == "qwen2.5:7b"
    assert cfg.ingestion.model == "qwen2.5:3b"
@pytest.mark.unit
def test_config_rejects_tilde_vault_path(tmp_path):
    from lit_monitor.core.config import Config
    paths = _make_paths_yaml(tmp_path, vault_path="~/MyVault")
    extraction = _make_extraction_yaml(tmp_path)
    with pytest.raises(ValueError, match="full absolute path"):
        Config(paths_yaml=paths, extraction_yaml=extraction)
@pytest.mark.unit
def test_config_rejects_empty_vault_path(tmp_path):
    from lit_monitor.core.config import Config
    paths = _make_paths_yaml(tmp_path, vault_path="")
    extraction = _make_extraction_yaml(tmp_path)
    with pytest.raises(ValueError, match="not set"):
        Config(paths_yaml=paths, extraction_yaml=extraction)
@pytest.mark.unit
def test_config_expands_tilde_in_storage_path(tmp_path):
    from lit_monitor.core.config import Config
    p = tmp_path / "paths.yaml"
    _write_yaml(p, {
        "zotero": {
            "library_id": "1",
            "local_storage_path": "~/Zotero/storage",
        },
        "obsidian": {"vault_path": str(tmp_path / "vault")},
        "state_db": {"path": str(tmp_path / "state.db")},
        "logs": {"path": str(tmp_path / "logs")},
    })
    extraction = _make_extraction_yaml(tmp_path)
    cfg = Config(paths_yaml=p, extraction_yaml=extraction)
    # Should be expanded — not start with ~
    assert not cfg.zotero.local_storage_path.startswith("~")
# ---------------------------------------------------------------------------
# StateDB tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_state_db_upsert_and_retrieve(tmp_path):

    from lit_monitor.core.state_db import StateDB
    db = StateDB(tmp_path / "test.db")
    db.upsert_paper({
        "doi": "10.1000/test",
        "title": "Test Paper",
        "authors": ["Smith, John"],
        "year": 2024,
        "source_type": "paper",
        "status": "pending",
    })
    row = db.get_paper("10.1000/test")
    assert row is not None
    assert row["title"] == "Test Paper"
    assert row["status"] == "pending"
@pytest.mark.unit
def test_state_db_mark_status(tmp_path):
    from lit_monitor.core.state_db import StateDB
    db = StateDB(tmp_path / "test.db")
    db.upsert_paper({"doi": "10.1/x", "status": "pending", "source_type": "paper"})
    db.mark_status("10.1/x", "extraction_complete")
    row = db.get_paper("10.1/x")
    assert row["status"] == "extraction_complete"
@pytest.mark.unit
def test_state_db_get_all_by_source_type(tmp_path):
    from lit_monitor.core.state_db import StateDB
    db = StateDB(tmp_path / "test.db")
    db.upsert_paper({"doi": "10.1/paper1", "source_type": "paper"})
    db.upsert_paper({"doi": "10.1/paper2", "source_type": "paper"})
    db.upsert_paper({"doi": "isbn:111", "source_type": "book"})
    papers = db.get_all_by_source_type("paper")
    assert len(papers) == 2
    books = db.get_all_by_source_type("book")
    assert len(books) == 1
@pytest.mark.unit
def test_state_db_known_dois(tmp_path):
    from lit_monitor.core.state_db import StateDB

    db = StateDB(tmp_path / "test.db")
    db.upsert_paper({"doi": "10.1/a", "source_type": "paper"})
    db.upsert_paper({"doi": "10.1/b", "source_type": "paper"})
    known = db.known_dois()
    assert "10.1/a" in known
    assert "10.1/b" in known
    assert "10.1/c" not in known
@pytest.mark.unit
def test_state_db_brain_build_pass_tracking(tmp_path):
    from lit_monitor.core.state_db import StateDB
    db = StateDB(tmp_path / "test.db")
    db.upsert_brain_build_progress("ZKEY1", "10.1/test")
    db.mark_brain_build_pass("ZKEY1", 1)
    db.mark_brain_build_pass("ZKEY1", 2)
    prog = db.get_brain_build_progress("ZKEY1")
    assert prog["pass1_complete"] == 1
    assert prog["pass2_complete"] == 1
    assert prog["pass3_complete"] == 0
    assert prog["fully_complete"] == 0
    # Complete pass 3 — should mark fully_complete
    db.mark_brain_build_pass("ZKEY1", 3)
    prog = db.get_brain_build_progress("ZKEY1")
    assert prog["fully_complete"] == 1
# ---------------------------------------------------------------------------
# I4 — schema-aware fully_complete in brain_build_progress
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_mark_brain_build_pass_paper_flips_fully_complete_at_pass_3(tmp_path):
    """Paper schema (3 passes): fully_complete must not fire after pass 2."""
    from lit_monitor.core.state_db import StateDB
    db = StateDB(tmp_path / "test.db")
    db.upsert_brain_build_progress("ZKEY_I4A", "10.1/paper")
    db.mark_brain_build_pass("ZKEY_I4A", 1, content_type="paper")
    db.mark_brain_build_pass("ZKEY_I4A", 2, content_type="paper")
    prog = db.get_brain_build_progress("ZKEY_I4A")
    assert prog["pass1_complete"] == 1
    assert prog["pass2_complete"] == 1
    assert prog["fully_complete"] == 0  # still waiting for pass 3
    db.mark_brain_build_pass("ZKEY_I4A", 3, content_type="paper")
    prog = db.get_brain_build_progress("ZKEY_I4A")
    assert prog["fully_complete"] == 1
# ---------------------------------------------------------------------------
# M4 — state_db SQL/migration robustness
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_mark_brain_build_pass_rejects_invalid_pass_num(tmp_path):
    """M4: pass_num outside the static column map must raise ValueError.

    Defends against a regression where upstream validation might be dropped:
    the static `_PASS_COMPLETE_COLUMNS` lookup must still refuse unknown
    pass numbers instead of building a column name from the integer.
    """
    from lit_monitor.core.state_db import StateDB
    db = StateDB(tmp_path / "test.db")
    db.upsert_brain_build_progress("ZKEY_M4", "10.1/m4")
    with pytest.raises(ValueError):
        db.mark_brain_build_pass("ZKEY_M4", 99)


@pytest.mark.unit
def test_state_db_init_is_idempotent(tmp_path):
    """M4: instantiating StateDB twice on the same path must not raise.

    The PRAGMA-based migration pre-check must skip ALTERs whose columns
    already exist on a second init, replacing the older catch-and-suppress
    pattern that relied on SQLite error-message string-matching.
    """
    from lit_monitor.core.state_db import StateDB
    db_path = tmp_path / "test.db"
    StateDB(db_path)
    # Second init on the same file — would previously have hit ALTER TABLE
    # "duplicate column" errors; with PRAGMA pre-check it is a no-op.
    StateDB(db_path)
# ---------------------------------------------------------------------------
# Zotero client path construction
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_zotero_path_construction(tmp_path):
    """
    ZoteroClient.get_attachment_local_path() should build the correct path
    without making any API calls.
    """
    from lit_monitor.core.zotero_client import ZoteroClient
    # Patch pyzotero.Zotero so no real API call is made
    with patch("pyzotero.zotero.Zotero.__init__", return_value=None):
        client = ZoteroClient(
            library_id="12345",
            api_key="fake_key",
            local_storage_path=str(tmp_path / "storage"),
        )
    path = client.get_attachment_local_path("ABCDEF12", "paper.pdf")
    assert path == tmp_path / "storage" / "ABCDEF12" / "paper.pdf"
@pytest.mark.unit
def test_zotero_extract_authors():
    from lit_monitor.core.zotero_client import ZoteroClient

    data = {
        "creators": [
            {"creatorType": "author", "lastName": "Smith", "firstName": "John"},
            {"creatorType": "author", "lastName": "Jones", "firstName": "Alice"},
            {"creatorType": "editor", "lastName": "Brown", "firstName": "Bob"},
        ]
    }
    authors = ZoteroClient.extract_authors(data)
    assert authors == ["Smith, John", "Jones, Alice", "Brown, Bob"]

@pytest.mark.unit
def test_zotero_get_current_version(tmp_path):
    """get_current_version() calls last_modified_version() as a method and returns int.

    Regression test for G6: in pyzotero ≥1.11, last_modified_version is a method.
    The old code called int() on the unbound method object, raising TypeError.
    """
    from lit_monitor.core.zotero_client import ZoteroClient

    with patch("pyzotero.zotero.Zotero.__init__", return_value=None):
        client = ZoteroClient(
            library_id="12345",
            api_key="fake_key",
            local_storage_path=str(tmp_path / "storage"),
        )
    # last_modified_version must be called as a method, not read as an attribute.
    client._zot.last_modified_version = MagicMock(return_value=42)
    version = client.get_current_version()
    assert version == 42
    client._zot.last_modified_version.assert_called_once_with()

@pytest.mark.unit
def test_zotero_get_current_version_raises_on_non_int(tmp_path):
    """get_current_version() raises RuntimeError when pyzotero returns a non-int.

    Ensures future pyzotero API changes surface with a clear error message
    rather than a cryptic downstream TypeError.
    """
    from lit_monitor.core.zotero_client import ZoteroClient

    with patch("pyzotero.zotero.Zotero.__init__", return_value=None):
        client = ZoteroClient(
            library_id="12345",
            api_key="fake_key",
            local_storage_path=str(tmp_path / "storage"),
        )
    client._zot.last_modified_version = MagicMock(return_value="not-an-int")
    with pytest.raises(RuntimeError, match="pyzotero last_modified_version\\(\\) returned"):
        client.get_current_version()


@pytest.mark.unit
def test_zotero_paginate_early_exit(tmp_path):
    """L2: _paginate stops fetching once `limit` items have been collected.

    Against a 50k-item library, the previous "fetch everything then slice"
    pattern made hundreds of unnecessary round-trips when the caller only
    wanted, say, 10 items. The fix plumbs ``limit`` into ``_paginate`` so
    pagination breaks early.
    """
    from lit_monitor.core.zotero_client import ZoteroClient

    with patch("pyzotero.zotero.Zotero.__init__", return_value=None):
        client = ZoteroClient(library_id="12345", api_key="fake_key")

    # fetch_fn returns a full 100-item page every call. Without the limit
    # plumbing, this loop would only stop on an empty page (never, in this
    # test) — so the early-exit branch is what bounds the call count.
    call_count = {"n": 0}

    def fake_fetch(start, page_size):
        call_count["n"] += 1
        return [{"key": f"K{start + i}"} for i in range(page_size)]

    items = client._paginate(fake_fetch, page_size=100, limit=10)
    # Pagination must stop after the first full page (>= limit collected).
    assert call_count["n"] == 1
    # The returned list may contain >= limit; slicing is the caller's job.
    assert len(items) >= 10


@pytest.mark.unit
def test_zotero_create_note_reads_first_successful_value(tmp_path):
    """L2: create_note iterates result["successful"].values() instead of "0".

    pyzotero's create_items result is keyed by the input item's index as a
    string. Hardcoding "0" is fragile if pyzotero ever changes the keying
    scheme; reading the first value is robust.
    """
    from lit_monitor.core.zotero_client import ZoteroClient

    with patch("pyzotero.zotero.Zotero.__init__", return_value=None):
        client = ZoteroClient(library_id="12345", api_key="fake_key")

    client._zot.item_template = MagicMock(return_value={"itemType": "note", "note": ""})
    client._zot.create_items = MagicMock(
        return_value={
            # Use a non-"0" key to prove we are no longer indexing by "0".
            "successful": {"7": {"key": "NEWNOTEKEY"}},
            "failed": {},
            "unchanged": {},
            "success": {"7": "NEWNOTEKEY"},
        }
    )
    new_key = client.create_note("PARENTKEY", "<p>hi</p>")
    assert new_key == "NEWNOTEKEY"


@pytest.mark.unit
def test_zotero_create_note_raises_on_empty_successful(tmp_path):
    """L2 review: create_note raises a clear RuntimeError when pyzotero
    returns an empty ``successful`` dict (e.g. the create silently failed
    and pyzotero populated ``failed`` instead). Without this guard,
    ``next(iter(...))`` would raise StopIteration, which surfaces as a
    confusing error for the caller.
    """
    from lit_monitor.core.zotero_client import ZoteroClient

    with patch("pyzotero.zotero.Zotero.__init__", return_value=None):
        client = ZoteroClient(library_id="12345", api_key="fake_key")

    client._zot.item_template = MagicMock(return_value={"itemType": "note", "note": ""})
    client._zot.create_items = MagicMock(
        return_value={
            "successful": {},
            "failed": {"0": {"code": 400, "message": "Bad Request"}},
            "unchanged": {},
            "success": {},
        }
    )
    with pytest.raises(RuntimeError, match="no successful item"):
        client.create_note("PARENTKEY", "<p>hi</p>")


@pytest.mark.unit
def test_zotero_init_rejects_old_pyzotero(tmp_path):
    """L2: __init__ raises a clear RuntimeError if last_modified_version is missing.

    Discovery polling depends on ``last_modified_version()``; if a user has
    pyzotero <1.11 installed, we want a loud upgrade prompt at construction
    time rather than a cryptic AttributeError deep in the polling loop.
    """
    from lit_monitor.core.zotero_client import ZoteroClient

    class FakeOldZotero:
        def __init__(self, *_args, **_kwargs):
            # Intentionally no last_modified_version attribute.
            pass

    with patch("lit_monitor.core.zotero_client.zotero.Zotero", FakeOldZotero):
        with pytest.raises(RuntimeError, match="pyzotero too old"):
            ZoteroClient(library_id="12345", api_key="fake_key")
# ---------------------------------------------------------------------------
# strip_references_section (A5) / strip_end_matter (M2)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_strip_references_section_markdown_heading():
    """Standard markdown '## References' heading is stripped."""
    from lit_monitor.core.markdown_processor import strip_references_section
    body = "Introduction text.\n\nMethods text.\n\nDiscussion.\n" * 5  # >70% of doc
    refs = "\n## References\n\nSmith 2020. Journal of X.\nJones 2018. Nature.\n"
    result = strip_references_section(body + refs)
    assert "Smith 2020" not in result
    assert "Introduction text." in result

@pytest.mark.unit
def test_strip_references_section_allcaps_heading():
    """REFERENCES (all-caps, no markdown) is stripped."""
    from lit_monitor.core.markdown_processor import strip_references_section
    body = "Some paper content.\n" * 20
    refs = "\nREFERENCES\n\n1. Author A. Title. 2019.\n"
    result = strip_references_section(body + refs)
    assert "Author A" not in result
    assert "Some paper content." in result

@pytest.mark.unit
def test_strip_references_section_bibliography_heading():
    """'Bibliography' heading variant is stripped."""
    from lit_monitor.core.markdown_processor import strip_references_section
    body = "Paper body.\n" * 20
    refs = "\n# Bibliography\n\nDoe 2021. Some Journal.\n"
    result = strip_references_section(body + refs)
    assert "Doe 2021" not in result

@pytest.mark.unit
def test_strip_references_section_works_cited():
    """'Works Cited' heading is stripped."""
    from lit_monitor.core.markdown_processor import strip_references_section
    body = "Paper body.\n" * 20
    refs = "\nWorks Cited\n\nAuthor X. Title. 2020.\n"
    result = strip_references_section(body + refs)
    assert "Author X" not in result

@pytest.mark.unit
def test_strip_references_section_no_heading_unchanged():
    """Text without a reference heading is returned unchanged."""
    from lit_monitor.core.markdown_processor import strip_references_section
    text = "This paper has no bibliography section.\nJust body text.\n" * 10
    result = strip_references_section(text)
    assert result == text

@pytest.mark.unit
def test_strip_references_section_early_heading_not_stripped():
    """A 'References' heading in the first 70% of the document must NOT be stripped."""
    from lit_monitor.core.markdown_processor import strip_references_section
    # Put the heading at the very start — well within the first 70%
    early = "## References\n\nEarly inline references section.\n"
    body = "Main paper body text.\n" * 30  # large enough to push early heading < 70%
    result = strip_references_section(early + body)
    # The early heading should NOT trigger stripping — body must be present
    assert "Main paper body text." in result

@pytest.mark.unit
def test_strip_references_section_empty_string():
    """Empty string returns empty string without error."""
    from lit_monitor.core.markdown_processor import strip_references_section
    assert strip_references_section("") == ""

@pytest.mark.unit
def test_strip_references_section_heading_with_colon():
    """'References:' (trailing colon) is stripped."""
    from lit_monitor.core.markdown_processor import strip_references_section
    body = "Body text.\n" * 20
    refs = "\nReferences:\n\nSmith 2019.\n"
    result = strip_references_section(body + refs)
    assert "Smith 2019" not in result

# ---------------------------------------------------------------------------
# strip_end_matter (M2) — direct unit tests for each end-matter category
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_strip_end_matter_acknowledgements():
    """'Acknowledgements' heading in last 30% is stripped."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "Paper body content.\n" * 20
    tail = "\n## Acknowledgements\n\nWe thank the funding agency.\n"
    result = strip_end_matter(body + tail)
    assert "We thank the funding agency" not in result
    assert "Paper body content." in result

@pytest.mark.unit
def test_strip_end_matter_acknowledgments_alternate_spelling():
    """'Acknowledgments' (US spelling, no 'e') is stripped."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "Paper body content.\n" * 20
    tail = "\nAcknowledgments\n\nThe authors acknowledge NIH support.\n"
    result = strip_end_matter(body + tail)
    assert "NIH support" not in result

@pytest.mark.unit
def test_strip_end_matter_funding():
    """'Funding' heading is stripped."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "Main results are described here.\n" * 20
    tail = "\n# Funding\n\nThis work was funded by Grant 12345.\n"
    result = strip_end_matter(body + tail)
    assert "Grant 12345" not in result
    assert "Main results" in result

@pytest.mark.unit
def test_strip_end_matter_conflict_of_interest():
    """'Conflict of Interest' heading is stripped."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "Study design and methods.\n" * 20
    tail = "\n## Conflict of Interest\n\nThe authors declare no conflict.\n"
    result = strip_end_matter(body + tail)
    assert "authors declare no conflict" not in result

@pytest.mark.unit
def test_strip_end_matter_author_contributions():
    """'Author Contributions' heading is stripped."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "Experimental results follow.\n" * 20
    tail = "\n## Author Contributions\n\nJ.S. designed the experiments.\n"
    result = strip_end_matter(body + tail)
    assert "J.S. designed" not in result

@pytest.mark.unit
def test_strip_end_matter_data_availability():
    """'Data Availability Statement' heading is stripped."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "Discussion of findings.\n" * 20
    tail = "\n### Data Availability Statement\n\nData available on request.\n"
    result = strip_end_matter(body + tail)
    assert "Data available on request" not in result

@pytest.mark.unit
def test_strip_end_matter_code_availability():
    """'Code Availability' heading is stripped."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "Conclusions are drawn here.\n" * 20
    tail = "\n## Code Availability\n\nCode at https://github.com/example.\n"
    result = strip_end_matter(body + tail)
    assert "github.com/example" not in result

@pytest.mark.unit
def test_strip_end_matter_supplementary_information():
    """'Supplementary Information' heading is stripped."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "The paper presents novel findings.\n" * 20
    tail = "\n# Supplementary Information\n\nFigure S1 shows the control data.\n"
    result = strip_end_matter(body + tail)
    assert "Figure S1" not in result

@pytest.mark.unit
def test_strip_end_matter_also_strips_references():
    """strip_end_matter is a superset of strip_references_section — references stripped too."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    body = "Paper body.\n" * 20
    tail = "\n## References\n\nSmith 2020. Journal of X.\n"
    result = strip_end_matter(body + tail)
    assert "Smith 2020" not in result
    assert "Paper body." in result

@pytest.mark.unit
def test_strip_end_matter_early_heading_not_stripped():
    """A heading in the first 70% of the document must NOT trigger stripping."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    early = "## Acknowledgements\n\nSome inline thank-you.\n"
    body = "Main results section.\n" * 30
    result = strip_end_matter(early + body)
    assert "Main results section." in result

@pytest.mark.unit
def test_strip_end_matter_empty_string():
    """Empty string is returned unchanged."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    assert strip_end_matter("") == ""

@pytest.mark.unit
def test_strip_end_matter_no_end_matter_unchanged():
    """Text with no end-matter headings is returned unchanged."""
    from lit_monitor.core.markdown_processor import strip_end_matter
    text = "Introduction.\nMethods.\nResults.\nConclusion.\n" * 5
    assert strip_end_matter(text) == text


@pytest.mark.unit
def test_strip_end_matter_long_review_window_constant(monkeypatch):
    """L5: the 30% search window is a named constant; long reviews still strip.

    Construct a long synthetic review whose end-matter heading sits at ~60%
    of the document (i.e. *outside* the default last-30% search window but
    *inside* a hypothetical 50% window). With the default
    ``_END_MATTER_SEARCH_WINDOW = 0.30`` the heading is missed (limitation
    documented). Bumping the constant to 0.50 catches it — proving the
    threshold is the single knob controlling this behaviour and that the
    long-review end-matter path still works when widened.
    """
    from lit_monitor.core import markdown_processor
    from lit_monitor.core.markdown_processor import strip_end_matter

    # Build a long doc: 60% body, then the end-matter heading + tail, then
    # 40% of trailing references-like junk. Heading position ≈ 60% of total.
    head_chunk = "Review body paragraph with citations [1].\n" * 600
    tail_chunk = "Smith 2020. J Foo.\nJones 2018. Nature.\n" * 400
    end_matter = "\n## Acknowledgements\n\nWe thank the funders.\n"
    text = head_chunk + end_matter + tail_chunk

    # Sanity-check heading position: must sit between 50% and 70%.
    heading_pos = text.index("## Acknowledgements") / len(text)
    assert 0.50 < heading_pos < 0.70, f"fixture drift: heading at {heading_pos:.2%}"

    # Default (30%) window: heading is in the first 70% → NOT stripped.
    assert "We thank the funders." in strip_end_matter(text)

    # Widen the constant to 50% → heading is now within the search window
    # and the long review's end-matter gets stripped as expected.
    monkeypatch.setattr(markdown_processor, "_END_MATTER_SEARCH_WINDOW", 0.50)
    stripped = strip_end_matter(text)
    assert "We thank the funders." not in stripped
    assert "Review body paragraph with citations [1]." in stripped


@pytest.mark.unit
def test_strip_end_matter_pathological_input_completes_promptly():
    """ReDoS regression: heading regex stays linear on crafted input.

    The end-matter heading regex previously used ambiguous quantifiers
    (``acknowledg(?:e?ment|e?ments)``, ``conflict(?:s)?``). After the de-ReDoS
    rewrite (``acknowledge?ments?``, ``conflicts?``) a pathological string —
    a near-heading prefix followed by a long run that never completes the
    heading — must return promptly and correctly (input unchanged: no heading
    matched).
    """
    import time

    from lit_monitor.core.markdown_processor import strip_end_matter

    # A long run that looks like it's starting a heading but never closes it,
    # placed in the last-30% window. Must not trigger catastrophic backtracking.
    body = "Paper body text.\n" * 100
    payload = body + "\nacknowledge" + "e" * 200_000
    start = time.perf_counter()
    result = strip_end_matter(payload)
    elapsed = time.perf_counter() - start
    # No valid end-matter heading line ⇒ text returned unchanged.
    assert result == payload
    assert elapsed < 2.0, f"strip_end_matter too slow: {elapsed:.2f}s (ReDoS?)"


# ---------------------------------------------------------------------------
# N13 — StateDB: N7 cleanup logs WARNING on failure instead of silently passing
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_state_db_n7_cleanup_logs_warning_on_failure(tmp_path, caplog):
    """N13: when the N7 stale-row DELETE raises, a WARNING is logged and startup completes.

    Approach: construct a real DB, remove r10_cleanup_done so the N7 block runs,
    install a BEFORE DELETE trigger on papers that raises, then construct a new
    StateDB — the cleanup DELETE will fail, which should produce a WARNING rather
    than silently passing.
    """
    import logging
    import sqlite3

    from lit_monitor.core.state_db import StateDB

    db_path = tmp_path / "test.db"
    # First init — sets up schema and marks r10_cleanup_done.
    StateDB(db_path)

    # Remove the done flag so the cleanup block will run again on next init.
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM kv_store WHERE key = 'r10_cleanup_done'")
    # Insert a stale 'book' row so the DELETE will actually match rows
    # (triggers only fire when rows are present to delete).
    conn.execute(
        "INSERT INTO papers (doi, title, source_type) VALUES ('10.0/stale', 'Stale Book', 'book')"
    )
    # Install a trigger that makes any DELETE on papers fail.
    conn.execute(
        "CREATE TRIGGER force_delete_fail BEFORE DELETE ON papers "
        "BEGIN SELECT RAISE(FAIL, 'test-forced failure'); END"
    )
    conn.commit()
    conn.close()

    with caplog.at_level(logging.WARNING, logger="lit_monitor.core.state_db"):
        StateDB(db_path)  # _init_schema fires the N7 cleanup → trigger raises → WARNING

    assert any("N7 stale-row cleanup failed" in r.message for r in caplog.records), (
        "Expected WARNING about N7 cleanup failure, but none was emitted"
    )
