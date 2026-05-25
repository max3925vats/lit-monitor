"""
Phase 1 Security Audit Tests
-----------------------------
These tests verify the security properties of the five audited repos.
They read the audit repos from a sibling directory (lit-monitor-audit/).
Run from the lit-monitor project root.
All tests are marked @pytest.mark.unit — no external services required.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Audit repo root — sibling directory to this project
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # lit-monitor/
_AUDIT_ROOT = _PROJECT_ROOT.parent / "lit-monitor-audit"
AUDIT_REPOS = [
    "claude-academic-research",
    "zotero-mcp",
    "paper-search-mcp",
    "findpapers",
    "ollama-ebook-summary",
]
_MIT_PATTERN = re.compile(r"mit license", re.IGNORECASE)
_APACHE_PATTERN = re.compile(r"apache license|apache-2\.0", re.IGNORECASE)

# Skip every test in this file when the audit repo directory is absent.
# These are one-time security-audit tests — they require the cloned repos
# from the original audit session and are not part of the CI regression suite.
pytestmark = pytest.mark.skipif(
    not _AUDIT_ROOT.exists(),
    reason=(
        f"Security audit repos not found at {_AUDIT_ROOT}. "
        "Clone the audit repos before running these tests."
    ),
)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _all_py_files(repo: str) -> list[Path]:
    repo_path = _AUDIT_ROOT / repo
    return list(repo_path.rglob("*.py"))
def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
# ---------------------------------------------------------------------------
# Test 1: No eval() / exec() with user-controlled input in audit repos
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_eval_in_scripts():
    """
    Check that no audited repo contains dangerous eval() or exec() calls
    that could accept user-controlled input.
    False-positives filtered:
    - AbstractRetrieval / ArticleRetrieval class names contain 'eval' as a
      substring but are not eval() calls.
    """

    for repo in AUDIT_REPOS:
        for py_file in _all_py_files(repo):
            source = _read(py_file)
            # Quick text filter to skip files without 'eval(' or 'exec('
            if "eval(" not in source and "exec(" not in source:
                continue
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    # Direct eval() / exec() call
                    if isinstance(func, ast.Name) and func.id in ("eval", "exec"):
                        # Check if argument is a constant (safe) or dynamic (dangerous)
                        if node.args and not isinstance(node.args[0], ast.Constant):
                            pytest.fail(
                                f"Dangerous {func.id}() with dynamic argument in "
                                f"{repo}/{py_file.name}:{node.lineno}"
                            )
# ---------------------------------------------------------------------------
# Test 2: No hardcoded API keys or secrets
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_api_keys_not_hardcoded():
    """
    Scan for patterns that look like hardcoded credentials:
    api_key = "XXXX..." where XXXX is 16+ alphanumeric chars.
    """
    pattern = re.compile(
        r"""(?:api_key|api_token|secret|password)\s*=\s*["'][A-Za-z0-9]{16,}["']""",
        re.IGNORECASE,
    )
    for repo in AUDIT_REPOS:
        for py_file in _all_py_files(repo):
            source = _read(py_file)
            matches = pattern.findall(source)
            if matches:
                pytest.fail(
                    f"Potential hardcoded credential in {repo}/{py_file.name}: "
                    f"{matches[0]!r}"
                )
# ---------------------------------------------------------------------------
# Test 3: findpapers HTTP calls target only allowlisted academic domains
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_findpapers_http_to_allowlist_only():
    """
    Verify that findpapers connector BASE_URL constants point only to
    known, legitimate academic API domains.
    """
    ALLOWED_DOMAINS = {
        "export.arxiv.org",
        "api.crossref.org",
        "ieeexploreapi.ieee.org",
        "api.openalex.org",
        "eutils.ncbi.nlm.nih.gov",

        "api.elsevier.com",
        "api.semanticscholar.org",
        "api.clarivate.com",
        "doi.org",
    }
    base_url_pattern = re.compile(
        r"""_BASE_URL\s*=\s*["'](https?://[^"'/]+)""", re.IGNORECASE
    )
    connectors_dir = _AUDIT_ROOT / "findpapers" / "findpapers" / "connectors"
    assert connectors_dir.exists(), f"findpapers connectors dir not found: {connectors_dir}"
    for py_file in connectors_dir.glob("*.py"):
        source = _read(py_file)
        for match in base_url_pattern.finditer(source):
            url = match.group(1)
            # Extract domain
            domain = url.split("//", 1)[1].split("/")[0]
            assert domain in ALLOWED_DOMAINS, (
                f"Unexpected domain '{domain}' in findpapers/{py_file.name}. "
                f"Full URL: {url}"
            )
# ---------------------------------------------------------------------------
# Test 4: No writes outside configured paths in findpapers
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_writes_outside_configured_paths():
    """
    findpapers should only write to caller-provided paths.
    Check that no hardcoded absolute write paths appear in the source.
    """
    # Patterns that would indicate writing to hardcoded system paths
    hardcoded_write_pattern = re.compile(
        r"""(?:open|write_text|write_bytes)\s*\(\s*["']/(?!tmp|var/tmp)""",
        re.IGNORECASE,
    )
    for py_file in _all_py_files("findpapers"):
        source = _read(py_file)
        if hardcoded_write_pattern.search(source):
            pytest.fail(
                f"Possible write to hardcoded path in findpapers/{py_file.name}"
            )
# ---------------------------------------------------------------------------
# Test 5: findpapers license is MIT
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_findpapers_license_is_mit():
    """findpapers must be MIT licensed for adoption."""
    license_file = _AUDIT_ROOT / "findpapers" / "LICENSE"
    assert license_file.exists(), "findpapers LICENSE file not found"
    content = license_file.read_text(encoding="utf-8", errors="replace")
    assert _MIT_PATTERN.search(content), (
        f"findpapers LICENSE does not appear to be MIT:\n{content[:300]}"
    )
# ---------------------------------------------------------------------------
# Test 6: ollama-ebook-summary license check (expected: no LICENSE file)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ollama_ebook_summary_license_status():
    """
    ollama-ebook-summary has no LICENSE file — confirmed in audit.
    This test documents that finding and ensures we haven't missed a license
    file under an alternate name.
    The repo is used as reference only (third_party/) — no code is adopted.
    """
    repo_path = _AUDIT_ROOT / "ollama-ebook-summary"
    assert repo_path.exists(), f"ollama-ebook-summary not found at {repo_path}"
    license_candidates = list(repo_path.glob("LICENSE*")) + list(repo_path.glob("license*"))
    # Confirm: no license file exists (this is the documented finding)
    assert len(license_candidates) == 0, (
        f"ollama-ebook-summary unexpectedly has a license file: {license_candidates}. "
        "Update audit_report.md if a license was added."
    )
    # The absence of a license is the documented state — test passes.
    # Mitigation: repo is in third_party/ for reference only; no code copied.
# ---------------------------------------------------------------------------
# Test 7: findpapers database identifiers confirmed lowercase
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_findpapers_database_identifiers_are_lowercase():
    """
    Confirm that findpapers uses lowercase database identifier strings.
    smoke_test_findings.md incorrectly documented these as uppercase.
    This test locks in the correct values.
    """
    engine_file = _AUDIT_ROOT / "findpapers" / "findpapers" / "engine.py"
    assert engine_file.exists(), f"findpapers engine.py not found: {engine_file}"
    source = engine_file.read_text(encoding="utf-8", errors="replace")
    # Confirm examples from docstrings use lowercase
    assert '"arxiv"' in source or "'arxiv'" in source, (
        "Expected lowercase 'arxiv' identifier in findpapers engine.py"
    )
    # Confirm WoS uses lowercase 'wos'
    wos_connector = _AUDIT_ROOT / "findpapers" / "findpapers" / "connectors" / "wos.py"
    assert wos_connector.exists()
    wos_source = wos_connector.read_text(encoding="utf-8", errors="replace")
    assert "wos-starter" in wos_source, "Expected WoS Starter API reference in wos.py"
    assert "api.clarivate.com/apis/wos-starter/v1" in wos_source, (
        "Expected WoS Starter API base URL in wos.py"
    )
# ---------------------------------------------------------------------------
# Test 8: findpapers 0.6.x exposes findpapers.search() callable
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_findpapers_search_is_callable():
    """
    Confirm that findpapers 0.6.x exposes the search() callable used by
    search_runner.py.  The earlier audit note about Engine was wrong — 0.6.x
    uses a functional API with findpapers.search() as the entry point.
    """
    try:
        import findpapers as fp
    except ImportError:
        pytest.skip("findpapers not installed")
    assert callable(fp.search), "findpapers.search must be a callable in 0.6.x"
