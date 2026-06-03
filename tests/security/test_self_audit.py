"""
Self Security Audit (B12 / bundle E6)
-------------------------------------
This module audits lit-monitor's OWN source tree under ``scripts/`` for a set
of concrete security anti-patterns. Unlike ``test_audit.py`` (which audits the
five EXTERNAL cloned repos and is gated behind LIT_MONITOR_LIVE), this file:

  * runs in the DEFAULT test suite — no LIT_MONITOR_LIVE gate,
  * needs NO network and NO external repos — it only reads source files,
  * has ONE test per anti-pattern so a failure pinpoints the class of issue.

The checks mirror the spirit of the project security rules:
  * no hardcoded secrets / API keys / tokens in source,
  * no ``eval()`` / ``exec()`` (dynamic-code execution),
  * no disabled TLS verification,
  * SQL through parameterized placeholders only (no value-position string
    interpolation into ``.execute()``),
  * no ``subprocess(..., shell=True)``,
  * the user's secrets file (``~/.config/lit-monitor/config.toml``) is only
    read by an explicit allowlist of credential-loading modules.

Each check COLLECTS all violations and asserts the list is empty with a
``file:line`` message, so a future regression is caught with a clear pointer.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Source tree under audit
# ---------------------------------------------------------------------------
_TEST_DIR = Path(__file__).parent
_PROJECT_ROOT = _TEST_DIR.parent.parent  # lit-monitor/
_SCRIPTS_ROOT = _PROJECT_ROOT / "scripts"


def _source_files() -> list[Path]:
    """All Python source files under scripts/, excluding bytecode caches."""
    return [
        p
        for p in sorted(_SCRIPTS_ROOT.rglob("*.py"))
        if "__pycache__" not in p.parts
    ]


def _rel(p: Path) -> str:
    """Path relative to the project root for readable failure messages."""
    try:
        return str(p.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(p)


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _parse(p: Path, source: str) -> ast.Module | None:
    try:
        return ast.parse(source, filename=str(p))
    except SyntaxError:
        return None


def test_scripts_tree_is_present() -> None:
    """Sanity guard: the audit only means something if it finds source files.

    Protects against a refactor that moves scripts/ and silently turns every
    other check into a vacuous pass over an empty file list.
    """
    files = _source_files()
    assert _SCRIPTS_ROOT.is_dir(), f"scripts/ not found at {_SCRIPTS_ROOT}"
    assert len(files) > 50, (
        f"Expected the scripts/ tree to contain many source files, found "
        f"{len(files)}. Did the source layout move?"
    )


# ---------------------------------------------------------------------------
# Check 1: No hardcoded secrets / API keys / tokens
# ---------------------------------------------------------------------------
# Two complementary detectors:
#   (a) provider-specific credential SHAPES that are unmistakable when they
#       appear as a string literal (live key prefixes),
#   (b) a long opaque value assigned to a variable named like a secret.
# Both are scoped to *string literals* in real source, and known placeholder
# values ("your-...", "xxx...", "<...>", env-var lookups) are excluded.
# ---------------------------------------------------------------------------

# (a) Live credential prefixes. These are not substrings of normal identifiers.
#   sk-... / sk-ant-...  : OpenAI / Anthropic keys
#   ghp_... / gho_...    : GitHub tokens
#   AKIA...              : AWS access key IDs (20 chars, uppercase A-Z0-9)
#   xoxb-/xoxa-/...      : Slack tokens
_LIVE_KEY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
]

# (b) A long opaque token assigned to a secret-named variable, e.g.
#   api_key = "Zm9vYmFy....=="   (>= 24 chars of base64-ish content)
# Identifier on the LHS must look like a credential holder.
_SECRET_ASSIGN_PATTERN = re.compile(
    r"""(?P<name>\b[A-Za-z_][A-Za-z0-9_]*"""
    r"""(?:_key|_token|_secret|_password|apikey|api_key|secret|password)\b)"""
    r"""\s*[:=]\s*"""
    r"""["'](?P<value>[A-Za-z0-9+/]{24,}={0,2})["']""",
    re.IGNORECASE,
)

# Substrings that mark an obvious placeholder / non-secret, so check (b) does
# not fire on documentation, examples, or env-var indirection.
_PLACEHOLDER_MARKERS = (
    "your",
    "xxx",
    "example",
    "placeholder",
    "changeme",
    "dummy",
    "fake",
    "test",
    "<",
    "...",
    "redacted",
    "os.environ",
    "getenv",
)


def _looks_like_placeholder(value: str) -> bool:
    low = value.lower()
    return any(m in low for m in _PLACEHOLDER_MARKERS)


def test_no_hardcoded_secrets() -> None:
    """No live API-key/token shapes or long secret-named literals in scripts/."""
    violations: list[str] = []
    for path in _source_files():
        source = _read(path)
        for lineno, line in enumerate(source.splitlines(), start=1):
            for pat in _LIVE_KEY_PATTERNS:
                m = pat.search(line)
                if m and not _looks_like_placeholder(m.group(0)):
                    violations.append(
                        f"{_rel(path)}:{lineno}: live-key shape {m.group(0)!r}"
                    )
            m2 = _SECRET_ASSIGN_PATTERN.search(line)
            if m2 and not _looks_like_placeholder(m2.group("value")):
                violations.append(
                    f"{_rel(path)}:{lineno}: secret-named literal "
                    f"{m2.group('name')!r} = <{len(m2.group('value'))}-char literal>"
                )
    assert not violations, "Possible hardcoded secrets in scripts/:\n" + "\n".join(
        violations
    )


# ---------------------------------------------------------------------------
# Check 2: No eval() / exec()
# ---------------------------------------------------------------------------
# AST-based: only direct calls to the builtins eval()/exec() count, so
# identifiers that merely contain "eval"/"exec" as a substring (e.g.
# AbstractRetrieval, _exec_plan) do not trip the check.
# No allowlist entry is needed today — scripts/ contains zero such calls.
# ---------------------------------------------------------------------------
def test_no_eval_or_exec() -> None:
    """scripts/ must not call the eval()/exec() builtins."""
    violations: list[str] = []
    for path in _source_files():
        source = _read(path)
        if "eval(" not in source and "exec(" not in source:
            continue
        tree = _parse(path, source)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("eval", "exec")
            ):
                violations.append(
                    f"{_rel(path)}:{node.lineno}: {node.func.id}() call"
                )
    assert not violations, (
        "Dynamic-code execution (eval/exec) found in scripts/:\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Check 3: No disabled TLS verification
# ---------------------------------------------------------------------------
# Text-level patterns, matched against non-comment content so a docstring
# explaining the rule cannot trip it.
# ---------------------------------------------------------------------------
_TLS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"verify\s*=\s*False"),
    re.compile(r"ssl\._create_unverified_context"),
    re.compile(r"check_hostname\s*=\s*False"),
    re.compile(r"CERT_NONE"),
]


def test_no_disabled_tls() -> None:
    """No source under scripts/ disables TLS certificate verification."""
    violations: list[str] = []
    for path in _source_files():
        source = _read(path)
        for lineno, line in enumerate(source.splitlines(), start=1):
            code = line.split("#", 1)[0]  # ignore trailing comments
            for pat in _TLS_PATTERNS:
                if pat.search(code):
                    violations.append(f"{_rel(path)}:{lineno}: {line.strip()}")
    assert not violations, "Disabled TLS verification in scripts/:\n" + "\n".join(
        violations
    )


# ---------------------------------------------------------------------------
# Check 4: Parameterized SQL only
# ---------------------------------------------------------------------------
# AST-based. Flag a call to ``.execute(...)`` / ``.executemany(...)`` whose
# FIRST argument is an f-string (ast.JoinedStr) that contains an interpolated
# expression (ast.FormattedValue) — i.e. SQL text built by interpolation rather
# than a ``?`` placeholder + params tuple.
#
# Two narrow, justified exemptions so the check is robust on the current clean
# tree (these are NOT value-injection — they are unavoidable IDENTIFIER
# interpolation or integer-coerced clauses that SQL/Cypher cannot parameterize):
#
#   * The interpolated value is wrapped in ``int(...)``  → e.g. f" LIMIT {int(limit)}"
#     (already coerced to an int; no injection surface).
#   * The interpolated value is a *plain Name* AND the SQL is a DDL/introspection
#     statement that names a table/label (PRAGMA / DROP TABLE / Kuzu CALL
#     table_info / MATCH (n:{label})). Identifiers cannot use ``?`` placeholders.
#
# Any OTHER interpolation (a value into VALUES/WHERE/SET, a method call, an
# attribute, arithmetic, etc.) is flagged as a SQL-injection smell. This catches
# the clear case the task targets while not going red on the safe identifier
# interpolation that exists today.
# ---------------------------------------------------------------------------
_SQL_EXEC_METHODS = {"execute", "executemany"}

# Keywords that mark a statement whose interpolation is an identifier slot,
# not a value slot. Matched case-insensitively against the f-string's literal
# fragments. Kept deliberately tight.
_IDENTIFIER_DDL_KEYWORDS = (
    "pragma",
    "drop table",
    "create table",
    "alter table",
    "table_info",
    "show_connection",
    "match (",  # Kuzu node-count: MATCH (n:{label}) ...
    "]->()",  # Kuzu edge-count introspection: MATCH ()-[r:{rel}]->()
)


def _formatted_values(joined: ast.JoinedStr) -> list[ast.FormattedValue]:
    return [v for v in joined.values if isinstance(v, ast.FormattedValue)]


def _literal_text(joined: ast.JoinedStr) -> str:
    parts = [
        v.value
        for v in joined.values
        if isinstance(v, ast.Constant) and isinstance(v.value, str)
    ]
    return " ".join(parts).lower()


def _is_int_coerced(expr: ast.expr) -> bool:
    """True for int(...)  — already coerced, safe to inline."""
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "int"
    )


def _is_plain_identifier(expr: ast.expr) -> bool:
    """True for a bare Name (e.g. {table}/{rel}) used as an identifier slot."""
    return isinstance(expr, ast.Name)


# Conventional variable names for a string of ``?`` placeholders, e.g.
#   placeholders = ",".join("?" * len(ids))
#   conn.execute(f"... WHERE id IN ({placeholders})", ids)
# This is the standard safe idiom for a parameterized ``IN (...)`` clause: the
# f-string interpolates only ``?`` marks, never values (values go in params).
# Restricting the exemption to these exact names keeps it from masking a real
# value interpolation, which is never named "placeholders".
_PLACEHOLDER_VAR_NAMES = {"placeholders", "placeholder"}


def _is_placeholder_var(expr: ast.expr) -> bool:
    """True for a bare Name conventionally holding a ``?,?,?`` placeholder string."""
    return isinstance(expr, ast.Name) and expr.id in _PLACEHOLDER_VAR_NAMES


def test_sql_is_parameterized() -> None:
    """No SQL/Cypher value built by f-string interpolation inside .execute().

    Identifier-only interpolation (table/label names; int-coerced LIMIT) is
    explicitly exempted because SQL placeholders cannot name identifiers.
    """
    violations: list[str] = []
    for path in _source_files():
        source = _read(path)
        if ".execute" not in source:
            continue
        tree = _parse(path, source)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _SQL_EXEC_METHODS
                and node.args
            ):
                continue
            first = node.args[0]
            if not isinstance(first, ast.JoinedStr):
                continue  # literal str or variable → fine
            fmts = _formatted_values(first)
            if not fmts:
                continue  # f-string with no interpolation → effectively literal

            is_identifier_ddl = any(
                kw in _literal_text(first) for kw in _IDENTIFIER_DDL_KEYWORDS
            )
            for fv in fmts:
                expr = fv.value
                if _is_int_coerced(expr):
                    continue  # f"... LIMIT {int(limit)}" — coerced, safe
                if _is_placeholder_var(expr):
                    continue  # f"... IN ({placeholders})" — ?-marks, not values
                if is_identifier_ddl and _is_plain_identifier(expr):
                    continue  # f"PRAGMA table_info({table})" — identifier slot
                # Anything else interpolated into SQL is a smell.
                rendered = ast.dump(expr)
                violations.append(
                    f"{_rel(path)}:{node.lineno}: interpolated SQL expression "
                    f"({rendered}) inside .{node.func.attr}() — use a ? placeholder"
                )
    assert not violations, (
        "Possible SQL-injection (f-string SQL) in scripts/:\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Check 5: No subprocess(..., shell=True)
# ---------------------------------------------------------------------------
# AST-based: any call passing shell=True (truthy literal) as a keyword. The
# project spawns children exclusively via the argv-list form
# (subprocess.run/Popen / asyncio.create_subprocess_exec), so no allowlist is
# needed today.
# ---------------------------------------------------------------------------
def test_no_shell_true() -> None:
    """No subprocess call in scripts/ uses shell=True."""
    violations: list[str] = []
    for path in _source_files():
        source = _read(path)
        if "shell" not in source:
            continue
        tree = _parse(path, source)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "shell":
                    continue
                val = kw.value
                if isinstance(val, ast.Constant) and val.value is True:
                    violations.append(
                        f"{_rel(path)}:{node.lineno}: subprocess call with shell=True"
                    )
    assert not violations, "shell=True usage in scripts/:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Check 6: Secrets file read only by an explicit allowlist
# ---------------------------------------------------------------------------
# The user's credential file ~/.config/lit-monitor/config.toml must only be
# *opened/parsed* by known credential-loading modules. A NEW module that starts
# reading it directly is the anti-pattern this catches.
#
# Allowlist (verified to legitimately load credentials, 2026-06):
#   scripts/setup/check_configured.py  — presence/parse check the project
#                                        CLAUDE.md designates as THE reader.
#   scripts/server/config_io.py        — load_secrets (setup wizard).
#   scripts/cli.py                     — _load_secrets (env-key hydration).
#   scripts/server/runtime.py          — _load_secrets (server boot).
#   scripts/search/search_runner.py    — _load_api_secrets (Scopus/IEEE/email).
# Every other module either passes the already-loaded dict around (e.g.
# routes/ingest.py reads runtime.secrets) or only mentions the path in
# docstrings — neither opens the file, so neither is on the allowlist.
# ---------------------------------------------------------------------------
_SECRETS_READ_ALLOWLIST = {
    "scripts/setup/check_configured.py",
    "scripts/server/config_io.py",
    "scripts/cli.py",
    "scripts/server/runtime.py",
    "scripts/search/search_runner.py",
}

# A module "reads the secrets file directly" if it both references the
# config.toml path AND performs a TOML load / binary open in the same file.
_TOML_LOAD_MARKERS = ("tomllib.load", "toml.load")


def test_secrets_file_read_only_by_allowlist() -> None:
    """Only allowlisted credential-loading modules may open config.toml."""
    violations: list[str] = []
    for path in _source_files():
        source = _read(path)
        if "config.toml" not in source:
            continue
        loads_toml = any(m in source for m in _TOML_LOAD_MARKERS)
        opens_binary = ('.open("rb")' in source) or (".open('rb')" in source)
        if not (loads_toml or opens_binary):
            continue  # only mentions the path (docstring/string), does not read it
        rel = _rel(path)
        if rel not in _SECRETS_READ_ALLOWLIST:
            violations.append(
                f"{rel}: reads the secrets file (config.toml) directly but is "
                f"not in the allowlist. Route credential loading through an "
                f"existing loader or add this file to the allowlist with a "
                f"justification."
            )
    assert not violations, (
        "Unallowlisted direct read of the secrets file:\n" + "\n".join(violations)
    )
