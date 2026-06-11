"""B3: Cypher safety guard.

Read-only sanitizer for the ``run_cypher`` MCP tool.  Strips comments
first, then rejects any mutation keyword.  Appends LIMIT 100 (or a
custom hard_limit) if no LIMIT clause is present.

Defense layers (each addresses a documented bypass class):

1. **Comment stripping before keyword matching.**
   ``_strip_comments`` (a linear single-pass scanner) strips
   ``// line-comments`` and ``/* block-comments */`` (blocks may span
   multiple lines) *before* ``_BAN`` runs.
   Effect: a keyword that lives entirely inside a comment is safe;
   ``// CREATE foo`` and ``/* DELETE x */`` both pass.
   A keyword that is *outside* every comment is still caught even if
   comments are present.

2. **Word-boundary checks on banned keywords.**
   ``_BAN`` uses ``\\b`` anchors so identifier *substrings* that happen
   to start with a banned keyword don't trigger a false rejection.
   ``created_at``, ``merged``, ``dropped_at``, ``deleted_by`` all pass.
   Bare ``CREATE``, ``DELETE``, etc. are caught.

3. **Case-insensitive matching (re.IGNORECASE).**
   ``CrEaTe``, ``cReAtE``, ``set``, ``merge``, ``load csv`` — all blocked.

4. **Multi-token LOAD CSV pattern.**
   ``LOAD\\s+CSV`` with ``\\s+`` catches any whitespace between the two
   tokens (space, tab, newline), covering the LOAD\\nCSV variant.

5. **LIMIT detection is case-insensitive.**
   ``limit 5`` (lowercase) is treated as an existing LIMIT so we don't
   double-append.

6. **Trailing semicolon stripping before LIMIT injection.**
   ``MATCH p RETURN p;`` → ``MATCH p RETURN p LIMIT 100`` (no ``; LIMIT``
   syntax error).

Known false-positive (documented, accepted):
   A Cypher **string literal** that contains a forbidden keyword, such as
   ``MATCH (p) WHERE p.title = 'CREATE TABLE' RETURN p``, will be rejected
   because the guard operates on raw text after comment stripping and does
   not parse Cypher string boundaries.  This is the deliberate
   safety-over-precision tradeoff: we prefer a false rejection over a
   missed mutation.  Users who need to match on literal 'CREATE'-containing
   strings should parameterize the query on the caller side.

Unicode look-alikes (e.g. ``CRΕATE`` with Greek Ε):
   Not matched by ``_BAN`` — they won't match ``\\bCREATE\\b`` and Kuzu
   won't accept them either, so no bypass risk exists.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class CypherSafetyError(ValueError):
    """Raised by :func:`guard` when the query contains a forbidden mutation keyword.

    Subclasses :class:`ValueError` so MCP's existing ``ValueError`` handler
    routes it cleanly to a ``TextContent("Error: ...")`` response instead of
    crashing the server.
    """


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Comment stripping is done by ``_strip_comments`` (a linear scanner) rather
# than a regex.  The previous ``//[^\n]*|/\*.*?\*/`` pattern is O(n²) on crafted
# input like ``(/*x)*n`` — many unterminated block-comment openers force the
# lazy ``.*?`` to rescan to end-of-string at every position (a polynomial-ReDoS,
# CodeQL py/polynomial-redos).  A single forward pass over the string is O(n)
# and reproduces the exact same semantics (see ``_strip_comments`` docstring).


def _strip_comments(query: str) -> str:
    """Remove ``//`` line comments and ``/* */`` block comments in linear time.

    Single forward scan, O(len(query)).  Semantics match the prior regex
    (``//[^\\n]*|/\\*.*?\\*/`` with ``re.DOTALL``) exactly:

      * ``// …`` is dropped up to (but not including) the next newline.
      * ``/* … */`` (possibly multi-line) is dropped, terminator included.
      * An **unterminated** ``/*`` (no closing ``*/``) is NOT stripped — the
        text is preserved so any banned keyword inside it stays visible to the
        ``_BAN`` check and is still rejected.  This mirrors the lazy regex,
        which required a closing ``*/`` to match.
    """
    out: list[str] = []
    i = 0
    n = len(query)
    while i < n:
        c = query[i]
        if c == "/" and i + 1 < n and query[i + 1] == "/":
            # Line comment: skip to the next newline (newline itself is kept).
            j = query.find("\n", i + 2)
            i = n if j == -1 else j
        elif c == "/" and i + 1 < n and query[i + 1] == "*":
            # Block comment: find the matching */. If absent, keep the text
            # (matches the lazy-regex behaviour — no terminator ⇒ no strip).
            end = query.find("*/", i + 2)
            if end == -1:
                out.append(query[i:])
                break
            i = end + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)

# Banned mutation keywords.  \\b word boundaries prevent false matches on
# identifier substrings (e.g. "created_at", "dropped_at", "deleted_by").
# LOAD\\s+CSV uses \\s+ to catch any whitespace variant (tab, newline, etc.).
#
# CALL is banned because Kuzu CALL procedures can have side effects, and the
# run_cypher tool is read-only by contract. The internal read-only procedures
# (CALL show_tables / table_info / show_connection) are issued directly via
# conn.execute() in schema_describer.py and migrations.py — they NEVER route
# through this guard, so banning CALL here breaks no legitimate read path.
_BAN: re.Pattern[str] = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|DROP|ALTER|REMOVE|CALL|LOAD\s+CSV)\b",
    re.IGNORECASE,
)

# Detect an existing LIMIT N clause so we don't append a second one.
# Case-insensitive to match lowercase 'limit 5' as well.
_LIMIT: re.Pattern[str] = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def guard(query: str, hard_limit: int = 100) -> str:
    """Validate *query* as read-only Cypher; return query with LIMIT injected if absent.

    Processing order (each step is necessary for safety):

    1. Type + emptiness check.
    2. Strip comments from a *copy* of the query (the original is kept for
       the LIMIT-injection step so we return a clean string without comment
       residue on the safe side, but the original's LIMIT clause is checked
       so users who put LIMIT after a comment don't get double-appended).
    3. Run ``_BAN`` against the stripped text only.
    4. Check ``_LIMIT`` against the stripped text (catches LIMIT inside or
       outside comments consistently).
    5. If no LIMIT: strip trailing whitespace + semicolons from the original
       query and append `` LIMIT {hard_limit}``.
    6. Return the (possibly LIMIT-appended) query string.

    Args:
        query:      Raw Cypher query string from the MCP caller.
        hard_limit: LIMIT value to inject when none is present (default 100).

    Returns:
        The query string, possibly with `` LIMIT {hard_limit}`` appended.

    Raises:
        CypherSafetyError: When *query* is not a non-empty string or contains
            a forbidden mutation keyword (after comment stripping).
    """
    # --- 1. Type and emptiness check -----------------------------------------
    if not isinstance(query, str):
        raise CypherSafetyError(
            f"query must be a non-empty string, got {type(query).__name__!r}"
        )
    if not query.strip():
        raise CypherSafetyError("query is empty")

    # --- 2. Strip comments from a working copy --------------------------------
    # We run ALL safety checks against the stripped text so that keywords
    # inside comments are invisible to _BAN, but keywords *outside* comments
    # (even if surrounded by comments) are still caught.
    stripped: str = _strip_comments(query)

    # --- 3. Keyword check (on comment-stripped text) --------------------------
    match = _BAN.search(stripped)
    if match:
        keyword = match.group(0).strip()
        raise CypherSafetyError(
            f"banned mutation keyword {keyword!r} — "
            "run_cypher is a read-only MCP tool; "
            "use the ingest / relink / re-extract endpoints for mutations"
        )

    # --- 4 & 5. LIMIT injection -----------------------------------------------
    # Check stripped text for an existing LIMIT so that a LIMIT hiding after
    # a comment isn't missed or double-counted.
    if not _LIMIT.search(stripped):
        # No LIMIT found: append one.
        # Strip trailing semicolons (and surrounding whitespace) from the
        # *original* query before appending to avoid "RETURN p; LIMIT 100"
        # syntax errors in Kuzu.
        cleaned: str = query.rstrip().rstrip(";").rstrip()
        return f"{cleaned} LIMIT {hard_limit}"

    # LIMIT already present — return the original query unchanged.
    return query
