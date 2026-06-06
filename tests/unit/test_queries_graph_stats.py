"""FG-1: unit tests for corpus-wide graph stat queries.

Covers ``get_entity_counts_by_type`` and ``get_entity_counts_by_source`` in
``scripts/api/queries.py``.  Each function has a happy-path test (grouped rows
coerced into a ``dict[str, int]``) and an error-path test (a backend whose
``_conn.execute`` raises must yield ``{}`` rather than propagate).

The fake graph-db stubs replicate the exact Kuzu result-iteration protocol used
by ``get_corpus_stats``: ``while res.has_next(): res.get_next()``.
"""
from __future__ import annotations

from typing import Any

from scripts.api.queries import (
    get_entity_counts_by_source,
    get_entity_counts_by_type,
)


class _FakeResult:
    """Minimal Kuzu QueryResult stand-in.

    Yields the supplied grouped rows one at a time via the
    ``has_next()`` / ``get_next()`` protocol, then reports exhaustion.
    """

    def __init__(self, rows: list[tuple[Any, Any]]) -> None:
        self._rows = list(rows)
        self._idx = 0

    def has_next(self) -> bool:
        return self._idx < len(self._rows)

    def get_next(self) -> tuple[Any, Any]:
        row = self._rows[self._idx]
        self._idx += 1
        return row


class _FakeConn:
    """Stand-in for GraphDB._conn — execute() returns a canned result."""

    def __init__(self, rows: list[tuple[Any, Any]]) -> None:
        self._rows = rows

    def execute(self, query: str, *args: Any, **kwargs: Any) -> _FakeResult:
        # Each call gets a fresh result so the cursor starts at row 0.
        return _FakeResult(self._rows)


class FakeGraph:
    """Fake GraphDB exposing a ._conn.execute that returns grouped rows."""

    def __init__(self, rows: list[tuple[Any, Any]]) -> None:
        self._conn = _FakeConn(rows)


class _BrokenConn:
    """._conn.execute always raises — exercises the defensive try/except."""

    def execute(self, query: str, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("kuzu boom")


class BrokenGraph:
    """Fake GraphDB whose _conn.execute raises on every query."""

    def __init__(self) -> None:
        self._conn = _BrokenConn()


# ---------------------------------------------------------------------------
# get_entity_counts_by_type
# ---------------------------------------------------------------------------


def test_entity_counts_by_type_groups_and_coerces() -> None:
    g = FakeGraph(rows=[("Method", 3), ("Material", 2)])
    out = get_entity_counts_by_type(g)
    assert out == {"Method": 3, "Material": 2}
    assert all(isinstance(v, int) for v in out.values())
    assert all(isinstance(k, str) for k in out)


def test_entity_counts_by_type_empty_on_error() -> None:
    g = BrokenGraph()  # ._conn.execute raises
    assert get_entity_counts_by_type(g) == {}  # must NOT raise


# ---------------------------------------------------------------------------
# get_entity_counts_by_source
# ---------------------------------------------------------------------------


def test_entity_counts_by_source_groups() -> None:
    g = FakeGraph(rows=[("schema", 5), ("biobert", 2), ("llm_cloud", 1)])
    assert get_entity_counts_by_source(g) == {
        "schema": 5,
        "biobert": 2,
        "llm_cloud": 1,
    }


def test_entity_counts_by_source_empty_on_error() -> None:
    assert get_entity_counts_by_source(BrokenGraph()) == {}
