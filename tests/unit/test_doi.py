"""AR-4: unit tests for the canonical DOI normalizer (scripts/core/doi.py).

DOIs are case-insensitive per the DOI spec, and URL-wrapping
(``https://doi.org/10.x/y``) is cosmetic. ``normalize_doi`` collapses both so
that dedup keys and RRF fusion keys treat the same paper as one entry.
"""
from __future__ import annotations

import pytest

from scripts.core.doi import normalize_doi


@pytest.mark.parametrize("raw,expected", [
    ("10.1234/AbC", "10.1234/abc"),                       # case fold
    ("https://doi.org/10.1234/abc", "10.1234/abc"),       # https URL wrap
    ("http://dx.doi.org/10.1234/ABC", "10.1234/abc"),     # http dx URL wrap + case
    ("  10.1234/abc  ", "10.1234/abc"),                   # surrounding whitespace
    ("10.1234/abc/", "10.1234/abc"),                      # trailing slash
    ("doi:10.1234/abc", "10.1234/abc"),                   # doi: prefix
    ("DOI: 10.1234/abc", "10.1234/abc"),                  # DOI: prefix with space
    ("", ""),                                             # empty string
    (None, ""),                                           # None (defensive)
])
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


def test_normalize_doi_idempotent():
    """Normalizing an already-normalized DOI is a no-op (stable canonical form)."""
    once = normalize_doi("https://doi.org/10.1/X")
    assert normalize_doi(once) == once
    assert once == "10.1/x"


def test_url_wrap_and_bare_collide():
    """The core AR-4 guarantee: URL-wrapped vs bare forms of the same DOI map
    to ONE canonical key (so dedup / fusion treat them as the same paper)."""
    assert normalize_doi("10.1/a") == normalize_doi("https://doi.org/10.1/A")
