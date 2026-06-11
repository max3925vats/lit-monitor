"""AR-4: the single canonical DOI normalizer for lit-monitor.

DOI identity previously had no single owner: different boundaries lowercased
DOIs but did not strip URL-wrapping, so ``https://doi.org/10.1/Y`` and
``10.1/y`` were treated as different papers at the search-dedup and RRF-fusion
boundaries. This module owns the one canonical form.

Rationale:
  - **Case** — DOIs are case-insensitive per the DOI Handbook / ISO 26324, so
    the canonical form is lowercased.
  - **URL-wrapping** — ``https://doi.org/<doi>`` (and the legacy
    ``http://dx.doi.org/<doi>``) is a display/resolver convenience; the bare
    DOI is the identity. We strip the resolver prefix.
  - **``doi:`` prefix** — some sources emit ``doi:10.x/y`` or ``DOI: 10.x/y``;
    strip it so the bare DOI remains.
  - **Trailing slash / whitespace** — cosmetic; stripped.

``normalize_doi`` NEVER raises — it is meant to run at every comparison
boundary, where a single malformed input must not abort a pipeline.
"""
from __future__ import annotations

import re

__all__ = ["normalize_doi"]

# Leading resolver URL: http(s)://(dx.)doi.org/  — case-insensitive scheme/host.
_URL_PREFIX_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)

# Leading "doi:" / "DOI:" with optional following whitespace.
_DOI_PREFIX_RE = re.compile(r"^doi:\s*", re.IGNORECASE)


def normalize_doi(raw: str | None) -> str:
    """Return the canonical, comparison-safe form of a DOI.

    The canonical form is lowercased, URL-unwrapped, and free of a ``doi:``
    prefix, surrounding whitespace, and a trailing slash. Falsy input returns
    the empty string. This function never raises.

    Args:
        raw: A DOI in any common surface form — bare (``10.1234/abc``),
            URL-wrapped (``https://doi.org/10.1234/abc``), ``doi:``-prefixed,
            or with case/whitespace/trailing-slash noise. ``None`` is allowed.

    Returns:
        The canonical DOI string, or ``""`` for ``None``/empty input.

    Examples:
        >>> normalize_doi("10.1234/AbC")
        '10.1234/abc'
        >>> normalize_doi("https://doi.org/10.1234/abc")
        '10.1234/abc'
        >>> normalize_doi("DOI: 10.1234/abc")
        '10.1234/abc'
        >>> normalize_doi(None)
        ''
    """
    if not raw:
        return ""
    s = raw.strip()
    # Order matters: a "doi:" prefix never wraps a URL, but strip it first so a
    # hypothetical "doi: https://..." still reaches the URL stripper.
    s = _DOI_PREFIX_RE.sub("", s, count=1).strip()
    s = _URL_PREFIX_RE.sub("", s, count=1)
    s = s.rstrip("/")
    return s.lower()
