"""Shared constants for the scripts.search module."""

FINDPAPERS_TIMEOUT_SECONDS = 180
"""Timeout cap for a single _findpapers.search() call (per topic / researcher).

Note: findpapers has no cooperative cancellation; this only bounds when we
stop actively waiting and log a warning. The underlying thread may continue
running until findpapers' internal retries complete.
"""

S2_TIMEOUT_SECONDS = 30
"""Timeout for Semantic Scholar API calls (citation_graph + semantic_scholar)."""
