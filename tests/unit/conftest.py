"""
Shared pytest fixtures for unit tests.

The main job here is to keep ChromaDB's process-wide singletons from leaking
across test modules.  ChromaDB ``PersistentClient`` caches a ``System`` object
per persist_path in ``SharedSystemClient._identifier_to_system``; when an
earlier test's ``tmp_path`` has been cleaned up, a later test that asks for a
*different* persist_path can still hit a stale handle through that cache and
fail with ``code: 14 unable to open database file`` from the Rust bindings.

Clearing the cache before every test is cheap (it is a no-op when empty) and
makes ChromaDB-backed tests order-independent.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_strict_mode():
    """Reset the strict-mode global state before and after every unit test.

    ``set_strict()`` writes to a module-level variable.  Without this fixture,
    tests that call ``set_strict(True)`` (e.g. TestStrictModeFlag and
    TestLoadSecretsStrictMode) would leak into subsequent tests and turn
    expected-warning sites into unexpected RuntimeErrors.
    """
    import scripts.core.strict_mode as _sm
    _sm._strict_override = None
    yield
    _sm._strict_override = None


@pytest.fixture(autouse=True)
def _reset_chromadb_shared_system_cache():
    """Drop ChromaDB's cached System objects before every unit test."""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except Exception:
        # chromadb not importable in this env (defensive — should always be present)
        yield
        return

    # Wipe cache before the test runs so any prior pollution is gone.
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        # On older chromadb the helper may not exist; fall back to the dict.
        cache = getattr(SharedSystemClient, "_identifier_to_system", None)
        if cache is not None:
            cache.clear()
    yield
    # And again after, so the next test in the file isn't tripped up by THIS one.
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        cache = getattr(SharedSystemClient, "_identifier_to_system", None)
        if cache is not None:
            cache.clear()
