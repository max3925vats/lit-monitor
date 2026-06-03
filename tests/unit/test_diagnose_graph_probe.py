"""P6.15: _check_graph_extra is a read-only diagnostic — it must probe
writability without creating the persist-dir parent on disk."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.setup import diagnose as _diag


@pytest.mark.unit
def test_check_graph_extra_does_not_create_dirs(tmp_path, monkeypatch) -> None:
    """When kuzu is importable, the probe must not mkdir a missing parent."""
    # Ensure the [graph] extra "appears installed" without requiring kuzu: the
    # import is `import kuzu` inside the function, so inject a dummy module.
    monkeypatch.setitem(sys.modules, "kuzu", object())

    missing_parent = tmp_path / "does_not_exist" / "graph.kuzu"

    # Force the config-resolved persist dir to our nonexistent path.
    class _Graph:
        persist_dir = str(missing_parent)

    class _Retrieval:
        graph_db = _Graph()

    class _Cfg:
        retrieval = _Retrieval()

    with patch.object(_diag._config_mod, "get_config", return_value=_Cfg()):
        ok, msg = _diag._check_graph_extra()

    # tmp_path itself is a writable existing ancestor → ok, and crucially the
    # missing intermediate directory must NOT have been created.
    assert ok is True, msg
    assert not (tmp_path / "does_not_exist").exists()


@pytest.mark.unit
def test_check_graph_extra_unwritable_parent_reports_failure(
    tmp_path, monkeypatch
) -> None:
    """An existing-but-unwritable nearest ancestor yields ok=False."""
    monkeypatch.setitem(sys.modules, "kuzu", object())
    target = tmp_path / "sub" / "graph.kuzu"

    class _Graph:
        persist_dir = str(target)

    class _Retrieval:
        graph_db = _Graph()

    class _Cfg:
        retrieval = _Retrieval()

    # tmp_path exists; pretend it is not writable.
    real_access = __import__("os").access

    def _fake_access(path, mode):
        if Path(path) == tmp_path:
            return False
        return real_access(path, mode)

    with patch.object(_diag._config_mod, "get_config", return_value=_Cfg()), \
         patch("scripts.setup.diagnose.os.access", side_effect=_fake_access):
        ok, msg = _diag._check_graph_extra()

    assert ok is False
    assert "not writable" in msg
