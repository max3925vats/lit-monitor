"""Guard: the live_server fixture must NOT resolve to the real repo config dir,
so a form-submitting visual test can never clobber the user's live config."""
import os

import pytest

pytestmark = pytest.mark.visual


def test_live_server_uses_isolated_config(live_server):
    root = os.environ.get("LIT_MONITOR_ROOT", "")
    assert root, "live_server must set LIT_MONITOR_ROOT to an isolated dir"
    assert "Literature_search_assistant/lit-monitor/config" not in os.path.realpath(
        os.path.join(root, "config")
    ), "live_server config must not be the real repo config dir"
