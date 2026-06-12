import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest
import uvicorn


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="session")
def live_server():
    # --- CONFIG SAFETY: isolate to a temp copy of the real config ----------
    # NOTE: LIT_MONITOR_ROOT redirects the CONFIG DIR only (config/*.yaml). It does
    # NOT redirect the secrets path (~/.config/lit-monitor/config.toml — hard-wired
    # to Path.home() in setup/_paths.py + runtime.py). The visual suite is safe
    # because no test drives a write route (only /settings + /setup step-1 write,
    # and neither is submitted here — Playwright form tests target /settings, whose
    # extraction.yaml save IS under the redirected config dir). If a future visual
    # test ever drives the Setup credentials step, it would write the user's REAL
    # config.toml — isolate the secrets path too before adding such a test.
    from lit_monitor.core.config import config_dir

    real_cfg = config_dir()
    tmp_root = Path(tempfile.mkdtemp(prefix="lm-visual-cfg-"))
    (tmp_root / "config").mkdir(parents=True, exist_ok=True)
    if real_cfg.exists():
        for f in real_cfg.glob("*.yaml"):
            shutil.copy2(f, tmp_root / "config" / f.name)
    prev_root = os.environ.get("LIT_MONITOR_ROOT")
    prev_dev = os.environ.get("LIT_MONITOR_DEV")
    os.environ["LIT_MONITOR_ROOT"] = str(tmp_root)
    os.environ["LIT_MONITOR_DEV"] = "1"  # enables /dev + the gallery spike routes

    from lit_monitor.server.app import create_app
    from lit_monitor.server.runtime import reset_runtime

    reset_runtime()
    app = create_app()
    port = _free_port()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(80):
        if server.started:
            break
        time.sleep(0.1)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(timeout=5)
        if prev_root is None:
            os.environ.pop("LIT_MONITOR_ROOT", None)
        else:
            os.environ["LIT_MONITOR_ROOT"] = prev_root
        if prev_dev is None:
            os.environ.pop("LIT_MONITOR_DEV", None)
        else:
            os.environ["LIT_MONITOR_DEV"] = prev_dev
        reset_runtime()
        shutil.rmtree(tmp_root, ignore_errors=True)
