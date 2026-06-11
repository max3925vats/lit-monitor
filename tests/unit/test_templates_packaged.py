"""CWD-independent render smoke: the server resolves its packaged templates
regardless of the working directory.

This guards the wheel-drops-templates bug: ``app.py`` resolves templates via
``HERE = Path(__file__).resolve().parent`` (inside the installed package), so a
``create_app()`` render must succeed even when CWD is somewhere with no
``templates/`` dir on disk.
"""


def test_base_template_resolves(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from fastapi.testclient import TestClient

    from lit_monitor.server.app import create_app

    r = TestClient(create_app()).get("/")
    assert r.status_code in (200, 302, 307)
