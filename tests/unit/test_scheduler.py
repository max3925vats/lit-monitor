"""Unit tests for scripts.server.scheduler.

All tests mock ``subprocess.run`` so launchctl/systemctl are never invoked,
and use ``monkeypatch`` to redirect install paths into ``tmp_path``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lit_monitor.server import scheduler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def macos_tmp_dirs(tmp_path, monkeypatch):
    """Redirect launchd paths into tmp_path and force platform=macos."""
    launch_dir = tmp_path / "LaunchAgents"
    plist = launch_dir / "com.litmonitor.discovery.plist"
    monkeypatch.setattr(scheduler, "_LAUNCHD_DIR", launch_dir)
    monkeypatch.setattr(scheduler, "_LAUNCHD_PLIST", plist)
    monkeypatch.setattr(scheduler, "detect_platform", lambda: "macos")
    return tmp_path


@pytest.fixture
def linux_tmp_dirs(tmp_path, monkeypatch):
    """Redirect systemd paths into tmp_path and force platform=linux."""
    sysd_dir = tmp_path / "systemd"
    timer = sysd_dir / "lit-monitor-discovery.timer"
    service = sysd_dir / "lit-monitor-discovery.service"
    monkeypatch.setattr(scheduler, "_SYSTEMD_DIR", sysd_dir)
    monkeypatch.setattr(scheduler, "_SYSTEMD_TIMER", timer)
    monkeypatch.setattr(scheduler, "_SYSTEMD_SERVICE", service)
    monkeypatch.setattr(scheduler, "detect_platform", lambda: "linux")
    return tmp_path


# ---------------------------------------------------------------------------
# ScheduleSpec.parse validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_schedulespec_parse_validates_day():
    with pytest.raises(ValueError):
        scheduler.ScheduleSpec.parse("xyz", "10:00")


@pytest.mark.unit
def test_schedulespec_parse_validates_time():
    with pytest.raises(ValueError):
        scheduler.ScheduleSpec.parse("mon", "25:99")


# ---------------------------------------------------------------------------
# write_schedule per platform
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_macos_renders_plist_and_calls_launchctl(macos_tmp_dirs):
    spec = scheduler.ScheduleSpec.parse("mon", "08:00")
    with patch("scripts.server.scheduler.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        path = scheduler.write_schedule(spec)
    assert path.exists()
    text = path.read_text()
    assert "<key>Weekday</key>" in text
    assert "<integer>1</integer>" in text  # Mon == launchd weekday 1
    assert "<key>Hour</key>" in text
    assert "<integer>8</integer>" in text
    # launchctl load was invoked.
    cmds = [c.args[0] for c in mock_run.call_args_list]
    assert any(cmd[0] == "launchctl" and "load" in cmd for cmd in cmds)


@pytest.mark.unit
def test_write_linux_renders_timer_and_calls_systemctl(linux_tmp_dirs):
    spec = scheduler.ScheduleSpec.parse("fri", "14:30")
    with patch("scripts.server.scheduler.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        path = scheduler.write_schedule(spec)
    assert path.exists()
    text = path.read_text()
    assert "OnCalendar=Fri" in text
    assert "14:30:00" in text
    # The companion .service file was also written.
    assert scheduler._SYSTEMD_SERVICE.exists()
    # systemctl --user enable --now was invoked.
    cmds = [c.args[0] for c in mock_run.call_args_list]
    assert any("enable" in cmd for cmd in cmds)


@pytest.mark.unit
def test_write_macos_uses_atomic_write(macos_tmp_dirs):
    """P1.2: the macOS plist is written via atomic_write_text (crash-safe).

    Asserts the helper is invoked with the plist path + rendered content, that
    the final file is valid, and that no ``.tmp`` dropping is left behind.
    """
    spec = scheduler.ScheduleSpec.parse("mon", "08:00")
    with (
        patch("scripts.server.scheduler.subprocess.run") as mock_run,
        patch(
            "scripts.server.scheduler.atomic_write_text",
            wraps=scheduler.atomic_write_text,
        ) as mock_atomic,
    ):
        mock_run.return_value.returncode = 0
        path = scheduler.write_schedule(spec)

    # atomic_write_text was used for the plist write.
    assert mock_atomic.call_count == 1
    called_path, called_content = mock_atomic.call_args.args[:2]
    assert called_path == scheduler._LAUNCHD_PLIST
    assert "<key>Weekday</key>" in called_content
    # Final file is valid and complete; no temp leftovers.
    assert path.exists()
    assert "<key>Weekday</key>" in path.read_text()
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.unit
def test_write_linux_uses_atomic_write_for_both_units(linux_tmp_dirs):
    """P1.2: both systemd timer and service files are written atomically."""
    spec = scheduler.ScheduleSpec.parse("fri", "14:30")
    with (
        patch("scripts.server.scheduler.subprocess.run") as mock_run,
        patch(
            "scripts.server.scheduler.atomic_write_text",
            wraps=scheduler.atomic_write_text,
        ) as mock_atomic,
    ):
        mock_run.return_value.returncode = 0
        scheduler.write_schedule(spec)

    # Two atomic writes: timer + service.
    written_paths = {c.args[0] for c in mock_atomic.call_args_list}
    assert written_paths == {scheduler._SYSTEMD_TIMER, scheduler._SYSTEMD_SERVICE}
    assert scheduler._SYSTEMD_TIMER.exists()
    assert scheduler._SYSTEMD_SERVICE.exists()
    assert "OnCalendar=Fri" in scheduler._SYSTEMD_TIMER.read_text()
    assert not list(scheduler._SYSTEMD_TIMER.parent.glob("*.tmp"))


# ---------------------------------------------------------------------------
# read_schedule round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_macos_round_trips(macos_tmp_dirs):
    spec = scheduler.ScheduleSpec.parse("wed", "09:15")
    with patch("scripts.server.scheduler.subprocess.run"):
        scheduler.write_schedule(spec)
    got = scheduler.read_schedule()
    assert got is not None
    assert got.day_of_week == "wed"
    assert got.time == "09:15"


# ---------------------------------------------------------------------------
# remove_schedule
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_remove_macos_unlinks_plist(macos_tmp_dirs):
    spec = scheduler.ScheduleSpec.parse("mon", "08:00")
    with patch("scripts.server.scheduler.subprocess.run"):
        scheduler.write_schedule(spec)
    assert scheduler._LAUNCHD_PLIST.exists()
    with patch("scripts.server.scheduler.subprocess.run"):
        scheduler.remove_schedule()
    assert not scheduler._LAUNCHD_PLIST.exists()
