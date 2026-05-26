"""Platform-aware recurring-schedule reader/writer for the discovery pipeline.

Manages an OS-level schedule that invokes ``lit-monitor run`` on a recurring
day-of-week cadence (weekly by default). On macOS the schedule lives in a
launchd LaunchAgent plist; on Linux it is a systemd user timer + service pair.

This module is *library code*. It does not invoke ``launchctl`` or
``systemctl`` at import time; only the public ``write_schedule`` and
``remove_schedule`` functions do, and both are unit-tested with
``subprocess.run`` mocked. Smoke testing of the real invocation lives in
F5.2.
"""

from __future__ import annotations

import logging
import platform
import plistlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
Platform = Literal["macos", "linux", "unsupported"]

# Canonical day-of-week ordering. Used only to validate user input — both
# launchd and systemd have their own day encodings (see below).
_DAY_TO_NUM = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 0}

# launchd uses 0=Sun ... 6=Sat for the StartCalendarInterval Weekday key.
_DAY_TO_LAUNCHD = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}

# systemd OnCalendar uses 3-letter abbreviations: Mon, Tue, etc.
_DAY_TO_SYSTEMD = {
    "mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu",
    "fri": "Fri", "sat": "Sat", "sun": "Sun",
}
_SYSTEMD_TO_DAY = {v: k for k, v in _DAY_TO_SYSTEMD.items()}

# Matches the OnCalendar= line written by our systemd.timer.j2 template.
_ONCALENDAR_RE = re.compile(
    r"^OnCalendar\s*=\s*(\S+)\s+\*-\*-\*\s+(\d{2}):(\d{2}):\d{2}",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Install paths (module-level so tests can monkeypatch them)
# ---------------------------------------------------------------------------

_LAUNCHD_DIR = Path.home() / "Library" / "LaunchAgents"
_LAUNCHD_PLIST = _LAUNCHD_DIR / "com.litmonitor.discovery.plist"

_SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
_SYSTEMD_TIMER = _SYSTEMD_DIR / "lit-monitor-discovery.timer"
_SYSTEMD_SERVICE = _SYSTEMD_DIR / "lit-monitor-discovery.service"

# ---------------------------------------------------------------------------
# Jinja environment
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).resolve().parent / "scheduler_templates"
# Only the plist template needs XML escaping; ini-style files don't.
_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(["plist.j2"]),
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleSpec:
    """A recurring schedule: which day of the week, what HH:MM (24h local)."""

    day_of_week: DayOfWeek
    time: str  # "HH:MM"
    raw: str | None = None  # Raw plist/timer text; populated by read_schedule.

    @classmethod
    def parse(cls, day: str, time: str) -> ScheduleSpec:
        """Validate user-supplied day + time and return a ScheduleSpec.

        Raises ValueError if the day is unknown or the time is not HH:MM 24h.
        """
        day = day.lower().strip()
        if day not in _DAY_TO_NUM:
            raise ValueError(
                f"day_of_week must be one of {list(_DAY_TO_NUM)}, got {day!r}"
            )
        time = time.strip()
        try:
            h_str, m_str = time.split(":")
            h, m = int(h_str), int(m_str)
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ValueError
        except (ValueError, IndexError) as e:
            raise ValueError(f"time must be HH:MM (24h), got {time!r}") from e
        return cls(day_of_week=day, time=time)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_platform() -> Platform:
    """Return ``"macos"``, ``"linux"``, or ``"unsupported"``."""
    sysname = platform.system()
    if sysname == "Darwin":
        return "macos"
    if sysname == "Linux":
        return "linux"
    return "unsupported"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Repo root = two levels up from this file (scripts/server/scheduler.py)."""
    return Path(__file__).resolve().parents[2]


def _uv_path() -> str:
    """Find the ``uv`` binary; fall back to literal ``"uv"`` if which fails."""
    p = shutil.which("uv")
    return p or "uv"


# ---------------------------------------------------------------------------
# read_schedule
# ---------------------------------------------------------------------------


def read_schedule() -> ScheduleSpec | None:
    """Read the currently-installed schedule, if any.

    Returns None if no schedule file exists for the current platform.
    Raises NotImplementedError on unsupported platforms.
    """
    plat = detect_platform()
    if plat == "macos":
        if not _LAUNCHD_PLIST.exists():
            return None
        with _LAUNCHD_PLIST.open("rb") as fh:
            raw_bytes = fh.read()
        data = plistlib.loads(raw_bytes)
        cal = data.get("StartCalendarInterval", {})
        weekday = cal.get("Weekday")
        hour = cal.get("Hour")
        minute = cal.get("Minute")
        if weekday is None or hour is None or minute is None:
            return None
        day = next((k for k, v in _DAY_TO_LAUNCHD.items() if v == weekday), None)
        if day is None:
            return None
        return ScheduleSpec(
            day_of_week=day,  # type: ignore[arg-type]
            time=f"{int(hour):02d}:{int(minute):02d}",
            raw=raw_bytes.decode("utf-8", errors="replace"),
        )

    if plat == "linux":
        if not _SYSTEMD_TIMER.exists():
            return None
        text = _SYSTEMD_TIMER.read_text(encoding="utf-8")
        m = _ONCALENDAR_RE.search(text)
        if not m:
            return None
        day_str, hh, mm = m.group(1), m.group(2), m.group(3)
        day = _SYSTEMD_TO_DAY.get(day_str)
        if day is None:
            return None
        return ScheduleSpec(
            day_of_week=day,  # type: ignore[arg-type]
            time=f"{hh}:{mm}",
            raw=text,
        )

    raise NotImplementedError(f"unsupported platform: {plat}")


# ---------------------------------------------------------------------------
# write_schedule
# ---------------------------------------------------------------------------


def write_schedule(spec: ScheduleSpec) -> Path:
    """Install the schedule for the current platform.

    Renders the appropriate template, writes the file under the user's home,
    and runs the platform-specific load/enable command.

    Returns the path of the written file. Raises NotImplementedError on
    unsupported platforms; raises subprocess.CalledProcessError if
    launchctl/systemctl returns non-zero.
    """
    plat = detect_platform()
    repo = _repo_root()

    if plat == "macos":
        hour = int(spec.time.split(":")[0])
        minute = int(spec.time.split(":")[1])
        ctx = {
            "label": "com.litmonitor.discovery",
            "command": _uv_path(),
            "args": ["run", "lit-monitor", "run"],
            "working_dir": str(repo),
            "day_num": _DAY_TO_LAUNCHD[spec.day_of_week],
            "hour": hour,
            "minute": minute,
            "stdout_path": str(repo / "logs" / "discovery-launchd.stdout.log"),
            "stderr_path": str(repo / "logs" / "discovery-launchd.stderr.log"),
        }
        _LAUNCHD_DIR.mkdir(parents=True, exist_ok=True)
        rendered = _ENV.get_template("launchd.plist.j2").render(**ctx)
        _LAUNCHD_PLIST.write_text(rendered, encoding="utf-8")
        # Unload any previous incarnation (best-effort), then load the new one.
        subprocess.run(
            ["launchctl", "unload", str(_LAUNCHD_PLIST)],
            capture_output=True, check=False,
        )
        subprocess.run(
            ["launchctl", "load", str(_LAUNCHD_PLIST)],
            capture_output=True, check=True,
        )
        return _LAUNCHD_PLIST

    if plat == "linux":
        ctx = {
            "day_of_week_systemd": _DAY_TO_SYSTEMD[spec.day_of_week],
            "time": spec.time,
            "command": _uv_path(),
            "args": ["run", "lit-monitor", "run"],
            "working_dir": str(repo),
            "stdout_path": str(repo / "logs" / "discovery-systemd.stdout.log"),
            "stderr_path": str(repo / "logs" / "discovery-systemd.stderr.log"),
        }
        _SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
        timer_text = _ENV.get_template("systemd.timer.j2").render(**ctx)
        service_text = _ENV.get_template("systemd.service.j2").render(**ctx)
        _SYSTEMD_TIMER.write_text(timer_text, encoding="utf-8")
        _SYSTEMD_SERVICE.write_text(service_text, encoding="utf-8")
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "lit-monitor-discovery.timer"],
            capture_output=True, check=True,
        )
        return _SYSTEMD_TIMER

    raise NotImplementedError(f"unsupported platform: {plat}")


# ---------------------------------------------------------------------------
# remove_schedule
# ---------------------------------------------------------------------------


def remove_schedule() -> None:
    """Remove the installed schedule for the current platform.

    Idempotent: if nothing is installed, returns silently.
    Raises NotImplementedError on unsupported platforms.
    """
    plat = detect_platform()
    if plat == "macos":
        if _LAUNCHD_PLIST.exists():
            subprocess.run(
                ["launchctl", "unload", str(_LAUNCHD_PLIST)],
                capture_output=True, check=False,
            )
            _LAUNCHD_PLIST.unlink()
        return
    if plat == "linux":
        if _SYSTEMD_TIMER.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now",
                 "lit-monitor-discovery.timer"],
                capture_output=True, check=False,
            )
            _SYSTEMD_TIMER.unlink(missing_ok=True)
            _SYSTEMD_SERVICE.unlink(missing_ok=True)
        return
    raise NotImplementedError(f"unsupported platform: {plat}")


__all__ = [
    "DayOfWeek",
    "Platform",
    "ScheduleSpec",
    "detect_platform",
    "read_schedule",
    "remove_schedule",
    "write_schedule",
]
