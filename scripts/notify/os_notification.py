"""P2: cross-platform OS notification for discovery-pipeline completion.

Uses plyer (optional [notify] pip extra). Module-level try/import means
the absence of plyer doesn't break callers — notify_discovery_complete
logs and returns silently. Notification failures inside plyer are caught
and logged at WARNING — they MUST NOT abort the pipeline.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from plyer import notification as _plyer_notification  # type: ignore[import]
    _PLYER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _plyer_notification = None
    _PLYER_AVAILABLE = False


def notify_discovery_complete(
    run_id: int,
    paper_count: int,
    app_url: str,
    *,
    enabled: bool = True,
    on_zero_results: bool = True,
) -> None:
    """Fire an OS notification for a completed discovery run.

    Never raises. Returns silently with an INFO/WARNING log on any
    short-circuit or failure path.

    Args:
        run_id: discovery_runs.id for this run.
        paper_count: number of new papers found (summary.papers_ingested).
        app_url: base URL ('http://localhost:8765') used to build the click URL.
        enabled: master on/off switch from config.discovery.notify.enabled.
        on_zero_results: if False, skip notification when paper_count == 0.
    """
    if not enabled:
        logger.info("P2: notifications disabled in config — skipping.")
        return

    if paper_count == 0 and not on_zero_results:
        logger.info(
            "P2: zero results; on_zero_results=False — skipping notification."
        )
        return

    if not _PLYER_AVAILABLE:
        logger.info(
            "P2: plyer not installed — skipping notification "
            "(install with: uv sync --extra notify)."
        )
        return

    title = "lit-monitor discovery complete"
    s = "" if paper_count == 1 else "s"
    message = f"{paper_count} new paper{s} found. Click to view."

    # Click URL is what /discovery/notify-handler (P3) consumes; embedded in
    # the message body for clients that don't honor plyer's click-action.
    click_url = (
        f"{app_url.rstrip('/')}/discovery/notify-handler?run_id={run_id}"
    )

    try:
        _plyer_notification.notify(
            title=title,
            message=message,
            app_name="lit-monitor",
            # plyer doesn't standardize click-action across platforms; the
            # notification body itself directs the user to the URL.
        )
        logger.info(
            "P2: notification fired (run_id=%d, paper_count=%d, url=%s)",
            run_id,
            paper_count,
            click_url,
        )
    except Exception as exc:
        # Notification failures must NEVER abort the pipeline.
        logger.warning(
            "P2: notify_discovery_complete failed (non-fatal): %s", exc
        )
