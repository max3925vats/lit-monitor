"""Dev-only component gallery + the HTMX×Shoelace form-association spike. Mounted only
under `serve --dev`. A throwaway verification surface (not shipped behaviour)."""
from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/dev/sl-probe", response_class=HTMLResponse)
def sl_probe(request: Request):
    from lit_monitor.server.app import templates  # noqa: PLC0415
    return templates.TemplateResponse(request, "dev/_sl_probe.html", {})


@router.post("/dev/sl-probe/echo", response_class=HTMLResponse)
def sl_probe_echo(probe: str = Form(default=""), flag: str = Form(default="")):
    return HTMLResponse(f'<div id="echo">probe={probe}|flag={flag}</div>')
