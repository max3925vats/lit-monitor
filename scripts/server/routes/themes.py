"""Bundle H (v0.9): cluster/theme review HTTP endpoints + HTML pages.

Routes:
    GET  /api/themes                          — list all active clusters
    GET  /api/themes/{cluster_id}             — single cluster + members
    POST /api/themes/{cluster_id}/write-back/tags        — tags dry-run / confirm
    POST /api/themes/{cluster_id}/write-back/collections — collections dry-run / confirm
    GET  /themes                              — HTML cluster review index
    GET  /themes/{cluster_id}                 — HTML per-cluster detail
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from scripts.server.runtime import get_runtime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["themes"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_db() -> Any | None:
    """Return state_db or None on failure."""
    try:
        return get_runtime().state_db
    except Exception:
        return None


def _get_templates():
    """Lazy import to avoid circular dependency at module load."""
    from scripts.server.app import templates
    return templates


def _get_active_clusters(db: Any) -> list[dict]:
    """Return non-archived cluster rows ordered by display_name."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT id, display_name, n_papers, cohesion_score, computed_at "
            "FROM clusters WHERE archived = 0 "
            "ORDER BY display_name"
        ).fetchall()
    return [dict(r) for r in rows]


def _get_cluster_by_id(db: Any, cluster_id: int) -> dict | None:
    """Return a single cluster row or None."""
    with db._connect() as conn:
        row = conn.execute(
            "SELECT id, display_name, n_papers, cohesion_score, computed_at "
            "FROM clusters WHERE id = ? AND archived = 0",
            (cluster_id,),
        ).fetchone()
    return dict(row) if row else None


def _get_cluster_papers(db: Any, cluster_id: int) -> list[dict]:
    """Return papers in a cluster joined with discovery score data.

    Returns dicts with: doi, title, score, rationale, score_breakdown_json,
    ingested, distance_to_centroid.
    """
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT
                ca.doi,
                p.title,
                ca.distance_to_centroid
            FROM cluster_assignments ca
            LEFT JOIN papers p ON p.doi = ca.doi
            WHERE ca.cluster_id = ?
            ORDER BY ca.distance_to_centroid ASC
            """,
            (cluster_id,),
        ).fetchall()
    result = []
    for r in rows:
        result.append(
            {
                "doi": r[0],
                "title": r[1],
                "distance_to_centroid": r[2],
                # No score/rationale/score_breakdown from the library (these are
                # extraction-only fields); set safe defaults for the partial.
                "score": None,
                "rationale": None,
                "score_breakdown_json": None,
                "ingested": 0,
            }
        )
    return result


# ---------------------------------------------------------------------------
# GET /api/themes — list clusters
# ---------------------------------------------------------------------------

@router.get("/api/themes")
def list_themes() -> list[dict]:
    """List all active (non-archived) cluster rows.

    Returns:
        List of dicts with: id, display_name, n_papers, cohesion_score, computed_at.
    """
    db = _safe_db()
    if db is None:
        return []
    try:
        return _get_active_clusters(db)
    except Exception as exc:
        logger.warning("list_themes failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# GET /api/themes/{cluster_id} — single cluster + members
# ---------------------------------------------------------------------------

@router.get("/api/themes/{cluster_id}")
def get_theme(cluster_id: int) -> dict:
    """Return one cluster and its member papers.

    Returns:
        Dict with keys: cluster (metadata dict) + papers (list of member dicts).

    Raises:
        HTTPException(404): If cluster_id does not exist or is archived.
        HTTPException(503): If state DB is unavailable.
    """
    db = _safe_db()
    if db is None:
        raise HTTPException(status_code=503, detail="State DB unavailable")

    cluster = _get_cluster_by_id(db, cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Theme {cluster_id} not found")

    papers = _get_cluster_papers(db, cluster_id)
    return {"cluster": cluster, "papers": papers}


# ---------------------------------------------------------------------------
# POST /api/themes/{cluster_id}/write-back/{mode}
# ---------------------------------------------------------------------------

@router.post("/api/themes/{cluster_id}/write-back/{mode}")
def write_back_cluster(
    cluster_id: int,
    mode: str,
    confirm: bool = Query(default=False),
) -> dict:
    """Dry-run or confirmed Zotero write-back for a cluster's tags / collections.

    Path params:
        cluster_id: Cluster primary key.
        mode:       'tags' or 'collections'.

    Query params:
        confirm: False (default) → dry-run preview. True → live write.

    Returns:
        {"dry_run": bool, "diff": [...]} — preview of what would be written.
        When confirm=True the diff is the set of items actually written.

    Raises:
        HTTPException(400): If mode is not 'tags' or 'collections'.
        HTTPException(404): If cluster not found.
        HTTPException(503): If state DB or Zotero unavailable.
    """
    if mode not in ("tags", "collections"):
        raise HTTPException(
            status_code=400,
            detail=f"mode must be 'tags' or 'collections', got {mode!r}",
        )

    db = _safe_db()
    if db is None:
        raise HTTPException(status_code=503, detail="State DB unavailable")

    cluster = _get_cluster_by_id(db, cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Theme {cluster_id} not found")

    dry_run = not confirm

    try:
        zot = get_runtime().zotero_client
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Zotero unavailable: {exc}") from exc

    try:
        if mode == "tags":
            from scripts.clustering.write_back import push_tags_to_zotero
            report = push_tags_to_zotero(db, zot, dry_run=dry_run)
        else:
            from scripts.clustering.write_back import push_collections_to_zotero
            report = push_collections_to_zotero(db, zot, dry_run=dry_run)
    except Exception as exc:
        # A3-5 info-leak guard: log the full exception (with traceback) server-
        # side, but return only a generic detail to the client — str(exc) here
        # can embed Zotero internals / filesystem paths.
        logger.error(
            "write_back_cluster %s/%s failed", cluster_id, mode, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal error") from exc

    return {"dry_run": dry_run, "diff": report}


# ---------------------------------------------------------------------------
# GET /themes — HTML cluster review index
# ---------------------------------------------------------------------------

@router.get("/themes", response_class=HTMLResponse)
def themes_index(request: Request) -> HTMLResponse:
    """HTML cluster review index — list of themes with write-back buttons."""
    db = _safe_db()
    themes: list[dict] = []
    if db is not None:
        try:
            themes = _get_active_clusters(db)
        except Exception as exc:
            logger.warning("themes index: cluster query failed: %s", exc)

    templates = _get_templates()
    return templates.TemplateResponse(
        request,
        "themes/index.html",
        {"themes": themes},
    )


# ---------------------------------------------------------------------------
# GET /themes/{cluster_id} — HTML per-cluster detail
# ---------------------------------------------------------------------------

@router.get("/themes/{cluster_id}", response_class=HTMLResponse)
def themes_detail(request: Request, cluster_id: int) -> HTMLResponse:
    """HTML per-cluster detail — papers in cluster + assignment distance."""
    db = _safe_db()
    if db is None:
        raise HTTPException(status_code=503, detail="State DB unavailable")

    cluster = _get_cluster_by_id(db, cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Theme {cluster_id} not found")

    papers = _get_cluster_papers(db, cluster_id)

    # Bundle H: read web_ui.show_feedback_buttons config flag (default False).
    show_feedback_buttons = False
    try:
        from scripts.server.config_io import load_config
        cfg = load_config("extraction")
        show_feedback_buttons = bool(
            cfg.get("web_ui", {}).get("show_feedback_buttons", False)
        )
    except Exception:
        pass

    templates = _get_templates()
    return templates.TemplateResponse(
        request,
        "themes/detail.html",
        {
            "cluster": cluster,
            "papers": papers,
            "show_feedback_buttons": show_feedback_buttons,
        },
    )
