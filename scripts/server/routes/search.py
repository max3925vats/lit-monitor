"""H10: POST /api/search — free-text paper retrieval endpoint.

Delegates entirely to scripts.api.queries.get_papers_by_query, which is the
single implementation shared with the MCP tools (find_papers_by_query and
find_papers_by_query_hybrid).  HTTP and MCP surfaces are guaranteed to return
the same results for the same inputs.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

# Import at module level so tests can monkeypatch this binding directly.
from scripts.api.queries import get_papers_by_query

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    """Request body for POST /api/search."""

    query: str
    mode: Literal["vector", "graph", "hybrid"] = "vector"
    k: int = 20

    @field_validator("query")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()

    @field_validator("k")
    @classmethod
    def _k_in_range(cls, v: int) -> int:
        if not (1 <= v <= 100):
            raise ValueError("k must be in [1, 100]")
        return v


@router.post("/api/search")
def search(body: SearchRequest) -> list[dict[str, Any]]:
    """Free-text paper search.

    Returns up to ``k`` papers ranked by the chosen retrieval mode:
    - **vector**: cosine similarity via ChromaDB.
    - **graph**: entity-resolve + MENTIONS overlap via KuzuDB.
    - **hybrid**: both legs fused via Reciprocal Rank Fusion (RRF).

    Each result has at minimum ``{doi, title, score}``.
    """
    return get_papers_by_query(body.query, mode=body.mode, k=body.k)
