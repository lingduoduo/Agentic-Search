"""HTTP router exposing the conversation-memory service under /api/memory."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.internal.auth import user_from_headers
from src.internal.memory import service
from src.internal.memory.service import DEFAULT_MEMORY_USER_ID, maybe_build_encoder


class _SaveRequest(BaseModel):
    text: str = Field(..., min_length=1)


class _SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=100)


class _ConsolidateRequest(BaseModel):
    resolve_conflicts: bool = True


class _CurateRequest(BaseModel):
    session_id: str | None = None


def create_memory_router(
    db, llm=None, *, default_user_id: str = DEFAULT_MEMORY_USER_ID
) -> APIRouter:
    router = APIRouter(prefix="/api/memory", tags=["memory"])

    def _uid(request: Request) -> str:
        user = user_from_headers(request.headers)
        return user.id if user is not None else default_user_id

    @router.post("/save")
    def save(body: _SaveRequest, request: Request) -> dict:
        return {"memory_id": service.save_memory(db, _uid(request), body.text)}

    @router.get("/list")
    def list_memories(request: Request) -> dict:
        records = db.get_user_memory_records(_uid(request))
        return {
            "memories": [
                {"id": r.id, "text": r.memory_text, "updated_at": r.updated_at}
                for r in records
            ]
        }

    @router.post("/search")
    def search(body: _SearchRequest, request: Request) -> dict:
        hits = service.search_memories(
            db,
            _uid(request),
            body.query,
            max_results=body.max_results,
            encoder=maybe_build_encoder(),
        )
        return {
            "results": [
                {"id": r.id, "text": r.memory_text, "score": s} for r, s in hits
            ]
        }

    @router.post("/consolidate")
    def consolidate(body: _ConsolidateRequest, request: Request) -> dict:
        return {
            "report": service.consolidate_memories(
                db, _uid(request), resolve_conflicts=body.resolve_conflicts
            )
        }

    @router.get("/profile")
    def get_profile(request: Request) -> dict:
        return {
            "profile": [asdict(e) for e in service.get_user_profile(db, _uid(request))]
        }

    @router.post("/profile/generate")
    def generate_profile(request: Request) -> dict:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM not configured")
        entries = service.generate_user_profile(db, _uid(request), llm)
        return {"profile": [asdict(e) for e in entries]}

    @router.post("/curate")
    async def curate(body: _CurateRequest, request: Request) -> dict:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM not configured")
        return await service.curate_from_conversation(
            db, _uid(request), llm, session_id=body.session_id
        )

    return router
