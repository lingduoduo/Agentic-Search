"""Standard answer admin API.

py.
SQLAlchemy ORM imports replaced with AgenticSearchStore.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import Response

from src.internal.auth import AuthenticatedUser
from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.servers.manage.models import StandardAnswer
from src.internal.servers.manage.models import StandardAnswerCategory
from src.internal.servers.manage.models import StandardAnswerCategoryCreationRequest
from src.internal.servers.manage.models import StandardAnswerCreationRequest
from src.internal.servers._auth import make_require_admin


def create_manage_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter with standard-answer admin endpoints."""

    router = APIRouter(prefix="/manage", tags=["manage"])

    _require_admin = make_require_admin(app_settings)

    # -----------------------------------------------------------------------
    # Standard answers
    # -----------------------------------------------------------------------

    @router.post("/admin/standard-answer", status_code=201)
    def create_standard_answer(
        req: StandardAnswerCreationRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> StandardAnswer:
        record = store.insert_standard_answer(
            keyword=req.keyword,
            answer=req.answer,
            category_ids=req.categories,
            match_regex=req.match_regex,
            match_any_keywords=req.match_any_keywords,
        )
        return StandardAnswer.from_record(record)

    @router.get("/admin/standard-answer")
    def list_standard_answers(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[StandardAnswer]:
        return [StandardAnswer.from_record(r) for r in store.fetch_standard_answers()]

    @router.patch("/admin/standard-answer/{answer_id}")
    def patch_standard_answer(
        answer_id: str,
        req: StandardAnswerCreationRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> StandardAnswer:
        if store.fetch_standard_answer(answer_id) is None:
            raise HTTPException(status_code=404, detail="Standard answer not found.")
        record = store.update_standard_answer(
            answer_id=answer_id,
            keyword=req.keyword,
            answer=req.answer,
            category_ids=req.categories,
            match_regex=req.match_regex,
            match_any_keywords=req.match_any_keywords,
        )
        return StandardAnswer.from_record(record)

    @router.delete(
        "/admin/standard-answer/{answer_id}",
        status_code=204,
        response_class=Response,
        response_model=None,
    )
    def delete_standard_answer(
        answer_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> Response:
        store.remove_standard_answer(answer_id)
        return Response(status_code=204)

    # -----------------------------------------------------------------------
    # Standard answer categories
    # -----------------------------------------------------------------------

    @router.post("/admin/standard-answer/category", status_code=201)
    def create_standard_answer_category(
        req: StandardAnswerCategoryCreationRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> StandardAnswerCategory:
        record = store.insert_standard_answer_category(name=req.name)
        return StandardAnswerCategory.from_record(record)

    @router.get("/admin/standard-answer/category")
    def list_standard_answer_categories(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[StandardAnswerCategory]:
        return [
            StandardAnswerCategory.from_record(r)
            for r in store.fetch_standard_answer_categories()
        ]

    @router.patch("/admin/standard-answer/category/{category_id}")
    def patch_standard_answer_category(
        category_id: str,
        req: StandardAnswerCategoryCreationRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> StandardAnswerCategory:
        if store.fetch_standard_answer_category(category_id) is None:
            raise HTTPException(
                status_code=404, detail="Standard answer category not found."
            )
        record = store.update_standard_answer_category(
            category_id=category_id, name=req.name
        )
        return StandardAnswerCategory.from_record(record)

    return router


__all__ = ["create_manage_router"]
