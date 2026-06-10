from typing import Any

from sqlalchemy.orm import Session

from src.backend.context.search.models import IndexFilters

# Stub: repo-local user type (real model lives in src.backend.db.models)
User = Any


def build_access_filters_for_user(user: User, session: Session) -> list[str]:
    raise NotImplementedError(
        "build_access_filters_for_user requires a DB-backed User model and ACL query"
    )


def build_user_only_filters(user: User, db_session: Session) -> IndexFilters:
    user_acl_filters = build_access_filters_for_user(user, db_session)
    return IndexFilters(
        source_type=None,
        document_set=None,
        time_cutoff=None,
        tags=None,
        access_control_list=user_acl_filters,
    )
