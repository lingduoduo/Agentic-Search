"""Base Data Access Layer (DAL) for database operations.

Subclasses add domain-specific query methods while inheriting session
management. Supports both external sessions (FastAPI Depends) and
self-managed sessions (background tasks).

Example (FastAPI)::

    def get_scim_dal(db_session: Session = Depends(get_session)) -> ScimDAL:
        return ScimDAL(db_session)

Example (background task)::

    with ScimDAL.from_tenant("tenant_abc") as dal:
        dal.create_user_mapping(...)
        dal.commit()
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from src.backend.db.engine.sql_engine import get_session_with_tenant


class DAL:
    def __init__(self, db_session: Session) -> None:
        self._session = db_session

    @property
    def session(self) -> Session:
        return self._session

    def commit(self) -> None:
        self._session.commit()

    def flush(self) -> None:
        self._session.flush()

    def rollback(self) -> None:
        self._session.rollback()

    @classmethod
    @contextmanager
    def from_tenant(cls, tenant_id: str) -> Generator[DAL, None, None]:
        """Create a DAL with a self-managed session for the given tenant."""
        with get_session_with_tenant(tenant_id=tenant_id) as session:
            yield cls(session)
