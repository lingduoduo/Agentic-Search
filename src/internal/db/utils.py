"""DB utility helpers — error classification and patch sentinels."""

from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Final
from typing import TypeGuard
from typing import TypeVar

from psycopg2 import errorcodes
from psycopg2 import OperationalError
from psycopg2.errors import ForeignKeyViolation
from psycopg2.errors import UniqueViolation
from sqlalchemy.exc import IntegrityError

_T = TypeVar("_T")


class UnsetType:
    """Sentinel distinguishing 'not provided' from None/falsy in patch helpers."""

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET: Final[UnsetType] = UnsetType()


def is_set(value: _T | UnsetType) -> TypeGuard[_T]:
    return not isinstance(value, UnsetType)


def none_as_unset(value: _T | None) -> _T | UnsetType:
    """Map a request field's None (omitted) to UNSET. Only for non-nullable patch fields."""
    return UNSET if value is None else value


def is_unique_violation(exc: IntegrityError, constraint: str) -> bool:
    orig = exc.orig
    return (
        isinstance(orig, UniqueViolation)
        and getattr(orig.diag, "constraint_name", None) == constraint
    )


def is_fk_violation(exc: IntegrityError) -> bool:
    return isinstance(exc.orig, ForeignKeyViolation)


RETRYABLE_PG_CODES = {
    errorcodes.SERIALIZATION_FAILURE,
    errorcodes.DEADLOCK_DETECTED,
    errorcodes.CONNECTION_EXCEPTION,
    errorcodes.CONNECTION_DOES_NOT_EXIST,
    errorcodes.CONNECTION_FAILURE,
    errorcodes.TRANSACTION_ROLLBACK,
}


def is_retryable_sqlalchemy_error(exc: BaseException) -> bool:
    if isinstance(exc, OperationalError):
        pgcode = getattr(getattr(exc, "orig", None), "pgcode", None)
        return pgcode in RETRYABLE_PG_CODES
    return False


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


def model_to_dict(model: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy model instance to a plain dict."""
    from sqlalchemy import inspect as sa_inspect

    return {c.key: getattr(model, c.key) for c in sa_inspect(model).mapper.column_attrs}
