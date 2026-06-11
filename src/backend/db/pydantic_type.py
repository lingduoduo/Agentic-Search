from typing import Any
from typing import Optional
from typing import Type

from pydantic import BaseModel
from pydantic import TypeAdapter
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator


class PydanticType(TypeDecorator):
    impl = JSONB

    def __init__(
        self, pydantic_model: Type[BaseModel], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._adapter: TypeAdapter[BaseModel] = TypeAdapter(pydantic_model)

    def process_bind_param(
        self,
        value: Optional[BaseModel],
        dialect: Any,  # noqa: ARG002
    ) -> Optional[dict]:
        if value is not None:
            return self._adapter.dump_python(value, mode="json")
        return None

    def process_result_value(
        self,
        value: Optional[dict],
        dialect: Any,  # noqa: ARG002
    ) -> Optional[BaseModel]:
        if value is not None:
            return self._adapter.validate_python(value)
        return None


class PydanticListType(TypeDecorator):
    impl = JSONB

    def __init__(
        self, pydantic_model: Type[BaseModel], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._adapter: TypeAdapter[list[BaseModel]] = TypeAdapter(list[pydantic_model])

    def process_bind_param(
        self,
        value: Optional[list[BaseModel]],
        dialect: Any,  # noqa: ARG002
    ) -> Optional[list[dict]]:
        if value is not None:
            return self._adapter.dump_python(value, mode="json")
        return None

    def process_result_value(
        self,
        value: Optional[list[dict]],
        dialect: Any,  # noqa: ARG002
    ) -> Optional[list[BaseModel]]:
        if value is not None:
            return self._adapter.validate_python(value)
        return None
