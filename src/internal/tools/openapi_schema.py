from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Simplified type names accepted in this schema format → JSON Schema type strings
_TO_JSON_SCHEMA_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


class ParameterType(str, Enum):
    """Supported parameter types in the simplified schema format."""

    STR: str = "str"
    INT: str = "int"
    FLOAT: str = "float"
    BOOL: str = "bool"


ParameterTypeMap = {
    ParameterType.STR: str,
    ParameterType.INT: int,
    ParameterType.FLOAT: float,
    ParameterType.BOOL: bool,
}


class ParameterIn(str, Enum):
    """Supported parameter locations."""

    PATH: str = "path"
    QUERY: str = "query"
    HEADER: str = "header"
    COOKIE: str = "cookie"
    REQUEST_BODY: str = "request_body"


class OpenAPISchema(BaseModel):
    """Validated simplified OpenAPI schema for user-supplied tool specs.

    Accepted format::

        {
            "server": "https://api.example.com",
            "description": "My tool provider",
            "paths": {
                "/search": {
                    "get": {
                        "operationId": "search",
                        "description": "Search the index",
                        "parameters": [
                            {"name": "q", "in": "query", "description": "Query",
                             "required": true, "type": "str"}
                        ]
                    }
                }
            }
        }

    Call ``to_openapi_json()`` to get a standard OpenAPI JSON string that
    ``tool_registry.register_from_openapi()`` (and ``parse_openapi_schema()``)
    can consume directly.
    """

    server: str = Field(
        default="",
        validate_default=True,
        description="Base URL of the tool provider service",
    )
    description: str = Field(
        default="",
        validate_default=True,
        description="Description of the tool provider",
    )
    paths: dict[str, dict] = Field(
        default_factory=dict,
        validate_default=True,
        description="Path definitions",
    )

    @field_validator("server", mode="before")
    @classmethod
    def validate_server(cls, server: str) -> str:
        if not server:
            raise ValueError("'server' cannot be empty")
        return server

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, description: str) -> str:
        if not description:
            raise ValueError("'description' cannot be empty")
        return description

    @field_validator("paths", mode="before")
    @classmethod
    def validate_paths(cls, paths: dict[str, dict]) -> dict[str, dict]:
        if not paths or not isinstance(paths, dict):
            raise ValueError("'paths' must be a non-empty dict")

        _valid_methods = {"get", "post"}
        _valid_ins = {item.value for item in ParameterIn}
        _valid_types = {item.value for item in ParameterType}

        operation_ids: list[str] = []
        validated: dict[str, dict] = {}

        for path, path_item in paths.items():
            validated[path] = {}
            for method, operation in path_item.items():
                if method not in _valid_methods:
                    continue
                if not isinstance(operation.get("description"), str):
                    raise ValueError(
                        f"{method.upper()} {path}: operation must have a 'description' string"
                    )
                if not isinstance(operation.get("operationId"), str):
                    raise ValueError(
                        f"{method.upper()} {path}: operation must have an 'operationId' string"
                    )
                if not isinstance(operation.get("parameters", []), list):
                    raise ValueError(
                        f"{method.upper()} {path}: 'parameters' must be a list or omitted"
                    )
                op_id = operation["operationId"]
                if op_id in operation_ids:
                    raise ValueError(
                        f"'operationId' must be unique; duplicate: {op_id!r}"
                    )
                operation_ids.append(op_id)

                validated_params = []
                for param in operation.get("parameters", []):
                    if not isinstance(param.get("name"), str):
                        raise ValueError("parameter.name must be a non-empty string")
                    if not isinstance(param.get("description"), str):
                        raise ValueError(
                            "parameter.description must be a non-empty string"
                        )
                    if not isinstance(param.get("required"), bool):
                        raise ValueError("parameter.required must be a boolean")
                    if param.get("in") not in _valid_ins:
                        raise ValueError(
                            f"parameter.in must be one of: {'/'.join(sorted(_valid_ins))}"
                        )
                    if param.get("type") not in _valid_types:
                        raise ValueError(
                            f"parameter.type must be one of: {'/'.join(sorted(_valid_types))}"
                        )
                    validated_params.append(
                        {
                            "name": param["name"],
                            "in": param["in"],
                            "description": param["description"],
                            "required": param["required"],
                            "type": param["type"],
                        }
                    )

                validated[path][method] = {
                    "description": operation["description"],
                    "operationId": operation["operationId"],
                    "parameters": validated_params,
                }

        return validated

    def to_openapi_json(self) -> str:
        """Serialize to a standard OpenAPI JSON string.

        Converts the simplified schema into the envelope that
        ``parse_openapi_schema()`` and ``tool_registry.register_from_openapi()``
        already understand:

        - ``server`` → ``servers[0].url``
        - ``description`` → ``info.title`` / ``info.description``
        - simplified ``parameters[].type`` → ``parameters[].schema.type``
        """
        paths: dict[str, dict] = {}
        for path, path_item in self.paths.items():
            paths[path] = {}
            for method, operation in path_item.items():
                openapi_params = [
                    {
                        "name": p["name"],
                        "in": p["in"],
                        "description": p["description"],
                        "required": p["required"],
                        "schema": {
                            "type": _TO_JSON_SCHEMA_TYPE.get(p["type"], "string")
                        },
                    }
                    for p in operation.get("parameters", [])
                ]
                paths[path][method] = {
                    "operationId": operation["operationId"],
                    "description": operation["description"],
                    "parameters": openapi_params,
                }

        doc = {
            "openapi": "3.0.0",
            "info": {"title": self.description, "description": self.description},
            "servers": [{"url": self.server}],
            "paths": paths,
        }
        return json.dumps(doc)
