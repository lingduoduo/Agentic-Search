"""Pydantic models for the manage API."""

from __future__ import annotations

from pydantic import BaseModel


class StandardAnswerCategoryCreationRequest(BaseModel):
    name: str


class StandardAnswerCategory(BaseModel):
    id: str
    name: str

    @classmethod
    def from_record(cls, record: dict) -> "StandardAnswerCategory":
        return cls(id=str(record["id"]), name=str(record["name"]))


class StandardAnswerCreationRequest(BaseModel):
    keyword: str
    answer: str
    categories: list[str] = []
    match_regex: bool = False
    match_any_keywords: bool = False


class StandardAnswer(BaseModel):
    id: str
    keyword: str
    answer: str
    categories: list[str]
    match_regex: bool
    match_any_keywords: bool

    @classmethod
    def from_record(cls, record: dict) -> "StandardAnswer":
        return cls(
            id=str(record["id"]),
            keyword=str(record["keyword"]),
            answer=str(record["answer"]),
            categories=[str(c) for c in record.get("categories", [])],
            match_regex=bool(record.get("match_regex", False)),
            match_any_keywords=bool(record.get("match_any_keywords", False)),
        )


__all__ = [
    "StandardAnswer",
    "StandardAnswerCategory",
    "StandardAnswerCategoryCreationRequest",
    "StandardAnswerCreationRequest",
]
