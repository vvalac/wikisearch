from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class SafetyResult(BaseModel):
    category: Literal["clean", "misuse", "harmful"]
    reason: str


class WikiPage(BaseModel):
    title: str
    summary: str
    url: str


class SearchIteration(BaseModel):
    query: str
    pages: list[WikiPage]


class SearchOutput(BaseModel):
    iterations: list[SearchIteration]


class WikiResponse(BaseModel):
    answer: str
    sources: list[str]
