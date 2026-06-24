from __future__ import annotations
import json
from typing import Literal
from pydantic import BaseModel, field_validator


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


class WikiFact(BaseModel):
    summary: str
    source: str | None = None  # None when the model fails to supply a URL; filtered before main agent sees it


class WikiSearchResult(BaseModel):
    facts: list[WikiFact]


class WikiResponse(BaseModel):
    answer: str
    sources: list[str] = []

    @field_validator("sources", mode="before")
    @classmethod
    def _normalise_sources(cls, v: object) -> list[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (ValueError, TypeError):
                pass
            return [v]
        if isinstance(v, list):
            result: list[str] = []
            for item in v:
                if isinstance(item, str) and item.startswith("["):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, list):
                            result.extend(str(x) for x in parsed)
                            continue
                    except (ValueError, TypeError):
                        pass
                result.append(str(item))
            return result
        return v  # type: ignore[return-value]
