from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable

import wikipediaapi
from langfuse import get_client, observe
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

from models import SafetyResult, SearchIteration, WikiPage, WikiResponse
from prompts import get as get_prompt

_USER_AGENT = "wikisearch/0.1 (adamscook@gmail.com)"
_wiki = wikipediaapi.Wikipedia(user_agent=_USER_AGENT, language="en")


def _fetch_page(title: str) -> WikiPage | None:
    page = _wiki.page(title)
    if not page.exists():
        return None
    return WikiPage(title=page.title, summary=page.summary[:4000], url=page.fullurl)


# ---------------------------------------------------------------------------
# Stream types
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    message: str


@dataclass
class CleanResult:
    response: WikiResponse
    new_history: list[ModelMessage]


@dataclass
class MisuseResult:
    safety_reason: str


@dataclass
class HarmfulResult:
    safety_reason: str


QueryOutcome = CleanResult | MisuseResult | HarmfulResult
StreamEvent = Checkpoint | QueryOutcome


# ---------------------------------------------------------------------------
# Step 2 — Safety filter (private)
# ---------------------------------------------------------------------------

_safety_agent: Agent[None, SafetyResult] = Agent(
    model="anthropic:claude-haiku-4-5",
    output_type=SafetyResult,
    defer_model_check=True,
)


@_safety_agent.system_prompt
def _safety_system_prompt() -> str:
    return get_prompt("safety-filter")


@observe(as_type="generation", name="safety-filter")
async def _filter_harm(query: str) -> SafetyResult:
    prompt = get_client().get_prompt("safety-filter")
    get_client().update_current_generation(prompt=prompt)
    result = await _safety_agent.run(query)
    return result.output


# ---------------------------------------------------------------------------
# Steps 3 + 4 + 5 — Main agent with Wikipedia search tool (private)
# ---------------------------------------------------------------------------

@dataclass
class _MainDeps:
    on_status: Callable[[str], None] | None
    iterations: list[SearchIteration] = field(default_factory=list)
    searches_done: int = 0


@dataclass
class _SearchDeps:
    on_status: Callable[[str], None] | None
    iterations: list[SearchIteration] = field(default_factory=list)
    attempt: int = 0


_search_agent: Agent[_SearchDeps, str] = Agent(
    model="anthropic:claude-sonnet-4-6",
    output_type=str,
    deps_type=_SearchDeps,
    defer_model_check=True,
)


@_search_agent.system_prompt
def _search_system_prompt() -> str:
    return get_prompt("search-agent")


@_search_agent.tool
async def search_wikipedia(ctx: RunContext[_SearchDeps], title: str) -> str:
    """Search Wikipedia for an article by title. Returns page content or a not-found message."""
    if ctx.deps.attempt >= 5:
        return "No relevant Wikipedia content found."
    ctx.deps.attempt += 1
    if ctx.deps.on_status:
        ctx.deps.on_status(f"Searching Wikipedia: '{title}'…")

    page = _fetch_page(title)
    iteration = SearchIteration(query=title, pages=[page] if page else [])
    ctx.deps.iterations.append(iteration)

    if not page:
        return f"No Wikipedia page found for '{title}'. Try a different or simpler title."
    return f"Title: {page.title}\nURL: {page.url}\n\n{page.summary}"


@observe(as_type="generation", name="search-agent")
async def _run_search(query: str, deps: _MainDeps) -> str:
    prompt = get_client().get_prompt("search-agent")
    get_client().update_current_generation(prompt=prompt)
    search_deps = _SearchDeps(on_status=deps.on_status)
    result = await _search_agent.run(query, deps=search_deps)
    deps.iterations.extend(search_deps.iterations)
    return result.output


_main_agent: Agent[_MainDeps, WikiResponse] = Agent(
    model="anthropic:claude-opus-4-8",
    output_type=WikiResponse,
    deps_type=_MainDeps,
    defer_model_check=True,
)


@_main_agent.system_prompt
def _main_system_prompt() -> str:
    return get_prompt("main-agent")


@_main_agent.tool
async def wikipedia_search(ctx: RunContext[_MainDeps], query: str) -> str:
    """Search Wikipedia for relevant content. Pass the best article title as query."""
    if ctx.deps.searches_done >= 1:
        return "Search limit reached. Synthesise an answer from the content already retrieved."
    ctx.deps.searches_done += 1
    return await _run_search(query, ctx.deps)


@observe(as_type="generation", name="main-agent")
async def _process_query(
    query: str,
    history: list[ModelMessage],
    on_status: Callable[[str], None] | None = None,
) -> tuple[WikiResponse, list[ModelMessage]]:
    prompt = get_client().get_prompt("main-agent")
    get_client().update_current_generation(prompt=prompt)
    deps = _MainDeps(on_status=on_status)
    result = await _main_agent.run(query, message_history=history, deps=deps)
    return result.output, result.all_messages()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_query(
    query: str,
    history: list[ModelMessage],
    skip_safety: bool = False,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Async generator: yields Checkpoint events then a final QueryOutcome.
    Callers iterate with `async for` — each yield hands control back to the
    event loop so UIs get a render slot between status updates.
    """
    with get_client().start_as_current_observation(name="wikisearch-query", input=query):
        if not skip_safety:
            yield Checkpoint("Checking safety…")
            safety = await _filter_harm(query)
            if safety.category == "misuse":
                yield MisuseResult(safety_reason=safety.reason)
                return
            if safety.category == "harmful":
                yield HarmfulResult(safety_reason=safety.reason)
                return

        # Bridge on_status callbacks into the generator stream via a queue
        # so Wikipedia search titles appear as checkpoints in real time.
        status_q: asyncio.Queue[str] = asyncio.Queue()

        def on_status(text: str) -> None:
            status_q.put_nowait(text)

        yield Checkpoint("Thinking…")
        task = asyncio.create_task(_process_query(query, history, on_status=on_status))

        while not task.done():
            try:
                yield Checkpoint(status_q.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0)

        response, new_history = await task
        yield CleanResult(response=response, new_history=new_history)
