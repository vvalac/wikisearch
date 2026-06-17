"""
WikiSearch POC — hardcoded query, Wikipedia fetch, Anthropic LLM grounded response.

Flow: query → Wikipedia fetch → pydantic-ai agent (claude-opus-4-8) → structured output
"""

import asyncio
from dotenv import load_dotenv
import wikipediaapi
from pydantic import BaseModel
from pydantic_ai import Agent

load_dotenv()


QUERY = "What is the largest ocean on Earth?"
WIKI_PAGE = "Pacific Ocean"  # hardcoded for POC; in real app this comes from LLM tool call


# --- Pydantic models ---

class WikiSearchResult(BaseModel):
    title: str
    summary: str
    url: str


class WikiSearchResponse(BaseModel):
    query_used: str
    results: list[WikiSearchResult]


class FinalResponse(BaseModel):
    answer: str
    sources: list[str]


# --- Wikipedia fetch ---

def fetch_wikipedia(page_title: str) -> WikiSearchResponse:
    wiki = wikipediaapi.Wikipedia(
        user_agent="wikisearch-poc/0.1",
        language="en",
    )
    page = wiki.page(page_title)
    results = []
    if page.exists():
        results.append(WikiSearchResult(
            title=page.title,
            summary=page.summary[:3000],
            url=page.fullurl,
        ))
    return WikiSearchResponse(query_used=page_title, results=results)


# --- Main ---

async def main() -> None:
    print(f"Query: {QUERY}\n")
    print(f"Fetching Wikipedia: '{WIKI_PAGE}'...")

    wiki_response = fetch_wikipedia(WIKI_PAGE)

    if not wiki_response.results:
        print("No Wikipedia content found.")
        return

    for r in wiki_response.results:
        print(f"  Found: {r.title}")
        print(f"  URL:   {r.url}\n")

    context = "\n\n".join(
        f"Title: {r.title}\nURL: {r.url}\n\n{r.summary}"
        for r in wiki_response.results
    )

    agent: Agent[None, FinalResponse] = Agent(
        model="anthropic:claude-sonnet-4-6",
        output_type=FinalResponse,
        system_prompt=(
            "You are WikiSearch. Answer the user's question using only the Wikipedia "
            "content provided. Be concise and factual. Always populate 'sources' with "
            "the URLs of any Wikipedia articles you drew from."
        ),
    )

    print("Calling LLM...")
    result = await agent.run(
        f"Question: {QUERY}\n\nWikipedia content:\n{context}"
    )

    print("\n=== Answer ===")
    print(result.output.answer)
    print("\n=== Sources ===")
    for url in result.output.sources:
        print(f"  {url}")


if __name__ == "__main__":
    asyncio.run(main())

# for testing in Jupyter, use: await main()