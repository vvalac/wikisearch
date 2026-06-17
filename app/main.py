from dotenv import load_dotenv
from langfuse import get_client
from pydantic_ai.agent import Agent

load_dotenv()  # must run before any other import so env vars are set for LangFuse + prompts

get_client()           # initialise OTEL → LangFuse exporter
Agent.instrument_all() # PydanticAI emits OTEL spans for every agent call

from tui import WikiSearchApp  # noqa: E402 — intentionally after LangFuse init


def run() -> None:
    WikiSearchApp().run()


if __name__ == "__main__":
    run()
