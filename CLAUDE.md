# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WikiSearch is an Anthropic LLM-powered agentic chat tool that answers user questions grounded in Wikipedia data. It runs as a Textual TUI, uses Pydantic-AI for agent orchestration, and refuses all non-Wikipedia use cases.

## Environment & Commands

The project uses [UV](https://docs.astral.sh/uv/) for environment management.

```bash
uv run python -m app          # run the app
uv run pytest                 # run all tests
uv run pytest tests/test_foo.py::test_bar  # run a single test
uv run textual run app/main.py  # launch TUI in dev mode with Textual devtools
```

## Architecture

The app lives in `app/`. The process flow has five sequential steps:

1. **Initial query** — user input captured in the TUI
2. **Safety filter** — independent LLM call that classifies input as misuse, harmful, or clean
   - Misuse → politely reject and prompt the user to rephrase
   - Harmful → ask user permission before continuing; cancel if declined
3. **Process query** — main Pydantic-AI agent step; rephrases the query if needed and calls the `search_wikipedia` tool
4. **`search_wikipedia` tool** — wraps the `Wikipedia-API` library; uses a small model to judge result relevance; retries up to 3 times; returns a Pydantic model with results and sources
5. **Process response** — synthesizes a grounded answer from the original query and Wikipedia results; returns a Pydantic model with the response text and linked Wikipedia sources

### Key design constraints

- All LLM calls use the Anthropic API via Pydantic-AI.
- Every step is traced in LangFuse; spans should be labeled so a layperson can read a trace.
- Final responses carry a binary LangFuse score on two criteria: (1) helpful, (2) grounded in Wikipedia data.
- Pydantic base models enforce schema at every step boundary.

## Observability & Evals

- LangFuse handles tracing and scoring (env vars: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).
- `eval/` holds 10 reusable test prompts (JSON) including at least one harms case and one misuse case.
- `eval/golden_samples/` stores JSON outputs from human-validated "golden" runs.

## Directory Layout

```
app/           # primary application code
data/
  spec/        # PRD and architecture diagram
eval/          # test prompts (JSON) and golden_samples/
```
