# WikiSearch
## PRD for v0

#### Vision Statement
WikiSearch is an Anthropic LLM-powered, agentic chat tool that answers generic user questions grounded in Wikipedia-based data. It is written in python, and operated via a simple anthropic-themed (orange, pixel art crab, etc) TUI. It is designed for simple question and answering, not long-horizon chatbot tasks. It only answers questions via a search of Wikipedia data and refuses any other use case.

#### Safety Statement
WikiSearch is primarily designed to be HELPFUL. Understanding that Wikipedia could have some harmful information, the user is always prompted that a query may contain harmful information and permission is requested before completing a harmful query.
All AI systems are subject to potential abuse. Input to the system is parsed before it is fully processed and rejected if misuse is observed.

#### Technology
WikiSearch is written in Python and operated via the command line. It is primarily targeted toward MacOS devices and is not designed to work on Windows. The TUI is written using textual. The environment is managed with UV.
API access is required to pull results from Wikipedia via the Wikimedia API. The python library, Wikipedia-API may be used instead of curling the Wikimedia API directly.
An architecture diagram is shown in wikisearch-diagram-v0.png.
Traceability and evaluation is handled via LangFuse.

##### Process Flow
1. A user asks an initial question (initial query step)
2. The question is sent as-written to an independent safety filter. The filter is prompted to detect misuse (prompt injection, platform abuse such as asking for coding support, etc) and harms. (harms filter step)
2.a. If classified as misuse, the system politely rejects. The user is reminded of the use case for WikiSearch and asked to try again, possibly rephrasing their question.
2.b. If classified as harms, the user is prompted to agree to view harmful content. If the user declines, a message is sent that is courteous and states that the query has been canceled. If the user agrees, the message is sent forward to the process query step. (user permission check step)
3. The process query step is the main LLM-powered step in the flow. It rephrases the user's question if necessary, and calls the search_wikipedia tool to retrieve grounded information.
4. The search_wikipedia step returns a pydantic base model with relevant wikipedia results and their associated sources. There may be more than one result. The tool leverages a small model to determine if results are relevant, and can trigger additional searches, up to three times before it is forced to return data to the primary workflow.
5. The process_response step takes in as context the user's original query and data from search_wikipedia to craft a grounded answer to the user's question. The response is sent as a pydantic base model that includes the response and grounded sources. The sources are linked in such a way that a user may navigate to wikipedia and view the grounding information independently.

##### Evals and Observability
WikiSearch uses Pydantic-AI for its primary functionality. Pydantic base models are used to enforce schema . LangFuse is integrated in such a way that full system traces are parsed each time the workflow is run. LangFuse spans are tagged in such a way that it is obvious to a layperson what is being tracked where. A binary scoring system is set for final output looking at two criteria:
1. Is the output helpful?
2. Is the output grounded with wiki data?
When a golden example is observed by human operators, the JSON output of that run will be stored in the eval/golden_samples directory. A set of 10 prompts, in JSON format, will be stored in the eval parent folder for apples-to-apples repeated testing. At least one of these prompts will include harms, and at least one will include misuse, in order to test filters.

##### Directory Structure
The WikiSearch directory is set up to have an app folder where primary functionality exists. A data folder holds samples of wikipedia data output and spec information like this PRD and any associated workflow diagrams. An eval folder holds golden samples once they exist, and a set of 10 prompts that can be rerun repeatedly.
