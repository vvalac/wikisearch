# WikiSearch
## PRD for v0

#### Vision Statement
WikiSearch is an Anthropic LLM-powered, agentic chat tool that answers generic user questions grounded in Wikipedia-based data. It is written in Python and operated via a simple Anthropic-themed (orange, pixel art crab) TUI. It is designed for simple question-and-answering with natural follow-up support — not long-horizon, free-flowing chat. It only answers questions via a search of Wikipedia data and refuses all other use cases.

#### Safety Statement
WikiSearch is primarily designed to be HELPFUL. Understanding that Wikipedia could contain some harmful information, the user is prompted that a query may contain harmful content and permission is requested before completing such a query. Input is parsed before full processing and rejected if misuse is detected.

#### Technology
WikiSearch is written in Python and operated via the command line. It is primarily targeted toward macOS and is not designed for Windows. The TUI is built with Textual. The environment is managed with UV.

The Wikipedia-API Python library is used to query Wikipedia (via the Wikimedia API). Pydantic-AI is used for all LLM-powered steps. LangFuse handles traceability and evaluation.

An architecture diagram is provided in `wikisearch-diagram-v0.png`.

---

##### Process Flow

**1. Initial Query**
The user types a question into the TUI. The TUI displays a status indicator while each downstream step is running.

**2. filter_harm (Safety Filter)**
The query is sent as-written to an independent LLM call that classifies it into one of three categories:

- **Misuse** (prompt injection, off-topic requests such as coding help, platform abuse): The system politely rejects the query and informs the user of WikiSearch's intended use case, asking them to rephrase. This is allowed once per session turn. If the very next message is also classified as misuse, the session is fully terminated and the user is asked to restart.
- **Harmful** (content that could be dangerous or sensitive): The user is asked for explicit permission to proceed. If they decline, the query is canceled with a courteous message. If they agree, the query advances to step 3.
- **Clean**: The query advances immediately to step 3.

**3. process_query (Main Workflow Agent)**
The primary Pydantic-AI agent step. It rephrases the user's question if necessary for better search results, then calls `search_wikipedia` as a tool. This agent operates with conversation history scoped to the last 5–10 messages, enabling natural follow-up questions without requiring context condensation.

**4. search_wikipedia (Sub-Agent Tool)**
`search_wikipedia` is exposed as a tool to `process_query` but is implemented internally as a Pydantic-AI sub-agent running a smaller model (Sonnet by default; Haiku is the target once evals confirm quality parity). It operates with a fixed maximum number of turns (3) to prevent runaway loops.

Each turn, the sub-agent:
- Issues a Wikipedia search via the Wikipedia-API library
- Evaluates relevance of the returned results
- Decides whether to search again (rephrasing or refining the query) or return

All search iterations are accumulated in an array, each entry containing the search query, raw results, and source URLs. This full array is returned to `process_query` as a Pydantic base model.

**5. process_response**
Takes as context the user's original query and the accumulated Wikipedia results from `search_wikipedia`. Produces a grounded answer as a Pydantic base model containing:
- The response text
- The Wikipedia sources consulted, formatted as navigable links so the user can independently verify the information

---

##### TUI Behavior

- **On load**: Display a random fun fact about crabs.
- **Conversation**: Single scrollable conversation pane. No sidebar. No persistent conversation history panel (out of scope for v0).
- **Status indicators**: Visible, real-time indicators for each step in the pipeline (e.g., "Checking safety…", "Searching Wikipedia…", "Thinking…"). Wikipedia searches must be clearly surfaced to the user as they happen.
- **Sources**: Every response includes the Wikipedia sources that were consulted, displayed as clickable links.
- **Conversation length**: Scoped to 5–10 messages. This is a deliberate POC constraint — no context condensation is required.

---

##### Evals and Observability

WikiSearch uses LangFuse for full system tracing. Every pipeline run generates a trace with clearly labeled spans (readable by a layperson). A binary scoring rubric is applied to final output on two criteria:
1. Is the output helpful?
2. Is the output grounded in Wikipedia data?

When a golden example is validated by a human operator, its JSON output is stored in `eval/golden_samples/`. A set of 10 test prompts (JSON) is stored in `eval/` for repeatable apples-to-apples testing. At least one prompt tests harmful content handling; at least one tests misuse detection.

The initial target model for `search_wikipedia`'s relevance-checking sub-agent is Sonnet, chosen to ensure quality out of the box. Haiku is the intended long-term target; evals will determine when that downgrade is safe.

---

##### Directory Structure

```
app/       # primary application code
data/
  spec/    # PRD and architecture diagram
eval/
  golden_samples/   # human-validated golden run outputs
```
