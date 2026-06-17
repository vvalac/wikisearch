from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Label, Markdown, Static

from crab_facts import random_fact
from models import WikiResponse
from pipeline import Checkpoint, CleanResult, HarmfulResult, MisuseResult, run_query
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

_EVAL_DIR = Path(__file__).parent.parent / "eval"


def _trim_history(messages: list[ModelMessage], keep: int = 10) -> list[ModelMessage]:
    """Slice history to `keep` messages, then advance to the first real user turn.

    A raw tail-slice can orphan a tool_result whose tool_use was cut off,
    causing Anthropic's API to 400. Starting from a UserPromptPart guarantees
    the history begins at a clean exchange boundary.
    """
    trimmed = messages[-keep:]
    for i, msg in enumerate(trimmed):
        if isinstance(msg, ModelRequest) and any(
            isinstance(p, UserPromptPart) for p in msg.parts
        ):
            return trimmed[i:]
    return trimmed

_HELP_MD = """\
**WikiSearch — commands**

`/help` — show this message
`/eval all` — run all eval prompts in the TUI
`/eval harms` — run only harmful & misuse evals
`/exit` — quit

Ask any Wikipedia question to get started.
"""

_ORANGE = "#F26522"
_ORANGE_BG = "#2a1200"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class CrabFact(Static):
    DEFAULT_CSS = f"""
    CrabFact {{
        background: $surface;
        border: round {_ORANGE};
        margin: 1 2 0 2;
        padding: 0 2;
        color: #666666;
    }}
    """

    def compose(self) -> ComposeResult:
        yield Label(f"🦀  {random_fact()}")


class UserBubble(Widget):
    """Right-leaning user message."""
    DEFAULT_CSS = f"""
    UserBubble {{
        margin: 1 1 0 20;
        padding: 0 2;
        background: {_ORANGE_BG};
        border: round {_ORANGE};
        height: auto;
    }}
    UserBubble Label {{
        width: 1fr;
    }}
    UserBubble .sender {{
        color: {_ORANGE};
        text-style: bold;
    }}
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Label("You", classes="sender")
        yield Label(self._text, markup=False)


class AssistantBubble(Widget):
    """Left-leaning assistant response."""
    DEFAULT_CSS = f"""
    AssistantBubble {{
        margin: 0 20 1 1;
        padding: 0 2;
        background: $panel;
        border: round $primary;
        height: auto;
    }}
    AssistantBubble .sender {{
        color: {_ORANGE};
        text-style: bold;
        margin-bottom: 1;
    }}
    AssistantBubble Markdown {{
        background: $panel;
        padding: 0;
        margin: 0;
    }}
    AssistantBubble .sources-header {{
        color: #aaaaaa;
        margin-top: 1;
        text-style: bold;
    }}
    AssistantBubble .source-link {{
        color: $primary;
    }}
    """

    def __init__(self, response: WikiResponse) -> None:
        super().__init__()
        self._response = response

    def compose(self) -> ComposeResult:
        yield Label("WikiSearch", classes="sender")
        yield Markdown(self._response.answer)
        if self._response.sources:
            yield Label("Sources", classes="sources-header")
            for url in self._response.sources:
                yield Label(f"  {url}", markup=False, classes="source-link")


class SystemMessage(Static):
    """Warning / system-level message."""
    DEFAULT_CSS = """
    SystemMessage {
        margin: 1 2;
        padding: 0 1;
        color: $warning;
        border-left: thick $warning;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(text, markup=False)


class HelpMessage(Widget):
    """Rendered help text for /help."""
    DEFAULT_CSS = f"""
    HelpMessage {{
        margin: 1 2;
        padding: 0 2;
        border-left: thick {_ORANGE};
        height: auto;
    }}
    HelpMessage Markdown {{
        background: transparent;
        padding: 0;
        margin: 0;
    }}
    """

    def compose(self) -> ComposeResult:
        yield Markdown(_HELP_MD)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class WikiSearchApp(App):
    TITLE = "WikiSearch"
    CSS = f"""
    Header {{
        background: {_ORANGE};
        color: #ffffff;
    }}
    #conversation {{
        height: 1fr;
        overflow-y: auto;
        padding-bottom: 1;
    }}
    #status {{
        margin: 0 20 1 1;
        padding: 0 2;
        color: #aaaaaa;
    }}
    Input {{
        margin: 0 1 1 1;
        border: tall $panel-lighten-2;
    }}
    Input:focus {{
        border: tall {_ORANGE};
    }}
    """
    BINDINGS: ClassVar = [("ctrl+c", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._history: list[ModelMessage] = []
        self._consecutive_misuse: int = 0
        self._input_locked: bool = False
        self._pending_harmful_query: str = ""
        self._awaiting_harm_confirm: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="conversation"):
            yield CrabFact()
        yield Input(placeholder="Ask a Wikipedia question… (type /help for commands)", id="query-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#query-input", Input).focus()

    def _add_widget(self, widget: Widget) -> None:
        container = self.query_one("#conversation", ScrollableContainer)
        container.mount(widget)
        container.scroll_end(animate=False)

    @on(Input.Submitted, "#query-input")
    def on_submit(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query or self._input_locked:
            return
        event.input.clear()
        if query.startswith("/"):
            self._dispatch_slash(query)
            return
        self._input_locked = True
        self._add_widget(UserBubble(query))
        self._run_pipeline(query)

    def _dispatch_slash(self, raw: str) -> None:
        parts = raw.strip().split()
        cmd = parts[0].lower()
        args = [p.lower() for p in parts[1:]]

        if cmd == "/exit":
            self.exit()
        elif cmd == "/help":
            self._add_widget(HelpMessage())
        elif cmd == "/eval":
            sub = args[0] if args else "all"
            if sub not in ("all", "harms"):
                self._add_widget(SystemMessage("Usage: /eval all  |  /eval harms"))
                return
            self._run_eval(sub)
        else:
            self._add_widget(SystemMessage(
                f"Unknown command '{cmd}'. Type /help for available commands."
            ))

    @work(exclusive=True)
    async def _run_pipeline(self, query: str) -> None:
        try:
            await self._pipeline(query)
        finally:
            self._input_locked = False
            try:
                self.query_one("#query-input", Input).focus()
            except NoMatches:
                pass

    async def _pipeline(self, query: str) -> None:
        skip_safety = False
        if self._awaiting_harm_confirm:
            self._awaiting_harm_confirm = False
            if query.strip().lower() == "yes":
                query = self._pending_harmful_query
                skip_safety = True

        container = self.query_one("#conversation", ScrollableContainer)
        status = Static("Checking safety…", id="status")
        await container.mount(status)
        container.scroll_end(animate=False)

        try:
            outcome: CleanResult | MisuseResult | HarmfulResult | None = None
            async for event in run_query(query, self._history, skip_safety=skip_safety):
                if isinstance(event, Checkpoint):
                    status.update(event.message)
                else:
                    outcome = event

            if isinstance(outcome, MisuseResult):
                self._consecutive_misuse += 1
                if self._consecutive_misuse >= 2:
                    self._add_widget(SystemMessage(
                        "WikiSearch only answers questions about topics found on Wikipedia. "
                        "Two misuse attempts detected — please restart the session."
                    ))
                    self._input_locked = True
                    return
                self._add_widget(SystemMessage(
                    "That looks like a non-Wikipedia request. "
                    "WikiSearch only answers factual questions grounded in Wikipedia. "
                    "Please rephrase your question."
                ))

            elif isinstance(outcome, HarmfulResult):
                self._pending_harmful_query = query
                self._awaiting_harm_confirm = True
                self._add_widget(SystemMessage(
                    "This question may involve sensitive content. "
                    "Reply 'yes' to proceed anyway, or rephrase your question."
                ))

            elif isinstance(outcome, CleanResult):
                self._consecutive_misuse = 0
                self._history = _trim_history(outcome.new_history)
                self._add_widget(AssistantBubble(outcome.response))

        finally:
            await status.remove()
            container.scroll_end(animate=False)

    @work(exclusive=True)
    async def _run_eval(self, subcommand: str) -> None:
        self._input_locked = True
        try:
            samples = [
                json.loads(p.read_text())
                for p in sorted(_EVAL_DIR.glob("*.json"))
            ]
            if subcommand == "harms":
                samples = [
                    s for s in samples
                    if s["expected_safety"] in ("harmful", "misuse")
                ]

            self._add_widget(SystemMessage(
                f"/eval {subcommand} — running {len(samples)} prompts…"
            ))

            results: list[dict] = []
            container = self.query_one("#conversation", ScrollableContainer)

            for sample in samples:
                prompt = sample["prompt"]
                expected = sample["expected_safety"]

                self._add_widget(UserBubble(f"[eval] {prompt}"))

                status = Static("Checking safety…", id="status")
                await container.mount(status)
                container.scroll_end(animate=False)

                outcome: CleanResult | MisuseResult | HarmfulResult | None = None
                try:
                    async for event in run_query(prompt, []):
                        if isinstance(event, Checkpoint):
                            status.update(event.message)
                        else:
                            outcome = event
                finally:
                    await status.remove()

                if isinstance(outcome, CleanResult):
                    actual = "clean"
                    self._add_widget(AssistantBubble(outcome.response))
                elif isinstance(outcome, MisuseResult):
                    actual = "misuse"
                    self._add_widget(SystemMessage(f"Blocked (misuse): {outcome.safety_reason}"))
                elif isinstance(outcome, HarmfulResult):
                    actual = "harmful"
                    self._add_widget(SystemMessage(f"Blocked (harmful): {outcome.safety_reason}"))
                else:
                    actual = "unknown"

                results.append({
                    "id": sample["id"],
                    "expected": expected,
                    "actual": actual,
                    "match": actual == expected,
                })

            correct = sum(1 for r in results if r["match"])
            lines = [f"Eval done — {correct}/{len(results)} correct"]
            for r in results:
                icon = "✓" if r["match"] else "✗"
                lines.append(f"  {icon}  {r['id']}  (expected {r['expected']}, got {r['actual']})")
            self._add_widget(SystemMessage("\n".join(lines)))

        finally:
            self._input_locked = False
            try:
                self.query_one("#query-input", Input).focus()
            except NoMatches:
                pass
