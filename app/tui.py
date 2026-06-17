from __future__ import annotations

from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Label, LoadingIndicator, Markdown, Static

from crab_facts import random_fact
from models import WikiResponse
from pipeline import filter_harm, process_query
from pydantic_ai.messages import ModelMessage

_ORANGE = "#F26522"
_ORANGE_DIM = "#7a3311"
_ORANGE_BG = "#2a1200"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class CrabFact(Static):
    DEFAULT_CSS = f"""
    CrabFact {{
        background: {_ORANGE_BG};
        border: round {_ORANGE};
        margin: 1 2 0 2;
        padding: 0 2;
        color: #aaaaaa;
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


class ThinkingBubble(Widget):
    """Inline spinner shown while the pipeline is running. Replaced by AssistantBubble."""
    DEFAULT_CSS = f"""
    ThinkingBubble {{
        margin: 0 20 1 1;
        padding: 0 1;
        background: $panel;
        border: round $panel-lighten-2;
        height: 3;
        layout: horizontal;
    }}
    ThinkingBubble LoadingIndicator {{
        width: 4;
        height: 3;
        color: {_ORANGE};
    }}
    ThinkingBubble #stage {{
        height: 3;
        content-align: left middle;
        color: #aaaaaa;
        padding-left: 1;
    }}
    """

    status: reactive[str] = reactive("Checking safety…")

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()
        yield Label(self.status, id="stage")

    def watch_status(self, value: str) -> None:
        try:
            self.query_one("#stage", Label).update(value)
        except NoMatches:
            pass


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
        yield Input(placeholder="Ask a Wikipedia question…", id="query-input")
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
        self._input_locked = True
        self._add_widget(UserBubble(query))
        self._run_pipeline(query)

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
        # Handle harm confirmation from a previous turn
        if self._awaiting_harm_confirm:
            if query.strip().lower() == "yes":
                query = self._pending_harmful_query
            self._awaiting_harm_confirm = False

        # Inline thinking indicator appears immediately
        thinking = ThinkingBubble()
        self._add_widget(thinking)
        container = self.query_one("#conversation", ScrollableContainer)

        try:
            # Step 2 — safety filter
            thinking.status = "Checking safety…"
            safety = await filter_harm(query)

            if safety.category == "misuse":
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
                return

            if safety.category == "harmful":
                self._pending_harmful_query = query
                self._awaiting_harm_confirm = True
                self._add_widget(SystemMessage(
                    "This question may involve sensitive content. "
                    "Reply 'yes' to proceed anyway, or rephrase your question."
                ))
                return

            self._consecutive_misuse = 0

            # Steps 3–5 — search + respond
            thinking.status = "Thinking…"

            def on_status(text: str) -> None:
                thinking.status = text

            response, new_history, _ = await process_query(
                query, self._history, on_status=on_status
            )
            self._history = new_history[-10:]
            self._add_widget(AssistantBubble(response))

        finally:
            await thinking.remove()
            container.scroll_end(animate=False)
