from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Input, Static, Label
from textual import on


class ChatApp(App):
    """Simple chat interface with message display and input"""
    CSS = """
    Screen {
        background: transparent;
    }

    #chat_area {
        height: 1fr;
        padding: 1;
        background: transparent;
    }

    #input_area {
        height: auto;
        dock: bottom;
        border: round grey;
        background: transparent;
    }

    Input {
        background: transparent;
    }

    Static {
        background: transparent;
    }

    .message {
        background: transparent;
    }
    """

    def compose(self) -> ComposeResult:
        # Main chat display area
        with ScrollableContainer(id="chat_area"):
            yield Static("Welcome to the chat! Type something below and press Enter.\n", classes="message")
        # Input area at bottom
        with Horizontal(id="input_area"):
            yield Label("> ")
            yield Input(placeholder="Type your message here...", compact=True)

    def on_ready(self) -> None:
        """Called when the app is ready - focus the input"""
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def handle_message(self, event: Input.Submitted) -> None:
        """Handle when user submits a message"""
        message = event.value.strip()
        if message:  # Only process non-empty messages
            # Add the message to chat area
            chat_area = self.query_one("#chat_area")
            chat_area.mount(Static(f"● {message}\n", classes="message"))
            # Clear the input
            event.input.clear()
            # Scroll to bottom to show latest message
            chat_area.scroll_end(animate=False)


if __name__ == "__main__":
    app = ChatApp()
    app.run()