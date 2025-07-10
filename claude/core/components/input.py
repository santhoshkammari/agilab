import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Input, Static, Label
from textual import on
from rich.panel import Panel
from rich.text import Text

from claude.tools import tools
from claude.llm import Ollama

# Orange color mapping from chart
ORANGE_COLORS = {
    0: "#8A3324",  # Burnt Amber
    1: "#B06500",  # Ginger
    2: "#CD7F32",  # Bronze
    3: "#D78C3D",  # Rustic Orange
    4: "#FF6E00",  # Hot Orange
    5: "#FF9913",  # Goldfish
    6: "#FFAD00",  # Neon Orange
    7: "#FFBD31",  # Bumblebee Orange
    8: "#D16002",  # Marmalade
    9: "#E77D22",  # Pepper Orange
    10: "#E89149",  # Jasper Orange
    11: "#EABD8C",  # Dark Topaz
    12: "#ED9121",  # Carrot
    13: "#FDAE44",  # Safflower Orange
    14: "#FAB972",  # Calm Orange
    15: "#FED8B1",  # Light Orange
    16: "#CC5500",  # Burnt Orange
    17: "#E27A53",  # Dusty Orange
    18: "#F4AB6A",  # Aesthetic Orange
    19: "#FEE8D6"  # Orange Paper
}


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
        overflow-y: hidden;
    }

    #input_area {
        height: auto;
        border: round grey;
        background: transparent;
    }

    #footer {
        height: 1;
        dock: bottom;
        background: transparent;
    }

    #footer-left {
        text-align: left;
    }

    #footer-right {
        text-align: right;
        color: #915FF0;
    }

    Input {
        background: transparent;
    }

    Static {
        background: transparent;
    }

    .message {
        background: transparent;
        color: grey;
        text-style: italic;
    }
    
    .ai-response {
        background: transparent;
    }
    
    .streaming {
        background: transparent;
        color: #E77D22;
    }
    """

    def __init__(self):
        super().__init__()
        self.llm = Ollama(host="http://192.168.170.76:11434")
        self.current_streaming_widget = None

    def compose(self) -> ComposeResult:
        # Main chat display area
        with ScrollableContainer(id="chat_area"):
            # Create welcome panel with Rich styling
            welcome_text = Text()
            welcome_text.append("✻ ", style=ORANGE_COLORS[17])
            welcome_text.append("Welcome to ")
            welcome_text.append("Plaude Pode", style="bold orange1")
            welcome_text.append("!\n\n/help for help, /status for your current setup\n\n")
            welcome_text.append(f"cwd: {Path(__file__).parent}")

            # Use border style from orange mapping (18 = Aesthetic Orange)
            welcome_panel = Panel(
                welcome_text,
                border_style=ORANGE_COLORS[17],
                expand=False
            )
            yield Static(welcome_panel, classes="message")
        # Input area at bottom
        with Horizontal(id="input_area"):
            yield Label("> ")
            yield Input(placeholder="Type your message here...", compact=True)
        # Footer
        with Horizontal(id="footer"):
            # yield Static("auto accept-on", id="footer-left")
            # yield Static("Try claude doctor or npm i -g @anthropic-ai/claude-code", id="footer-right")
            yield Static("⏵⏵ auto-accept edits on", id="footer-right")


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
            chat_area.mount(Static(f"\n> {message}", classes="message"))
            # Clear the input
            event.input.clear()

            chat_area.scroll_end(animate=False)
            # Start AI response streaming
            self.call_later(self.start_ai_response, message)

    async def start_ai_response(self, query: str):
        """Start streaming AI response using Ollama"""
        chat_area = self.query_one("#chat_area")

        # Create a new widget for the AI response
        self.current_streaming_widget = Static("", classes="ai-response")
        chat_area.mount(self.current_streaming_widget)

        # Stream the response using Ollama
        response_text = "\n● "
        try:
            for chunk in self.llm(prompt=query, model="qwen2.5:7b-instruct", tools=tools):
                content = chunk['message']['content']
                if content:
                    response_text += content
                    self.current_streaming_widget.update(response_text)
                    self.refresh()
                    await asyncio.sleep(0.001)
        except Exception as e:
            error_text = f"\n🤖 Error: {str(e)}"
            self.current_streaming_widget.update(error_text)

        # Mark streaming as complete
        self.current_streaming_widget.remove_class("streaming")
        self.current_streaming_widget = None


if __name__ == "__main__":
    app = ChatApp()
    app.run()
