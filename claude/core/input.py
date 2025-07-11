import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Input, Static, Label, Markdown, ProgressBar
from textual import on
from rich.panel import Panel
from rich.text import Text

from claude.tools import tools
from claude.llm import Ollama
from claude.core.utils import ORANGE_COLORS, get_contextual_thinking_words


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
        overflow-y: auto;
        scrollbar-size: 0 0;
    }

    #input_area {
        height: auto;
        border: round grey;
        background: transparent;
    }

    #status_bar {
        height: 1;
        background: transparent;
        padding: 0 1;
    }

    #status_indicator {
        color: #E27A53;
        text-style: bold;
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
    
    Markdown {
        background: transparent;
        padding: 0;
        margin: 0;
    }
    
    .streaming {
        background: transparent;
        color: #E77D22;
    }
    
    .tool-executing {
        background: transparent;
    }
    
    .tool-completed {
        background: transparent;
    }
    """

    def __init__(self):
        super().__init__()
        self.llm = Ollama(host="http://192.168.170.76:11434")
        self.tool_widgets = {}  # Track tool execution widgets

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="chat_area"):
            welcome_panel = Panel(
                renderable=Text.from_markup(f"[{ORANGE_COLORS[17]}]✻ [/][bold]Welcome to [/][bold orange1]Plaude Pode[/]!\n\n"
                                 f"/help for help, /status for your current setup\n\ncwd: {Path(__file__).parent}"),
                border_style=ORANGE_COLORS[17],
                expand=False
            )
            yield Static(welcome_panel, classes="message")
        # Status bar above input
        with Horizontal(id="status_bar"):
            yield Static("", id="status_indicator")
        # Input area at bottom
        with Horizontal(id="input_area"):
            yield Label("> ")
            yield Input(placeholder="Type your message here...", compact=True, value="read app.py")
        # Footer
        with Horizontal(id="footer"):
            yield Static("⏵⏵ auto-accept edits on", id="footer-right")


    def on_ready(self) -> None:
        """Called when the app is ready - focus the input"""
        self.query_one(Input).focus()
        self.theme="gruvbox"

    @on(Input.Submitted)
    def handle_message(self, event: Input.Submitted) -> None:
        """Handle when user submits a message"""
        message = event.value.strip()
        if message:  # Only process non-empty messages
            # Add the message to chat area
            chat_area = self.query_one("#chat_area")
            chat_area.mount(Static(f"> {message}\n", classes="message"))
            # Clear the input
            event.input.clear()

            chat_area.scroll_end(animate=False)
            self.refresh()
            # Start AI response streaming
            self.call_later(self.start_ai_response, message)

    async def start_ai_response(self, query: str):
        """Start streaming AI response using Ollama with markdown rendering"""
        chat_area = self.query_one("#chat_area")
        status_indicator = self.query_one("#status_indicator")

        # Stream the response using Ollama
        response_text = ""
        thinking_mode = False
        flower_chars = ["✻", "✺", "✵", "✴", "❋", "❊", "❉", "❈", "❇", "❆", "❅", "❄"]
        flower_index = 0
        
        # Generate contextual thinking words based on user input
        thinking_words = get_contextual_thinking_words(query)
        thinking_word_index = 0
        
        import random
        import time
        
        # Track start time for elapsed seconds
        start_time = time.time()
        
        # Set initial status
        status_indicator.update("● Generating response...")
        
        try:
            async for chunk in self._stream_ollama_response(query):
                content = chunk['message']['content']
                tool_calls = chunk.message.tool_calls
                if content:
                    # Check for thinking mode start
                    if '<think>' in content:
                        thinking_mode = True
                        continue
                    
                    # Check for thinking mode end
                    if '</think>' in content:
                        thinking_mode = False
                        # Update status for normal content generation
                        elapsed_seconds = int(time.time() - start_time)
                        status_indicator.update(f"● Generating response... {elapsed_seconds}s")
                        continue
                    
                    # If in thinking mode, show flower animation in status bar
                    if thinking_mode:
                        flower_index = (flower_index + 1) % len(flower_chars)
                        # Change thinking word every 5 flower cycles to slow it down
                        if flower_index % 5 == 0:
                            thinking_word_index = (thinking_word_index + 1) % len(thinking_words)
                        current_thinking_word = thinking_words[thinking_word_index]
                        status_indicator.update(f"{flower_chars[flower_index]} {current_thinking_word}")
                        await asyncio.sleep(0.3)
                    else:
                        # Collect content but don't render markdown yet
                        response_text += content
                        elapsed_seconds = int(time.time() - start_time)
                        flower_index = (flower_index + 1) % len(flower_chars)
                        # Change thinking word every 5 flower cycles to slow it down
                        if flower_index % 5 == 0:
                            thinking_word_index = (thinking_word_index + 1) % len(thinking_words)
                        current_thinking_word = thinking_words[thinking_word_index]
                        status_indicator.update(f"{flower_chars[flower_index]} {current_thinking_word} [grey]({elapsed_seconds}s)[/grey]")
                        #await asyncio.sleep(0.001)
                        #await asyncio.sleep(0.4) # mimicing large language model
                elif tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.function.name.split('_')[0].title()
                        tool_args = list(tc.function.arguments.values())[0] if tc.function.arguments else ""
                        tool_id = f"{tc.function.name}_{id(tc)}"
                        
                        # Create tool widget with animation
                        await self.create_tool_widget(tool_name, tool_args, tool_id)
                        
                        # Simulate tool execution completion after some time
                        # In real implementation, this would be triggered by actual tool completion
                        self.set_timer(2, lambda: self.complete_tool_execution(tool_id,tool_args,'Read 297 lines'))

        except Exception as e:
            error_text = f"🤖 **Error**: {str(e)}"
            chat_area.mount(Static(error_text, classes="ai-response"))
            status_indicator.update("")
            return

        # Create final markdown widget with collected content
        if response_text.strip():
            markdown_widget = Markdown("● " + response_text.strip(), classes="ai-response")
            chat_area.mount(markdown_widget)
        
        # Ensure scroll to end after markdown rendering
        self.call_after_refresh(lambda: chat_area.scroll_end(animate=False))
        
        # Reset status to empty
        status_indicator.update("")
    
    async def create_tool_widget(self, tool_name: str, tool_args: str, tool_id: str):
        """Create a widget for tool execution"""
        from rich.text import Text
        chat_area = self.query_one("#chat_area")
        status_indicator = self.query_one("#status_indicator")
        
        # Create rich text with grey dot and bold tool name
        tool_text = Text()
        tool_text.append("● ", style="bright_black")
        tool_text.append(tool_name.title(), style="bold")
        tool_text.append(f"({tool_args})", style="default")
        
        tool_widget = Static(tool_text, classes="tool-executing")
        chat_area.mount(tool_widget)
        self.tool_widgets[tool_id] = tool_widget
        
        # Update status bar to show tool progress
        status_indicator.update(f"● Executing {tool_name.title()}...")
        
        # Scroll to end
        self.call_after_refresh(lambda: chat_area.scroll_end(animate=False))
        
    
    def complete_tool_execution(self, tool_id: str, tool_args,result: str = ""):
        """Mark tool execution as completed with green dot"""
        if tool_id in self.tool_widgets:
            from rich.text import Text
            widget = self.tool_widgets[tool_id]
            tool_info = tool_id.split('_')[0].title()
            status_indicator = self.query_one("#status_indicator")
            chat_area = self.query_one("#chat_area")
            
            # Create rich text with green dot and bold tool name, and result on same widget
            tool_text = Text()
            tool_text.append("● ", style="dim #5cf074")
            tool_text.append(tool_info, style="bold")
            tool_text.append(f"({tool_args})", style="default")
            
            # Add result text on next line if available
            if result and result.strip():
                tool_text.append(f"\n  ⎿ {result}\n", style="default")

            widget.update(tool_text)
            widget.remove_class("tool-executing")
            widget.add_class("tool-completed")
            
            # Scroll to end after tool completion
            self.call_after_refresh(lambda: chat_area.scroll_end(animate=False))
            
            # Clear status bar
            status_indicator.update("")
            
            # Remove from tracking
            del self.tool_widgets[tool_id]

    async def _stream_ollama_response(self, query: str):
        """Convert synchronous Ollama generator to async generator"""
        for chunk in self.llm(prompt=query+"/no_think", model="qwen3:4b", tools=tools):
            yield chunk
            await asyncio.sleep(0)  # Yield control to event loop


if __name__ == "__main__":
    app = ChatApp()
    app.run()
