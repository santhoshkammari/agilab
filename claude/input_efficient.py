
from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult, App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, OptionList, Label, Input

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
    BINDINGS = [
        Binding("shift+tab", "cycle_mode", "Cycle Mode", priority=True),
        Binding("ctrl+c", "clear_input", "Clear Input", priority=True),
        Binding("escape", "interrupt_conversation", "Interrupt Conversation", priority=True),
        Binding("ctrl+d", "clear_input", "Clear Input", priority=True),
        Binding("ctrl+l", "clear_screen", "Clear Screen", priority=True),
        Binding("shift+up", "previous_command", "Previous Command", priority=True),
        Binding("shift+down", "next_command", "Next Command", priority=True),
    ]

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
            border: round #7E7E80;
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
            height: 10vh;
            dock: bottom;
            background: transparent;
        }

        #footer-left {
            text-align: left;
            width: auto;
        }

        #footer-right {
            text-align: right;
            width: 1fr;
        }

        .mode-bypass {
            color: #fabd2f;
        }

        .mode-plan {
            color: #458588;
        }

        .mode-auto-edit {
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

        .welcome {
            background: transparent;
            color: grey;
        }

        .ai-response {
            background: transparent;
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

        #permission_area {
            height: auto;
            background: transparent;
            border: round #458588;
            padding: 1;
            margin: 0;
        }

        #permission_message {
            color: #a89984;
            text-style: italic;
            padding: 0 0 1 0;
            background: transparent;
        }

        #permission_options {
            background: transparent;
            border: none;
            height: auto;
            scrollbar-size: 0 0;
        }

        #permission_options:focus {
            border: none;
        }

        .option-list--option {
            background: transparent;
            color: #a89984;
            padding: 0 1;
            height: 1;
        }

        .option-list--option-highlighted {
            background: transparent;
            color: #fabd2f;
            text-style: bold;
        }

        #command_palette {
            height: auto;
            background: transparent;
            border: round #915FF0;
            padding: 1;
            margin: 0;
        }

        #command_palette_message {
            color: #a89984;
            text-style: italic;
            padding: 0 0 1 0;
            background: transparent;
        }

        #command_options {
            background: transparent;
            border: none;
            height: auto;
            scrollbar-size: 0 0;
        }

        #command_options:focus {
            border: none;
        }

        #provider_selection {
            height: auto;
            background: transparent;
            border: round #E27A53;
            padding: 1;
            margin: 0;
        }

        #provider_selection_message {
            color: #a89984;
            text-style: italic;
            padding: 0 0 1 0;
            background: transparent;
        }

        #provider_options {
            background: transparent;
            border: none;
            height: auto;
            scrollbar-size: 0 0;
        }

        #provider_options:focus {
            border: none;
        }

        #model_selection {
            height: auto;
            background: transparent;
            border: round #fabd2f;
            padding: 1;
            margin: 0;
        }

        #model_selection_message {
            color: #a89984;
            text-style: italic;
            padding: 0 0 1 0;
            background: transparent;
        }

        #model_options {
            background: transparent;
            border: none;
            height: auto;
            scrollbar-size: 0 0;
        }

        #model_options:focus {
            border: none;
        }

        """

    def __init__(self,cwd):
        super().__init__()
        self.mode_idx = 0
        self.permission_mode = 'default'
        self.modes = ['default', 'auto-accept-edits', 'bypass-permissions', 'plan-mode']
        self.permission_maps = {"default":"[grey]  ? for shortcuts[/grey]",
                                'auto-accept-edits':"   ⏵⏵ auto-accept edits on   ",
                                'bypass-permissions':"   Bypassing Permissions   ",
                                'plan-mode':"   ⏸ plan mode on   "}
        self.cwd = cwd

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="chat_area"):
            welcome_panel = Panel(
                renderable=Text.from_markup(
                    f"[{ORANGE_COLORS[17]}]✻ [/][white]Welcome to [/][bold white]Claude Code[/]!\n\n"
                    f"[italic]/help for help, /status for your current setup\n\ncwd: {self.cwd}[/italic]"),
                border_style=ORANGE_COLORS[17],
                expand=False,
                padding=(0, 2, 0, 1)
            )
            yield Static(welcome_panel, classes="welcome")

        # Status bar above input
        with Horizontal(id="status_bar"):
            yield Static("", id="status_indicator")

        # Permission area (initially hidden)
        with Vertical(id="permission_area"):
            yield Static("", id="permission_message")
            yield OptionList(
                "1. Yes",
                "2. Yes, and don't ask again this session",
                "3. No, and tell Claude what to do differently",
                id="permission_options"
            )

        # Command palette (initially hidden)
        with Vertical(id="command_palette"):
            yield Static("Command Palette - Select an option:", id="command_palette_message")
            yield OptionList(
                "/host <url> - Set LLM host URL",
                "/provider - Switch LLM provider",
                "/think - Enable thinking mode",
                "/no-think - Disable thinking mode",
                "/status - Show current configuration",
                "/clear - Clear conversation history",
                id="command_options"
            )

        # Provider selection (initially hidden)
        with Vertical(id="provider_selection"):
            yield Static("Select Provider:", id="provider_selection_message")
            yield OptionList(
                "google - Google Generative AI (Gemini models)",
                "ollama - Local Ollama server",
                "openrouter - OpenRouter API (multiple models)",
                "vllm - Local vLLM server",
                id="provider_options"
            )

        # Model selection (initially hidden)
        with Vertical(id="model_selection"):
            yield Static("Select Model:", id="model_selection_message")
            yield OptionList(
                id="model_options"
            )

        # Input area at bottom
        with Horizontal(id="input_area"):
            yield Label(" > ")
            yield Input(placeholder='Try "write a test for input.py"', compact=True,
                        value="create a todo and then task: websearch for latest AI news and then fetch the first URL to summarize")

        # Footer
        with Horizontal(id="footer"):
            yield Static(self.permission_maps["default"], id="footer-left")
            yield Static("", id="footer-right")



    def action_cycle_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.modes)
        self.permission_mode = self.modes[self.mode_idx]

        footer_left = self.query_one("#footer-left")
        footer_left.update(self.permission_maps[self.permission_mode])

        footer_left.remove_class("mode-bypass", "mode-plan", "mode-auto-edit")
        if self.permission_mode == 'bypass-permissions':
            footer_left.add_class("mode-bypass")
        elif self.permission_mode == 'plan-mode':
            footer_left.add_class("mode-plan")
        elif self.permission_mode == 'auto-accept-edits':
            footer_left.add_class("mode-auto-edit")


    def on_ready(self) -> None:
        """Called when the app is ready - focus the input"""
        self.theme = "gruvbox"
        for _ in ["permission_area","command_palette","provider_selection","model_selection"]:
            self.query_one(f"#{_}").display=False
        self.query_one(Input).focus()
        self.query_one("#footer-right").update("Try claude doctor or npm i -g @anthropic-ai/claude-code")