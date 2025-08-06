
from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult, App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, OptionList, Label, Input

# Only keep used colors
COLORS = {
    "orange": "#E27A53",
    "streaming": "#E77D22",
    "blue": "#458588",
    "purple": "#915FF0",
    "yellow": "#fabd2f",
    "grey": "#a89984"
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

    CSS = f"""
        Screen, Static, Input {{ background: transparent; }}
        
        #chat_area {{ height: 1fr; padding: 1; overflow-y: auto; scrollbar-size: 0 0; }}
        #input_area {{ height: auto; border: round #7E7E80; }}
        #status_bar {{ height: 1; padding: 0 1; }}
        #status_indicator {{ color: {COLORS["orange"]}; text-style: bold; }}
        #footer {{ height: 10vh; dock: bottom; }}
        #footer-left {{ text-align: left; width: auto; }}
        #footer-right {{ text-align: right; width: 1fr; }}
        
        .mode-bypass {{ color: {COLORS["yellow"]}; }}
        .mode-plan {{ color: {COLORS["blue"]}; }}
        .mode-auto-edit {{ color: {COLORS["purple"]}; }}
        .message, .welcome {{ color: grey; text-style: italic; }}
        .streaming {{ color: {COLORS["streaming"]}; }}
        
        .selection-area {{ height: auto; padding: 1; margin: 0; }}
        .selection-area Static {{ color: {COLORS["grey"]}; text-style: italic; padding: 0 0 1 0; }}
        .selection-area OptionList {{ border: none; height: auto; scrollbar-size: 0 0; }}
        .selection-area OptionList:focus {{ border: none; }}
        
        #permission_area {{ border: round {COLORS["blue"]}; }}
        #command_palette {{ border: round {COLORS["purple"]}; }}
        #provider_selection {{ border: round {COLORS["orange"]}; }}
        #model_selection {{ border: round {COLORS["yellow"]}; }}
        
        .option-list--option {{ color: {COLORS["grey"]}; padding: 0 1; height: 1; }}
        .option-list--option-highlighted {{ color: {COLORS["yellow"]}; text-style: bold; }}
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
        yield from self._create_chat_area()
        yield from self._create_status_bar()
        yield from self._create_selection_areas()
        yield from self._create_input_area()
        yield from self._create_footer()
    
    def _create_chat_area(self):
        with ScrollableContainer(id="chat_area"):
            welcome = Panel(
                Text.from_markup(f"[{COLORS['orange']}]✻ [/][white]Welcome to [/][bold white]Claude Code[/]!\n\n"
                               f"[italic]/help for help, /status for your current setup\n\ncwd: {self.cwd}[/italic]"),
                border_style=COLORS['orange'], expand=False, padding=(0, 2, 0, 1)
            )
            yield Static(welcome, classes="welcome")
    
    def _create_status_bar(self):
        with Horizontal(id="status_bar"):
            yield Static("", id="status_indicator")
    
    def _create_selection_areas(self):
        areas = [
            ("permission_area", "permission_message", "permission_options", 
             ["", ["1. Yes", "2. Yes, and don't ask again this session", "3. No, and tell Claude what to do differently"]]),
            ("command_palette", "command_palette_message", "command_options",
             ["Command Palette - Select an option:", ["/host <url> - Set LLM host URL", "/provider - Switch LLM provider", "/think - Enable thinking mode", "/no-think - Disable thinking mode", "/status - Show current configuration", "/clear - Clear conversation history"]]),
            ("provider_selection", "provider_selection_message", "provider_options",
             ["Select Provider:", ["google - Google Generative AI (Gemini models)", "ollama - Local Ollama server", "openrouter - OpenRouter API (multiple models)", "vllm - Local vLLM server"]]),
            ("model_selection", "model_selection_message", "model_options",
             ["Select Model:", []])
        ]
        
        for area_id, msg_id, opt_id, (msg, opts) in areas:
            with Vertical(id=area_id, classes="selection-area"):
                yield Static(msg, id=msg_id)
                yield OptionList(*opts, id=opt_id)
    
    def _create_input_area(self):
        with Horizontal(id="input_area"):
            yield Label(" > ")
            yield Input(placeholder='Try "write a test for input.py"', compact=True,
                       value="create a todo and then task: websearch for latest AI news and then fetch the first URL to summarize")
    
    def _create_footer(self):
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
        self.theme = "gruvbox"
        for area in ["permission_area", "command_palette", "provider_selection", "model_selection"]:
            self.query_one(f"#{area}").display = False
        self.query_one(Input).focus()
        fr = self.query_one("#footer-right")
        fr.update("Try claude doctor or npm i -g @anthropic-ai/claude-code")
        fr.add_class("mode-bypass")