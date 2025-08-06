
from rich.panel import Panel
from rich.text import Text
from textual import on
from textual.app import ComposeResult, App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, OptionList, Label, Input

class ChatApp(App):
    BINDINGS = [
        Binding("shift+tab", "cycle_mode", "Cycle Mode", priority=True),
        Binding("ctrl+c", "clear_input", "Clear Input", priority=True),
        Binding("escape", "interrupt_conversation", "Interrupt Conversation", priority=True),
        Binding("ctrl+l", "clear_screen", "Clear Screen", priority=True),
        Binding("shift+up", "previous_command", "Previous Command", priority=True),
        Binding("shift+down", "next_command", "Next Command", priority=True),
    ]

    CSS = """
        Screen, Static, Input { background: transparent; }
        #chat_area { height: 1fr; padding: 1; overflow-y: auto; scrollbar-size: 0 0; }
        #input_area { height: auto; border: round #7E7E80; }
        #status_bar { height: 1; padding: 0 1; }
        #status_indicator { color: #E27A53; text-style: bold; }
        #footer { height: 10vh; dock: bottom; }
        #footer-left { text-align: left; width: auto; }
        #footer-right { text-align: right; width: 1fr; }
        .mode-bypass { color: #fabd2f; }
        .mode-plan { color: #458588; }
        .mode-auto-edit { color: #915FF0; }
        .message, .welcome { color: grey; text-style: italic; }
        .streaming { color: #E77D22; }
        .selection-area { height: auto; padding: 1; margin: 0; }
        .selection-area Static { color: #a89984; text-style: italic; padding: 0 0 1 0; }
        .selection-area OptionList { border: none; height: auto; scrollbar-size: 0 0; }
        .selection-area OptionList:focus { border: none; }
        #permission_area { border: round #458588; }
        #command_palette { border: round #915FF0; }
        #provider_selection { border: round #E27A53; }
        #model_selection { border: round #fabd2f; }
        .option-list--option { color: #a89984; padding: 0 1; height: 1; }
        .option-list--option-highlighted { color: #fabd2f; text-style: bold; }
        """

    def __init__(self,cwd):
        super().__init__()
        self.command_history, self.history_index, self.mode_idx = [], -1, 0
        self.modes = ['default', 'auto-accept-edits', 'bypass-permissions', 'plan-mode']
        self.providers = ["gemini", "ollama", "openrouter", "vllm"]
        self.provider_name = "gemini"
        self.permission_maps = {
            "default": "[grey]  ? for shortcuts[/grey]",
            'auto-accept-edits': "   ⏵⏵ auto-accept edits on   ",
            'bypass-permissions': "   Bypassing Permissions   ",
            'plan-mode': "   ⏸ plan mode on   "
        }
        self.permission_mode, self.cwd = 'default', cwd

        self.llm = None
        self.model_loaded = False

    def compose(self) -> ComposeResult:
        for method in [self._create_chat_area, self._create_status_bar, self._create_selection_areas, 
                      self._create_input_area, self._create_footer]:
            yield from method()
    
    def _create_chat_area(self):
        with ScrollableContainer(id="chat_area"):
            yield Static(Panel(
                Text.from_markup(f"[#E27A53]✻ [/][white]Welcome to [/][bold white]Claude Code[/]!\n\n"
                               f"[italic]/help for help, /status for your current setup\n\ncwd: {self.cwd}[/italic]"),
                border_style="#E27A53", expand=False, padding=(0, 2, 0, 1)
            ), classes="welcome")
    
    def _create_status_bar(self):
        with Horizontal(id="status_bar"):
            yield Static("", id="status_indicator")
    
    def _create_selection_areas(self):
        areas = [
            ("permission_area", "", ["1. Yes", "2. Yes, and don't ask again this session", "3. No, and tell Claude what to do differently"]),
            ("command_palette", "Command Palette - Select an option:", ["/host <url> - Set LLM host URL", "/provider - Switch LLM provider", "/think - Enable thinking mode", "/no-think - Disable thinking mode", "/status - Show current configuration", "/clear - Clear conversation history"]),
            ("provider_selection", "Select Provider:", self.providers),
            ("model_selection", "Select Model:", [])
        ]
        
        for area_id, msg, opts in areas:
            with Vertical(id=area_id, classes="selection-area"):
                yield Static(msg, id=f"{area_id}_message")
                yield OptionList(*opts, id=f"{area_id.split('_')[0]}_options")
    
    def _create_input_area(self):
        with Horizontal(id="input_area"):
            yield Label(" > ")
            yield Input(placeholder='Try "write a test for input.py"', compact=True,
                       # value="create a todo and then task: websearch for latest AI news and then fetch the first URL to summarize"
                        value = "/provider"
                        )
    
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
        mode_classes = {'bypass-permissions': "mode-bypass", 'plan-mode': "mode-plan", 'auto-accept-edits': "mode-auto-edit"}
        if self.permission_mode in mode_classes:
            footer_left.add_class(mode_classes[self.permission_mode])

    def action_clear_input(self):
        self.query_one(Input).clear()

    def _navigate_history(self, direction: int):
        input_widget = self.query_one(Input)
        if not input_widget.has_focus or not self.command_history:
            return
        
        if self.history_index == -1:
            self.current_input_backup = input_widget.value
            self.history_index = len(self.command_history) - 1
        else:
            self.history_index = max(0, min(len(self.command_history) - 1, self.history_index + direction))
        
        if direction > 0 and self.history_index == len(self.command_history) - 1:
            input_widget.value = getattr(self, 'current_input_backup', '')
            self.history_index = -1
        else:
            input_widget.value = self.command_history[self.history_index]
        input_widget.cursor_position = len(input_widget.value)

    def action_previous_command(self) -> None:
        self._navigate_history(-1)

    def action_next_command(self) -> None:
        self._navigate_history(1)

    def _check_slash_command(self,value):
        if value == "/provider":
            provider_msg = self.query_one("#provider_selection_message")
            provider_msg.update(f"Select Provider (current: {self.provider_name}):")

            self.query_one("#provider_selection").display = True
            provider_options = self.query_one("#provider_options")
            provider_options.focus()

            try:
                current_index = self.providers.index(self.provider_name)
                provider_options.highlighted = current_index
            except ValueError:
                provider_options.highlighted = 0
            return True

        return False

    def execute_input(self,value):
        status = self._check_slash_command(value)
        if status:
            return




    @on(OptionList.OptionSelected, "#provider_options")
    def handle_provider_choice(self, event: OptionList.OptionSelected) -> None:
        """Handle provider choice from OptionList"""
        self.provider_name = self.providers[event.option_index]
        self.model_loaded = False  # Reset flag when provider changes
        self.hide_provider_selection()

    def hide_provider_selection(self):
        """Hide provider selection widget and show input"""
        self.query_one("#provider_selection").display = False
        self.query_one(Input).focus()
        self.load_model()

    def on_ready(self) -> None:
        self.theme = "gruvbox"
        areas = ["permission_area", "command_palette", "provider_selection", "model_selection"]
        for area in areas:
            self.query_one(f"#{area}").display = False
        self.query_one(Input).focus()
        fr = self.query_one("#footer-right")
        fr.update("Try claude doctor or npm i -g @anthropic-ai/claude-code")
        fr.add_class("mode-bypass")

    @on(Input.Submitted)
    def handle_message(self,event: Input.Submitted):
        if not self.model_loaded:
            self.load_model()
        self.command_history.append(event.value.strip())
        self.execute_input(event.value.strip())
        event.input.clear()

    def load_model(self):
        if self.model_loaded:
            return
            
        if self.provider_name == 'gemini':
            from flowgen.llm.gemini import GeminiAsync
            self.llm = GeminiAsync()
        elif self.provider_name == 'ollama':
            from flowgen.llm.llm import OllamaAsync
            self.llm = OllamaAsync()
        elif self.provider_name == 'openrouter':
            from flowgen.llm import OpenRouter
            self.llm = OpenRouter()
        elif self.provider_name == 'vllm':
            from flowgen.llm.llm import vLLMAsync
            self.llm = vLLMAsync()
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")
        
        self.model_loaded = True