import json

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual import on
from textual.app import ComposeResult, App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, OptionList, Label, Input
import asyncio
import time
import datetime

from flowgen.tools.web import tool_functions as wt

def get_random_status_message():
    """Simple status messages for thinking animation"""
    STATUS_MESSAGES = [
        'Accomplishing',
        'Actioning',
        'Actualizing',
        'Baking',
        'Brewing',
        'Calculating',
        'Cerebrating',
        'Churning',
        'Clauding',
        'Coalescing',
        'Cogitating',
        'Computing',
        'Combobulating',
        'Conjuring',
        'Considering',
        'Cooking',
        'Crafting',
        'Creating',
        'Crunching',
        'Deliberating',
        'Determining',
        'Doing',
        'Effecting',
        'Finagling',
        'Forging',
        'Forming',
        'Generating',
        'Hatching',
        'Herding',
        'Honking',
        'Hustling',
        'Ideating',
        'Inferring',
        'Manifesting',
        'Marinating',
        'Moseying',
        'Mulling',
        'Mustering',
        'Musing',
        'Noodling',
        'Percolating',
        'Pondering',
        'Processing',
        'Puttering',
        'Reticulating',
        'Ruminating',
        'Slepping',
        'Shucking',
        'Shimming',
        'Simmering',
        'Smooshing',
        'Spinning',
        'Stewing',
        'Synthesizing',
        'Thinking',
        'Tinkering',
        'Transmuting',
        'Unfurling',
        'Vibing',
        'Working',
    ]
    import random
    return random.choice(STATUS_MESSAGES)

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
        Screen, Static, Input, .ai-response { background: transparent; }
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
        self.tools = {**wt}
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
        self.display_toolname_maps = {
            'read_file': 'Read',
            'read_multiple_files': 'Multi Read',
            'write_file': 'Write',
            'edit_file': 'Edit',
            'multi_edit_file': 'Multi Edit',
            'bash_execute': 'Bash',
            'glob_find_files': 'Glob',
            'grep_search': 'Grep',
            'list_directory': 'LS',
            'fetch_url': 'Web Fetch',
            'async_web_search': 'Web Search',
            'todo_read': 'Todo Read',
            'todo_write': 'Update Todos',
            'apply_edit': 'Apply Edit',
            'discard_edit': 'Discard Edit'
        }
        self.permission_mode, self.cwd = 'default', cwd

        self.llm = None
        self.chat_history = []
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
                        value = "what is today's ai news?"
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
        
        if value == "/clear":
            self.chat_history.clear()
            chat_area = self.query_one("#chat_area")
            chat_area.remove_children()
            chat_area.mount(Static(Panel(
                Text.from_markup(f"[#E27A53]✻ [/][white]Welcome to [/][bold white]Claude Code[/]!\n\n"
                               f"[italic]/help for help, /status for your current setup\n\ncwd: {self.cwd}[/italic]"),
                border_style="#E27A53", expand=False, padding=(0, 2, 0, 1)
            ), classes="welcome"))
            return True

        return False

    def execute_input(self, input):
        status = self._check_slash_command(input)
        if status:
            return

        # Schedule async execution
        asyncio.create_task(self._execute_input_async(input))

    async def _execute_input_async(self, input):
        animation_task = asyncio.create_task(self.animate_thinking_status(input))
        self.chat_history.extend([{"role": "user", "content": input}])
        messages = self.chat_history.copy()
        chat_area = self.query_one("#chat_area")
        max_iterations = 5

        try:
            while max_iterations:
                max_iterations-=1

                response = await asyncio.get_event_loop().run_in_executor(None, self.llm, messages)
                if response['content']:
                    markdown_content = Markdown("● " + response['content'].strip())
                    markdown_widget = Static(markdown_content, classes="ai-response")
                    chat_area.mount(markdown_widget)
                    chat_area.scroll_end(animate=False)
                    self.refresh()

                messages.append({
                    "role": "assistant",
                    "content":response['content'],
                    "tool_calls": response['tool_calls']
                })

                if response['tool_calls']:
                    for t in response['tool_calls']:
                        name, args = t['function']['name'],json.loads(str(t['function']['arguments']))

                        tool_text = Text()
                        tool_text.append("● ", style="dim #5cf074")
                        tool_text.append(self.display_toolname_maps[name], style="bold")
                        if args:
                            tool_text.append(f'("{list(args.values())[0]}")', style="default")

                        st = time.perf_counter()
                        result = await self.tools[name](**args)
                        tt = time.perf_counter() - st
                        tool_text.append(f"\n  ⎿ {self.display_toolresult(name,result,tt)}\n", style="default")

                        markdown_widget = Static(tool_text, classes="ai-response")
                        chat_area.mount(markdown_widget)
                        chat_area.scroll_end(animate=False)
                        self.refresh()

                        messages.append({
                            "role": "tool",
                            "tool_call_id": t.get("id"),
                            "name": name,
                            "content": str(result)
                        })
                else:
                    break

        except Exception as e:
            raise e
        finally:
            animation_task.cancel()
            self.chat_history.extend(messages)
            try:
                await animation_task
            except asyncio.CancelledError:
                pass



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

        input = event.value.strip()
        self.command_history.append(input)
        chat_area = self.query_one("#chat_area")
        chat_area.mount(Static(f"\n> {input}\n", classes="message"))
        chat_area.scroll_end(animate=False)
        self.refresh()

        self.execute_input(input)
        event.input.clear()

    def load_model(self):
        if self.model_loaded:
            return

        tools = [*list(wt.values())]

        if self.provider_name == 'gemini':
            from flowgen.llm.gemini import Gemini
            self.llm = Gemini(tools=tools)
        elif self.provider_name == 'ollama':
            from flowgen.llm.llm import Ollama
            self.llm = Ollama(tools=tools)
        elif self.provider_name == 'openrouter':
            from flowgen.llm import OpenRouter
            self.llm = OpenRouter(tools=tools)
        elif self.provider_name == 'vllm':
            from flowgen.llm.llm import vLLM
            self.llm = vLLM(tools=tools)
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")
        
        self.model_loaded = True

    async def animate_thinking_status(self, query: str):
        """Animate the thinking status with flowers and comprehensive status info like Claude Code"""
        flower_chars = ["✻", "✺", "✵", "✴", "❋", "❊", "❉", "❈", "❇", "❆", "❅", "❄"]
        flower_index = 0

        # Get one random status message for this entire session
        status_message = get_random_status_message() + "..."

        status_indicator = self.query_one("#status_indicator")
        chat_area = self.query_one("#chat_area")
        start_time = time.time()

        try:
            while True:
                elapsed_seconds = int(time.time() - start_time)
                flower_index = (flower_index + 1) % len(flower_chars)

                # Get current time for timestamp
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                
                # Estimate token count (simple approximation: ~4 chars per token)
                estimated_tokens = len(query) // 4 if query else 0
                
                # Create comprehensive status message with grey formatting like Claude Code
                status_msg = (
                    f"{flower_chars[flower_index]} {status_message} "
                    f"[grey]({elapsed_seconds}s • {estimated_tokens} tokens • esc to interrupt)[/grey]"
                )

                status_indicator.update(status_msg)

                # Ensure scroll stays at bottom during long operations every few cycles
                if flower_index % 10 == 0:
                    chat_area.scroll_end(animate=False)

                await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            status_indicator.update("")
            raise

    def display_toolresult(self, name, result,timetaken):
        if name=='async_web_search':
            return f"Did 1 search in {int(timetaken)}s"
        return str(result)