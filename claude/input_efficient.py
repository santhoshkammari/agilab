import json
import random

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
from flowgen.tools.markdown import tool_functions as mt
from flowgen.tools.content_extract import tool_functions as ct
from flowgen.tools.fileops import tool_functions as ft

SYSTEM_PROMPT="""
You are Claude Code, Anthropic's official CLI for Claude.

You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

IMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.
IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.

# Tone and style
You should be concise, direct, and to the point.
You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail.
IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.
IMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.
Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.
Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as "The answer is <answer>.", "Here is the content of the file..." or "Based on the information provided, the answer is..." or "Here is what I will do next...".

When you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).
Remember that your output will be displayed on a command line interface. Your responses can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.
Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.
If you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.
Only use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.
IMPORTANT: Keep your responses short, since they will be displayed on a command line interface.

# Proactiveness
You are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:
- Doing the right thing when asked, including taking actions and follow-up actions
- Not surprising the user with actions you take without asking
For example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.

# Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).
- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.
- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.
- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.

# Code style
- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked

# Task Management
You have access to the TodoWrite tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.

# Doing tasks
The user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:
- Use the TodoWrite tool to plan the task if required
- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.
- Implement the solution using all tools available to you
- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.
- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (eg. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct. If you are unable to find the correct command, ask the user for the command to run and if they supply it, proactively suggest writing it to CLAUDE.md so that you will know to run it next time.
NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.

- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are NOT part of the user's provided input or the tool result.

# Tool usage policy
- When doing file search, prefer to use the Task tool in order to reduce context usage.
- You should proactively use the Task tool with specialized agents when the task at hand matches the agent's description.
- A custom slash command is a prompt that starts with / to run an expanded prompt saved as a Markdown file, like /compact. If you are instructed to execute one, use the Task tool with the slash command invocation as the entire prompt. Slash commands can take arguments; defer to user instructions.
- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run "git status" and "git diff", send a single message with two tool calls to run the calls in parallel.
- IMPORTANT: When using directory listing tools, ALWAYS use absolute paths. Never use relative paths like "." for current directory - instead use the absolute current working directory path provided in SystemInformation.

You MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.

IMPORTANT: Always use the TodoWrite tool to plan and track tasks throughout the conversation.

# Code References
When referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.

SystemInformation:
- Current working directory : {cwd}
"""

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
        self.tools = {**wt,**mt,**ct,**ft}
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
            'write_file': 'Write',
            'edit_file': 'Update',
            'multi_edit_file': 'Multi Edit',
            'bash_execute': 'Bash',
            'glob_find_files': 'Search',
            'grep_search': 'Search',
            'list_directory': 'List',
            'task_agent': 'Task',
            'todo_write': 'Update Todos',
            'extract_markdown_from_url': 'Extract Markdown',
            'markdown_analyzer_get_headers': 'MD Headers',
            'markdown_analyzer_get_paragraphs': 'MD Paragraphs',
            'markdown_analyzer_get_links': 'MD Links',
            'markdown_analyzer_get_code_blocks': 'MD Code Blocks',
            'markdown_analyzer_get_tables_metadata': 'MD Tables Info',
            'markdown_analyzer_get_table_by_line': 'MD Table',
            'markdown_analyzer_get_lists': 'MD Lists',
            'markdown_analyzer_get_overview': 'MD Overview',
            'apply_edit': 'Apply Edit',
            'discard_edit': 'Discard Edit'
        }
        self.permission_mode, self.cwd = 'default', cwd

        self.llm = None
        self.chat_history = [{"role":"system","content":SYSTEM_PROMPT.format(cwd=cwd)}]
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
        max_iterations = 15

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
                        # Check if the tool function is async
                        import inspect
                        if inspect.iscoroutinefunction(self.tools[name]):
                            result = await self.tools[name](**args)
                        else:
                            result = self.tools[name](**args)
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

        tools = [*list(self.tools.values())]

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

    def display_toolresult(self, name, result, timetaken):
        tt = f"{timetaken:.1f}s" if timetaken >= 1 else f"{timetaken:.2f}s"
        
        if name == 'async_web_search':
            return f"Did 1 search in {tt}"
        
        elif name == 'extract_markdown_from_url':
            if isinstance(result, str) and "Markdown Saved at" in result:
                filename = result.split("Markdown Saved at ")[-1]
                return f"Saved markdown to {filename} in {tt}"
            return f"Extracted markdown in {tt}"
        
        elif name == 'markdown_analyzer_get_headers':
            if isinstance(result, dict) and 'Header' in result:
                count = len(result['Header'])
                return f"Found {count} headers in {tt}"
            elif "No headers found" in str(result):
                return f"No headers found in {tt}"
            return f"Analyzed headers in {tt}"
        
        elif name == 'markdown_analyzer_get_paragraphs':
            if isinstance(result, dict) and 'Paragraph' in result:
                count = len(result['Paragraph'])
                return f"Extracted {count} paragraphs in {tt}"
            elif "No paragraphs found" in str(result):
                return f"No paragraphs found in {tt}"
            return f"Analyzed paragraphs in {tt}"
        
        elif name == 'markdown_analyzer_get_links':
            if isinstance(result, list):
                count = len(result)
                return f"Found {count} HTTP links in {tt}"
            elif "No HTTP links found" in str(result):
                return f"No HTTP links found in {tt}"
            return f"Analyzed links in {tt}"
        
        elif name == 'markdown_analyzer_get_code_blocks':
            if isinstance(result, dict) and 'Code block' in result:
                count = len(result['Code block'])
                return f"Found {count} code blocks in {tt}"
            elif "No code blocks found" in str(result):
                return f"No code blocks found in {tt}"
            return f"Analyzed code blocks in {tt}"
        
        elif name == 'markdown_analyzer_get_tables_metadata':
            if "No tables found" in str(result):
                return f"No tables found in {tt}"
            else:
                # Count tables from formatted output
                table_count = str(result).count("Table #") if "Table #" in str(result) else 0
                return f"Found {table_count} tables in {tt}"
        
        elif name == 'markdown_analyzer_get_table_by_line':
            if "No table at line" in str(result):
                return f"No table found at specified line in {tt}"
            return f"Extracted table in {tt}"
        
        elif name == 'markdown_analyzer_get_lists':
            if isinstance(result, dict):
                ordered_count = len(result.get('Ordered list', []))
                unordered_count = len(result.get('Unordered list', []))
                total_count = ordered_count + unordered_count
                return f"Found {total_count} lists in {tt}"
            elif "No lists found" in str(result):
                return f"No lists found in {tt}"
            return f"Analyzed lists in {tt}"
        
        elif name == 'markdown_analyzer_get_overview':
            if "Empty document found" in str(result):
                return f"Document is empty in {tt}"
            return f"Generated overview in {tt}"
        
        # FileOps tools - clean aesthetic display
        elif name == 'read_file':
            if isinstance(result, str):
                if "[System Reminder: File exists but has empty contents]" in result:
                    return "Read empty file"
                
                lines = result.count('\n') + 1 if result else 0
                # Extract first few characters of actual content (skip line numbers)
                content_lines = result.split('\n')[:3]
                preview_content = []
                
                for line in content_lines:
                    if '\t' in line:  # Has line number format
                        content_part = line.split('\t', 1)[1] if len(line.split('\t')) > 1 else line
                        preview_content.append(content_part[:60])
                    else:
                        preview_content.append(line[:60])
                
                if preview_content:
                    preview = '\n'.join(preview_content)
                    if len(preview) > 150:
                        preview = preview[:147] + "..."
                    return f"Read {lines} lines\n{preview}"
                
                return f"Read {lines} lines"
            return "Read file"
        
        
        elif name == 'write_file':
            if isinstance(result, dict) and 'preview_lines' in result:
                # Show content preview from structured return
                preview = '\n'.join(result['preview_lines'])
                if len(preview) > 150:
                    preview = preview[:147] + "..."
                return f"Write\n{preview}"
            elif isinstance(result, str):
                # Fallback for old format
                return "Write"
            return "Write"
        
        elif name == 'edit_file':
            if isinstance(result, dict) and 'diff_lines' in result:
                # Show diff from structured return
                diff_display = '\n'.join(result['diff_lines'][:4])  # Show max 4 diff lines
                count = result.get('changes_count', 1)
                if diff_display:
                    return f"Update ({count} {'change' if count == 1 else 'changes'})\n{diff_display}"
                else:
                    return f"Update ({count} {'change' if count == 1 else 'changes'})"
            elif isinstance(result, str):
                # Fallback for old format
                return "Update"
            return "Update"
        
        elif name == 'multi_edit_file':
            if isinstance(result, str):
                edit_count = result.count("Edit") if "Edit" in result else 1
                return f"Multi Edit ({edit_count} changes)"
            return "Multi Edit"
        
        elif name == 'bash_execute':
            if isinstance(result, dict):
                # New structured return format from tool_bash.py
                success = result.get('success', False)
                stdout = result.get('stdout', '')
                stderr = result.get('stderr', '')
                exit_code = result.get('exit_code', 0)
                exec_time = result.get('execution_time', 0)
                
                status_icon = "✓" if success else "✗"
                time_str = f"{exec_time:.2f}s" if exec_time < 1 else f"{exec_time:.1f}s"
                
                if success and stdout:
                    lines = stdout.strip().split('\n')
                    if len(lines) <= 2:
                        return f"Bash {status_icon} ({time_str})\n{stdout.strip()}"
                    else:
                        preview = '\n'.join(lines[:2])
                        return f"Bash {status_icon} ({time_str})\n{preview}\n... ({len(lines)-2} more lines)"
                elif not success:
                    error_preview = stderr[:100] + "..." if len(stderr) > 100 else stderr
                    return f"Bash {status_icon} (exit {exit_code}, {time_str})\n{error_preview}"
                else:
                    return f"Bash {status_icon} ({time_str})"
            elif isinstance(result, str):
                # Fallback for old string format
                lines = result.strip().split('\n')
                if len(lines) <= 3:
                    return f"Bash\n{result.strip()}"
                else:
                    preview = '\n'.join(lines[:2])
                    return f"Bash\n{preview}\n... ({len(lines)-2} more lines)"
            return "Bash"
        
        elif name == 'glob_find_files':
            if isinstance(result, list):
                count = len(result)
                return f"Found {count} files"
            elif isinstance(result, str) and "files found" in result.lower():
                # Try to extract count from string
                import re
                match = re.search(r'(\d+)', result)
                if match:
                    return f"Found {match.group(1)} files"
            return "Search completed"
        
        elif name == 'grep_search':
            if isinstance(result, str):
                if not result.strip():
                    return "No matches found"
                
                lines = result.strip().split('\n')
                count = len(lines)
                
                # Show a preview for content mode or count for others
                if count <= 3:
                    return f"Found {count} matches\n{result.strip()}"
                else:
                    preview = '\n'.join(lines[:2])
                    return f"Found {count} matches\n{preview}\n... ({count-2} more)"
            elif isinstance(result, list):
                count = len(result)
                return f"Found {count} matches"
            return "Search completed"
        
        elif name == 'list_directory':
            if isinstance(result, dict) and 'entries' in result:
                # New structured format from tool_ls.py
                entries = result['entries']
                count = len(entries)
                dirs = sum(1 for e in entries if e.get('type') == 'directory')
                files = count - dirs
                
                if count <= 5:
                    preview = '\n'.join(f"{'📁' if e.get('type') == 'directory' else '📄'} {e['name']}" for e in entries)
                    return f"Listed {count} items ({dirs} dirs, {files} files)\n{preview}"
                else:
                    preview = '\n'.join(f"{'📁' if e.get('type') == 'directory' else '📄'} {e['name']}" for e in entries[:3])
                    return f"Listed {count} items ({dirs} dirs, {files} files)\n{preview}\n... ({count-3} more)"
            elif isinstance(result, list):
                count = len(result)
                return f"Listed {count} paths"
            elif isinstance(result, str):
                # Handle error messages or fallback format
                if "FileNotFound" in result or "Not a directory" in result or "Permission denied" in result:
                    return f"List error: {result}"
                # Count items in string representation
                lines = [line for line in result.split('\n') if line.strip()]
                return f"Listed {len(lines)} paths"
            return "List"
        
        elif name == 'task_agent':
            if isinstance(result, str):
                if len(result) > 100:
                    return f"Task completed\n{result[:97]}..."
                return f"Task completed\n{result}"
            return "Task completed"
        
        
        elif name == 'todo_write':
            if isinstance(result, dict) and 'success' in result:
                # New structured format from tool_todowrite.py
                summary = result.get('summary', {})
                total = summary.get('total_tasks', 0)
                status_breakdown = summary.get('status_breakdown', {})
                current_task = summary.get('current_task')
                
                pending = status_breakdown.get('pending', 0)
                in_progress = status_breakdown.get('in_progress', 0)
                completed = status_breakdown.get('completed', 0)
                
                status_text = f"{total} tasks: {pending}⏸ {in_progress}⏳ {completed}✓"
                
                if current_task:
                    current_preview = current_task['content'][:50] + "..." if len(current_task['content']) > 50 else current_task['content']
                    return f"Update Todos\n{status_text}\nCurrent: {current_preview}"
                else:
                    return f"Update Todos\n{status_text}"
            elif isinstance(result, str):
                return "Update Todos"
            return "Update Todos"
        
        
        return str(result)