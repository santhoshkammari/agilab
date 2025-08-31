import json
import random
import os
import re
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from rich.theme import Theme
from io import StringIO
from textual import on
from textual.app import ComposeResult, App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, OptionList, Label, Input
from textual.widget import Widget
import asyncio
import time
import datetime

from flowgen.tools.web import tool_functions as wt
from flowgen.tools.markdown import tool_functions as mt
from flowgen.tools.content_extract import tool_functions as ct
from flowgen.tools.fileops import tool_functions as ft

SYSTEM_PROMPT="""
You are Claude Code, Anthropic's official CLI for Claude.
You are an AI assistant that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

IMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.
IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.

# Tone and style
You should be concise, direct, and to the point.
You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail.
IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.
IMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.
Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.
Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as "The answer is <answer>.", "Here is the content of the file..." or "Based on the information provided, the answer is..." or "Here is what I will do next...". Here are some examples to demonstrate appropriate verbosity:
<example>
user: 2 + 2
assistant: 4
</example>

<example>
user: what is 2+2?
assistant: 4
</example>

<example>
user: is 11 a prime number?
assistant: Yes
</example>

<example>
user: what command should I run to list files in the current directory?
assistant: ls
</example>

<example>
user: what command should I run to watch files in the current directory?
assistant: [runs ls to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]
npm run dev
</example>

<example>
user: How many golf balls fit inside a jetta?
assistant: 150000
</example>

<example>
user: what files are in the directory src/?
assistant: [runs ls and sees foo.c, bar.c, baz.c]
user: which file contains the implementation of foo?
assistant: src/foo.c
</example>
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

Examples:

<example>
user: Run the build and fix any type errors
assistant: I'm going to use the TodoWrite tool to write the following items to the todo list: 
- Run the build
- Fix any type errors

I'm now going to run the build using Bash.

Looks like I found 10 type errors. I'm going to use the TodoWrite tool to write 10 items to the todo list.

marking the first todo as in_progress

Let me start working on the first item...

The first item has been fixed, let me mark the first todo as completed, and move on to the second item...
..
..
</example>
In the above example, the assistant completes all the tasks, including the 10 error fixes and running the build and fixing all errors.

<example>
user: Help me write a new feature that allows users to track their usage metrics and export them to various formats

A: I'll help you implement a usage metrics tracking and export feature. Let me first use the TodoWrite tool to plan this task.
Adding the following todos to the todo list:
1. Research existing metrics tracking in the codebase
2. Design the metrics collection system
3. Implement core metrics tracking functionality
4. Create export functionality for different formats

Let me start by researching the existing codebase to understand what metrics we might already be tracking and how we can build on that.

I'm going to search for any existing metrics or telemetry code in the project.

I've found some existing telemetry code. Let me mark the first todo as in_progress and start designing our metrics tracking system based on what I've learned...

[Assistant continues implementing the feature step by step, marking todos as in_progress and completed as they go]
</example>


Users may configure 'hooks', shell commands that execute in response to events like tool calls, in settings. Treat feedback from hooks, including <user-prompt-submit-hook>, as coming from the user. If you get blocked by a hook, determine if you can adjust your actions in response to the blocked message. If not, ask the user to check their hooks configuration.

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

- When WebFetch returns a message about a redirect to a different host, you should immediately make a new WebFetch request with the redirect URL provided in the response.
- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run "git status" and "git diff", send a single message with two tool calls to run the calls in parallel.




Here is useful information about the environment you are running in:
<env>
Working directory: {cwd}
Is directory a git repo: Yes
Platform: linux
OS Version: Linux 6.8.0-65-generic
Today's date: 2025-08-08
</env>
You are powered by the model named Sonnet 4. The exact model ID is claude-sonnet-4-20250514.

Assistant knowledge cutoff is January 2025.


IMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.


IMPORTANT: Always use the TodoWrite tool to plan and track tasks throughout the conversation.

# Code References

When referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.

<example>
user: Where are errors from the client handled?
assistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.
</example>

Answer the user's request using the relevant tool(s), if they are available. Check that all the required parameters for each tool call are provided or can reasonably be inferred from context. IF there are no relevant tools or there are missing values for required parameters, ask the user to supply these values; otherwise proceed with the tool calls. If the user provides a specific value for a parameter (for example provided in quotes), make sure to use that value EXACTLY. DO NOT make up values for or ask about optional parameters. Carefully analyze descriptive terms in the request as they may indicate required parameter values that should be included even if not explicitly quoted.

SystemInformation:
- Current working directory : {cwd}

/no_think
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

class ToolPermissionWidget(Widget):
    def __init__(self, title: str, content: str, options: list = None):
        super().__init__()
        self.title = title
        self.content = content
        self.options = options or ["• Execute", "• Execute & remember choice", "• Redirect Claude"]
        
    def compose(self) -> ComposeResult:
        with Vertical(classes="selection-area"):
            yield Static(f"{self.title}: {self.content}", id="tool_permission_message")
            yield OptionList(*self.options, id="tool_permission_options")

class ChatApp(App):
    BINDINGS = [
        Binding("shift+tab", "cycle_mode", "Cycle Mode", priority=True),
        Binding("tab", "trigger_file_autocomplete", "File Autocomplete", priority=True),
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
        #status_bar { height: auto; padding: 1 1; text-align: left;}
        #status_indicator { color: #E27A53; text-style: bold; }
        #footer { height: 10vh; dock: bottom; }
        #footer-left { text-align: left; width: auto; }
        #footer-right { text-align: right; width: 1fr; }
        .mode-bypass { color: #FF697E; }
        .mode-plan { color: #458588; }
        .mode-auto-edit { color: #915FF0; }
        .message, .welcome { color: grey; text-style: italic; }
        .streaming { color: #E77D22; }
        .selection-area { height: auto; padding: 1; margin: 0; }
        .selection-area Static { color: #a89984; text-style: italic; padding: 0 0 1 0; }
        .selection-area OptionList { border: none; height: auto; scrollbar-size: 0 0; background: transparent;}
        .selection-area OptionList:focus { border: none; background: transparent; }
        #permission_area { border: round #458588; }
        #command_palette { border: round #915FF0; }
        #file_autocomplete { border: round #fabd2f; }
        #provider_selection,#tool_permission_area { border: round #E27A53;background: transparent; }
        #tool_permission_area_message { color: #83a598; text-style: italic; }
        #model_selection { border: round #fabd2f; }
        OptionList { background: transparent !important; }
        OptionList > .option-list--option { color: #a89984; padding: 0 1; height: 1; background: transparent !important;}
        OptionList > .option-list--option-highlighted { color: #fabd2f; text-style: bold; background: transparent !important; }
        
        /* Claude Code Markdown Theming */
        .ai-response Markdown {
            color: #ebdbb2;
        }
        .ai-response MarkdownH1 {
            color: #fabd2f;
            text-style: bold;
        }
        .ai-response MarkdownH2 {
            color: #83a598;
            text-style: bold;
        }
        .ai-response MarkdownH3 {
            color: #d3869b;
            text-style: bold;
        }
        .ai-response MarkdownCodeBlock {
            color: #ebdbb2;
            background: #3c3836;
        }
        .ai-response MarkdownCode {
            color: #fe8019;
            background: #3c3836;
        }
        .ai-response MarkdownBlockQuote {
            color: #928374;
            text-style: italic;
        }
        .ai-response MarkdownList {
            color: #b8bb26;
        }
        .ai-response MarkdownListItem {
            color: #ebdbb2;
        }
        .ai-response MarkdownLink {
            color: #458588;
            text-style: underline;
        }
        """

    def __init__(self,cwd):
        super().__init__()
        
        # Delete .claudecode folder if it exists at startup
        import shutil
        claudecode_path = Path(cwd) / ".claudecode"
        if claudecode_path.exists():
            shutil.rmtree(claudecode_path)
        
        self.tools = {**wt,**mt,**ct,**ft}
        self.command_history, self.history_index, self.mode_idx = [], -1, 0
        self.modes = ['default', 'auto-accept-edits', 'bypass-permissions', 'plan-mode']
        self.providers = ["gemini", "ollama", "openrouter", "vllm"]
        self.provider_name = "gemini"
        self.permission_maps = {
            "default": "[grey]  ? for shortcuts[/grey]",
            'auto-accept-edits': "   ⏵⏵ auto-accept edits on   ",
            'bypass-permissions': "   bypass permissions on   ",
            'plan-mode': "   ⏸ plan mode on   "
        }
        
        # Add interruption flag
        self.conversation_interrupted = False
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
            'web_fetch': 'WebFetch',
            'markdown_analyzer_get_headers': 'Extract Headers',
            'markdown_analyzer_get_paragraphs': 'Extract Paragraphs',
            'markdown_analyzer_get_links': 'Extract Links',
            'markdown_analyzer_get_code_blocks': 'Extract Code',
            'markdown_analyzer_get_tables_metadata': 'Extract Tables Info',
            'markdown_analyzer_get_table_by_line': 'Extract Table',
            'markdown_analyzer_get_lists': 'Extract Lists',
            'markdown_analyzer_get_overview': 'Extract Overview',
            'apply_edit': 'Apply Edit',
            'discard_edit': 'Discard Edit',
            'async_web_search':'Web Search'
        }
        self.permission_mode, self.cwd = 'default', cwd

        self.llm = None
        self.chat_history = [{"role":"system","content":SYSTEM_PROMPT.format(cwd=cwd)}]
        self.model_loaded = False
        self.pending_tool_calls = []
        self.current_tool_index = 0
        self.auto_approve_tools = set()  # Tools that user chose "don't ask again" for
        
        # Tools that don't need permission in default mode
        self.no_permission_tools = {'read_file', 'list_directory', 'glob_find_files', 'grep_search', 'todo_read', 'todo_write'}
        
        # Tools that are auto-approved in auto-accept-edits mode  
        self.auto_accept_edit_tools = {'write_file', 'edit_file', 'multi_edit_file'}
        
        
        # For file autocomplete
        self.is_showing_files = False
        self.current_word_start = 0
        self.current_word = ""
        self.all_files = []
        self._collect_all_files()
        
        # History file management
        self.history_file = Path.home() / ".claudecode" / "history"
        self._load_history_from_file()

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
            ("permission_area", "", ["• Execute", "• Execute & remember choice", "• Redirect Claude"]),
            ("command_palette", "Command Palette - Select an option:", ["/host <url> - Set LLM host URL", "/provider - Switch LLM provider", "/think - Enable thinking mode", "/no-think - Disable thinking mode", "/status - Show current configuration", "/clear - Clear conversation history"]),
            ("provider_selection", "Select Provider:", self.providers),
            ("model_selection", "Select Model:", []),
            ("file_autocomplete", "Files:", []),
            ("tool_permission_area", "", ["• Execute", "• Execute & remember choice", "• Redirect Claude"])
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
                       #  value = "create 5 todos for building a web scraper project"
                       #  value = ("Complete the task by creating 5 todods , first web serach for virat , next rohith, next get virat test scores, next get rohith test scores, finally show"
                       #           "in markdown table both guys stats, prefer using markdown analyzer")
                       #  value = 'read app.py'
                       #  value= 'using websearch tool, search about todays ai news'
                        value = "get virats test score exact from wiki"
                        )
    
    def _create_footer(self):
        with Horizontal(id="footer"):
            yield Static(self.permission_maps["default"], id="footer-left")
            yield Static("", id="footer-right")

    def action_cycle_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.modes)
        self.permission_mode = self.modes[self.mode_idx]
        footer_left = self.query_one("#footer-left")
        
        # Add grey "(shift+tab to cycle)" to all modes when cycling
        base_text = self.permission_maps[self.permission_mode]
        if self.permission_mode != "default":
            # Remove trailing spaces from base_text before adding the cycle text
            display_text = base_text.rstrip() + " [grey](shift+tab to cycle)[/grey]"
        else:
            display_text = base_text
        
        footer_left.update(display_text)
        footer_left.remove_class("mode-bypass", "mode-plan", "mode-auto-edit")
        mode_classes = {'bypass-permissions': "mode-bypass", 'plan-mode': "mode-plan", 'auto-accept-edits': "mode-auto-edit"}
        if self.permission_mode in mode_classes:
            footer_left.add_class(mode_classes[self.permission_mode])

    def action_clear_input(self):
        self.query_one(Input).clear()

    def action_interrupt_conversation(self):
        """Handle escape key - hide file autocomplete if showing, otherwise interrupt conversation"""
        if self.is_showing_files:
            self._hide_file_autocomplete()
        else:
            self.conversation_interrupted = True
            status_indicator = self.query_one("#status_indicator")
            status_indicator.update("[red]Interrupted by user[/red]")
            
            # Close any open permission dialogs
            self.hide_tool_permission()
            
            # Clear pending tool calls since conversation is interrupted
            self.pending_tool_calls = []
            self.current_tool_index = 0
            
            # Add interruption message to chat history for model context
            if self.chat_history and self.chat_history[-1].get("role") == "user":
                interrupt_msg = {
                    "role": "user", 
                    "content": "[CONVERSATION INTERRUPTED BY USER - Please acknowledge the interruption and wait for further instructions]"
                }
                self.chat_history.append(interrupt_msg)

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


    def _collect_all_files(self):
        """Collect all files recursively from current directory"""
        self.all_files = []
        try:
            for root, dirs, files in os.walk(self.cwd):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
                
                for file in files:
                    if not file.startswith('.'):  # Skip hidden files
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.cwd)
                        self.all_files.append(rel_path)
        except Exception:
            pass  # Silently ignore permission errors

    def _load_history_from_file(self):
        """Load command history from ~/.claudecode/history file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    # Load last 10 commands, filter out empty lines
                    self.command_history = [line for line in lines if line.strip()][-10:]
        except Exception:
            pass  # Silently ignore errors, start with empty history

    def _save_history_to_file(self):
        """Save command history to ~/.claudecode/history file"""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Keep only last 10 commands
            history_to_save = self.command_history[-10:]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(history_to_save))
        except Exception:
            pass  # Silently ignore errors

    def _add_to_history(self, command: str):
        """Add command to history and save to file"""
        if command.strip():
            # Remove duplicates and add to end
            if command in self.command_history:
                self.command_history.remove(command)
            self.command_history.append(command)
            
            # Keep only last 10 commands
            self.command_history = self.command_history[-10:]
            
            # Save to file
            self._save_history_to_file()

    def action_trigger_file_autocomplete(self):
        """Trigger file autocomplete on tab press"""
        input_widget = self.query_one(Input)
        if not input_widget.has_focus:
            return
            
        current_text = input_widget.value
        cursor_pos = input_widget.cursor_position
        
        # Find the last word before cursor
        words = current_text[:cursor_pos].split()
        if words:
            last_word = words[-1]
            # Find start position of last word
            word_start = current_text.rfind(last_word, 0, cursor_pos)
            
            self.current_word = last_word
            self.current_word_start = word_start
            
            # Find fuzzy matches
            matched_files = self._fuzzy_match_files(last_word)
            
            if matched_files:
                self._show_file_autocomplete(matched_files)

    def _fuzzy_match_files(self, query):
        """Simple fuzzy matching for files (basic implementation)"""
        if not query:
            return self.all_files[:10]  # Show first 10 if no query
            
        query = query.lower()
        matches = []
        
        for file_path in self.all_files:
            filename = os.path.basename(file_path).lower()
            dir_path = os.path.dirname(file_path).lower()
            
            # Exact filename match gets highest score
            if filename.startswith(query):
                score = 100 - len(filename) + len(query) * 10
                matches.append((score, file_path))
            # Substring in filename
            elif query in filename:
                score = 80 - len(filename) + len(query) * 5
                matches.append((score, file_path))
            # Substring in directory
            elif query in dir_path:
                score = 40 - len(dir_path) // 10
                matches.append((score, file_path))
            # Characters match in order (basic fuzzy)
            elif self._char_sequence_match(query, filename):
                score = 20
                matches.append((score, file_path))
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x[0], reverse=True)
        return [match[1] for match in matches[:15]]

    def _char_sequence_match(self, query, text):
        """Check if characters in query appear in order in text"""
        query_idx = 0
        for char in text:
            if query_idx < len(query) and char == query[query_idx]:
                query_idx += 1
        return query_idx == len(query)

    def _show_file_autocomplete(self, files):
        """Show file autocomplete options"""
        file_msg = self.query_one("#file_autocomplete_message")
        file_msg.update(f"Files matching '{self.current_word}':")
        
        # Update options
        file_options = self.query_one("#file_options")
        file_options.clear_options()
        
        # Add files to options (limit to avoid UI overflow)
        for file_path in files:
            file_options.add_option(file_path)
        
        # Show the autocomplete area
        self.query_one("#file_autocomplete").display = True
        file_options.focus()
        
        # Auto-select the first (most relevant) option
        if files:
            file_options.highlighted = 0
        
        self.is_showing_files = True

    def _hide_file_autocomplete(self):
        """Hide file autocomplete and return focus to input"""
        self.query_one("#file_autocomplete").display = False
        self.is_showing_files = False
        # Focus input after a brief moment to ensure proper cursor handling
        self.call_later(self.query_one(Input).focus)

    @on(OptionList.OptionSelected, "#file_options")
    def handle_file_selection(self, event: OptionList.OptionSelected) -> None:
        """Handle file selection from autocomplete"""
        selected_file = str(event.option.prompt)
        
        # Replace the current word with selected file path
        input_widget = self.query_one(Input)
        current_text = input_widget.value
        
        # Replace the word
        new_text = (current_text[:self.current_word_start] + 
                   selected_file + 
                   current_text[self.current_word_start + len(self.current_word):])
        
        # Calculate cursor position
        new_cursor_pos = self.current_word_start + len(selected_file)
        
        # Hide autocomplete first
        self.query_one("#file_autocomplete").display = False
        self.is_showing_files = False
        
        # Set the new text and cursor position in one sequence
        input_widget.value = new_text
        input_widget.focus()
        
        # Use a delayed approach to ensure cursor position is properly set
        def set_cursor_final():
            # Move cursor using actions to ensure no selection
            input_widget.action_end(select=False)  # Go to end first
            
            # Calculate how many positions to move back from end
            positions_from_end = len(new_text) - new_cursor_pos
            for _ in range(positions_from_end):
                input_widget.action_cursor_left(select=False)
        
        # Delay to ensure input is properly focused and ready
        self.call_later(set_cursor_final)
        
        # Add the final text to history
        self._add_to_history(new_text)

    def _extract_file_references(self, input_text):
        """Extract file references from user input text"""
        import re
        from pathlib import Path
        
        # Pattern to match file paths (relative or absolute)
        # Matches common file extensions and path patterns
        file_patterns = [
            r'\b[\w\-./]+\.(?:py|js|ts|tsx|jsx|html|css|scss|less|json|yaml|yml|xml|md|txt|ini|cfg|conf|log|sql|sh|bat|ps1|php|rb|go|rs|cpp|c|h|hpp|java|kt|scala|swift|m|mm|pl|r|R|mat|nb|ipynb)\b',
            r'\b(?:[./][\w\-./]*/?[\w\-]+\.[a-zA-Z0-9]+)\b',
            r'\b(?:/[\w\-./]*/?[\w\-]+\.[a-zA-Z0-9]+)\b'
        ]
        
        found_files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, input_text, re.IGNORECASE)
            for match in matches:
                # Convert to absolute path if relative
                if match.startswith('./'):
                    abs_path = str(Path(self.cwd) / match[2:])
                elif match.startswith('/'):
                    abs_path = match
                else:
                    abs_path = str(Path(self.cwd) / match)
                
                # Check if file exists
                if Path(abs_path).exists() and Path(abs_path).is_file():
                    found_files.add(abs_path)
        
        return list(found_files)

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
        # Reset interruption flag at start of new conversation
        self.conversation_interrupted = False
        animation_task = asyncio.create_task(self.animate_thinking_status(input))
        
        # Check for file references and auto-execute Read tool calls
        file_refs = self._extract_file_references(input)
        if file_refs:
            # Create mock assistant message with Read tool calls
            mock_tool_calls = []
            tool_results = []
            file_read_info = []  # Store info for display
            
            for file_path in file_refs:
                tool_call_id = f"call_{len(mock_tool_calls)}"
                mock_tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"file_path": file_path})
                    }
                })
                
                # Execute the read tool
                try:
                    result = self.tools["read_file"](file_path=file_path)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "read_file",
                        "content": str(result)
                    })
                    
                    # Store info for display
                    lines = str(result).count('\n') + 1 if result else 0
                    filename = Path(file_path).name
                    file_read_info.append(f"  ⎿ Read {filename} ({lines} lines)")
                    
                except Exception as e:
                    # Handle file read errors
                    error_result = f"Error reading {file_path}: {str(e)}"
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "read_file",
                        "content": error_result
                    })
                    
                    filename = Path(file_path).name
                    file_read_info.append(f"  ⎿ Error reading {filename}")
            
            # Add user message
            self.chat_history.extend([{"role": "user", "content": input}])
            
            # Add mock assistant message with tool calls
            self.chat_history.append({
                "role": "assistant",
                "content": "",
                "tool_calls": mock_tool_calls
            })
            
            # Add tool results
            self.chat_history.extend(tool_results)
            
            # Display enriched user message with auto-read files
            enriched_message = f"\n> {input}\n" + "\n".join(file_read_info) + "\n"
            
            chat_area = self.query_one("#chat_area")
            chat_area.mount(Static(enriched_message, classes="message"))
            chat_area.scroll_end(animate=False)
            self.refresh()
        else:
            self.chat_history.extend([{"role": "user", "content": input}])
            
            # Display regular user message
            chat_area = self.query_one("#chat_area")
            chat_area.mount(Static(f"\n> {input}\n", classes="message"))
            chat_area.scroll_end(animate=False)
            self.refresh()
        
        messages = self.chat_history.copy()
        chat_area = self.query_one("#chat_area")
        max_iterations = 15

        try:
            while max_iterations and not self.conversation_interrupted:
                max_iterations-=1

                response = await asyncio.get_event_loop().run_in_executor(None, self.llm, messages)
                if response['content'].strip():
                    formatted_response = self._format_claude_response(response['content'].strip())
                    markdown_widget = Static(formatted_response, classes="ai-response")
                    chat_area.mount(markdown_widget)
                    chat_area.scroll_end(animate=False)
                    self.refresh()

                messages.append({
                    "role": "assistant",
                    "content":response['content'],
                    "tool_calls": response['tool_calls']
                })

                if response['tool_calls'] and not self.conversation_interrupted:
                    # Check if we need permission for any tools
                    tools_needing_permission = [t for t in response['tool_calls'] 
                                               if self.needs_permission(t['function']['name'])]
                    
                    if tools_needing_permission and not self.pending_tool_calls:
                        self.pending_tool_calls = response['tool_calls']
                        self.current_tool_index = 0
                        await self.show_tool_permission()
                        break  # Wait for user permission

                    chat_area.scroll_end(animate=False)
                    self.refresh()

                    # Execute tools (either permission granted or bypassed)
                    tools_to_execute = self.pending_tool_calls if self.pending_tool_calls else response['tool_calls']
                    
                    for t in tools_to_execute:
                        if self.conversation_interrupted:
                            break
                            
                        name, args = t['function']['name'],json.loads(str(t['function']['arguments']))

                        tool_text = Text()
                        tool_text.append("● ", style="dim #5cf074")
                        tool_text.append(self.display_toolname_maps[name], style="bold")
                        if args and name not in ["todo_write"]:
                            tool_text.append(f'("{list(args.values())[0]}")', style="default")

                        st = time.perf_counter()
                        # Check if the tool function is async
                        import inspect
                        if inspect.iscoroutinefunction(self.tools[name]):
                            result = await self.tools[name](**args)
                        else:
                            result = self.tools[name](**args)
                        tt = time.perf_counter() - st
                        
                        # Handle Rich Text objects for todo_write specially
                        tool_result_display = self.display_toolresult(name,result,tt)
                        if name == 'todo_write' and hasattr(tool_result_display, 'append'):
                            markdown_widget = Static(tool_text, classes="ai-response")
                            chat_area.mount(markdown_widget)

                            # Mount the Rich Text todos separately
                            todo_widget = Static(tool_result_display, classes="ai-response")
                            chat_area.mount(todo_widget)
                        else:
                            # Regular string result
                            tool_text.append(f"\n  ⎿ {tool_result_display}\n", style="default")
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
                    
                    # Clear pending tool calls after execution
                    if self.pending_tool_calls:
                        self.pending_tool_calls = []
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



    async def show_tool_permission(self):
        """Show tool permission dialog for the current tool"""
        if not self.pending_tool_calls or self.current_tool_index >= len(self.pending_tool_calls):
            return
        
        current_tool = self.pending_tool_calls[self.current_tool_index]
        name = current_tool['function']['name']
        args = json.loads(str(current_tool['function']['arguments']))
        
        # Format tool info for display
        display_name = self.display_toolname_maps.get(name, name)
        
        # Create content description based on tool type
        if name == 'write_file':
            file_path = args.get('file_path', 'unknown')
            content_preview = args.get('content', '')
            content = f"Write to {file_path}\nContent: {content_preview}"
        elif name == 'bash_execute':
            command = args.get('command', 'unknown')
            content = f"Execute: {command}"
        elif name == 'read_file':
            file_path = args.get('file_path', 'unknown')
            content = f"Read: {file_path}"
        elif name == 'edit_file':
            file_path = args.get('file_path', 'unknown')
            old_string = args.get('old_string', '')
            new_string = args.get('new_string', '')
            # Show a preview of what will be changed
            old_preview = old_string[:50] + "..." if len(old_string) > 50 else old_string
            new_preview = new_string[:50] + "..." if len(new_string) > 50 else new_string
            content = f"Update: {file_path}\n```diff\n- {old_preview}\n+ {new_preview}\n```"
        else:
            if args:
                first_arg = str(list(args.values())[0])
                content = f"{display_name}: {first_arg}"
            else:
                content = display_name

        # Update permission message
        tool_msg = self.query_one("#tool_permission_area_message")
        tool_msg.update(f"Execute {display_name}?\n{content}")
        
        # Show permission area and focus
        self.query_one("#tool_permission_area").display = True
        tool_options = self.query_one("#tool_options")
        tool_options.focus()


    @on(OptionList.OptionSelected, "#tool_options")
    def handle_tool_permission_choice(self, event: OptionList.OptionSelected) -> None:
        """Handle tool permission choice"""
        choice = event.option_index
        
        if choice == 0:  # Yes
            self.hide_tool_permission()
            asyncio.create_task(self.continue_tool_execution())
        elif choice == 1:  # Yes, and don't ask again  
            if self.pending_tool_calls and self.current_tool_index < len(self.pending_tool_calls):
                tool_name = self.pending_tool_calls[self.current_tool_index]['function']['name']
                self.auto_approve_tools.add(tool_name)
            self.hide_tool_permission()
            asyncio.create_task(self.continue_tool_execution())
        else:  # No, tell Claude what to do
            self.pending_tool_calls = []
            self.hide_tool_permission()
            # Focus back to input for user to give new instructions
            self.query_one(Input).focus()

    def hide_tool_permission(self):
        """Hide tool permission area"""
        self.query_one("#tool_permission_area").display = False
        self.query_one(Input).focus()

    def needs_permission(self, tool_name):
        """Check if a tool needs permission based on current mode"""
        if self.permission_mode == 'bypass-permissions':
            return False
        elif self.permission_mode == 'plan-mode':
            return True  # Plan mode doesn't use tools, so always ask
        elif self.permission_mode == 'auto-accept-edits':
            # Auto-approve all tools except those explicitly requiring permission
            dangerous_tools = {'bash_execute'}  # Only bash needs permission
            if tool_name in dangerous_tools:
                return tool_name not in self.auto_approve_tools
            return False  # All other tools are auto-approved
        else:  # default mode
            # Only bash_execute and write operations need permission, everything else is automatic
            permission_required_tools = {'bash_execute', 'write_file', 'edit_file', 'multi_edit_file'}
            if tool_name in permission_required_tools:
                return tool_name not in self.auto_approve_tools
            return False  # All other tools are automatic

    async def continue_tool_execution(self):
        """Continue executing pending tools after permission granted"""
        if not self.pending_tool_calls:
            return
        
        # Continue the async execution from where we left off
        asyncio.create_task(self._continue_async_execution())

    async def _continue_async_execution(self):
        """Resume tool execution after permission granted"""
        # Continue executing the pending tools without re-processing input
        if not self.pending_tool_calls:
            return
            
        chat_area = self.query_one("#chat_area")
        messages = self.chat_history.copy()
        
        try:
            # Execute the pending tools
            for t in self.pending_tool_calls:
                name, args = t['function']['name'], json.loads(str(t['function']['arguments']))

                tool_text = Text()
                tool_text.append("● ", style="dim #5cf074")
                tool_text.append(self.display_toolname_maps[name], style="bold")
                if args and name not in ["todo_write"]:
                    tool_text.append(f'("{list(args.values())[0]}")', style="default")

                st = time.perf_counter()
                # Check if the tool function is async
                import inspect
                if inspect.iscoroutinefunction(self.tools[name]):
                    result = await self.tools[name](**args)
                else:
                    result = self.tools[name](**args)
                tt = time.perf_counter() - st
                
                # Handle Rich Text objects for todo_write specially
                tool_result_display = self.display_toolresult(name, result, tt)
                if name == 'todo_write' and hasattr(tool_result_display, 'append'):
                    markdown_widget = Static(tool_text, classes="ai-response")
                    chat_area.mount(markdown_widget)

                    # Mount the Rich Text todos separately
                    todo_widget = Static(tool_result_display, classes="ai-response")
                    chat_area.mount(todo_widget)
                else:
                    # Regular string result
                    tool_text.append(f"\n  ⎿ {tool_result_display}\n", style="default")
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
            
            # Clear pending tool calls after execution
            self.pending_tool_calls = []
            
            # Continue the conversation with the updated messages
            max_iterations = 15
            while max_iterations > 0 and not self.conversation_interrupted:
                max_iterations -= 1
                response = await asyncio.get_event_loop().run_in_executor(None, self.llm, messages)
                
                if response['content'].strip():
                    formatted_response = self._format_claude_response(response['content'].strip())
                    markdown_widget = Static(formatted_response, classes="ai-response")
                    chat_area.mount(markdown_widget)
                    chat_area.scroll_end(animate=False)
                    self.refresh()

                messages.append({
                    "role": "assistant",
                    "content": response['content'],
                    "tool_calls": response['tool_calls']
                })

                if response['tool_calls'] and not self.conversation_interrupted:
                    # Check if we need permission for any tools
                    tools_needing_permission = [t for t in response['tool_calls'] 
                                               if self.needs_permission(t['function']['name'])]
                    
                    if tools_needing_permission and not self.pending_tool_calls:
                        self.pending_tool_calls = response['tool_calls']
                        self.current_tool_index = 0
                        await self.show_tool_permission()
                        break  # Wait for user permission

                    # Execute tools (permission granted or bypassed)
                    for t in response['tool_calls']:
                        if self.conversation_interrupted:
                            break
                            
                        name, args = t['function']['name'], json.loads(str(t['function']['arguments']))

                        tool_text = Text()
                        tool_text.append("● ", style="dim #5cf074")
                        tool_text.append(self.display_toolname_maps[name], style="bold")
                        if args and name not in ["todo_write"]:
                            tool_text.append(f'("{list(args.values())[0]}")', style="default")

                        st = time.perf_counter()
                        # Check if the tool function is async
                        import inspect
                        if inspect.iscoroutinefunction(self.tools[name]):
                            result = await self.tools[name](**args)
                        else:
                            result = self.tools[name](**args)
                        tt = time.perf_counter() - st
                        
                        # Handle Rich Text objects for todo_write specially
                        tool_result_display = self.display_toolresult(name, result, tt)
                        if name == 'todo_write' and hasattr(tool_result_display, 'append'):
                            markdown_widget = Static(tool_text, classes="ai-response")
                            chat_area.mount(markdown_widget)

                            # Mount the Rich Text todos separately
                            todo_widget = Static(tool_result_display, classes="ai-response")
                            chat_area.mount(todo_widget)
                        else:
                            # Regular string result
                            tool_text.append(f"\n  ⎿ {tool_result_display}\n", style="default")
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
                    
            # Update chat history
            self.chat_history = messages
                    
        except Exception as e:
            raise e

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
        areas = ["permission_area", "command_palette", "provider_selection", "model_selection", "file_autocomplete", "tool_permission_area"]
        for area in areas:
            self.query_one(f"#{area}").display = False
        self.query_one(Input).focus()
        fr = self.query_one("#footer-right")
        fr.update("")

    @on(Input.Submitted)
    def handle_message(self,event: Input.Submitted):
        # Only handle if the input widget has focus and there's content
        if not event.input.has_focus or not event.value.strip():
            return
            
        if not self.model_loaded:
            self.load_model()

        input = event.value.strip()
        self._add_to_history(input)

        self.execute_input(input)
        event.input.clear()

    def load_model(self):
        if self.model_loaded:
            return

        tools = [*list(self.tools.values())]

        if self.provider_name == 'gemini':
            from flowgen import Gemini
            self.llm = Gemini(tools=tools)
        elif self.provider_name == 'ollama':
            from flowgen import Ollama
            self.llm = Ollama(tools=tools,host="192.168.170.76",model='qwen3:0.6b-fp16')
        elif self.provider_name == 'openrouter':
            from flowgen import OpenRouter
            self.llm = OpenRouter(tools=tools)
        elif self.provider_name == 'vllm':
            from flowgen import vLLM
            self.llm = vLLM(
                base_url="http://192.168.170.76:8077/v1",
                tools=tools,
                # model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
                model = "/home/ng6309/datascience/santhosh/models/Qwen__Qwen3-4B-Thinking-2507",
                timeout=600
            )
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

    def _format_claude_response(self, content):
        """Format assistant response to match Claude Code CLI style"""
        # Just return the markdown with code theme - Textual handles styling via CSS
        return Markdown(content.strip(), code_theme="gruvbox-dark")

    def display_toolresult(self, name, result, timetaken):
        tt = f"{timetaken:.1f}s" if timetaken >= 1 else f"{timetaken:.2f}s"
        
        if name == 'async_web_search':
            return f"Did 1 search in {tt}"
        
        elif name == 'web_fetch':
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
                # Show diff from structured return with code markdown formatting
                diff_lines = result['diff_lines']
                count = result.get('changes_count', 1)
                
                if diff_lines:
                    # Format diff with proper code markdown
                    # Show first few lines with proper formatting
                    formatted_diff_lines = []
                    shown_lines = 0
                    max_display_lines = 6
                    
                    for line in diff_lines:
                        if shown_lines >= max_display_lines:
                            if len(diff_lines) > max_display_lines:
                                formatted_diff_lines.append("...")
                            break
                        formatted_diff_lines.append(line)
                        shown_lines += 1
                    
                    diff_display = '\n'.join(formatted_diff_lines)
                    # Wrap in code block for proper syntax highlighting
                    return f"Update ({count} {'change' if count == 1 else 'changes'})\n```diff\n{diff_display}\n```"
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
                return f"Found {count} lines"
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
                return f"Listed {count} items ({dirs} dirs, {files} files)"
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
            if isinstance(result, dict) and 'todos' in result:
                # Format todos with styled checkboxes using Rich Text like input.py
                from rich.text import Text
                todos = result.get('todos', [])
                if not todos:
                    return "Update Todos\nNo todos"
                
                # Create Rich Text object for colored formatting
                formatted_todos = Text()
                
                # Sort todos by ID to maintain order
                for todo in sorted(todos, key=lambda x: x.get('id', '0')):
                    status = todo.get('status', 'pending')
                    content = todo.get('content', '')
                    
                    # Get checkbox symbol and color based on status
                    if status == 'completed':
                        checkbox = "☒"
                        color = "#5cf074"  # Green
                    elif status == 'in_progress':
                        checkbox = "☐"
                        color = "#b19cd9"  # Lavender/purple
                    else:  # pending
                        checkbox = "☐"
                        color = "#a89984"  # Grey
                    
                    # Format the todo line with colors
                    formatted_todos.append(f"  {checkbox} ", style=color)
                    formatted_todos.append(content, style=color)
                    formatted_todos.append("\n")
                
                return formatted_todos
            elif isinstance(result, dict) and 'success' in result:
                # Fallback to old format
                summary = result.get('summary', {})
                total = summary.get('total_tasks', 0)
                status_breakdown = summary.get('status_breakdown', {})
                
                pending = status_breakdown.get('pending', 0)
                in_progress = status_breakdown.get('in_progress', 0)
                completed = status_breakdown.get('completed', 0)
                
                status_text = f"{total} tasks: {pending}⏸ {in_progress}⏳ {completed}✓"
                return f"Update Todos\n{status_text}"
            elif isinstance(result, str):
                return "Update Todos"
            return "Update Todos"
        
        
        return str(result)
