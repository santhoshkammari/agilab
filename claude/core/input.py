import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Input, Static, Label, ProgressBar, OptionList
from textual import on
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from claude.tools import tools, tools_dict
from claude.llm import Ollama
from claude.core.utils import ORANGE_COLORS, get_contextual_thinking_words, SYSTEM_PROMPT

#host="http://192.168.170.76:11434"
host=None
NUM_CTX = 2048
MODEL= "qwen3:0.6b"


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
        background: #504945;
        color: #fabd2f;
        text-style: bold;
    }
    
    .option-list--option-hover {
        background: #3c3836;
        color: #ebdbb2;
    }
    """

    def __init__(self,cwd):
        super().__init__()
        self.llm = Ollama(host=host)
        self.tool_widgets = {}  # Track tool execution widgets
        self.cwd = cwd
        # Initialize conversation history with system prompt
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT.format(cwd=self.cwd)}
        ]
        # Permission system state
        self.pending_tool_call = None
        self.permission_granted = False
        self.auto_approve_tools = set()  # Tools that user chose "don't ask again" for
        self.waiting_for_permission = False

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="chat_area"):
            welcome_panel = Panel(
                renderable=Text.from_markup(f"[{ORANGE_COLORS[17]}]✻ [/][bold]Welcome to [/][bold orange1]Plaude Pode[/]!\n\n"
                                 f"/help for help, /status for your current setup\n\ncwd: {self.cwd}"),
                border_style=ORANGE_COLORS[17],
                expand=False
            )
            yield Static(welcome_panel, classes="message")
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
        
        # Input area at bottom
        with Horizontal(id="input_area"):
            yield Label("> ")
            yield Input(placeholder="Type your message here...", compact=True, value="read /home/ntlpt59/master/own/claude/claude/core/input.py")
        # Footer
        with Horizontal(id="footer"):
            yield Static("⏵⏵ auto-accept edits on   ", id="footer-right")


    def on_ready(self) -> None:
        """Called when the app is ready - focus the input"""
        self.query_one(Input).focus()
        self.theme="gruvbox"
        # Hide permission area initially
        self.query_one("#permission_area").display = False

    @on(OptionList.OptionSelected, "#permission_options")
    def handle_permission_choice(self, event: OptionList.OptionSelected) -> None:
        """Handle permission choice from OptionList"""
        choice_index = event.option_index
        
        if choice_index == 0:  # Yes
            self.permission_granted = True
            self.hide_permission_widget()
            self.call_later(self.execute_pending_tool)
        elif choice_index == 1:  # Yes, and don't ask again
            if self.pending_tool_call:
                tool_name = self.pending_tool_call.function.name
                self.auto_approve_tools.add(tool_name)
            self.permission_granted = True
            self.hide_permission_widget()
            self.call_later(self.execute_pending_tool)
        elif choice_index == 2:  # No
            self.permission_granted = False
            self.hide_permission_widget()
            # Add user message asking what to do instead
            self.conversation_history.append({
                "role": "user",
                "content": f"I don't want you to execute the {self.pending_tool_call.function.name} tool. Please suggest an alternative approach or ask me what you should do instead."
            })
            chat_area = self.query_one("#chat_area")
            chat_area.mount(Static(f"\n> I don't want you to execute the {self.pending_tool_call.function.name} tool. Please suggest an alternative.\n", classes="message"))
            self.pending_tool_call = None
            self.call_later(self.start_ai_response, "tool_rejected", self.conversation_history.copy())

    @on(Input.Submitted)
    def handle_message(self, event: Input.Submitted) -> None:
        """Handle when user submits a message"""
        # Don't process input if waiting for permission
        if self.waiting_for_permission:
            return
            
        query = event.value.strip()
        if query:  # Only process non-empty messages
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user", 
                "content": query + "/no_think"
            })
            
            # Add the message to chat area
            chat_area = self.query_one("#chat_area")
            chat_area.mount(Static(f"\n> {query}\n", classes="message"))
            # Clear the input
            event.input.clear()

            chat_area.scroll_end(animate=False)
            self.refresh()
            self.call_later(self.start_ai_response, query, self.conversation_history.copy())

    def show_permission_widget(self, tool_call):
        """Show permission widget and hide input"""
        self.pending_tool_call = tool_call
        self.waiting_for_permission = True
        
        # Update permission message
        tool_name = tool_call.function.name
        tool_args = list(tool_call.function.arguments.values())[0] if tool_call.function.arguments else ""
        
        permission_msg = self.query_one("#permission_message")
        permission_msg.update(f"Do you want to make this edit to {tool_name}({tool_args})?")
        
        # Show permission area and hide input
        self.query_one("#permission_area").display = True
        self.query_one("#input_area").display = False
        
        # Focus option list and highlight first option
        option_list = self.query_one("#permission_options")
        option_list.focus()
        option_list.highlighted = 0

    def hide_permission_widget(self):
        """Hide permission widget and show input"""
        self.waiting_for_permission = False
        self.query_one("#permission_area").display = False
        self.query_one("#input_area").display = True
        self.query_one(Input).focus()

    async def execute_pending_tool(self):
        """Execute the pending tool after permission is granted"""
        if self.pending_tool_call and self.permission_granted:
            tool_call = self.pending_tool_call
            tool_name = tool_call.function.name.split('_')[0].title()
            tool_args = list(tool_call.function.arguments.values())[0] if tool_call.function.arguments else ""
            tool_id = f"{tool_call.function.name}_{id(tool_call)}"
            
            # Create tool widget with animation
            await self.create_tool_widget(tool_name, tool_args, tool_id)
            
            # Execute the actual tool
            result = await self.execute_tool_and_get_result(tool_call, tool_id, tool_args)
            
            # Give time for user to see the tool completion
            await asyncio.sleep(0.1)
            
            # Add tool result to conversation history
            self.conversation_history.append({
                'role': 'tool',
                'content': str(result),
                'tool_name': tool_call.function.name
            })
            
            # Add user message asking for code explanation/summary
            if tool_call.function.name == 'read_file':
                self.conversation_history.append({
                    'role': 'user',
                    'content': 'Format response as: 1) Brief title in normal text (no # headers), 2) 3-4 bullet points using - for key components/features, 3) Short conclusion line. Keep it concise like Claude Code style.'
                })
            elif tool_call.function.name == 'list_directory':
                self.conversation_history.append({
                    'role': 'user', 
                    'content': 'Format response as: 1) Brief title in normal text (no # headers), 2) 3-4 bullet points using - for key contents, 3) Short summary line. Keep it concise.'
                })
            else:
                self.conversation_history.append({
                    'role': 'user',
                    'content': 'Format response as: 1) Brief title in normal text (no # headers), 2) 3-4 bullet points using - for key results, 3) Short conclusion. Keep it concise.'
                })
            
            # Continue with LLM response
            self.pending_tool_call = None
            self.permission_granted = False
            self.call_later(self.continue_ai_response)

    async def continue_ai_response(self):
        """Continue AI response after tool execution"""
        # Make another LLM call with tool results
        animation_task = asyncio.create_task(self.animate_thinking_status("tool_result", "Processing tool results..."))
        
        loop = asyncio.get_event_loop()
        final_response = await loop.run_in_executor(None, lambda: self.llm.chat(messages=self.conversation_history, model=MODEL, tools=tools, num_ctx=NUM_CTX))
        
        # Stop thinking animation
        animation_task.cancel()
        try:
            await animation_task
        except asyncio.CancelledError:
            pass
        
        # Display the final response
        if final_response.message.content and final_response.message.content.strip():
            # Remove thinking content before displaying
            clean_content = self.remove_thinking_content(final_response.message.content)
            if clean_content.strip():
                # Use rich Markdown for better rendering
                markdown_content = Markdown("● " + clean_content.strip())
                markdown_widget = Static(markdown_content, classes="ai-response")
                chat_area = self.query_one("#chat_area")
                chat_area.mount(markdown_widget)
                
                # Add assistant response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": clean_content.strip()
                })
        
        # Reset status
        status_indicator = self.query_one("#status_indicator")
        status_indicator.update("")
        
        # Ensure scroll to end
        chat_area = self.query_one("#chat_area")
        self.call_after_refresh(lambda: chat_area.scroll_end(animate=False))

    async def start_ai_response(self, query: str, messages: list):
        """Start AI response using Ollama chat method with tool call handling"""
        chat_area = self.query_one("#chat_area")
        status_indicator = self.query_one("#status_indicator")
        
        import time
        flower_chars = ["✻", "✺", "✵", "✴", "❋", "❊", "❉", "❈", "❇", "❆", "❅", "❄"]
        flower_index = 0
        
        # Generate contextual thinking words based on user input
        thinking_words = get_contextual_thinking_words(query)
        thinking_word_index = 0
        
        # Track start time for elapsed seconds
        start_time = time.time()
        
        # Set initial status
        status_indicator.update("● Generating response...")
        
        try:
            # Start thinking animation and make the LLM call in a separate thread
            animation_task = asyncio.create_task(self.animate_thinking_status(query, "Generating response..."))
            
            # Make the LLM call using chat method in a separate thread
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.llm.chat(messages=messages, model=MODEL ,num_ctx = NUM_CTX,tools=tools))
            
            # Stop thinking animation
            animation_task.cancel()
            try:
                await animation_task
            except asyncio.CancelledError:
                pass
            
            # Handle tool calls if present
            if response.message.tool_calls:
                # Add assistant message with tool calls to history
                self.conversation_history.append(response.message)
                
                # Process first tool call (handle one at a time for permissions)
                tc = response.message.tool_calls[0]
                tool_name = tc.function.name
                
                # Check if tool needs permission and wasn't auto-approved
                if tool_name in ['read_file', 'list_directory'] and tool_name not in self.auto_approve_tools:
                    # Show permission widget and wait for user decision
                    self.show_permission_widget(tc)
                    return
                else:
                    # Auto-approved or doesn't need permission - execute directly
                    tool_display_name = tc.function.name.split('_')[0].title()
                    tool_args = list(tc.function.arguments.values())[0] if tc.function.arguments else ""
                    tool_id = f"{tc.function.name}_{id(tc)}"
                    
                    # Create tool widget with animation
                    await self.create_tool_widget(tool_display_name, tool_args, tool_id)
                    
                    # Execute the actual tool
                    result = await self.execute_tool_and_get_result(tc, tool_id, tool_args)
                    
                    # Give time for user to see the tool completion
                    await asyncio.sleep(0.1)
                    
                    # Add tool result to conversation history
                    self.conversation_history.append({
                        'role': 'tool',
                        'content': str(result),
                        'tool_name': tc.function.name
                    })
                    
                    # Add user message asking for code explanation/summary
                    if tc.function.name == 'read_file':
                        self.conversation_history.append({
                            'role': 'user',
                            'content': 'Format response as: 1) Brief title in normal text (no # headers), 2) 3-4 bullet points using - for key components/features, 3) Short conclusion line. Keep it concise like Claude Code style.'
                        })
                    elif tc.function.name == 'list_directory':
                        self.conversation_history.append({
                            'role': 'user', 
                            'content': 'Format response as: 1) Brief title in normal text (no # headers), 2) 3-4 bullet points using - for key contents, 3) Short summary line. Keep it concise.'
                        })
                    else:
                        self.conversation_history.append({
                            'role': 'user',
                            'content': 'Format response as: 1) Brief title in normal text (no # headers), 2) 3-4 bullet points using - for key results, 3) Short conclusion. Keep it concise.'
                        })
                
                # Make another LLM call with tool results
                animation_task = asyncio.create_task(self.animate_thinking_status(query, "Processing tool results..."))
                
                loop = asyncio.get_event_loop()
                final_response = await loop.run_in_executor(None, lambda: self.llm.chat(messages=self.conversation_history, model=MODEL, tools=tools,
                    num_ctx = NUM_CTX))
                
                # Stop thinking animation
                animation_task.cancel()
                try:
                    await animation_task
                except asyncio.CancelledError:
                    pass
                
                # Display the final response
                if final_response.message.content and final_response.message.content.strip():
                    # Remove thinking content before displaying
                    clean_content = self.remove_thinking_content(final_response.message.content)
                    if clean_content.strip():
                        # Use rich Markdown for better rendering
                        markdown_content = Markdown("● " + clean_content.strip())
                        markdown_widget = Static(markdown_content, classes="ai-response")
                        chat_area.mount(markdown_widget)
                        
                        # Add assistant response to conversation history
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": clean_content.strip()
                        })
            else:
                # No tool calls, just display the response
                if response.message.content and response.message.content.strip():
                    # Remove thinking content before displaying
                    clean_content = self.remove_thinking_content(response.message.content)
                    if clean_content.strip():
                        # Use rich Markdown for better rendering
                        markdown_content = Markdown("● " + clean_content.strip())
                        markdown_widget = Static(markdown_content, classes="ai-response")
                        chat_area.mount(markdown_widget)
                        
                        # Add assistant response to conversation history
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": clean_content.strip()
                        })
                    
        except Exception as e:
            error_text = f"🤖 **Error**: {str(e)}"
            chat_area.mount(Static(error_text, classes="ai-response"))
            status_indicator.update("")
            return

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
        
    
    async def execute_tool(self, tool_call, tool_id: str, tool_args: str):
        """Execute the actual tool and complete the execution"""
        try:
            tool_name = tool_call.function.name
            
            # Get the tool method from the tools dictionary
            if tool_name in tools_dict:
                tool_method = tools_dict[tool_name]
                result = await tool_method(**tool_call.function.arguments)
                
                # Generate appropriate result text based on tool and result
                if tool_name == "read_file" and isinstance(result, dict) and 'lines' in result:
                    result_text = f"Read {result['lines']} lines"
                elif tool_name == "list_directory":
                    result_text = f"Listed {len(result)} items"
                else:
                    result_text = "done"
            else:
                result_text = f"Unknown tool: {tool_name}"
            
            self.complete_tool_execution(tool_id, tool_args, result_text)
                
        except Exception as e:
            self.complete_tool_execution(tool_id, tool_args, f"error: {str(e)}")
            
    async def execute_tool_and_get_result(self, tool_call, tool_id: str, tool_args: str):
        """Execute the actual tool and return the result"""
        try:
            tool_name = tool_call.function.name
            
            # Get the tool method from the tools dictionary
            if tool_name in tools_dict:
                tool_method = tools_dict[tool_name]
                result = await tool_method(**tool_call.function.arguments)
                
                # Generate appropriate result text based on tool and result
                if tool_name == "read_file" and isinstance(result, dict) and 'lines' in result:
                    result_text = f"Read {result['lines']} lines"
                elif tool_name == "list_directory":
                    result_text = f"Listed {len(result)} items"
                else:
                    result_text = "done"
                    
                self.complete_tool_execution(tool_id, tool_args, result_text)
                return result
            else:
                result_text = f"Unknown tool: {tool_name}"
                self.complete_tool_execution(tool_id, tool_args, result_text)
                return result_text
                
        except Exception as e:
            error_msg = f"error: {str(e)}"
            self.complete_tool_execution(tool_id, tool_args, error_msg)
            return error_msg
    
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

    def remove_thinking_content(self, content: str) -> str:
        """Remove <think>...</think> content from the response"""
        # Split by </think> and return the last part, properly stripped
        return content.split('</think>')[-1].strip()

    async def animate_thinking_status(self, query: str, base_message: str):
        """Animate the thinking status with flowers and contextual words"""
        import time
        flower_chars = ["✻", "✺", "✵", "✴", "❋", "❊", "❉", "❈", "❇", "❆", "❅", "❄"]
        flower_index = 0
        
        # Generate contextual thinking words based on user input
        thinking_words = get_contextual_thinking_words(query)
        thinking_word_index = 0
        
        status_indicator = self.query_one("#status_indicator")
        start_time = time.time()
        
        try:
            while True:
                elapsed_seconds = int(time.time() - start_time)
                flower_index = (flower_index + 1) % len(flower_chars)
                
                # Change thinking word every 5 flower cycles to slow it down
                if flower_index % 5 == 0:
                    thinking_word_index = (thinking_word_index + 1) % len(thinking_words)
                
                current_thinking_word = thinking_words[thinking_word_index]
                status_indicator.update(f"{flower_chars[flower_index]} {current_thinking_word} [grey]({elapsed_seconds}s)[/grey]")
                
                await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            status_indicator.update("")
            raise



if __name__ == "__main__":
    import sys
    cwd = sys.argv[1] if len(sys.argv) > 1 else "/home/ntlpt59/master/own/claude/claude/core"
    app = ChatApp(cwd)
    app.run()
    print("hai")
