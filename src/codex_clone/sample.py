"""
A terminal-based chatbot application built with prompt_toolkit.

Features:
- Multi-line input buffer with autocomplete support for `/model` command.
- Chat history display with styled formatting for user, bot, and system messages.
- Switch between multiple AI models (`Sonnet`, `Gpt-5`, `Qwen`, `LLama`) using `/model` command
  or interactive menu.
- Key bindings:
  * Ctrl+C: Quit application
  * Enter: Send message (or confirm model selection in menu)
  * Ctrl+J: Insert newline in input
  * Up/Down: Navigate model selection menu
  * Escape: Exit model menu
- Interactive model selection menu with descriptions for each model.
- Status bar showing available shortcuts.

Fixes applied:
1. Removed cursor in chat history by replacing `BufferControl` with `FormattedTextControl`.
2. Updated user message and input prompt color to a softer green (`#34D399`).

Usage:
Run the script directly to start the chatbot CLI:
    python chatbot.py
"""

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.containers import VSplit, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

class ModelCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith('/model'):
            models = ['Sonnet', 'Gpt-5', 'Qwen', 'LLama']
            word = text[6:].strip()  # Remove '/model'
            for model in models:
                if model.lower().startswith(word.lower()):
                    yield Completion(model, start_position=-len(word))

class ChatBot:
    def __init__(self):
        self.messages = []
        self.current_model = 'Sonnet'  # Default model
        self.models = ['Sonnet', 'Gpt-5', 'Qwen', 'LLama']
        self.show_model_menu = False
        self.selected_model_index = 0
        self.input_buffer = Buffer(
            multiline=True,
            completer=ModelCompleter(),
            complete_while_typing=True
        )
        self.setup_keybindings()
        self.setup_layout()

    def add_message(self, sender, message):
        if sender == "User":
            self.messages.append(f"▊ {message}")
        else:
            self.messages.append(message)
        self.update_chat_display()

    def get_chat_text(self):
        if not self.messages:
            return FormattedText([
                ('class:instruction', 'Welcome to ChatBot\n'),
                ('class:instruction', 'Type your message and press Enter to send\n'),
                ('class:instruction', 'Press Ctrl+Q to quit\n')
            ])

        formatted_messages = []
        recent_messages = self.messages[-20:] if len(self.messages) > 20 else self.messages
        for msg in recent_messages:
            if msg.startswith("▊ "):
                formatted_messages.append(('class:user', msg + '\n'))
            elif msg.startswith("System: ") or msg.startswith("Switched to ") or msg.startswith("Current model: ") or msg.startswith("Unknown model: "):
                formatted_messages.append(('class:system', msg + '\n'))
            else:
                formatted_messages.append(('class:bot', msg + '\n'))
        return FormattedText(formatted_messages)

    def update_chat_display(self):
        if hasattr(self, 'chat_control'):
            self.chat_control.text = self.get_chat_text()

    def process_input(self):
        user_input = self.input_buffer.text.strip()
        if user_input:
            if user_input.startswith('/model '):
                model_name = user_input[7:].strip()
                if model_name in self.models:
                    self.current_model = model_name
                    self.add_message("System", f"Switched to {model_name} model")
                else:
                    self.add_message("System", f"Unknown model: {model_name}. Available: {', '.join(self.models)}")
            elif user_input == '/model':
                self.show_model_menu = True
                self.selected_model_index = 0
                self.update_layout()
                return
            else:
                self.add_message("User", user_input)
                bot_response = self.generate_response(user_input)
                self.add_message("Bot", bot_response)
            self.input_buffer.text = ""

    def generate_response(self, user_input):
        user_input_lower = user_input.lower()
        if "hello" in user_input_lower or "hi" in user_input_lower:
            return f"[{self.current_model}] Hello! How can I help you today?"
        elif "how are you" in user_input_lower:
            return f"[{self.current_model}] I'm doing great! Thanks for asking."
        elif "bye" in user_input_lower or "goodbye" in user_input_lower:
            return f"[{self.current_model}] Goodbye! Have a great day!"
        elif "help" in user_input_lower:
            return f"[{self.current_model}] I'm here to chat! Try /model to switch models or just say hello."
        else:
            return f"[{self.current_model}] You said: '{user_input}'. That's interesting! Tell me more."

    def setup_keybindings(self):
        self.kb = KeyBindings()

        @self.kb.add('c-c')
        def exit_(event):
            event.app.exit()

        @self.kb.add('enter')
        def process_message(event):
            if self.show_model_menu:
                selected_model = self.models[self.selected_model_index]
                old_model = self.current_model
                self.current_model = selected_model
                self.add_message("System", f"Model updated: {old_model} → {selected_model}")
                self.show_model_menu = False
                self.input_buffer.text = ""
                self.update_layout()
                event.app.layout.focus(self.input_buffer)
            else:
                self.process_input()

        @self.kb.add('c-j')
        def newline(event):
            self.input_buffer.insert_text('\n')

        @self.kb.add('up')
        def move_up(event):
            if self.show_model_menu:
                self.selected_model_index = (self.selected_model_index - 1) % len(self.models)
                self.update_layout()

        @self.kb.add('down')
        def move_down(event):
            if self.show_model_menu:
                self.selected_model_index = (self.selected_model_index + 1) % len(self.models)
                self.update_layout()

        @self.kb.add('escape')
        def hide_menu(event):
            if self.show_model_menu:
                self.show_model_menu = False
                self.input_buffer.text = ""
                self.update_layout()
                event.app.layout.focus(self.input_buffer)

    def get_model_menu_text(self):
        if not self.show_model_menu:
            return FormattedText([])

        menu_items = []
        menu_items.append(('class:menu-header', 'Select model and reasoning level\n'))
        menu_items.append(('class:menu-subtitle', 'Switch between OpenAI models for this and future Codex CLI session\n\n'))

        for i, model in enumerate(self.models):
            if i == self.selected_model_index:
                menu_items.append(('class:menu-selected', f'  {i+1}. {model} '))
                if model == 'Sonnet':
                    menu_items.append(('class:menu-description', '— fastest responses with limited reasoning; ideal for coding, instructions, or lightweight tasks\n'))
                elif model == 'Gpt-5':
                    menu_items.append(('class:menu-description', '— balances speed with some reasoning; useful for straightforward queries and short explanations\n'))
                elif model == 'Qwen':
                    menu_items.append(('class:menu-description', '— default setting; provides a solid balance of reasoning depth and latency for general-purpose tasks\n'))
                elif model == 'LLama':
                    menu_items.append(('class:menu-description', '— maximizes reasoning depth for complex or ambiguous problems\n'))
            else:
                menu_items.append(('class:menu-item', f'  {i+1}. {model}\n'))

        menu_items.append(('class:menu-footer', '\nPress Enter to confirm or Esc to go back'))
        return FormattedText(menu_items)

    def setup_layout(self):
        self.chat_control = FormattedTextControl(
            text=lambda: self.get_chat_text(),
            show_cursor=False
        )

        self.chat_window = Window(
            content=self.chat_control,
            wrap_lines=True,
            always_hide_cursor=True
        )

        self.model_menu_control = FormattedTextControl(
            text=self.get_model_menu_text(),
            show_cursor=False
        )

        self.model_menu_window = Window(
            content=self.model_menu_control,
            wrap_lines=True,
            dont_extend_height=True,
            dont_extend_width=True
        )

        input_prompt = Window(
            content=FormattedTextControl(
                text=FormattedText([('class:input-prompt', '▊ ')])
            ),
            width=2,
            dont_extend_height=True
        )

        input_window = Window(
            content=BufferControl(
                buffer=self.input_buffer,
                input_processors=[]
            ),
            height=Dimension(min=1, max=4),
            wrap_lines=True,
            dont_extend_height=True
        )

        input_area = VSplit([
            input_prompt,
            input_window
        ])

        status_bar = Window(
            content=FormattedTextControl(
                text=FormattedText([
                    ('class:status-text', ' '),
                    ('class:status-key', 'send'),
                    ('class:status-text', '   '),
                    ('class:status-shortcut', 'Ctrl+J'),
                    ('class:status-text', ' newline   '),
                    ('class:status-shortcut', 'Ctrl+T'),
                    ('class:status-text', ' transcript   '),
                    ('class:status-shortcut', 'Ctrl+C'),
                    ('class:status-text', ' quit')
                ])
            ),
            height=1,
            dont_extend_height=True
        )

        if self.show_model_menu:
            root_container = HSplit([
                self.model_menu_window,
                status_bar
            ])
        else:
            root_container = HSplit([
                self.chat_window,
                Window(height=1, char='-', dont_extend_height=True),
                input_area,
                status_bar
            ])

        self.layout = Layout(root_container)

    def update_layout(self):
        self.model_menu_control.text = self.get_model_menu_text()
        self.setup_layout()
        if hasattr(self, 'app'):
            self.app.layout = self.layout

    def run(self):
        self.add_message("Bot", "Hello! How can I help you today?")

        from prompt_toolkit.styles import Style
        style = Style.from_dict({
            'user': '#34D399',              # softer green for user messages
            'input-prompt': '#34D399',      # same green for ▊ prompt
            'bot': '#ffffff',
            'system': '#ffaa00',
            'instruction': '#888888',
            'status-text': 'bg:#2d2d2d #ffffff',
            'status-key': 'bg:#2d2d2d #ffffff bold',
            'status-shortcut': 'bg:#2d2d2d #888888',
            'menu-header': '#ffffff bold',
            'menu-subtitle': '#888888',
            'menu-item': '#ffffff',
            'menu-selected': '#000000 bg:#ffffff',
            'menu-description': '#888888',
            'menu-footer': '#888888',
        })

        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            full_screen=True,
            style=style
        )

        self.app.layout.focus(self.input_buffer)
        self.app.run()

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run()