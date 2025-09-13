from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.containers import VSplit, HSplit, Window, ScrollOffsets
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.application import get_app
from datetime import datetime
import os

class ChatBot:
    def __init__(self):
        self.messages = []
        self.input_buffer = Buffer()
        self.setup_keybindings()
        self.setup_layout()
        
    def add_message(self, sender, message):
        timestamp = datetime.now().strftime("%H:%M")
        self.messages.append(f"[{timestamp}] {sender}: {message}")
        self.update_chat_display()
        
    def get_chat_text(self):
        if not self.messages:
            return FormattedText([
                ('class:welcome', '🤖 Welcome to the ChatBot!\n'),
                ('class:instruction', 'Type your message and press Enter to send.\n'),
                ('class:instruction', 'Press Ctrl+Q to quit.\n')
            ])
        
        formatted_messages = []
        # Show last 20 messages
        recent_messages = self.messages[-20:] if len(self.messages) > 20 else self.messages
        
        for msg in recent_messages:
            if "User:" in msg:
                formatted_messages.append(('class:user', msg + '\n'))
            else:
                formatted_messages.append(('class:bot', msg + '\n'))
        
        # Add extra newline at the end to prevent the gap issue
        if formatted_messages:
            formatted_messages.append(('', '\n'))
        
        return FormattedText(formatted_messages)
    
    def update_chat_display(self):
        # Update the buffer text and move cursor to end
        if hasattr(self, 'chat_buffer'):
            # Create text from messages - simple approach
            text_lines = []
            recent_messages = self.messages[-50:] if len(self.messages) > 50 else self.messages
            
            for msg in recent_messages:
                text_lines.append(msg)
            
            # Set buffer text and move cursor to end
            self.chat_buffer.text = '\n'.join(text_lines)
            # Move cursor to end to show latest messages at bottom
            self.chat_buffer.cursor_position = len(self.chat_buffer.text)
        else:
            self.chat_control.text = self.get_chat_text()
        
    def process_input(self):
        user_input = self.input_buffer.text.strip()
        if user_input:
            self.add_message("User", user_input)
            # Simple bot response
            bot_response = self.generate_response(user_input)
            self.add_message("Bot", bot_response)
            self.input_buffer.text = ""
            
    def generate_response(self, user_input):
        user_input_lower = user_input.lower()
        if "hello" in user_input_lower or "hi" in user_input_lower:
            return "Hello! How can I help you today?"
        elif "how are you" in user_input_lower:
            return "I'm doing great! Thanks for asking."
        elif "bye" in user_input_lower or "goodbye" in user_input_lower:
            return "Goodbye! Have a great day!"
        elif "help" in user_input_lower:
            return "I'm here to chat! Try asking me about anything or just say hello."
        else:
            return f"You said: '{user_input}'. That's interesting! Tell me more."
    
    def setup_keybindings(self):
        self.kb = KeyBindings()
        
        @self.kb.add('c-q')
        def exit_(event):
            event.app.exit()
            
        @self.kb.add('enter')
        def process_message(event):
            self.process_input()
    
    def setup_layout(self):
        self.chat_control = FormattedTextControl(
            text=self.get_chat_text(),
            show_cursor=False
        )
        
        # Chat history area with buffer to show messages from bottom
        self.chat_buffer = Buffer(multiline=True)
        
        self.chat_window = Window(
            content=BufferControl(
                buffer=self.chat_buffer,
                focusable=False
            ),
            wrap_lines=True,
            always_hide_cursor=True,
            dont_extend_height=False,
            dont_extend_width=False
        )
        
        # Input area
        input_window = Window(
            content=BufferControl(
                buffer=self.input_buffer,
                input_processors=[],
            ),
            height=1,
            wrap_lines=True,
            dont_extend_height=True
        )
        
        # Input label
        input_label = Window(
            content=FormattedTextControl(
                text=FormattedText([('class:input-label', '> ')])
            ),
            width=2,
            dont_extend_height=True
        )
        
        # Status bar
        status_bar = Window(
            content=FormattedTextControl(
                text=FormattedText([
                    ('class:status', ' ChatBot v1.0 | Ctrl+Q to quit | Enter to send ')
                ])
            ),
            height=1,
            dont_extend_height=True
        )
        
        # Input container
        input_container = VSplit([
            input_label,
            input_window
        ])
        
        # Main container
        root_container = HSplit([
            self.chat_window,
            Window(height=1, char='-', dont_extend_height=True),  # separator
            input_container,
            status_bar
        ])
        
        self.layout = Layout(root_container)
    
    def run(self):
        # Add welcome message
        self.add_message("Bot", "Hello! I'm your friendly chatbot. How can I help you today?")
        
        # Style configuration
        from prompt_toolkit.styles import Style
        style = Style.from_dict({
            'user': '#00aa00 bold',      # Green for user messages
            'bot': '#0088ff',            # Blue for bot messages  
            'welcome': '#ffaa00 bold',   # Orange for welcome
            'instruction': '#888888',    # Gray for instructions
            'input-label': '#00aa00 bold', # Green for input prompt
            'status': 'bg:#444444 #ffffff'  # Status bar styling
        })
        
        app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            full_screen=True,
            style=style
        )
        
        app.run()

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run()
