"""
Custom Scout-style textbox component with context button and send button.
"""
import gradio as gr
from typing import Callable, Any


class ScoutTextbox:
    """Custom textbox component styled like Scout.new website"""
    
    def __init__(
        self,
        placeholder: str = "Type your message here...",
        value: str = "",
        lines: int = 1,
        max_lines: int = 3,
        context_button_text: str = "@ Add context",
        send_button_icon: str = "↑",
        on_submit: Callable[[str], Any] | None = None,
        on_context: Callable[[], Any] | None = None,
    ):
        self.placeholder = placeholder
        self.value = value
        self.lines = lines
        self.max_lines = max_lines
        self.context_button_text = context_button_text
        self.send_button_icon = send_button_icon
        self.on_submit = on_submit
        self.on_context = on_context
        
        # Create the component structure
        self._create_components()
    
    def _create_components(self):
        """Create the internal Gradio components"""
        with gr.Column(elem_classes=["scout-textbox-container"]) as self.container:
            # Main textbox row
            self.textbox = gr.Textbox(
                value=self.value,
                placeholder=self.placeholder,
                lines=self.lines,
                max_lines=self.max_lines,
                container=False,
                show_label=False,
                elem_classes=["scout-main-textbox"],
                interactive=True,
            )
            
            # Button row
            with gr.Row(elem_classes=["scout-button-row"]):
                with gr.Column(scale=1):
                    self.context_button = gr.Button(
                        value=self.context_button_text,
                        variant="secondary",
                        size="sm",
                        elem_classes=["scout-context-button"],
                    )
                
                with gr.Column(scale=0, min_width=50):
                    self.send_button = gr.Button(
                        value=self.send_button_icon,
                        variant="primary",
                        size="sm",
                        elem_classes=["scout-send-button"],
                    )
        
        # Set up event handlers
        if self.on_submit:
            self.send_button.click(
                fn=self._handle_submit,
                inputs=[self.textbox],
                outputs=[self.textbox],
            )
            self.textbox.submit(
                fn=self._handle_submit,
                inputs=[self.textbox],
                outputs=[self.textbox],
            )
        
        if self.on_context:
            self.context_button.click(
                fn=self.on_context,
                inputs=[],
                outputs=[],
            )
    
    def _handle_submit(self, text: str):
        """Handle submit action"""
        if self.on_submit and text.strip():
            self.on_submit(text)
            return ""  # Clear textbox after submit
        return text
    
    def get_css(self) -> str:
        """Return custom CSS for Scout-style appearance"""
        return """
        .scout-textbox-container {
            border: 1px solid #F2F4F6;
            border-radius: 12px;
            padding: 12px;
            background: #FFFFFF;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .scout-main-textbox textarea {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            font-size: 16px;
            line-height: 1.5;
            resize: none;
        }
        
        .scout-main-textbox textarea:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        
        .scout-button-row {
            margin-top: 8px;
            gap: 8px;
            align-items: center;
        }
        
        .scout-context-button {
            background: #EAF6FF !important;
            border: 1px solid #EAF6FF !important;
            color: #57BAFF !important;
            font-size: 14px;
            font-weight: 500;
            padding: 6px 12px;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        
        .scout-context-button:hover {
            background: #D5EFFF !important;
            border-color: #D5EFFF !important;
        }
        
        .scout-send-button {
            background: #57BAFF !important;
            border: 1px solid #57BAFF !important;
            color: white !important;
            font-size: 16px;
            font-weight: 600;
            padding: 8px 12px;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            min-width: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }
        
        .scout-send-button:hover {
            background: #4AABF0 !important;
            border-color: #4AABF0 !important;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(87, 186, 255, 0.3);
        }
        
        .scout-textbox-container:focus-within {
            border-color: #57BAFF;
            box-shadow: 0 0 0 3px rgba(87, 186, 255, 0.1);
        }
        """


def create_scout_textbox(
    placeholder: str = "Type your message here...",
    on_submit: Callable[[str], Any] | None = None,
    on_context: Callable[[], Any] | None = None,
) -> ScoutTextbox:
    """
    Create a Scout-style textbox component.
    
    Args:
        placeholder: Placeholder text for the textbox
        on_submit: Function to call when message is submitted
        on_context: Function to call when context button is clicked
    
    Returns:
        ScoutTextbox instance
    """
    return ScoutTextbox(
        placeholder=placeholder,
        on_submit=on_submit,
        on_context=on_context,
    )