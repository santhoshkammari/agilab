"""
Custom Scout-style textbox component inheriting from Gradio Component class.
"""
import gradio as gr
from gradio.components.base import Component
from gradio.events import Events
from typing import Any, Callable, Literal


class ScoutTextbox(Component):
    """
    Custom textbox component styled like Scout.new website with context button and send button.
    """
    
    EVENTS = [
        Events.change,
        Events.input,
        Events.select,
        Events.submit,
        Events.focus,
        Events.blur,
        Events.stop,
        Events.copy,
    ]
    
    def __init__(
        self,
        value: str = "",
        *,
        placeholder: str = "Type your message here...",
        lines: int = 1,
        max_lines: int = 3,
        context_button_text: str = "@ Add context",
        send_button_icon: str = "↑",
        label: str | None = None,
        info: str | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
        preserved_by_key: list[str] | str | None = "value",
    ):
        self.placeholder = placeholder
        self.lines = lines
        self.max_lines = max_lines
        self.context_button_text = context_button_text
        self.send_button_icon = send_button_icon
        
        if elem_classes is None:
            elem_classes = []
        elif isinstance(elem_classes, str):
            elem_classes = [elem_classes]
        
        elem_classes.extend(["scout-textbox-component"])
        
        super().__init__(
            value=value,
            label=label,
            info=info,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
            preserved_by_key=preserved_by_key,
        )
    
    def preprocess(self, payload: str | None) -> str | None:
        """
        Parameters:
            payload: the text entered in the textarea.
        Returns:
            Passes text value as a str into the function.
        """
        return None if payload is None else str(payload)
    
    def postprocess(self, value: str | None) -> str | None:
        """
        Parameters:
            value: Expects a str returned from function and sets textarea value to it.
        Returns:
            The value to display in the textarea.
        """
        return None if value is None else str(value)
    
    def api_info(self) -> dict[str, Any]:
        return {"type": "string"}
    
    def example_payload(self) -> Any:
        return "Hello!!"
    
    def example_value(self) -> Any:
        return "Hello!!"
    
    def example_inputs(self) -> Any:
        return "Hello!!"
    
    def flag(self, payload: Any, flag_dir: str = "") -> str:
        return str(payload) if payload is not None else ""
    
    def read_from_flag(self, payload: Any) -> str:
        return str(payload) if payload is not None else ""
    
    @property
    def skip_api(self) -> bool:
        return False
    
    def process_example(self, value):
        return value
    
    def get_config(self):
        """Get component configuration for frontend"""
        config = super().get_config()
        config.update({
            "placeholder": self.placeholder,
            "lines": self.lines,
            "max_lines": self.max_lines,
            "context_button_text": self.context_button_text,
            "send_button_icon": self.send_button_icon,
            "component_type": "scout_textbox",
        })
        return config


def create_scout_textbox_ui(
    placeholder: str = "Type your message here...",
    context_handler: Callable = None,
    send_handler: Callable = None,
) -> tuple[gr.Textbox, gr.Button, gr.Button]:
    """
    Create Scout-style textbox UI components using standard Gradio components.
    
    Returns:
        tuple: (textbox, context_button, send_button)
    """
    
    # CSS for Scout styling
    scout_css = """
    .scout-textbox-wrapper {
        border: 1px solid #F2F4F6;
        border-radius: 12px;
        padding: 12px;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 0;
    }
    
    .scout-textbox-wrapper .scout-main-input textarea {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        font-size: 16px;
        line-height: 1.5;
        resize: none;
        outline: none !important;
    }
    
    .scout-textbox-wrapper .scout-main-input {
        margin-bottom: 8px !important;
    }
    
    .scout-button-row {
        gap: 8px;
        align-items: center;
        margin: 0;
    }
    
    .scout-context-btn {
        background: #EAF6FF !important;
        border: 1px solid #EAF6FF !important;
        color: #57BAFF !important;
        font-size: 14px;
        font-weight: 500;
        padding: 6px 12px !important;
        border-radius: 10px !important;
        transition: all 0.2s ease;
        height: auto !important;
        min-height: 32px !important;
    }
    
    .scout-context-btn:hover {
        background: #D5EFFF !important;
        border-color: #D5EFFF !important;
    }
    
    .scout-send-btn {
        background: #57BAFF !important;
        border: 1px solid #57BAFF !importanttart chatting with the AI assistant...;
        color: white !important;
        font-size: 16px;
        font-weight: 600;
        padding: 8px !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        min-height: 40px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
    }
    
    .scout-send-btn:hover {
        background: #4AABF0 !important;
        border-color: #4AABF0 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(87, 186, 255, 0.3);
    }
    
    .scout-textbox-wrapper:focus-within {
        border-color: #57BAFF;
        box-shadow: 0 0 0 3px rgba(87, 186, 255, 0.1);
    }
    """
    
    with gr.Column(elem_classes=["scout-textbox-wrapper"],):
        # Main textbox
        textbox = gr.Textbox(
            placeholder=placeholder,
            lines=1,
            max_lines=3,
            container=False,
            show_label=False,
            elem_classes=["scout-main-input"],
            interactive=True,
        )
        
        # Button row
        with gr.Row(elem_classes=["scout-button-row"]):
            context_button = gr.Button(
                "@ Add context",
                variant="secondary",
                size="sm",
                elem_classes=["scout-context-btn"],
                scale=0,
            )
            gr.Markdown()
            gr.Markdown()

            send_button = gr.Button(
                "↑",
                variant="primary",
                size="sm",
                elem_classes=["scout-send-btn"],
                scale=0,
                min_width=40,
            )
    
    return textbox, context_button, send_button, scout_css