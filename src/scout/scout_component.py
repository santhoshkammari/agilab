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
) -> tuple[gr.Textbox, gr.Button, gr.Button, gr.Button, gr.Button]:
    """
    Create Scout-style textbox UI components using standard Gradio components.
    
    Returns:
        tuple: (textbox, context_button, send_button, mode_toggle, settings_button)
    """
    
    # CSS for Scout styling
    scout_css = """
    .scout-textbox-wrapper {
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
        border-radius: 18px !important;
        padding: 14px 18px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 
            0 2px 8px rgba(0, 0, 0, 0.12),
            0 6px 20px rgba(0, 0, 0, 0.06),
            0 1px 3px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    
    .scout-textbox-wrapper > * {
        border-radius: inherit !important;
    }
    
    .scout-textbox-wrapper .scout-main-input textarea,
    .scout-textbox-wrapper .scout-main-input input {
        border: none !important;
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        font-size: 16px;
        line-height: 1.5;
        resize: none;
        outline: none !important;
    }
    
    .scout-textbox-wrapper .scout-main-input {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    .scout-textbox-wrapper .scout-main-input {
        margin-bottom: 6px !important;
    }
    
    .scout-button-row {
        gap: 8px;
        align-items: center;
        margin: 0;
    }
    
    .scout-context-btn {
        background: rgba(0, 122, 255, 0.08) !important;
        border: 1px solid rgba(0, 122, 255, 0.12) !important;
        color: #007AFF !important;
        font-size: 14px;
        font-weight: 500;
        padding: 8px 14px !important;
        border-radius: 12px !important;
        transition: all 0.15s ease;
        height: auto !important;
        min-height: 32px !important;
        box-shadow: 0 1px 3px rgba(0, 122, 255, 0.08) !important;
    }
    
    .scout-context-btn:hover {
        background: rgba(0, 122, 255, 0.12) !important;
        border-color: rgba(0, 122, 255, 0.18) !important;
        transform: translateY(-0.5px);
    }
    
    .scout-send-btn {
        background: #007AFF !important;
        border: none !important;
        color: white !important;
        font-size: 18px;
        font-weight: 600;
        padding: 0 !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        min-height: 36px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease;
        box-shadow: 
            0 1px 3px rgba(0, 122, 255, 0.3),
            0 2px 6px rgba(0, 122, 255, 0.15) !important;
    }
    
    .scout-send-btn:hover {
        background: #0056CC !important;
        transform: scale(1.05);
        box-shadow: 
            0 2px 6px rgba(0, 122, 255, 0.4),
            0 4px 12px rgba(0, 122, 255, 0.2) !important;
    }
    
    .scout-mode-toggle {
        background: linear-gradient(135deg, #007AFF, #0056CC) !important;
        border: none !important;
        color: white !important;
        font-size: 14px;
        font-weight: 600;
        padding: 8px 16px !important;
        border-radius: 16px !important;
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        height: auto !important;
        min-height: 32px !important;
        min-width: 70px !important;
        box-shadow: 
            0 1px 3px rgba(0, 122, 255, 0.3),
            0 2px 8px rgba(0, 122, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .scout-mode-toggle::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), transparent);
        pointer-events: none;
    }
    
    .scout-mode-toggle:hover {
        transform: translateY(-1px);
        box-shadow: 
            0 2px 6px rgba(0, 122, 255, 0.4),
            0 4px 16px rgba(0, 122, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
    }
    
    .scout-mode-toggle[data-mode="Plan"] {
        background: rgba(120, 120, 128, 0.16) !important;
        color: rgba(60, 60, 67, 0.6) !important;
        box-shadow: 
            0 1px 3px rgba(0, 0, 0, 0.1),
            0 2px 8px rgba(0, 0, 0, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(30px);
    }
    
    .scout-mode-toggle[data-mode="Plan"]:hover {
        background: rgba(120, 120, 128, 0.24) !important;
        box-shadow: 
            0 2px 6px rgba(0, 0, 0, 0.15),
            0 4px 16px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
    }
    
    .scout-textbox-wrapper:focus-within {
        border: 1px solid rgba(0, 122, 255, 0.15) !important;
        box-shadow: 
            0 3px 12px rgba(0, 0, 0, 0.15),
            0 8px 25px rgba(0, 0, 0, 0.08),
            0 2px 6px rgba(0, 122, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Plan mode textbox styling - iOS glass inactive */
    .scout-textbox-wrapper[data-mode="Plan"] {
        background: rgba(242, 242, 247, 0.7) !important;
        border: 1px solid rgba(0, 0, 0, 0.04) !important;
        box-shadow: 
            0 2px 8px rgba(0, 0, 0, 0.1),
            0 6px 20px rgba(0, 0, 0, 0.05),
            0 1px 3px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(40px);
    }
    
    .scout-textbox-wrapper[data-mode="Plan"]:focus-within {
        background: rgba(242, 242, 247, 0.8) !important;
        box-shadow: 
            0 3px 12px rgba(0, 0, 0, 0.15),
            0 10px 30px rgba(0, 0, 0, 0.08),
            0 2px 6px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Context button in Plan mode */
    .scout-textbox-wrapper[data-mode="Plan"] .scout-context-btn {
        background: rgba(120, 120, 128, 0.12) !important;
        border: 1px solid rgba(120, 120, 128, 0.16) !important;
        color: rgba(60, 60, 67, 0.6) !important;
    }
    
    .scout-textbox-wrapper[data-mode="Plan"] .scout-context-btn:hover {
        background: rgba(120, 120, 128, 0.18) !important;
        border-color: rgba(120, 120, 128, 0.24) !important;
    }
    
    .scout-settings-btn {
        background: rgba(120, 120, 128, 0.12) !important;
        border: 1px solid rgba(120, 120, 128, 0.16) !important;
        color: rgba(60, 60, 67, 0.6) !important;
        font-size: 16px;
        font-weight: 500;
        padding: 0 !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease;
        box-shadow: 0 1px 3px rgba(120, 120, 128, 0.12) !important;
    }
    
    .scout-settings-btn:hover {
        background: rgba(120, 120, 128, 0.18) !important;
        border-color: rgba(120, 120, 128, 0.24) !important;
        transform: rotate(45deg);
    }
    
    /* Remove borders from all textbox and input elements */
    .gradio-textbox,
    .gradio-textbox textarea,
    .gradio-textbox input,
    div[data-testid="textbox"],
    div[data-testid="textbox"] textarea,
    div[data-testid="textbox"] input {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Remove borders from chatbot output and make assistant messages transparent */
    .gradio-chatbot,
    .gradio-chatbot > div,
    .gradio-chatbot .message,
    .gradio-chatbot .message-wrap,
    .gradio-chatbot .chatbot,
    .gradio-chatbot .panel,
    div[data-testid="chatbot"],
    div[data-testid="chatbot"] > div,
    div[data-testid="chatbot"] .message,
    div[data-testid="chatbot"] .message-wrap {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Make assistant messages transparent */
    .gradio-chatbot .message.bot,
    .gradio-chatbot .message-wrap.bot,
    div[data-testid="chatbot"] .message.bot,
    div[data-testid="chatbot"] .message-wrap.bot,
    .gradio-chatbot .message[data-role="assistant"],
    div[data-testid="chatbot"] .message[data-role="assistant"],
    .gradio-chatbot .bot,
    div[data-testid="chatbot"] .bot,
    .gradio-chatbot .message-row.bot,
    .gradio-chatbot .message-bubble.bot,
    div[data-testid="chatbot"] .message-row.bot,
    div[data-testid="chatbot"] .message-bubble.bot {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Remove borders from all containers and blocks */
    .gradio-container,
    .gradio-block,
    .block,
    .container {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Glass-style Task Cards for Workspace */
    .task-column {
        padding: 16px !important;
        min-height: 400px !important;
        max-height: 600px !important;
        overflow-y: auto !important;
    }
    
    .task-card.glass-card {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.08),
            0 4px 16px rgba(0, 0, 0, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    
    .task-card.glass-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.12),
            0 8px 24px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .task-card.completed {
        border-left: 4px solid #34C759 !important;
        background: rgba(52, 199, 89, 0.05) !important;
    }
    
    .task-card.ongoing {
        border-left: 4px solid #007AFF !important;
        background: rgba(0, 122, 255, 0.05) !important;
    }
    
    .task-card-header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 12px !important;
    }
    
    .task-status-icon {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    .task-session-id {
        font-family: 'SF Mono', Monaco, monospace !important;
        font-size: 12px !important;
        color: #8E8E93 !important;
        background: rgba(142, 142, 147, 0.12) !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
    }
    
    .task-card-content {
        margin-bottom: 16px !important;
    }
    
    .task-message {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #1D1D1F !important;
        margin-bottom: 12px !important;
        line-height: 1.4 !important;
    }
    
    .task-details {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
    }
    
    .task-details > div {
        font-size: 12px !important;
        color: #8E8E93 !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    
    .task-card-actions {
        display: flex !important;
        gap: 8px !important;
        justify-content: flex-end !important;
    }
    
    .task-card-action {
        background: rgba(0, 122, 255, 0.1) !important;
        border: 1px solid rgba(0, 122, 255, 0.2) !important;
        color: #007AFF !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    .task-card-action:hover {
        background: rgba(0, 122, 255, 0.15) !important;
        border-color: rgba(0, 122, 255, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    .task-card-action.stop-btn {
        background: rgba(255, 59, 48, 0.1) !important;
        border-color: rgba(255, 59, 48, 0.2) !important;
        color: #FF3B30 !important;
    }
    
    .task-card-action.stop-btn:hover {
        background: rgba(255, 59, 48, 0.15) !important;
        border-color: rgba(255, 59, 48, 0.3) !important;
    }
    
    .empty-column, .error-column {
        text-align: center !important;
        padding: 32px 16px !important;
        color: #8E8E93 !important;
        font-size: 14px !important;
        font-style: italic !important;
    }
    """
    with gr.Row():
        gr.Markdown()
        with gr.Column(elem_classes=["scout-textbox-wrapper"],scale=3):
            # Main textbox
            textbox = gr.Textbox(
                placeholder=placeholder,
                show_label=False,
                container=False,
                lines=1,
                max_lines=2,
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
                
                mode_toggle = gr.Button(
                    "Scout",
                    variant="secondary",
                    size="sm",
                    elem_classes=["scout-mode-toggle"],
                    scale=0,
                )
                
                gr.Markdown()
                
                settings_button = gr.Button(
                    "⚙️",
                    variant="secondary",
                    size="sm",
                    elem_classes=["scout-settings-btn"],
                    scale=0,
                )

                send_button = gr.Button(
                    "↑",
                    variant="primary",
                    size="sm",
                    elem_classes=["scout-send-btn"],
                    scale=0,
                )

        gr.Markdown()

    return textbox, context_button, send_button, mode_toggle, settings_button, scout_css
