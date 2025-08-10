
import gradio as gr
import sys
import os

# Add the parent directory to sys.path to import HuggingFaceLLM
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../llm'))
sys.path.insert(0, "/home/ntlpt59/master/own/flowgen/flowgen/llm")

from huggingface_main import HuggingFaceLLM

# Initialize the HuggingFace LLM
llm = HuggingFaceLLM(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    device=None
)

# Anthropic-style theme colors
theme_colors = {
    "primary": "#000000",        # Anthropic black
    "secondary": "#ffffff",      # Clean white
    "accent": "#CD7F32",         # Anthropic orange/copper
    "background": "#ffffff",     # Clean white background
    "surface": "#f8f8f8",        # Light gray surface
    "border": "#e5e5e5",         # Light border
    "text": "#000000",           # Black text
    "text_secondary": "#666666", # Gray text
    "chat_user": "#000000",      # User message background
    "chat_assistant": "#f8f8f8"  # Assistant message background
}

def chat_with_llm(message, history):
    """Chat function that handles conversation with HuggingFace LLM"""
    try:
        # Convert Gradio history format to LLM message format
        messages = []
        
        # Add conversation history
        if history:
            for human, ai in history:
                if human:
                    messages.append({"role": "user", "content": human})
                if ai:
                    messages.append({"role": "assistant", "content": ai})
        
        # Add current message
        if message:
            messages.append({"role": "user", "content": message})
        
        # Get response from LLM
        response = llm(messages)
        return response.get('content', 'Sorry, I could not generate a response.')
        
    except Exception as e:
        return f"Error: {str(e)}"

def chat_with_streaming(message, history):
    """Streaming chat function"""
    try:
        # Convert history format
        messages = []
        if history:
            for human, ai in history:
                if human:
                    messages.append({"role": "user", "content": human})
                if ai:
                    messages.append({"role": "assistant", "content": ai})
        
        if message:
            messages.append({"role": "user", "content": message})
        
        # Get streaming response
        stream_response = llm(messages, stream=True)
        
        # Handle streaming
        partial_message = ""
        if hasattr(stream_response, '__iter__'):
            for chunk in stream_response:
                if chunk and 'content' in chunk:
                    partial_message += chunk['content']
                    yield partial_message
        else:
            # Fallback for non-streaming
            response = llm(messages)
            yield response.get('content', 'Sorry, I could not generate a response.')
            
    except Exception as e:
        yield f"Error: {str(e)}"

# Create custom CSS with Anthropic-inspired styling
custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    --anthropic-black: {theme_colors['primary']};
    --anthropic-white: {theme_colors['secondary']};
    --anthropic-orange: {theme_colors['accent']};
    --anthropic-surface: {theme_colors['surface']};
    --anthropic-border: {theme_colors['border']};
}}

/* Global container styling */
.gradio-container {{
    background-color: {theme_colors['background']} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}}

/* Header styling */
.header-title {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    color: {theme_colors['text']} !important;
    margin: 2rem 0 1rem 0 !important;
    letter-spacing: -0.02em !important;
}}

.subheader {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    font-size: 1.1rem !important;
    color: {theme_colors['text_secondary']} !important;
    margin-bottom: 2rem !important;
}}

/* Chat container */
.chat-container {{
    border: 1px solid {theme_colors['border']} !important;
    border-radius: 12px !important;
    background: {theme_colors['background']} !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}}

/* Chat messages */
.message {{
    border-radius: 8px !important;
    margin: 8px 0 !important;
    padding: 12px 16px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}}

/* User messages */
.user.message {{
    background-color: {theme_colors['chat_user']} !important;
    color: {theme_colors['secondary']} !important;
    margin-left: 20% !important;
}}

/* Assistant messages */
.bot.message {{
    background-color: {theme_colors['chat_assistant']} !important;
    color: {theme_colors['text']} !important;
    margin-right: 20% !important;
    border: 1px solid {theme_colors['border']} !important;
}}

/* Input styling */
.input-container {{
    border: 1px solid {theme_colors['border']} !important;
    border-radius: 8px !important;
    background: {theme_colors['background']} !important;
    transition: border-color 0.2s ease !important;
}}

.input-container:focus-within {{
    border-color: {theme_colors['accent']} !important;
    box-shadow: 0 0 0 1px {theme_colors['accent']} !important;
}}

.input-box textarea {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    border: none !important;
    background: transparent !important;
    resize: none !important;
    padding: 12px 16px !important;
}}

.input-box textarea::placeholder {{
    color: {theme_colors['text_secondary']} !important;
    font-style: normal !important;
}}

/* Button styling */
.send-button {{
    background-color: {theme_colors['primary']} !important;
    color: {theme_colors['secondary']} !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}}

.send-button:hover {{
    background-color: {theme_colors['text_secondary']} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
}}

/* Secondary buttons */
.secondary-button {{
    background-color: {theme_colors['background']} !important;
    color: {theme_colors['text']} !important;
    border: 1px solid {theme_colors['border']} !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}}

.secondary-button:hover {{
    background-color: {theme_colors['surface']} !important;
    border-color: {theme_colors['accent']} !important;
}}

/* Controls section */
.controls-section {{
    margin-top: 1rem !important;
    padding: 1rem 0 !important;
    border-top: 1px solid {theme_colors['border']} !important;
}}

/* Examples styling */
.examples-container {{
    background-color: {theme_colors['surface']} !important;
    border: 1px solid {theme_colors['border']} !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    margin-top: 2rem !important;
}}

.examples-title {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: {theme_colors['text']} !important;
    margin-bottom: 1rem !important;
}}

.example-button {{
    background-color: {theme_colors['background']} !important;
    color: {theme_colors['text']} !important;
    border: 1px solid {theme_colors['border']} !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    margin: 4px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}}

.example-button:hover {{
    background-color: {theme_colors['accent']} !important;
    color: {theme_colors['secondary']} !important;
    border-color: {theme_colors['accent']} !important;
}}

/* Checkbox and toggle styling */
.checkbox-container {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: {theme_colors['text']} !important;
}}

/* Status indicator */
.status-indicator {{
    display: inline-block !important;
    width: 8px !important;
    height: 8px !important;
    background-color: {theme_colors['accent']} !important;
    border-radius: 50% !important;
    margin-right: 8px !important;
}}
"""

# Create the Gradio interface with Anthropic styling
with gr.Blocks(css=custom_css, title="Claude-Style Chat", theme=gr.themes.Base()) as demo:
    # Header section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span class="status-indicator"></span>
                    <h1 class="header-title">ANTHROPIC</h1>
                </div>
            """)
            gr.Markdown(
                f"**Build with Claude** • Powered by {llm._model_name}",
                elem_classes="subheader"
            )
    
    # Main chat interface
    with gr.Column(elem_classes="chat-section"):
        chatbot = gr.Chatbot(
            height=450,
            bubble_full_width=False,
            show_copy_button=True,
            avatar_images=None,
            elem_classes="chat-container"
        )
        
        # Input section with Anthropic styling
        with gr.Row(elem_classes="input-section"):
            with gr.Column(scale=5, elem_classes="input-container"):
                msg = gr.Textbox(
                    placeholder="Ask Claude anything...",
                    show_label=False,
                    lines=2,
                    max_lines=4,
                    elem_classes="input-box"
                )
            with gr.Column(scale=1, min_width=100):
                submit_btn = gr.Button(
                    "Send", 
                    variant="primary",
                    elem_classes="send-button"
                )
        
        # Controls section
        with gr.Row(elem_classes="controls-section"):
            with gr.Column(scale=1):
                clear_btn = gr.Button(
                    "New conversation", 
                    variant="secondary",
                    elem_classes="secondary-button"
                )
            with gr.Column(scale=1):
                streaming_toggle = gr.Checkbox(
                    label="🔄 Enable streaming", 
                    value=True,
                    elem_classes="checkbox-container"
                )
    
    # Handle message submission
    def respond(message, history, use_streaming):
        if use_streaming:
            # Use streaming response
            bot_message = ""
            for partial in chat_with_streaming(message, history):
                bot_message = partial
                yield history + [(message, bot_message)], ""
        else:
            # Use regular response
            bot_message = chat_with_llm(message, history)
            yield history + [(message, bot_message)], ""
    
    # Event handlers
    msg.submit(respond, [msg, chatbot, streaming_toggle], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot, streaming_toggle], [chatbot, msg])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    # Examples section with Anthropic styling
    with gr.Column(elem_classes="examples-container"):
        gr.HTML('<div class="examples-title">Get started</div>')
        gr.Examples(
            examples=[
                "Help me code a Python function",
                "Explain machine learning concepts", 
                "Write and review my code",
                "Brainstorm creative solutions"
            ],
            inputs=msg,
            label=None,
            elem_id="examples-grid"
        )
        
        # Footer with model info
        gr.HTML(f"""
            <div style="margin-top: 2rem; padding: 1rem; border-top: 1px solid var(--anthropic-border); 
                        font-family: 'Inter', sans-serif; font-size: 0.8rem; color: var(--anthropic-border);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>Model: <strong>{llm._model_name}</strong></span>
                    <span>🔒 Secure • 🚀 Fast • 🧠 Intelligent</span>
                </div>
            </div>
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
