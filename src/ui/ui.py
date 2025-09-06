# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio",
#     "openai",
#     "requests",
# ]

# ///
import argparse
import re
import requests
import json

import gradio as gr
from .css import theme
from .scout_component import create_scout_textbox_ui
import sys
import os


def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is 22°{unit[0].upper()}, partly cloudy with light breeze."

def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b


def fetch_models(endpoint):
    """Fetch available models from the /models endpoint."""
    try:
        base_url = f"http://{endpoint}" if not endpoint.startswith("http") else endpoint
        
        response = requests.get(f"{base_url}/models", timeout=5)
        response.raise_for_status()
        
        models_data = response.json()
        if "data" in models_data:
            return [model["id"] for model in models_data["data"]]
        return []
    except Exception:
        return []


def update_models(endpoint):
    """Update model choices based on endpoint."""
    models = fetch_models(endpoint)
    if models:
        return gr.Dropdown(choices=models, value=models[0])
    else:
        return gr.Dropdown(choices=["Qwen/Qwen3-8B"], value="Qwen/Qwen3-8B")


def chat_function(message, history, endpoint, model_name, key, temperature, top_p, max_tokens, think_start_token):
    """
    Chat function using direct API calls for streaming.
    """
    base_url = f"http://{endpoint}" if not endpoint.startswith("http") else endpoint

    # Convert history to messages format
    messages = []
    for msg in history:
        if msg["role"] in ["user", "assistant"]:
            msg_content = msg["content"]
            if "</think>" in msg_content:
                msg_content = msg_content.split("</think>")[-1]
            messages.append({"role": msg["role"], "content": msg_content})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Prepare tools for the API
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get weather for"},
                        "unit": {"type": "string", "description": "Temperature unit (celsius or fahrenheit)", "default": "celsius"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "calculate_sum",
                "description": "Calculate the sum of two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ]
    
    try:
        # Make request to your API
        chat_request = {
            "messages": messages,
            "tools": tools,
            "options": {
                "stream": True,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": int(max_tokens)
            }
        }
        
        response = requests.post(
            f"{base_url}/chat",
            json=chat_request,
            stream=True,
            timeout=30
        )
        
        if not response.ok:
            yield [gr.ChatMessage(role="assistant", content=f"API Error: {response.status_code} - {response.text}")]
            return
        
        result = []
        current_content = ""
        tool_calls = []
        completion_data = None
        
        # Parse SSE stream from your API
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data: '):
                continue
                
            data_str = line[6:]  # Remove 'data: ' prefix
            
            if data_str == '[DONE]':
                break
                
            try:
                data = json.loads(data_str)
                
                # Handle streaming content
                if 'content' in data:
                    current_content += data['content']
                    
                    # Update or add main response with streaming content
                    main_msg_exists = any(msg.metadata is None or msg.metadata.get('title') is None for msg in result)
                    if main_msg_exists:
                        # Update existing main message
                        for i, msg in enumerate(result):
                            if msg.metadata is None or msg.metadata.get('title') is None:
                                result[i] = gr.ChatMessage(role="assistant", content=current_content)
                                break
                    else:
                        # Add new main message
                        result.append(gr.ChatMessage(role="assistant", content=current_content))
                
                # Handle completion data with tool calls
                elif 'completion' in data:
                    completion_data = data['completion']
                    
                    # Extract final content
                    if completion_data.get('choices'):
                        final_content = completion_data['choices'][0]['message'].get('content', '')
                        
                        # Handle tool calls if present
                        if completion_data.get('tool_calls'):
                            tool_calls = completion_data['tool_calls']
                            
                            # Show tool calls
                            for i, tool_call in enumerate(tool_calls):
                                tool_name = tool_call['name']
                                tool_args = tool_call['arguments']
                                
                                # Execute tool locally to show result
                                try:
                                    if tool_name == 'get_weather':
                                        tool_result = get_weather(**tool_args)
                                    elif tool_name == 'calculate_sum':
                                        tool_result = calculate_sum(**tool_args)
                                    else:
                                        tool_result = f"Unknown tool: {tool_name}"
                                        
                                    tool_content = f"**Function:** {tool_name}\n**Arguments:** {json.dumps(tool_args, indent=2)}\n**Result:** {tool_result}"
                                    
                                    result.append(
                                        gr.ChatMessage(
                                            role="assistant",
                                            content=tool_content,
                                            metadata={"title": f"🔧 {tool_name}", "status": "done"}
                                        )
                                    )
                                except Exception as e:
                                    result.append(
                                        gr.ChatMessage(
                                            role="assistant",
                                            content=f"**Function:** {tool_name}\n**Arguments:** {json.dumps(tool_args, indent=2)}\n**Error:** {str(e)}",
                                            metadata={"title": f"🔧 {tool_name}", "status": "error"}
                                        )
                                    )
                        
                        # Update final content (remove tool call tags if any)
                        final_content = re.sub(r'<tool_call>.*?</tool_call>', '', final_content, flags=re.DOTALL).strip()
                        
                        if final_content:
                            # Remove any existing main message and add clean final message
                            result = [msg for msg in result if msg.metadata is not None and msg.metadata.get('title') is not None]
                            result.append(gr.ChatMessage(role="assistant", content=final_content))
                
                # Handle errors
                elif 'error' in data:
                    result.append(gr.ChatMessage(role="assistant", content=f"Error: {data['error']}"))
                    break
                    
            except json.JSONDecodeError:
                continue
            
            # Yield current state
            if result:
                yield result

    except Exception as e:
        yield [
            gr.ChatMessage(
                role="assistant", content=f"Error: {str(e)}\n\nPlease check your endpoint and model configuration."
            )
        ]


# Custom Scout textbox will be created in the demo function



def create_demo():
    """Create and configure the Gradio interface."""

    with gr.Blocks(title="LLM Chat Interface", theme=theme) as demo:
        with gr.Sidebar(open=False):
            endpoint_input = gr.Textbox(
                label="API Endpoint",
                value="0.0.0.0:8000",
                placeholder="Enter endpoint (e.g., localhost:8000)",
                info="URL or address of the OpenAI-compatible API",
            )
            
            refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
            
            model_input = gr.Dropdown(
                label="Model Name",
                choices=["Qwen/Qwen3-8B"],
                value="Qwen/Qwen3-8B",
                info="Name of the model to use",
                allow_custom_value=True,
            )
            
            key_input = gr.Textbox(
                label="API Key",
                value="EMPTY",
                placeholder="Enter API key",
                info="API key for the OpenAI-compatible API",
            )

            gr.Markdown("### Generation Parameters")
            temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
            top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top P")
            max_tokens_slider = gr.Slider(minimum=16, maximum=8192, value=2048, step=16, label="Max Tokens")
            
            gr.Markdown("### Thinking Options")
            think_start_toggle = gr.Checkbox(
                label="Think Start Token",
                value=False,
                info="Add '<think> ' token for models that need explicit thinking activation"
            )



        # Create placeholder markdown (visible initially)
        placeholder_md = gr.Markdown("", visible=True, height=280)
        
        # Create custom chatbot
        chatbot = gr.Chatbot(
            height=640,
            show_copy_button=False,
            placeholder="START HERE",
            type="messages",
            render_markdown=True,
            show_label=False,
            show_share_button=False,
            visible=False,
        )
        
        # Create Scout-style textbox
        scout_textbox, context_button, send_button, scout_css = create_scout_textbox_ui(
            placeholder="Create a website based on my vibes"
        )
        
        # Apply Scout CSS and hide footer
        demo.load(lambda: None, js=f"""
            function() {{
                const style = document.createElement('style');
                style.textContent = `{scout_css}
                
                /* Hide Gradio footer */
                .footer {{
                    display: none !important;
                }}
                footer {{
                    display: none !important;
                }}
                .gradio-container footer {{
                    display: none !important;
                }}
                
                /* Additional border removal - more aggressive targeting */
                * {{
                    border: none !important;
                    box-shadow: none !important;
                }}
                
                /* Restore only necessary button styles */
                .scout-context-btn,
                .scout-send-btn {{
                    border: 1px solid transparent !important;
                }}
                
                .scout-context-btn {{
                    background: #EAF6FF !important;
                    border-color: #EAF6FF !important;
                }}
                
                .scout-send-btn {{
                    background: #57BAFF !important;
                    border-color: #57BAFF !important;
                }}
                
                /* AGGRESSIVE: Remove ALL grey/colored backgrounds from chatbot */
                .gradio-chatbot *,
                div[data-testid="chatbot"] *,
                .chatbot *,
                [class*="chatbot"] *,
                [class*="message"] *,
                [class*="bubble"] * {{
                    background: transparent !important;
                    background-color: transparent !important;
                }}
                
                /* Specifically target common Gradio chatbot classes */
                .gradio-chatbot .prose,
                .gradio-chatbot .message-wrap,
                .gradio-chatbot .message-row,
                .gradio-chatbot .message,
                .gradio-chatbot .bot,
                .gradio-chatbot .assistant,
                div[data-testid="chatbot"] .prose,
                div[data-testid="chatbot"] .message-wrap,
                div[data-testid="chatbot"] .message-row, 
                div[data-testid="chatbot"] .message,
                div[data-testid="chatbot"] .bot,
                div[data-testid="chatbot"] .assistant {{
                    background: transparent !important;
                    background-color: transparent !important;
                }}
                
                /* Target any element with grey-ish background colors */
                *[style*="background-color: rgb(243, 244, 246)"],
                *[style*="background-color: #f3f4f6"],
                *[style*="background-color: rgb(249, 250, 251)"],
                *[style*="background-color: #f9fafb"],
                *[style*="background: rgb(243, 244, 246)"],
                *[style*="background: #f3f4f6"] {{
                    background: transparent !important;
                    background-color: transparent !important;
                
                /* Keep thinking messages with light grey background */
                .gradio-chatbot .message[data-title="💭 Thinking"],
                div[data-testid="chatbot"] .message[data-title="💭 Thinking"],
                .gradio-chatbot .message-wrap[data-title="💭 Thinking"],
                div[data-testid="chatbot"] .message-wrap[data-title="💭 Thinking"] {{
                    background: #F8F9FA !important;
                    background-color: #F8F9FA !important;
                    border-radius: 8px !important;
                    padding: 12px !important;
                    margin: 4px 0 !important;
                }}`;
                document.head.appendChild(style);
            }}
        """)
        
        # Custom chat function wrapper
        def handle_chat(message, history, endpoint, model, key, temp, top_p, max_tokens, think_start_token):
            if not message.strip():
                return history, "", gr.update()
            
            # Add user message to history
            new_history = history + [{"role": "user", "content": message}]
            
            # Make chatbot visible and hide placeholder when first message is sent
            chatbot_update = gr.update(visible=True)
            placeholder_update = gr.update(visible=False)
            
            # Call the chat function with streaming
            response_gen = chat_function(
                message, history, endpoint, model, key, temp, top_p, max_tokens, think_start_token
            )
            
            # Stream the response and build complete conversation history
            for response_messages in response_gen:
                # Build complete history: previous conversation + user message + current responses
                complete_history = new_history + response_messages
                yield complete_history, "", placeholder_update
        
        # Connect send button and textbox submit
        send_button.click(
            fn=handle_chat,
            inputs=[
                scout_textbox, 
                chatbot,
                endpoint_input,
                model_input,
                key_input,
                temp_slider,
                top_p_slider,
                max_tokens_slider,
                think_start_toggle
            ],
            outputs=[chatbot, scout_textbox, placeholder_md],
        )
        
        scout_textbox.submit(
            fn=handle_chat,
            inputs=[
                scout_textbox, 
                chatbot,
                endpoint_input,
                model_input,
                key_input,
                temp_slider,
                top_p_slider,
                max_tokens_slider,
                think_start_toggle
            ],
            outputs=[chatbot, scout_textbox, placeholder_md],
        )
        
        # Connect the refresh button to update models
        refresh_models_btn.click(
            fn=update_models,
            inputs=[endpoint_input],
            outputs=[model_input]
        )

        # Connect context button (placeholder functionality)
        def handle_context():
            gr.Info("Context feature coming soon!")
            return
        
        context_button.click(
            fn=handle_context,
            inputs=[],
            outputs=[]
        )

    return demo


def main():
    """Main function to launch the Gradio app."""
    parser = argparse.ArgumentParser(description="LLM Chat UI with OpenAI-compatible endpoint")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()

    # Create the demo
    demo = create_demo()

    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=args.port,
        share=False,  # Share by default unless --no-share is specified
        show_api=False,
    )


if __name__ == "__main__":
    main()
