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

import gradio as gr
from openai import OpenAI
from css import theme
from scout_component import create_scout_textbox_ui


def fetch_models(endpoint):
    """Fetch available models from the /models endpoint."""
    try:
        base_url = f"http://{endpoint}/v1" if not endpoint.startswith("http") else endpoint
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        
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


def chat_function(message, history, endpoint, model_name, key, temperature, top_p, max_tokens):
    """
    Chat function that communicates with OpenAI-compatible endpoint.
    """
    client = OpenAI(api_key=key, base_url=f"http://{endpoint}/v1" if not endpoint.startswith("http") else endpoint)

    messages = []

    for msg in history:
        if msg["role"] in ["user", "assistant"]:
            msg_content = msg["content"]
            if "</think>" in msg_content:
                msg_content = msg_content.split("</think>")[-1]
            messages.append({"role": msg["role"], "content": msg_content})

    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(max_tokens),
        )

        # Initialize buffers
        current_buffer = ""
        in_thinking = False
        thinking_content = ""
        final_content = ""
        thinking_blocks = []
        seen_non_whitespace = False

        # Process the stream
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                current_buffer += content

                # Process buffer for complete tags
                processed = True
                while processed:
                    processed = False

                    if not in_thinking:
                        # First check if we have enough content to make decisions
                        # Look for first non-whitespace content
                        non_ws_match = re.search(r"\S", current_buffer)

                        if non_ws_match:
                            # We found non-whitespace, now check what it is
                            seen_non_whitespace = True

                            # Check if the non-whitespace starts with a thinking tag
                            think_at_start = re.match(r"^(\s*)<think>", current_buffer)
                            if think_at_start:
                                # Save any whitespace before thinking
                                final_content += think_at_start.group(1)
                                current_buffer = current_buffer[think_at_start.end() :]
                                in_thinking = True
                                thinking_content = ""
                                processed = True
                                continue

                            # Check for thinking tag later in buffer
                            think_match = re.search(r"<think>", current_buffer)
                            if think_match:
                                # Save content before thinking
                                final_content += current_buffer[: think_match.start()]
                                current_buffer = current_buffer[think_match.end() :]
                                in_thinking = True
                                thinking_content = ""
                                processed = True
                                continue

                        # Check if we might be building up to a thinking tag
                        if not seen_non_whitespace or re.search(r"<t?h?i?n?k?$", current_buffer):
                            # Don't process yet - might be partial tag
                            break

                        # Safe to add current buffer to final content
                        final_content += current_buffer
                        current_buffer = ""

                    else:
                        # In thinking mode - look for closing tag
                        end_match = re.search(r"</think>", current_buffer)
                        if end_match:
                            # Extract thinking content
                            thinking_content += current_buffer[: end_match.start()]
                            current_buffer = current_buffer[end_match.end() :]
                            in_thinking = False

                            # Store the thinking block
                            if thinking_content.strip():
                                thinking_blocks.append(thinking_content.strip())
                            thinking_content = ""
                            processed = True
                            continue

                        # Check if buffer might contain partial closing tag
                        if re.search(r"</t?h?i?n?k?$", current_buffer):
                            # Hold the partial tag, add the rest to thinking
                            partial_match = re.search(r"</t?h?i?n?k?$", current_buffer)
                            thinking_content += current_buffer[: partial_match.start()]
                            current_buffer = current_buffer[partial_match.start() :]
                            break

                        # No closing tag or partial - accumulate all
                        thinking_content += current_buffer
                        current_buffer = ""

                # Build and yield current state
                result = []

                # Add completed thinking blocks
                for i, think_block in enumerate(thinking_blocks):
                    result.append(
                        gr.ChatMessage(
                            role="assistant", content=think_block, metadata={"title": "💭 Thinking", "status": "done"}
                        )
                    )

                # Add current thinking if in progress
                if in_thinking and thinking_content:
                    result.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=thinking_content,
                            metadata={"title": "💭 Thinking", "status": "pending"},
                        )
                    )

                # Only add main response if we have non-whitespace content
                # and we're not potentially waiting for a thinking tag
                if final_content.strip() and seen_non_whitespace:
                    # Only show if we're not in the middle of a potential tag
                    if not re.search(r"^\s*<t?h?i?n?k?$", current_buffer):
                        result.append(gr.ChatMessage(role="assistant", content=final_content.strip()))

                # Only yield if we have content
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



        # Create custom chatbot
        chatbot = gr.Chatbot(
            height=400,
            show_copy_button=True,
            placeholder="START HERE",
            type="messages",
            render_markdown=True,
            show_label=False
        )
        
        # Create Scout-style textbox
        scout_textbox, context_button, send_button, scout_css = create_scout_textbox_ui(
            placeholder="Create a website based on my vibes"
        )
        
        # Apply Scout CSS
        demo.load(lambda: None, js=f"""
            function() {{
                const style = document.createElement('style');
                style.textContent = `{scout_css}`;
                document.head.appendChild(style);
            }}
        """)
        
        # Custom chat function wrapper
        def handle_chat(message, history, endpoint, model, key, temp, top_p, max_tokens):
            if not message.strip():
                return history, ""
            
            # Call the chat function with streaming
            response_gen = chat_function(
                message, history, endpoint, model, key, temp, top_p, max_tokens
            )
            
            # Stream the response
            for response_messages in response_gen:
                yield response_messages, ""
        
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
                max_tokens_slider
            ],
            outputs=[chatbot, scout_textbox],
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
                max_tokens_slider
            ],
            outputs=[chatbot, scout_textbox],
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
