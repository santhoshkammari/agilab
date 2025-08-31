# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio",
#     "claude-code-sdk",
# ]

# ///
import argparse
import re
import json
import asyncio
import time
from typing import AsyncGenerator

import gradio as gr
from css import theme
from scout_component import create_scout_textbox_ui
from ssa import claude_code
from chat_manager import ChatManager




async def chat_function_async(message, history, session_id=None):
    """
    Chat function using claudecode for proper AI assistance.
    """
    try:
        result = []
        current_content = ""
        extracted_session_id = session_id
        last_yield_time = time.time()
        
        # Stream responses from claudecode
        async for event in claude_code(message, session_id=session_id):
            event_type = event.get('type', 'unknown')
            
            # Skip KeepConnectionLiveMessage type but still yield to keep connection alive
            if event_type == 'KeepConnectionLiveMessage':
                yield result, extracted_session_id
                continue
            
            # Extract session_id from SystemMessage
            if event_type == 'SystemMessage' and not extracted_session_id:
                data = event.get('data', {})
                if data.get('session_id'):
                    extracted_session_id = data['session_id']
            
            elif event_type == 'AssistantMessage':
                content = event.get('content', [])
                if content and len(content) > 0:
                    # Handle text content
                    if content[0].get('text'):
                        text_content = content[0]['text']
                        current_content += text_content
                        
                        # Update or add main response
                        main_msg_exists = any(msg.metadata is None or msg.metadata.get('title') is None for msg in result)
                        if main_msg_exists:
                            for i, msg in enumerate(result):
                                if msg.metadata is None or msg.metadata.get('title') is None:
                                    result[i] = gr.ChatMessage(role="assistant", content=current_content)
                                    break
                        else:
                            result.append(gr.ChatMessage(role="assistant", content=current_content))
                    
                    # Handle tool use
                    elif content[0].get('name'):
                        tool_name = content[0]['name']
                        tool_input = content[0].get('input', {})
                        
                        result.append(
                            gr.ChatMessage(
                                role="assistant",
                                content=f"Using {tool_name}...\n```json\n{json.dumps(tool_input, indent=2)}\n```",
                                metadata={"title": f"🔧 {tool_name}", "status": "pending"},
                            )
                        )
            
            elif event_type == 'UserMessage':
                # Handle tool results
                user_content = event.get('content', [])
                if user_content and len(user_content) > 0:
                    tool_use_id = user_content[0].get('tool_use_id')
                    if tool_use_id:
                        tool_result_content = user_content[0].get('content', [])
                        if tool_result_content and len(tool_result_content) > 0:
                            result_text = tool_result_content[0].get('text', '')
                            
                            # Find and update the corresponding pending tool message
                            for i, msg in enumerate(result):
                                if (msg.metadata and 
                                    msg.metadata.get('status') == 'pending' and
                                    msg.metadata.get('title', '').startswith('🔧')):
                                    
                                    tool_name = msg.metadata['title'].replace('🔧 ', '')
                                    tool_content = f"**Tool:** {tool_name}\n**Result:** {result_text[:500]}{'...' if len(result_text) > 500 else ''}"
                                    result[i] = gr.ChatMessage(
                                        role="assistant", 
                                        content=tool_content, 
                                        metadata={"title": f"🔧 {tool_name}", "status": "done"}
                                    )
                                    break
            
            elif event_type == 'ResultMessage':
                # Final result - ensure we have a clean main response
                final_result = event.get('result', '')
                if final_result.strip():
                    # Remove any pending tool messages and ensure clean final response
                    result = [msg for msg in result if not (msg.metadata and msg.metadata.get('status') == 'pending')]
                    
                    # Update or add final response
                    main_msg_exists = any(msg.metadata is None or msg.metadata.get('title') is None for msg in result)
                    if main_msg_exists:
                        for i, msg in enumerate(result):
                            if msg.metadata is None or msg.metadata.get('title') is None:
                                result[i] = gr.ChatMessage(role="assistant", content=final_result)
                                break
                    else:
                        result.append(gr.ChatMessage(role="assistant", content=final_result))
            
            # CRITICAL: Always yield on every iteration (following Qwen-Agent pattern)
            # This prevents timeout regardless of content changes
            yield result, extracted_session_id

    except asyncio.TimeoutError:
        yield [
            gr.ChatMessage(
                role="assistant", content="⏱️ Request timed out. The model is taking longer than expected. Please try again or break your request into smaller parts."
            )
        ], session_id
    except Exception as e:
        yield [
            gr.ChatMessage(
                role="assistant", content=f"⚠️ Error: {str(e)}\n\nPlease check your claude-code setup."
            )
        ], session_id


def chat_function(message, history, session_id=None):
    """Sync wrapper for async chat function."""
    import asyncio
    
    async def run_chat():
        results = []
        final_session_id = session_id
        async for result, session_id_update in chat_function_async(message, history, session_id):
            results.append((result, session_id_update))
            if session_id_update:
                final_session_id = session_id_update
        return results, final_session_id
    
    try:
        # Run in existing event loop if available, otherwise create new one
        try:
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we need to use run_in_executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_chat())
                results, final_session_id = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            results, final_session_id = asyncio.run(run_chat())
        
        # Yield all results with final session_id
        for result, _ in results:
            yield result, final_session_id
            
    except asyncio.TimeoutError:
        yield [
            gr.ChatMessage(
                role="assistant", content="⏱️ Request timed out. The model is taking longer than expected. Please try again or break your request into smaller parts."
            )
        ], session_id
    except Exception as e:
        yield [
            gr.ChatMessage(
                role="assistant", content=f"⚠️ Error: {str(e)}\\n\\nPlease check your claude-code setup."
            )
        ], session_id


# Custom Scout textbox will be created in the demo function



def create_demo():
    """Create and configure the Gradio interface."""
    
    # Initialize chat manager
    chat_manager = ChatManager()

    with gr.Blocks(title="Scout - AI Code Assistant", theme=theme) as demo:
        # State variables
        current_chat_id = gr.State(None)
        current_session_id = gr.State(None)
        
        with gr.Sidebar(open=False):
            gr.Markdown("## 🔍 Scout Chats")
            
            # New chat button
            new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm")
            
            # Chat list
            chat_list = gr.Radio(
                choices=[],
                label="Chat History",
                interactive=True,
                elem_classes=["chat-list"]
            )
            
            # Delete chat button
            delete_chat_btn = gr.Button("🗑️ Delete Chat", variant="secondary", size="sm")
        

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
                }}
                
                /* Title header styling */
                .title-header {{
                    text-align: center !important;
                    font-size: 2.5rem !important;
                    font-weight: 600 !important;
                    margin: 2rem 0 !important;
                    color: #1F2937 !important;
                }}
                
                .title-header h1 {{
                    margin: 0 !important;
                    background: linear-gradient(135deg, #57BAFF, #4AABF0) !important;
                    -webkit-background-clip: text !important;
                    -webkit-text-fill-color: transparent !important;
                    background-clip: text !important;
                }}`;
                document.head.appendChild(style);
            }}
        """)
        
        # Chat management functions
        def load_chat_list():
            """Load and format chat list for the radio component."""
            chats = chat_manager.get_chats()
            if not chats:
                return gr.update(choices=[], value=None)
            
            choices = [(f"{chat['title']} ({chat['updated_at'][:16]})", chat['id']) for chat in chats]
            return gr.update(choices=choices, value=chats[0]['id'] if chats else None)
        
        def create_new_chat():
            """Create a new chat and update the UI."""
            chat_id = chat_manager.create_chat()
            return (
                load_chat_list(),  # Update chat list
                [],  # Clear chatbot
                gr.update(visible=False),  # Hide chatbot
                gr.update(visible=True),   # Show placeholder
                chat_id,  # Update current_chat_id
                None      # Reset current_session_id
            )
        
        def load_selected_chat(chat_id):
            """Load a selected chat."""
            if not chat_id:
                return [], gr.update(visible=False), gr.update(visible=True), None
            
            chat = chat_manager.get_chat(chat_id)
            if chat:
                messages = chat['messages']
                session_id = chat['session_id']
                if messages:
                    return (
                        messages,  # Load chat history
                        gr.update(visible=True),   # Show chatbot
                        gr.update(visible=False),  # Hide placeholder
                        session_id  # Update session_id
                    )
            return [], gr.update(visible=False), gr.update(visible=True), None
        
        def delete_selected_chat(chat_id):
            """Delete the selected chat."""
            if chat_id:
                chat_manager.delete_chat(chat_id)
            return (
                load_chat_list(),  # Update chat list
                [],  # Clear chatbot
                gr.update(visible=False),  # Hide chatbot
                gr.update(visible=True),   # Show placeholder
                None,  # Reset current_chat_id
                None   # Reset current_session_id
            )
        
        # Custom chat function wrapper
        def handle_chat(message, history, chat_id_state, session_id_state):
            if not message.strip():
                return history, "", gr.update(), gr.update(), chat_id_state, session_id_state
            
            # Add user message to history
            new_history = history + [{"role": "user", "content": message}]
            
            # Add thinking indicator immediately
            thinking_history = new_history + [{"role": "assistant", "content": "🤔 Thinking...", "metadata": {"title": "💭 Processing", "status": "pending"}}]
            
            # Make chatbot visible and hide placeholder when first message is sent
            chatbot_update = gr.update(visible=True)
            placeholder_update = gr.update(visible=False)
            
            # Create new chat if needed
            current_chat_id = chat_id_state
            current_session_id = session_id_state
            
            if not current_chat_id:
                current_chat_id = chat_manager.create_chat()
            
            # Show thinking indicator first
            yield thinking_history, "", chatbot_update, placeholder_update, current_chat_id, current_session_id
            
            # Call the chat function with streaming
            response_gen = chat_function(message, history, current_session_id)
            
            # Stream the response and build complete conversation history
            for response_messages, updated_session_id in response_gen:
                # Update session_id if we got a new one
                if updated_session_id and updated_session_id != current_session_id:
                    current_session_id = updated_session_id
                
                # Build complete history: previous conversation + user message + current responses
                complete_history = new_history + response_messages
                
                # Save to database (generate title from first message if new chat)
                if len(new_history) == 1:  # First message
                    title = message[:50] + "..." if len(message) > 50 else message
                    chat_manager.update_chat(current_chat_id, complete_history, title)
                else:
                    chat_manager.update_chat(current_chat_id, complete_history)
                
                yield complete_history, "", chatbot_update, placeholder_update, current_chat_id, current_session_id
        
        # Connect send button and textbox submit (following Qwen-Agent pattern)
        send_button.click(
            fn=handle_chat,
            inputs=[scout_textbox, chatbot, current_chat_id, current_session_id],
            outputs=[chatbot, scout_textbox, chatbot, placeholder_md, current_chat_id, current_session_id],
            queue=True,
        )
        
        scout_textbox.submit(
            fn=handle_chat,
            inputs=[scout_textbox, chatbot, current_chat_id, current_session_id],
            outputs=[chatbot, scout_textbox, chatbot, placeholder_md, current_chat_id, current_session_id],
            queue=True,
        )
        
        # Connect sidebar buttons
        new_chat_btn.click(
            fn=create_new_chat,
            inputs=[],
            outputs=[chat_list, chatbot, chatbot, placeholder_md, current_chat_id, current_session_id]
        )
        
        chat_list.change(
            fn=load_selected_chat,
            inputs=[chat_list],
            outputs=[chatbot, chatbot, placeholder_md, current_session_id]
        )
        
        delete_chat_btn.click(
            fn=delete_selected_chat,
            inputs=[current_chat_id],
            outputs=[chat_list, chatbot, chatbot, placeholder_md, current_chat_id, current_session_id]
        )
        
        # Load chat list on startup
        demo.load(
            fn=load_chat_list,
            inputs=[],
            outputs=[chat_list]
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

    # Enable queue to prevent timeout errors (following Qwen-Agent pattern)
    demo.queue(default_concurrency_limit=10)
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=args.port,
        share=False,  # Share by default unless --no-share is specified
        show_api=False,
    )


if __name__ == "__main__":
    main()
