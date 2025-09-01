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
from utils import status_messages


def chat_function_sync(message, history, session_id=None, mode="Scout"):
    """
    Sync chat function using threading to handle claude_code async calls.
    This avoids Gradio's async issues completely.
    """
    import concurrent.futures
    import threading
    import queue
    import asyncio
    
    # Create a queue to pass results between threads
    result_queue = queue.Queue()
    
    def run_claude_code_in_thread():
        """Run claude_code in a separate thread with its own event loop."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def collect_events():
                events = []
                try:
                    async for event in claude_code(message, session_id=session_id, mode=mode):
                        events.append(event)
                        # Put each event in queue immediately for streaming
                        result_queue.put(('event', event))
                except Exception as e:
                    result_queue.put(('error', e))
                finally:
                    result_queue.put(('done', None))
            
            # Run the async collection
            loop.run_until_complete(collect_events())
            loop.close()
            
        except Exception as e:
            result_queue.put(('error', e))
            result_queue.put(('done', None))
    
    # Start the thread
    thread = threading.Thread(target=run_claude_code_in_thread, daemon=True)
    thread.start()
    
    # Process events as they come from the queue
    result = []
    current_content = ""
    extracted_session_id = session_id
    
    status_index = 0
    
    while True:
        try:
            # Get next item from queue with timeout for heartbeat
            try:
                item_type, item_data = result_queue.get(timeout=1.0)
            except queue.Empty:
                # No event in 1 second, yield heartbeat with rotating fun message
                if not result:
                    # Show rotating status message
                    heartbeat_result = [
                        gr.ChatMessage(
                            role="assistant", 
                            content=status_messages[status_index % len(status_messages)]
                        )
                    ]
                    status_index += 1
                    yield heartbeat_result, extracted_session_id
                else:
                    # Update last message if it's just a status message (no real content yet)
                    if (len(result) == 1 and 
                        any(status in result[-1].content for status in ["🍳", "🤔", "⚡", "🔍", "🧠", "⚙️", "🎯", "📚", "🚀", "🔮"])):
                        # Update with next rotating message
                        result[-1] = gr.ChatMessage(
                            role="assistant", 
                            content=status_messages[status_index % len(status_messages)]
                        )
                        status_index += 1
                    yield result, extracted_session_id
                continue
                
            if item_type == 'done':
                break
            elif item_type == 'error':
                yield [
                    gr.ChatMessage(
                        role="assistant", content=f"⚠️ Error: {str(item_data)}\n\nPlease check your claude-code setup."
                    )
                ], session_id
                break
            elif item_type == 'event':
                event = item_data
                event_type = event.get('type', 'unknown')
                
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
                
                # Yield after processing each event
                yield result, extracted_session_id
        
        except Exception as e:
            yield [
                gr.ChatMessage(
                    role="assistant", content=f"⚠️ Error: {str(e)}\n\nPlease check your claude-code setup."
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
        current_mode = gr.State("Scout")
        
        with gr.Sidebar(open=True):
            gr.Markdown("## 🔍 Scout Chats")
            
            # New chat button
            new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm")
            
            # Chat list using HTML for full control
            chat_list_html = gr.HTML(
                value="<div class='custom-chat-list'><h3>Chat History</h3><div id='chat-items'></div></div>",
                elem_classes=["chat-list-container"]
            )
            
            # Hidden component to track selected chat
            selected_chat = gr.State(value=None)
            
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
        scout_textbox, context_button, send_button, mode_toggle, scout_css = create_scout_textbox_ui(
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
                
                /* iOS-style sidebar styling */
                .gradio-sidebar {{
                    background: #F8F9FA !important;
                    border-right: 1px solid #E5E7EB !important;
                    padding: 16px !important;
                }}
                
                /* Chat list iOS styling - target Gradio radio component */
                .chat-list {{
                    background: transparent !important;
                }}
                
                .chat-list .block-title {{
                    display: none !important;
                }}
                
                /* Custom chat list styling */
                .chat-list-container {{
                    background: transparent !important;
                    border: none !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }}
                
                .custom-chat-list {{
                    background: transparent !important;
                    border: none !important;
                    border-radius: 0 !important;
                    padding: 0 !important;
                    margin: 16px 0 !important;
                }}
                
                .custom-chat-list h3 {{
                    margin: 0 0 12px 0 !important;
                    font-size: 13px !important;
                    font-weight: 600 !important;
                    color: #8E8E93 !important;
                    text-transform: none !important;
                    letter-spacing: 0 !important;
                    padding-left: 8px !important;
                }}
                
                #chat-items {{
                    display: flex !important;
                    flex-direction: column !important;
                    gap: 2px !important;
                    max-height: 400px !important;
                    overflow-y: auto !important;
                    padding-right: 4px !important;
                }}
                
                /* Hide scrollbars completely for smooth iOS look */
                #chat-items::-webkit-scrollbar {{
                    display: none !important;
                }}
                
                #chat-items {{
                    scrollbar-width: none !important;
                    -ms-overflow-style: none !important;
                }}
                
                .chat-item {{
                    padding: 12px 14px !important;
                    background: rgba(255, 255, 255, 0.1) !important;
                    backdrop-filter: blur(20px) !important;
                    -webkit-backdrop-filter: blur(20px) !important;
                    border: 1px solid rgba(255, 255, 255, 0.15) !important;
                    border-radius: 10px !important;
                    cursor: pointer !important;
                    transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
                    font-size: 14px !important;
                    color: #1C1C1E !important;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
                }}
                
                .chat-item:hover {{
                    background: rgba(255, 255, 255, 0.2) !important;
                    border-color: rgba(255, 255, 255, 0.25) !important;
                    transform: translateY(-1px) !important;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
                }}
                
                .chat-item.selected {{
                    background: rgba(0, 122, 255, 0.85) !important;
                    color: white !important;
                    border-color: rgba(0, 122, 255, 0.4) !important;
                    box-shadow: 0 2px 12px rgba(0, 122, 255, 0.3) !important;
                }}
                
                /* Remove blue theme from all radio groups in sidebar */
                .gradio-sidebar fieldset,
                .gradio-sidebar .block {{
                    background: transparent !important;
                    background-color: transparent !important;
                }}
                
                /* Target the radio container */
                .chat-list .gradio-radio fieldset {{
                    border: none !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    display: flex !important;
                    flex-direction: column !important;
                    gap: 6px !important;
                    max-height: 400px !important;
                    overflow-y: auto !important;
                }}
                
                /* Hide scrollbars for chat list */
                .chat-list .gradio-radio fieldset::-webkit-scrollbar {{
                    display: none !important;
                }}
                
                .chat-list .gradio-radio fieldset {{
                    scrollbar-width: none !important;
                    -ms-overflow-style: none !important;
                }}
                
                /* Style each radio option - clean iOS style */
                .chat-list .gradio-radio fieldset > label {{
                    margin: 0 !important;
                    padding: 12px 16px !important;
                    background: white !important;
                    border: 1px solid #E5E7EB !important;
                    border-radius: 8px !important;
                    cursor: pointer !important;
                    transition: all 0.15s ease !important;
                    font-size: 14px !important;
                    font-weight: 500 !important;
                    color: #374151 !important;
                    display: block !important;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
                    line-height: 1.4 !important;
                    margin-bottom: 4px !important;
                }}
                
                .chat-list .gradio-radio fieldset > label:hover {{
                    background: #F9FAFB !important;
                    border-color: #D1D5DB !important;
                }}
                
                /* Hide radio input */
                .chat-list .gradio-radio fieldset > label > input[type="radio"] {{
                    display: none !important;
                }}
                
                /* Selected state - clean iOS blue */
                .chat-list .gradio-radio fieldset > label:has(input[type="radio"]:checked) {{
                    background: #007AFF !important;
                    color: white !important;
                    border: 1px solid #007AFF !important;
                    box-shadow: 0 2px 8px rgba(0,122,255,0.2) !important;
                }}
                
                /* Text content styling */
                .chat-list .gradio-radio fieldset > label > span {{
                    display: block !important;
                    width: 100% !important;
                    padding: 0 !important;
                    margin: 0 !important;
                    background: transparent !important;
                    border: none !important;
                    font-weight: inherit !important;
                    color: inherit !important;
                    text-align: left !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                }}
                
                /* New Chat button styling */
                .gradio-sidebar button[data-testid] {{
                    width: 100% !important;
                    margin-bottom: 16px !important;
                    background: #57BAFF !important;
                    border: none !important;
                    border-radius: 8px !important;
                    padding: 12px 16px !important;
                    font-weight: 600 !important;
                    color: white !important;
                    font-size: 14px !important;
                }}
                
                .gradio-sidebar button[data-testid]:hover {{
                    background: #4AABF0 !important;
                }}
                
                /* Delete button styling */
                .gradio-sidebar button:not([data-testid]) {{
                    width: 100% !important;
                    background: #FEF2F2 !important;
                    border: 1px solid #FECACA !important;
                    border-radius: 8px !important;
                    padding: 8px 16px !important;
                    color: #DC2626 !important;
                    font-size: 13px !important;
                    margin-top: 8px !important;
                }}
                
                .gradio-sidebar button:not([data-testid]):hover {{
                    background: #FEE2E2 !important;
                    border-color: #FCA5A5 !important;
                }}
                
                /* Sidebar heading */
                .gradio-sidebar h2 {{
                    font-size: 16px !important;
                    font-weight: 600 !important;
                    color: #374151 !important;
                    margin: 0 0 16px 0 !important;
                    padding: 0 !important;
                    border: none !important;
                    background: transparent !important;
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
                
                // Set initial mode attribute
                setTimeout(() => {{
                    const toggleBtn = document.querySelector('.scout-mode-toggle');
                    const textboxWrapper = document.querySelector('.scout-textbox-wrapper');
                    if (toggleBtn) {{
                        toggleBtn.setAttribute('data-mode', 'Scout');
                    }}
                    if (textboxWrapper) {{
                        textboxWrapper.setAttribute('data-mode', 'Scout');
                    }}
                }}, 500);
            }}
        """)
        
        # Chat management functions
        def load_chat_list():
            """Load and format chat list for the custom HTML component."""
            chats = chat_manager.get_chats()
            if not chats:
                return "<div class='custom-chat-list'><h3>Recents</h3><div id='chat-items'><p style='color: #9CA3AF; font-style: italic; padding: 20px; text-align: center;'>No chats yet</p></div></div>"
            
            items_html = ""
            for chat in chats:
                title = chat['title'][:50] + "..." if len(chat['title']) > 50 else chat['title']
                date = chat['updated_at'][:16]
                items_html += f"""
                <div class='chat-item' onclick='selectChat("{chat["id"]}")' data-chat-id='{chat["id"]}'>
                    <div style='font-weight: 500;'>{title}</div>
                    <div style='font-size: 12px; color: #9CA3AF; margin-top: 2px;'>{date}</div>
                </div>
                """
            
            return f"""
            <div class='custom-chat-list'>
                <h3>Recents</h3>
                <div id='chat-items'>{items_html}</div>
            </div>
            <script>
                function selectChat(chatId) {{
                    // Remove selected class from all items
                    document.querySelectorAll('.chat-item').forEach(item => item.classList.remove('selected'));
                    // Add selected class to clicked item
                    document.querySelector(`[data-chat-id="${{chatId}}"]`).classList.add('selected');
                    // Trigger Gradio event to load chat
                    window.gradio_config.components.find(c => c.props && c.props.elem_id === 'selected_chat_state').props.value = chatId;
                }}
            </script>
            """
        
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
        def handle_chat(message, history, chat_id_state, session_id_state, selected_mode):
            if not message.strip():
                return history, "", gr.update(), gr.update(), chat_id_state, session_id_state
            
            # Add user message to history
            new_history = history + [{"role": "user", "content": message}]
            
            # Add initial thinking indicator  
            thinking_history = new_history + [{"role": "assistant", "content": "🍳 Cooking up something good..."}]
            
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
            
            # Call the sync chat function with streaming, passing the selected mode
            response_gen = chat_function_sync(message, history, current_session_id, selected_mode)
            
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
        
        # Connect send button and textbox submit
        send_button.click(
            fn=handle_chat,
            inputs=[scout_textbox, chatbot, current_chat_id, current_session_id, current_mode],
            outputs=[chatbot, scout_textbox, chatbot, placeholder_md, current_chat_id, current_session_id],
            queue=True,
        )
        
        scout_textbox.submit(
            fn=handle_chat,
            inputs=[scout_textbox, chatbot, current_chat_id, current_session_id, current_mode],
            outputs=[chatbot, scout_textbox, chatbot, placeholder_md, current_chat_id, current_session_id],
            queue=True,
        )
        
        # Connect sidebar buttons
        new_chat_btn.click(
            fn=create_new_chat,
            inputs=[],
            outputs=[chat_list_html, chatbot, chatbot, placeholder_md, current_chat_id, current_session_id]
        )
        
        selected_chat.change(
            fn=load_selected_chat,
            inputs=[selected_chat],
            outputs=[chatbot, chatbot, placeholder_md, current_session_id]
        )
        
        delete_chat_btn.click(
            fn=delete_selected_chat,
            inputs=[current_chat_id],
            outputs=[chat_list_html, chatbot, chatbot, placeholder_md, current_chat_id, current_session_id]
        )
        
        # Load chat list on startup
        demo.load(
            fn=load_chat_list,
            inputs=[],
            outputs=[chat_list_html]
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
        
        # Toggle mode function
        def toggle_mode(current_mode_state):
            new_mode = "Plan" if current_mode_state == "Scout" else "Scout"
            return new_mode, gr.update(value=new_mode)
        
        mode_toggle.click(
            fn=toggle_mode,
            inputs=[current_mode],
            outputs=[current_mode, mode_toggle],
            js="""
            function(current_mode) {
                const new_mode = current_mode === "Scout" ? "Plan" : "Scout";
                setTimeout(() => {
                    const toggleBtn = document.querySelector('.scout-mode-toggle');
                    const textboxWrapper = document.querySelector('.scout-textbox-wrapper');
                    if (toggleBtn) {
                        toggleBtn.setAttribute('data-mode', new_mode);
                    }
                    if (textboxWrapper) {
                        textboxWrapper.setAttribute('data-mode', new_mode);
                    }
                }, 100);
                return new_mode;
            }
            """
        )

    return demo


def main():
    """Main function to launch the Gradio app."""
    parser = argparse.ArgumentParser(description="LLM Chat UI with OpenAI-compatible endpoint")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()

    # Create the demo
    demo = create_demo()

    # Enable queue to prevent timeout errors
    demo.queue(default_concurrency_limit=10)
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=args.port,
        share=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()
