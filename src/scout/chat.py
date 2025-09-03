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
import os
import subprocess
from typing import AsyncGenerator

from logger import get_logger

# Set up logger
logger = get_logger("chat", level="DEBUG")

import gradio as gr
from css import theme
from scout_component import create_scout_textbox_ui
from ssa import claude_code
from chat_manager import ChatManager
from utils import status_messages


def get_current_branch():
    """Get the current Git branch name."""
    try:
        result = subprocess.run(
            "git branch --show-current",
            shell=True, capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "main"  # fallback
    except:
        return "main"  # fallback


def get_directory_name(cwd_path=None):
    """Get the current directory name (basename only)."""
    try:
        if cwd_path and cwd_path.strip():
            return os.path.basename(cwd_path.rstrip('/'))
        return os.path.basename(os.getcwd())
    except:
        return "scout"  # fallback


def search_directories(query="", max_results=20):
    """Search for directories using find command with smart sorting."""
    if not query:
        return [os.path.expanduser("~"), "/tmp", "/opt", "/usr/local"]
    
    try:
        # Search in multiple locations
        result = subprocess.run(
            f"find ~ /opt /usr/local -type d -name '*{query}*' 2>/dev/null",
            shell=True, capture_output=True, text=True, timeout=3
        )
        dirs = [d.strip() for d in result.stdout.split('\n') if d.strip() and os.path.isdir(d.strip())]
        
        if not dirs:
            return [os.path.expanduser("~")]
        
        # Smart sorting function
        def sort_priority(path):
            path_lower = path.lower()
            
            # Priority 1: Normal project directories (highest priority)
            if not any(x in path_lower for x in ['.cache', '.local', 'site-packages', '.claude', '.config', '/usr/', '/opt/']):
                if query.lower() in os.path.basename(path).lower():
                    return (1, len(path))  # Normal dir with name match
                return (2, len(path))     # Normal dir
            
            # Priority 2: System/package directories  
            elif any(x in path_lower for x in ['site-packages', '/opt/', '/usr/']):
                return (3, len(path))
            
            # Priority 3: Hidden/cache directories (lowest priority)
            else:
                return (4, len(path))
        
        # Sort and return
        dirs.sort(key=sort_priority)
        return dirs[:max_results]
        
    except:
        return [os.path.expanduser("~")]


def chat_function_sync(message, history, session_id=None, mode="Scout", cwd="", append_system_prompt=""):
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
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def collect_events():
            events = []
            async for event in claude_code(message, session_id=session_id, mode=mode, cwd=cwd if cwd else None, append_system_prompt=append_system_prompt):
                events.append(event)
                # Put each event in queue immediately for streaming
                result_queue.put(('event', event))
            result_queue.put(('done', None))
        
        # Run the async collection
        loop.run_until_complete(collect_events())
        loop.close()
    
    # Start the thread
    thread = threading.Thread(target=run_claude_code_in_thread, daemon=True)
    thread.start()
    
    # Process events as they come from the queue
    result = []
    current_content = ""
    extracted_session_id = session_id
    
    status_index = 0
    
    while True:
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
                        logger.debug(f'=== TOOL RESULT DEBUG START ===')
                        logger.debug(f'tool_use_id: {tool_use_id}')
                        logger.debug(f'user_content[0]: {user_content[0]}')
                        logger.debug(f'tool_result_content type: {type(tool_result_content)}')
                        logger.debug(f'tool_result_content: {tool_result_content}')
                        if tool_result_content and len(tool_result_content) > 0:
                            logger.debug(f'tool_result_content[0] type: {type(tool_result_content[0])}')
                            logger.debug(f'tool_result_content[0]: {repr(tool_result_content[0])}')
                        logger.debug(f'=== TOOL RESULT DEBUG END ===\n')
                        if tool_result_content:
                            if isinstance(tool_result_content, str):
                                # Direct string content (Write, Read, etc.)
                                result_text = tool_result_content
                            elif isinstance(tool_result_content, list) and len(tool_result_content) > 0:
                                # List format (WebSearch, etc.)
                                if isinstance(tool_result_content[0], dict):
                                    result_text = tool_result_content[0].get('text', '')
                                else:
                                    result_text = str(tool_result_content[0])
                            elif isinstance(tool_result_content, dict):
                                # Direct dict format
                                result_text = tool_result_content.get('text', '')
                            else:
                                result_text = str(tool_result_content)
                            
                            # Skip empty or whitespace-only results
                            if not result_text or not result_text.strip():
                                continue
                            
                            # Find and update the corresponding pending tool message
                            for i, msg in enumerate(result):
                                if (msg.metadata and 
                                    msg.metadata.get('status') == 'pending' and
                                    msg.metadata.get('title', '').startswith('🔧')):
                                    
                                    tool_name = msg.metadata['title'].replace('🔧 ', '')
                                    
                                    # Format tool result based on content type
                                    if result_text.startswith('<tool_use_error>') and result_text.endswith('</tool_use_error>'):
                                        # Extract error message
                                        error_msg = result_text[16:-17]  # Remove <tool_use_error> tags
                                        tool_content = f"**Tool:** {tool_name}\n**Error:** {error_msg}"
                                    elif len(result_text) > 1000:
                                        # Show first part and indicate truncation
                                        tool_content = f"**Tool:** {tool_name}\n**Result:** {result_text[:1000]}...\n\n*[Result truncated for display]*"
                                    else:
                                        # Show full result
                                        tool_content = f"**Tool:** {tool_name}\n**Result:** {result_text}"
                                    
                                    result[i] = gr.ChatMessage(
                                        role="assistant", 
                                        content=tool_content, 
                                        metadata={"title": f"🔧 {tool_name}", "status": "done"}
                                    )
                                    break
            
            elif event_type == 'ResultMessage':
                # Final result - ensure we have a clean main response
                final_result = event.get('result', '')
                if final_result and final_result.strip():
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
        
        


def create_demo():
    """Create and configure the Gradio interface."""
    
    # Initialize chat manager
    chat_manager = ChatManager()

    with gr.Blocks(title="Scout - AI Code Assistant", theme=theme) as demo:
        # State variables
        current_chat_id = gr.State(None)
        current_session_id = gr.State(None)
        current_mode = gr.State("Scout")
        current_cwd = gr.State("")
        current_append_system_prompt = gr.State("")
        
        with gr.Sidebar(open=False):
            gr.Markdown("## 🔍 Scout Chats")
            
            # New chat button
            new_chat_btn = gr.Button("➕ New Chat", variant="primary", size="sm")
            
            # Chat list using dropdown
            chat_dropdown = gr.Dropdown(
                label="Recent Chats",
                choices=[],
                value=None,
                interactive=True,
                allow_custom_value=False
            )
            
            # Delete chat button
            delete_chat_btn = gr.Button("🗑️ Delete Chat", variant="secondary", size="sm")

        # Create placeholder markdown (visible initially)
        placeholder_md = gr.Markdown("Drop Ideas", visible=True, height="25vh", elem_classes="placeholder-content")
        
        # Create custom chatbot
        chatbot = gr.Chatbot(
            height="78vh",
            show_copy_button=False,
            placeholder="START HERE",
            type="messages",
            render_markdown=True,
            show_label=False,
            show_share_button=False,
            visible=False,
        )
        
        # Create right sidebar using Gradio's native Sidebar
        with gr.Sidebar(position="right", open=False) as right_sidebar:
            gr.Markdown("## ⚙️ Settings")
            
            # Add Claude Code configuration settings
            with gr.Group():
                gr.Markdown("### Claude Code Settings")
                cwd_textbox = gr.Dropdown(
                    label="Set Directory",
                    choices=search_directories(""),
                    value="",
                    allow_custom_value=True,
                    interactive=True,
                    filterable=True
                )
                append_system_prompt_textbox = gr.Textbox(
                    label="Additional System Prompt",
                    placeholder="Additional instructions for Claude...",
                    value="",
                    interactive=True,
                    lines=3
                )
            
            # Add some placeholder settings
            with gr.Group():
                gr.Markdown("### Chat Settings")
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["Claude-3", "Claude-2", "GPT-4"],
                    value="Claude-3",
                    interactive=True
                )
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    interactive=True
                )
        
        # Create Scout-style textbox
        scout_textbox, context_button, send_button, mode_toggle, settings_button, scout_css = create_scout_textbox_ui(
            placeholder="Create a website based on my vibes"
        )
        
        # Create info cards for directory and branch - positioned right above chat input
        with gr.Row(elem_classes=["scout-info-cards"]):
            with gr.Column(scale=5):
                gr.Markdown()
            with gr.Column(scale=4, min_width=0):
                # Combined directory and branch info in single markdown
                combined_info = gr.Markdown(
                    value=f"📂 **{get_directory_name()}**   **{get_current_branch()}🌿**",
                    elem_classes=["scout-info-card", "scout-combined-card"]
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
                
                /* Right sidebar styling using Gradio's native sidebar */
                .gradio-sidebar[data-position="right"] {{
                    background: #F8F9FA !important;
                    border-left: 1px solid #E5E7EB !important;
                    border-right: none !important;
                    padding: 16px !important;
                }}
                
                /* iOS-style dropdown styling - remove all borders and lines */
                .secondary-wrap.svelte-1hfxrpf,
                input[role="listbox"].svelte-1hfxrpf,
                .secondary-wrap.svelte-1hfxrpf *,
                input[role="listbox"].svelte-1hfxrpf * {{
                    background: #F9FAFB !important;
                    background-color: #F9FAFB !important;
                    border: none !important;
                    border-radius: 8px !important;
                    color: #374151 !important;
                    font-size: 14px !important;
                    margin-bottom: 8px !important;
                    box-shadow: none !important;
                    outline: none !important;
                }}
                
                /* Target the outer dropdown container and all related elements */
                .gradio-sidebar label,
                .gradio-sidebar label > div,
                .gradio-sidebar div[class*="dropdown"],
                .gradio-sidebar div[class*="wrap"] {{
                    background: #F9FAFB !important;
                    background-color: #F9FAFB !important;
                    border: none !important;
                    border-radius: 8px !important;
                    box-shadow: none !important;
                }}
                
                /* Aggressive border removal for Recent Chats area */
                .gradio-sidebar *[class*="svelte"] {{
                    border: none !important;
                    box-shadow: none !important;
                    outline: none !important;
                }}
                
                /* Ensure the Recent Chats label area also gets the background */
                .gradio-sidebar .block.svelte-1svsvh2,
                .gradio-sidebar .form.svelte-633qhp,
                .gradio-sidebar .container.svelte-1hfxrpf,
                .gradio-sidebar span.svelte-g2oxp3,
                .gradio-sidebar .block,
                .gradio-sidebar .form {{
                    background: #F9FAFB !important;
                    background-color: #F9FAFB !important;
                    border: none !important;
                    border-radius: 8px !important;
                    padding: 8px !important;
                    box-shadow: none !important;
                }}
                
                /* Hide dropdown arrow with maximum specificity */
                .icon-wrap.svelte-1hfxrpf,
                .icon-wrap svg,
                svg.dropdown-arrow.svelte-xjn76a,
                .dropdown-arrow.svelte-xjn76a {{
                    display: none !important;
                    visibility: hidden !important;
                    opacity: 0 !important;
                    width: 0 !important;
                    height: 0 !important;
                }}
                
                /* Style dropdown options when opened */
                .gradio-sidebar .gradio-dropdown .dropdown {{
                    background: #F9FAFB !important;
                    border: 1px solid #E5E7EB !important;
                    border-radius: 8px !important;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
                }}
                
                .gradio-sidebar .gradio-dropdown .option {{
                    background: #F9FAFB !important;
                    color: #374151 !important;
                    padding: 8px 12px !important;
                }}
                
                .gradio-sidebar .gradio-dropdown .option:hover {{
                    background: #F3F4F6 !important;
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
                }}
                
                /* Scout Info Cards - Position closer to chat input */
                .scout-info-cards {{
                    margin-top: 8px !important;
                    margin-bottom: 12px !important;
                }}
                
                /* Make the parent row flex for side-by-side layout */
                .row.scout-info-cards > .svelte-vuh1yp {{
                    display: flex !important;
                    gap: 0px !important;
                    justify-content: center !important;
                    align-items: center !important;
                }}
                
                /* Target the prose elements that wrap our cards */
                .prose.scout-info-card {{
                    background: rgba(0, 122, 255, 0.08) !important;
                    border: 1px solid rgba(0, 122, 255, 0.12) !important;
                    color: #007AFF !important;
                    font-size: 14px !important;
                    font-weight: 500 !important;
                    padding: 8px 14px !important;
                    border-radius: 12px !important;
                    transition: all 0.15s ease !important;
                    height: auto !important;
                    min-height: 32px !important;
                    box-shadow: 0 1px 3px rgba(0, 122, 255, 0.08) !important;
                    display: inline-flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    margin: 0 !important;
                }}
                
                .prose.scout-branch-card {{
                    background: rgba(52, 199, 89, 0.08) !important;
                    border: 1px solid rgba(52, 199, 89, 0.12) !important;
                }}
                
                .prose.scout-info-card:hover {{
                    background: rgba(0, 122, 255, 0.12) !important;
                    border-color: rgba(0, 122, 255, 0.18) !important;
                    transform: translateY(-0.5px) !important;
                }}
                
                .prose.scout-branch-card:hover {{
                    background: rgba(52, 199, 89, 0.12) !important;
                    border-color: rgba(52, 199, 89, 0.18) !important;
                }}
                
                .prose.scout-info-card p {{
                    margin: 0 !important;
                    padding: 0 !important;
                    font-size: 14px !important;
                    font-weight: 500 !important;
                    color: #007AFF !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    gap: 6px !important;
                }}
                
                .prose.scout-branch-card p {{
                    color: #34C759 !important;
                }}
                
                /* Placeholder content styling - target the actual Gradio markdown block structure */
                .block.placeholder-content,
                div.block.placeholder-content {{
                    display: flex !important;
                    align-items: flex-end !important;
                    justify-content: center !important;
                    height: 35vh !important;
                    min-height: 35vh !important;
                    width: 100% !important;
                    margin-bottom: 20px !important;
                }}
                
                .block.placeholder-content .prose,
                .block.placeholder-content div,
                div.block.placeholder-content .prose,
                div.block.placeholder-content div {{
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    height: 100% !important;
                    width: 100% !important;
                    margin: 0 !important;
                    padding: 0 !important;
                }}
                
                .block.placeholder-content p,
                div.block.placeholder-content p {{
                    margin: 0 !important;
                    padding: 0 !important;
                    font-size: 2.5rem !important;
                    font-weight: 400 !important;
                    color: #64748B !important;
                    text-align: center !important;
                    line-height: 1.2 !important;
                    letter-spacing: -0.02em !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                }}`;
                document.head.appendChild(style);
                
                // Set initial mode attribute and style placeholder
                setTimeout(() => {{
                    const toggleBtn = document.querySelector('.scout-mode-toggle');
                    const textboxWrapper = document.querySelector('.scout-textbox-wrapper');
                    if (toggleBtn) {{
                        toggleBtn.setAttribute('data-mode', 'Scout');
                    }}
                    if (textboxWrapper) {{
                        textboxWrapper.setAttribute('data-mode', 'Scout');
                    }}
                    
                    // Style the info cards to look like buttons
                    const cards = document.querySelectorAll('p');
                    let foundCards = [];
                    
                    for (let card of cards) {{
                        const text = card.textContent.trim();
                        if (text.includes('scout') || text.includes('chat_under_branch')) {{
                            let current = card;
                            while (current.parentElement) {{
                                current = current.parentElement;
                                if (current.classList.contains('scout-info-card')) {{
                                    foundCards.push({{
                                        text: text,
                                        element: current,
                                        classes: current.className
                                    }});
                                    break;
                                }}
                            }}
                        }}
                    }}
                    
                    // Apply styling directly to found elements
                    foundCards.forEach((cardInfo, index) => {{
                        const element = cardInfo.element;
                        
                        // Base button styling like "Add context"
                        element.style.cssText = `
                            background: rgba(0, 122, 255, 0.08) !important;
                            border: 1px solid rgba(0, 122, 255, 0.12) !important;
                            color: #007AFF !important;
                            font-size: 14px !important;
                            font-weight: 500 !important;
                            padding: 8px 14px !important;
                            border-radius: 12px !important;
                            transition: all 0.15s ease !important;
                            height: auto !important;
                            min-height: 32px !important;
                            box-shadow: 0 1px 3px rgba(0, 122, 255, 0.08) !important;
                            display: inline-flex !important;
                            align-items: center !important;
                            justify-content: center !important;
                            margin: 0 4px !important;
                            cursor: pointer !important;
                        `;
                        
                        // Branch card gets green styling
                        if (cardInfo.text.includes('chat_under_branch')) {{
                            element.style.background = 'rgba(52, 199, 89, 0.08) !important';
                            element.style.borderColor = 'rgba(52, 199, 89, 0.12) !important';
                            
                            const p = element.querySelector('p');
                            if (p) {{
                                p.style.color = '#34C759 !important';
                            }}
                        }}
                        
                        // Style the paragraph inside
                        const p = element.querySelector('p');
                        if (p) {{
                            p.style.cssText += `
                                margin: 0 !important;
                                padding: 0 !important;
                                font-size: 14px !important;
                                font-weight: 500 !important;
                                display: flex !important;
                                align-items: center !important;
                                justify-content: center !important;
                                gap: 6px !important;
                            `;
                        }}
                        
                        // Add hover effects
                        element.addEventListener('mouseenter', () => {{
                            if (cardInfo.text.includes('chat_under_branch')) {{
                                element.style.background = 'rgba(52, 199, 89, 0.12) !important';
                                element.style.borderColor = 'rgba(52, 199, 89, 0.18) !important';
                            }} else {{
                                element.style.background = 'rgba(0, 122, 255, 0.12) !important';
                                element.style.borderColor = 'rgba(0, 122, 255, 0.18) !important';
                            }}
                            element.style.transform = 'translateY(-0.5px) !important';
                        }});
                        
                        element.addEventListener('mouseleave', () => {{
                            if (cardInfo.text.includes('chat_under_branch')) {{
                                element.style.background = 'rgba(52, 199, 89, 0.08) !important';
                                element.style.borderColor = 'rgba(52, 199, 89, 0.12) !important';
                            }} else {{
                                element.style.background = 'rgba(0, 122, 255, 0.08) !important';
                                element.style.borderColor = 'rgba(0, 122, 255, 0.12) !important';
                            }}
                            element.style.transform = 'translateY(0px) !important';
                        }});
                    }});
                    
                    // Make the container row flex
                    const row = document.querySelector('.scout-info-cards');
                    if (row) {{
                        const container = row.querySelector('.svelte-vuh1yp');
                        if (container) {{
                            container.style.cssText = `
                                display: flex !important;
                                gap: 8px !important;
                                justify-content: center !important;
                                align-items: center !important;
                            `;
                        }}
                    }}
                    
                    // Style the placeholder content
                    const dropIdeas = document.querySelector('p');
                    if (dropIdeas && dropIdeas.textContent.trim() === 'Drop Ideas') {{
                        const container = dropIdeas.closest('.block.placeholder-content');
                        if (container) {{
                            container.style.cssText += `
                                display: flex !important;
                                align-items: flex-end !important;
                                justify-content: center !important;
                                height: 35vh !important;
                                min-height: 35vh !important;
                                width: 100% !important;
                                margin-bottom: 20px !important;
                            `;
                            
                            const prose = container.querySelector('.prose');
                            if (prose) {{
                                prose.style.cssText += `
                                    display: flex !important;
                                    align-items: center !important;
                                    justify-content: center !important;
                                    height: 100% !important;
                                    width: 100% !important;
                                    margin: 0 !important;
                                    padding: 0 !important;
                                `;
                            }}
                            
                            dropIdeas.style.cssText += `
                                margin: 0 !important;
                                padding: 0 !important;
                                font-size: 2.5rem !important;
                                font-weight: 400 !important;
                                color: #64748B !important;
                                text-align: center !important;
                                line-height: 1.2 !important;
                                letter-spacing: -0.02em !important;
                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                            `;
                        }}
                    }}
                }}, 500);
            }}
        """)
        
        # Directory search handler
        def update_directory_choices(current_value):
            """Update directory choices based on current input."""
            if current_value and len(current_value) >= 2:
                dirs = search_directories(current_value)
                return gr.update(choices=dirs)
            else:
                dirs = search_directories("")
                return gr.update(choices=dirs)
        
        def update_info_cards(cwd_value):
            """Update info cards with current directory and branch."""
            dir_name = get_directory_name(cwd_value)
            branch_name = get_current_branch()
            
            combined_content = f"🌱 **{dir_name}**      🌿 **{branch_name}**"
            
            return gr.update(value=combined_content)
        
        def update_directory_and_cards(cwd_value):
            """Update both directory choices and info cards."""
            choices_update = update_directory_choices(cwd_value)
            combined_update = update_info_cards(cwd_value)
            return choices_update, combined_update
        
        # Chat management functions
        def load_chat_list():
            """Load and format chat list for dropdown."""
            chats = chat_manager.get_chats()
            if not chats:
                return gr.update(choices=[], value=None)
            
            choices = []
            for chat in chats:
                title = chat['title'][:50] + "..." if len(chat['title']) > 50 else chat['title']
                choices.append((title, chat['id']))
            
            return gr.update(choices=choices, value=None)
        
        def create_new_chat():
            """Create a new chat and update the UI."""
            chat_id = chat_manager.create_chat()
            updated_dropdown = load_chat_list()
            return (
                updated_dropdown,  # Update chat dropdown
                [],  # Clear chatbot
                gr.update(visible=False),  # Hide chatbot
                gr.update(visible=True),   # Show placeholder
                chat_id,  # Update current_chat_id
                None      # Reset current_session_id
            )
        
        def load_selected_chat(chat_id):
            """Load a selected chat."""
            if not chat_id:
                return [], gr.update(visible=False), gr.update(visible=True), None, None
            
            chat = chat_manager.get_chat(chat_id)
            if chat:
                messages = chat['messages']
                session_id = chat['session_id']
                if messages:
                    return (
                        messages,  # Load chat history
                        gr.update(visible=True),   # Show chatbot
                        gr.update(visible=False),  # Hide placeholder
                        session_id,  # Update session_id
                        chat_id  # Update current_chat_id
                    )
            return [], gr.update(visible=False), gr.update(visible=True), None, None
        
        def delete_selected_chat(chat_id):
            """Delete the selected chat."""
            if chat_id:
                chat_manager.delete_chat(chat_id)
            updated_dropdown = load_chat_list()
            return (
                updated_dropdown,  # Update chat dropdown
                [],  # Clear chatbot
                gr.update(visible=False),  # Hide chatbot
                gr.update(visible=True),   # Show placeholder
                None,  # Reset current_chat_id
                None   # Reset current_session_id
            )
        
        # Custom chat function wrapper
        def handle_chat(message, history, chat_id_state, session_id_state, selected_mode, cwd_value, append_prompt_value):
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
            
            # Call the sync chat function with streaming, passing the selected mode and settings
            response_gen = chat_function_sync(message, history, current_session_id, selected_mode, cwd_value, append_prompt_value)
            
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
            inputs=[scout_textbox, chatbot, current_chat_id, current_session_id, current_mode, cwd_textbox, append_system_prompt_textbox],
            outputs=[chatbot, scout_textbox, chatbot, placeholder_md, current_chat_id, current_session_id],
            queue=True,
        )
        
        scout_textbox.submit(
            fn=handle_chat,
            inputs=[scout_textbox, chatbot, current_chat_id, current_session_id, current_mode, cwd_textbox, append_system_prompt_textbox],
            outputs=[chatbot, scout_textbox, chatbot, placeholder_md, current_chat_id, current_session_id],
            queue=True,
        )
        
        # Connect chat dropdown to load selected chat
        chat_dropdown.change(
            fn=load_selected_chat,
            inputs=[chat_dropdown],
            outputs=[chatbot, chatbot, placeholder_md, current_session_id, current_chat_id]
        )
        
        # Connect sidebar buttons
        new_chat_btn.click(
            fn=create_new_chat,
            inputs=[],
            outputs=[chat_dropdown, chatbot, chatbot, placeholder_md, current_chat_id, current_session_id],
            js="""
            function() {
                // Re-apply placeholder styling when new chat is created
                setTimeout(() => {
                    // Find the placeholder container first, then the paragraph
                    const container = document.querySelector('.block.placeholder-content');
                    if (container && container.style.display !== 'none') {
                        const dropIdeas = container.querySelector('p');
                        if (dropIdeas && dropIdeas.textContent.trim() === 'Drop Ideas') {
                            container.style.cssText += `
                                display: flex !important;
                                align-items: flex-end !important;
                                justify-content: center !important;
                                height: 35vh !important;
                                min-height: 35vh !important;
                                width: 100% !important;
                                margin-bottom: 20px !important;
                            `;
                            
                            const prose = container.querySelector('.prose');
                            if (prose) {
                                prose.style.cssText += `
                                    display: flex !important;
                                    align-items: center !important;
                                    justify-content: center !important;
                                    height: 100% !important;
                                    width: 100% !important;
                                    margin: 0 !important;
                                    padding: 0 !important;
                                `;
                            }
                            
                            dropIdeas.style.cssText += `
                                margin: 0 !important;
                                padding: 0 !important;
                                font-size: 2.5rem !important;
                                font-weight: 400 !important;
                                color: #64748B !important;
                                text-align: center !important;
                                line-height: 1.2 !important;
                                letter-spacing: -0.02em !important;
                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                            `;
                        }
                    }
                }, 100);
            }
            """
        )
        
        delete_chat_btn.click(
            fn=delete_selected_chat,
            inputs=[current_chat_id],
            outputs=[chat_dropdown, chatbot, chatbot, placeholder_md, current_chat_id, current_session_id]
        )
        
        # Connect directory search functionality
        cwd_textbox.change(
            fn=update_directory_and_cards,
            inputs=[cwd_textbox],
            outputs=[cwd_textbox, combined_info]
        )
        
        # Load chat list and info cards on startup
        def initialize_ui():
            chat_list = load_chat_list()
            combined_update = update_info_cards("")
            return chat_list, combined_update
        
        demo.load(
            fn=initialize_ui,
            inputs=[],
            outputs=[chat_dropdown, combined_info]
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
        
        # State to track right sidebar open/closed  
        right_sidebar_open = gr.State(False)
        
        # Function to toggle right sidebar state
        def toggle_right_sidebar_state(current_open):
            new_state = not current_open
            return new_state, gr.update(open=new_state)
        
        # Connect settings button to toggle right sidebar state
        settings_button.click(
            fn=toggle_right_sidebar_state,
            inputs=[right_sidebar_open],
            outputs=[right_sidebar_open, right_sidebar]
        )
        
        # Listen to state changes and trigger sidebar events
        right_sidebar_open.change(
            fn=lambda is_open: None,
            inputs=[right_sidebar_open],
            outputs=[],
            js="""
            function(is_open) {
                console.log('Triggering right sidebar toggle...');
                
                // Super direct approach: simulate Playwright's exact selection
                // Use Playwright's approach: getByRole('button', { name: 'Toggle Sidebar' }).nth(1)
                setTimeout(() => {
                    try {
                        // Get all buttons with "Toggle Sidebar" text
                        const toggleButtons = Array.from(document.querySelectorAll('button')).filter(btn => {
                            // Check both textContent and innerText for robustness  
                            const text = (btn.textContent || btn.innerText || '').trim();
                            return text === 'Toggle Sidebar';
                        });
                        
                        console.log('Found Toggle Sidebar buttons:', toggleButtons.length);
                        
                        if (toggleButtons.length >= 2) {
                            // Click the second one (index 1) - this is the right sidebar
                            const rightToggle = toggleButtons[1];
                            console.log('SUCCESS: Clicking RIGHT sidebar toggle button!');
                            rightToggle.click();
                        } else {
                            console.log('Fallback: Looking for buttons by position...');
                            // Absolute fallback: find button in right half of screen
                            const rightSideButtons = Array.from(document.querySelectorAll('button')).filter(btn => {
                                const rect = btn.getBoundingClientRect();
                                return rect.left > window.innerWidth / 2 && 
                                       (btn.getAttribute('aria-expanded') !== null || 
                                        (btn.textContent && btn.textContent.includes('Toggle')));
                            });
                            
                            if (rightSideButtons.length > 0) {
                                console.log('Found right-side button, clicking:', rightSideButtons[0]);
                                rightSideButtons[0].click();
                            }
                        }
                    } catch (error) {
                        console.error('Error in sidebar toggle:', error);
                    }
                }, 100);
            }
            """
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
