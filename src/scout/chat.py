# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio",
#     "claude-code-sdk",
#     "httpx",
#     "asyncio",
# ]
# ///
import argparse
import re
import json
import asyncio
import time
import os
import subprocess
import httpx
from typing import AsyncGenerator

from logger import get_logger

# Set up logger
logger = get_logger("chat", level="DEBUG")

import gradio as gr
from css import theme
from scout_component import create_scout_textbox_ui
from chat_manager import ChatManager
from utils import status_messages

# FastAPI configuration
API_BASE_URL = "http://0.0.0.0:8000"


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


async def start_chat_task(message, session_id=None, mode="Scout", cwd="", append_system_prompt=""):
    """Start a chat task using FastAPI backend with session awareness."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/chat", json={
            "message": message,
            "session_id": session_id,
            "mode": mode,
            "cwd": cwd,
            "append_system_prompt": append_system_prompt
        })
        response.raise_for_status()
        result = response.json()
        
        # Handle existing session case
        if result["status"] == "existing_session":
            logger.info(f"Using existing task {result['task_id']} for session {session_id}")
        
        return result["task_id"]

async def stream_chat_events(task_id):
    """Stream events from FastAPI backend using SSE."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("GET", f"{API_BASE_URL}/chat/{task_id}/stream") as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip():
                        try:
                            event_data = json.loads(data)
                            yield event_data
                        except json.JSONDecodeError:
                            continue
                elif line.startswith("event: "):
                    # SSE event type line, continue to next line for data
                    continue

def chat_function_sync(message, history, session_id=None, mode="Scout", cwd="", append_system_prompt=""):
    """
    Non-blocking chat function - sends request and returns immediately with task info.
    """
    import asyncio
    
    async def send_chat_request():
        try:
            # Start chat task
            task_id = await start_chat_task(
                message=message,
                session_id=session_id,
                mode=mode,
                cwd=cwd,
                append_system_prompt=append_system_prompt
            )
            
            # Wait for task to start and extract the real session_id
            # Poll until we get the session_id from events
            real_session_id = session_id  # fallback
            
            for attempt in range(8):  # Increased attempts
                await asyncio.sleep(0.8)  # Longer wait
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{API_BASE_URL}/chat/{task_id}/status")
                    response.raise_for_status()
                    task_data = response.json()
                    
                    logger.info(f"🔍 Attempt {attempt + 1}: task {task_id} status={task_data['status']}, events={len(task_data.get('events', []))}")
                    
                    # Extract session_id from SystemMessage events
                    for event in task_data.get("events", []):
                        if (event.get('type') == 'SystemMessage' and 
                            event.get('data', {}).get('session_id')):
                            real_session_id = event['data']['session_id']
                            logger.info(f"✅ Extracted real session_id: {real_session_id} from task {task_id}")
                            return task_id, real_session_id
                    
                    # If task is completed but no session_id found, something is wrong
                    if task_data["status"] in ["completed", "failed"]:
                        logger.error(f"❌ Task {task_id} completed but no session_id found in events!")
                        break
            
            # Return with fallback session_id if we couldn't extract it
            return task_id, real_session_id
                
        except Exception as e:
            logger.error(f"Failed to send chat request: {e}")
            return None, session_id
    
    # Run the async request - use asyncio.run for simplicity
    try:
        task_id, extracted_session_id = asyncio.run(send_chat_request())
    except Exception as e:
        logger.error(f"Failed to send chat request: {e}")
        task_id, extracted_session_id = None, session_id
    
    if task_id:
        # Return immediately with a "sent" indicator
        return [
            gr.ChatMessage(
                role="assistant", 
                content=f"💭 Message sent to background processing...\n*Task ID: {task_id[:8]}*"
            )
        ], extracted_session_id, task_id
    else:
        # Return error message
        return [
            gr.ChatMessage(
                role="assistant", 
                content="⚠️ Failed to send message. Please check your API server connection."
            )
        ], session_id, None
        
        


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
        current_active_task_id = gr.State(None)  # Track the currently active/running task
        current_active_session_id = gr.State(None)  # Track the session ID of the currently active task
        
        # Create main tabs
        with gr.Tabs() as main_tabs:
            # Scout tab with all existing functionality
            with gr.Tab("Scout"):
                with gr.Sidebar(open=True):
                    gr.Markdown("## 🪡 Scout Chats")
                    
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
                    
                    # Refresh chat results button and auto-refresh toggle
                    with gr.Row():
                        refresh_chat_btn = gr.Button("🔄 Refresh Results", variant="secondary", size="sm")
                        auto_refresh_chat_checkbox = gr.Checkbox(label="Auto-refresh every 3s", value=True)
                    

                    gr.Markdown("## 🪼 Modes")
                    mode_radio = gr.Radio(
                            ["Scout", "DeepResearch", "AgenticSearch"], 
                            label="Modes", show_label=False,
                            value='Scout'
                            )

                    # Latest 10 chats radio buttons
                    gr.Markdown("## ✨ Threads")
                    latest_chats_radio = gr.Radio(
                        label="Quick Access",
                        choices=[],
                        value=None,
                        interactive=True,
                        show_label=False,
                        container=False,
                        elem_classes=["sidebar-quick-chats"]
                    )

                # Create main content area with tight spacing
                with gr.Column():
                    # Flexible spacer to push content down
                    flexible_spacer = gr.Markdown("", elem_classes="flexible-spacer")
                    
                    # Centered placeholder with minimal bottom margin
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown()  # Left spacer
                        with gr.Column(scale=3):
                            placeholder_md = gr.Markdown(
                                "Drop Ideas", 
                                visible=True, 
                                elem_classes="placeholder-content-tight"
                            )
                        with gr.Column(scale=1):
                            gr.Markdown()  # Right spacer
                    
                    # Custom chatbot
                    chatbot = gr.Chatbot(
                        height="73vh",
                        show_copy_button=False,
                        placeholder="START HERE",
                        type="messages",
                        render_markdown=True,
                        show_label=False,
                        show_share_button=False,
                        visible=False,
                    )
                    
                    # Info cards removed from here - moved to right sidebar
                    
                    # Create Scout-style textbox
                    scout_textbox, context_button, send_button, mode_toggle, settings_button, textbox_wrapper, scout_css = create_scout_textbox_ui(
                        placeholder="Create a website based on my vibes"
                    )
                
                # Create right sidebar using Gradio's native Sidebar with tabs
                with gr.Sidebar(position="right", open=False) as right_sidebar:
                    with gr.Tabs(elem_classes=["sidebar-tabs"]):
                        # Settings Tab - contains all previous sidebar content
                        with gr.Tab("⚙️ Settings"):
                            # Directory and Branch Info
                            gr.Markdown("## 📁 Current Context")
                            combined_info = gr.Markdown(
                                value=f"📂 **{get_directory_name()}**\n🌿 **{get_current_branch()}**",
                                elem_classes=["sidebar-info-card"]
                            )
                            
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
                        
                        # Live Tasks Tab - contains task events stream
                        with gr.Tab("📊 Live Tasks"):
                            # iOS-style event viewer for sidebar - clean and borderless
                            sidebar_event_header = gr.Markdown("*Select a task to view its live events*", visible=True, elem_classes=["ios-task-viewer-header"])
                            
                            # Clean event display for sidebar - no heavy containers
                            sidebar_events_display = gr.Markdown("", visible=False, elem_classes=["ios-events-display"])
                            
                            # Minimal event statistics for sidebar
                            sidebar_event_stats = gr.Markdown("", visible=False, elem_classes=["ios-event-stats"])
                            
                            # Auto-refresh toggle with iOS styling for sidebar
                            sidebar_auto_refresh_events = gr.Checkbox(label="Auto-refresh events", value=True, visible=False, elem_classes=["ios-toggle"])
            
            # Workspace tab with task management
            with gr.Tab("Workspace"):
                # Auto-refresh and manual refresh
                with gr.Row():
                    gr.Markdown() # dummy spaces for ui alignment
                    gr.Markdown()
                    gr.Markdown()
                    refresh_tasks_btn = gr.Button("🔄 Refresh Tasks", variant="primary", size="sm")
                    auto_refresh_checkbox = gr.Checkbox(label="Auto-refresh every 2s", value=True)
                
                # Three-column layout with iOS-style task cards
                with gr.Row(equal_height=True):
                    # Left column - Completed Tasks
                    with gr.Column(scale=1):
                        gr.Markdown("### ✅ Completed")
                        completed_tasks_radio = gr.Radio(
                            choices=[],
                            value=None,
                            interactive=True,
                            show_label=False,
                            elem_classes=["ios-task-cards", "completed-tasks"]
                        )
                        # Action buttons for completed tasks
                        with gr.Row(visible=False) as completed_actions:
                            completed_view_btn = gr.Button("👁️ View Events", size="sm", elem_classes=["ios-action-btn"])
                            completed_delete_btn = gr.Button("🗑️ Delete", size="sm", variant="secondary", elem_classes=["ios-action-btn"])
                    
                    # Middle column - Ongoing Tasks  
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔄 Ongoing")
                        ongoing_tasks_radio = gr.Radio(
                            choices=[],
                            value=None,
                            interactive=True,
                            show_label=False,
                            elem_classes=["ios-task-cards", "ongoing-tasks"]
                        )
                        # Action buttons for ongoing tasks
                        with gr.Row(visible=False) as ongoing_actions:
                            ongoing_view_btn = gr.Button("👁️ View Events", size="sm", elem_classes=["ios-action-btn"])
                            ongoing_stop_btn = gr.Button("⏹️ Stop", size="sm", variant="secondary", elem_classes=["ios-action-btn"])
                            ongoing_delete_btn = gr.Button("🗑️ Delete", size="sm", variant="secondary", elem_classes=["ios-action-btn"])
                    
                    # Right column - Task Results/Events  
                    with gr.Column(scale=1):
                        
                        # iOS-style event viewer - clean and borderless
                        event_header = gr.Markdown("*Select a task to view its events and results*", visible=True, elem_classes=["ios-task-viewer-header"])
                        
                        # Clean event display - no heavy containers
                        events_display = gr.Markdown("", visible=False, elem_classes=["ios-events-display"])
                        
                        # Minimal event statistics
                        event_stats = gr.Markdown("", visible=False, elem_classes=["ios-event-stats"])
                        
                        # Auto-refresh toggle with iOS styling
                        auto_refresh_events = gr.Checkbox(label="Auto-refresh events", value=True, visible=False, elem_classes=["ios-toggle"])
                        
                        # Selected task ID state
                        selected_task_id = gr.State(None)
        
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
                }}

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
                    white-space: nowrap !important;
                    min-width: max-content !important;
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
                    white-space: nowrap !important;
                    min-width: max-content !important;
                }}
                
                .prose.scout-branch-card p {{
                    color: #34C759 !important;
                }}
                
                /* Flexible spacer for pushing content down */
                .flexible-spacer {{
                    flex: 1 !important;
                    min-height: 20vh !important;
                }}
                
                /* Tight placeholder content styling */
                .block.placeholder-content-tight,
                div.block.placeholder-content-tight {{
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    margin: 0 !important;
                    padding: 8px 0 !important;
                    width: 100% !important;
                }}
                
                .block.placeholder-content-tight .prose,
                .block.placeholder-content-tight div,
                div.block.placeholder-content-tight .prose,
                div.block.placeholder-content-tight div {{
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    width: 100% !important;
                }}
                
                .block.placeholder-content-tight p,
                div.block.placeholder-content-tight p {{
                    margin: 0 !important;
                    padding: 0 !important;
                    font-size: 2.5rem !important;
                    font-weight: 400 !important;
                    color: #64748B !important;
                    text-align: center !important;
                    line-height: 1.2 !important;
                    letter-spacing: -0.02em !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                }}
                
                /* Scout Info Cards - Position closer to chat input */
                .scout-info-cards {{
                    margin-top: 4px !important;
                    margin-bottom: 0px !important;
                }}
                
                /* Scout/Workspace tabs styling - Glass-themed aesthetic */
                
                /* Create glass-themed background container for tabs */
                .tab-wrapper.svelte-1tcem6n {{
                    background: rgba(255, 255, 255, 0.8) !important;
                    backdrop-filter: blur(20px) !important;
                    -webkit-backdrop-filter: blur(20px) !important;
                    border: 1px solid rgba(255, 255, 255, 0.2) !important;
                    border-radius: 12px !important;
                    padding: 3px !important;
                    margin: 16px 0 16px 16px !important;
                    display: inline-flex !important;
                    width: fit-content !important;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1), 
                                0 1px 3px rgba(0, 0, 0, 0.1),
                                inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
                }}
                
                /* Style individual tab buttons within the glass container */
                .tab-wrapper.svelte-1tcem6n button.svelte-1tcem6n {{
                    background: transparent !important;
                    border: none !important;
                    border-radius: 9px !important;
                    padding: 8px 16px !important;
                    margin: 0 1px !important;
                    font-size: 14px !important;
                    font-weight: 500 !important;
                    color: #64748B !important;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                    box-shadow: none !important;
                    cursor: pointer !important;
                    min-width: 80px !important;
                }}
                
                /* Selected tab styling - glass pill effect */
                .tab-wrapper.svelte-1tcem6n button.svelte-1tcem6n.selected,
                .tab-wrapper.svelte-1tcem6n button.svelte-1tcem6n[aria-selected="true"] {{
                    background: rgba(255, 255, 255, 0.95) !important;
                    backdrop-filter: blur(20px) !important;
                    -webkit-backdrop-filter: blur(20px) !important;
                    border: 1px solid rgba(255, 255, 255, 0.3) !important;
                    color: #1F2937 !important;
                    font-weight: 600 !important;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15),
                                0 1px 3px rgba(0, 0, 0, 0.1),
                                inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
                }}
                
                /* Hover effect for non-selected tabs */
                .tab-wrapper.svelte-1tcem6n button.svelte-1tcem6n:hover:not(.selected):not([aria-selected="true"]) {{
                    color: #475569 !important;
                    background: rgba(255, 255, 255, 0.4) !important;
                    backdrop-filter: blur(10px) !important;
                    -webkit-backdrop-filter: blur(10px) !important;
                }}
                
                /* Remove tab underlines completely */
                .tab-wrapper.svelte-1tcem6n button.svelte-1tcem6n::after,
                .tab-wrapper.svelte-1tcem6n button.svelte-1tcem6n::before,
                .tab-container.svelte-1tcem6n::after,
                .tab-container.svelte-1tcem6n::before,
                *[role="tab"]::after,
                *[role="tab"]::before {{
                    display: none !important;
                    content: none !important;
                    border: none !important;
                    border-bottom: none !important;
                }}
                
                /* Remove blue underline from selected tabs */
                .tab-wrapper.svelte-1tcem6n .selected {{
                    border-bottom: none !important;
                    text-decoration: none !important;
                }}
                
                /* Target any remaining underline elements */
                .tab-wrapper *,
                .tab-container * {{
                    border-bottom: none !important;
                    text-decoration: none !important;
                }}
                
                /* iOS Premium Task Viewer Styling - Glass Morphism & Clean Design */
                .ios-task-viewer-header {{
                    color: #8E8E93 !important;
                    font-size: 15px !important;
                    font-weight: 400 !important;
                    text-align: center !important;
                    margin: 24px 0 32px 0 !important;
                    padding: 0 16px !important;
                    line-height: 1.4 !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
                    background: transparent !important;
                    border: none !important;
                }}
                
                .ios-events-display {{
                    background: rgba(255, 255, 255, 0.85) !important;
                    backdrop-filter: blur(40px) saturate(180%) !important;
                    -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
                    border: none !important;
                    border-radius: 20px !important;
                    padding: 24px !important;
                    margin: 8px 0 16px 0 !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
                    font-size: 15px !important;
                    line-height: 1.5 !important;
                    color: #1D1D1F !important;
                    overflow-y: auto !important;
                    max-height: 500px !important;
                    white-space: pre-wrap !important;
                    word-wrap: break-word !important;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06), 
                                0 4px 16px rgba(0, 0, 0, 0.04),
                                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                }}
                
                .ios-events-display:hover {{
                    background: rgba(255, 255, 255, 0.92) !important;
                    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08), 
                                0 6px 20px rgba(0, 0, 0, 0.06),
                                inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
                    transform: translateY(-2px) !important;
                }}
                
                .ios-event-stats {{
                    background: rgba(248, 249, 250, 0.8) !important;
                    backdrop-filter: blur(20px) !important;
                    -webkit-backdrop-filter: blur(20px) !important;
                    border: none !important;
                    border-radius: 16px !important;
                    padding: 16px 20px !important;
                    margin-top: 16px !important;
                    font-size: 14px !important;
                    font-weight: 500 !important;
                    color: #6B7280 !important;
                    text-align: center !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04),
                                inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
                }}
                
                .ios-toggle {{
                    background: transparent !important;
                    border: none !important;
                    padding: 12px 0 !important;
                    margin-top: 20px !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
                    font-size: 15px !important;
                    color: #1D1D1F !important;
                }}
                
                /* Enhanced code blocks within events */
                .ios-events-display code {{
                    background: rgba(0, 122, 255, 0.08) !important;
                    color: #007AFF !important;
                    padding: 2px 6px !important;
                    border-radius: 6px !important;
                    font-size: 14px !important;
                    font-family: 'SF Mono', Monaco, monospace !important;
                    border: none !important;
                }}
                
                .ios-events-display strong {{
                    color: #1D1D1F !important;
                    font-weight: 600 !important;
                }}
                
                .ios-events-display em {{
                    color: #6B7280 !important;
                    font-style: normal !important;
                    font-weight: 400 !important;
                }}
                
                /* iOS Premium Sidebar Tabs Styling */
                .sidebar-tabs .tab-nav {{
                    background: rgba(255, 255, 255, 0.9) !important;
                    backdrop-filter: blur(20px) !important;
                    -webkit-backdrop-filter: blur(20px) !important;
                    border: none !important;
                    border-radius: 12px !important;
                    padding: 4px !important;
                    margin-bottom: 16px !important;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
                }}
                
                .sidebar-tabs .tab-nav button {{
                    background: transparent !important;
                    border: none !important;
                    border-radius: 8px !important;
                    padding: 8px 12px !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
                    font-size: 14px !important;
                    font-weight: 500 !important;
                    color: #8E8E93 !important;
                    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
                }}
                
                .sidebar-tabs .tab-nav button.selected {{
                    background: rgba(0, 122, 255, 0.1) !important;
                    color: #007AFF !important;
                    font-weight: 600 !important;
                }}
                
                .sidebar-tabs .tab-nav button:hover {{
                    background: rgba(0, 122, 255, 0.05) !important;
                    color: #007AFF !important;
                }}
                
                /* Workspace two-column layout */
                .workspace-columns {{
                    display: flex !important;
                    gap: 24px !important;
                    align-items: flex-start !important;
                }}
                
                .workspace-left-column {{
                    flex: 1 !important;
                    min-width: 0 !important;
                }}
                
                .workspace-right-column {{
                    flex: 1 !important;
                    min-width: 0 !important;
                }}
                
                /* Task card select button styling */
                button[data-testid*="select"] {{
                    background: #007AFF !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 4px !important;
                    padding: 4px 8px !important;
                    font-size: 12px !important;
                    min-width: 24px !important;
                }}
                
                button[data-testid*="select"]:hover {{
                    background: #0056CC !important;
                }}
                
                /* iOS-style Task Radio Cards */
                .ios-task-cards {{
                    padding: 8px 0 !important;
                }}
                
                /* Hide default radio buttons and style as iOS cards */
                .ios-task-cards .wrap,
                .ios-task-cards fieldset {{
                    border: none !important;
                    padding: 0 !important;
                    margin: 0 !important;
                    background: transparent !important;
                }}
                
                .ios-task-cards label {{
                    display: block !important;
                    background: rgba(255, 255, 255, 0.95) !important;
                    backdrop-filter: blur(20px) !important;
                    -webkit-backdrop-filter: blur(20px) !important;
                    border: 1px solid rgba(0, 0, 0, 0.06) !important;
                    border-radius: 12px !important;
                    padding: 16px !important;
                    margin: 8px 0 !important;
                    cursor: pointer !important;
                    transition: all 0.2s ease !important;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04), 
                                0 1px 3px rgba(0, 0, 0, 0.06) !important;
                    position: relative !important;
                    overflow: hidden !important;
                }}
                
                .ios-task-cards label:hover {{
                    transform: translateY(-1px) !important;
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08), 
                                0 2px 6px rgba(0, 0, 0, 0.1) !important;
                    border-color: rgba(0, 122, 255, 0.2) !important;
                }}
                
                /* Selected state - iOS blue accent */
                .ios-task-cards input[type="radio"]:checked + label,
                .ios-task-cards label:has(input[type="radio"]:checked) {{
                    background: rgba(0, 122, 255, 0.08) !important;
                    border-color: rgba(0, 122, 255, 0.3) !important;
                    transform: translateY(-1px) !important;
                    box-shadow: 0 4px 16px rgba(0, 122, 255, 0.15), 
                                0 2px 6px rgba(0, 122, 255, 0.1) !important;
                }}
                
                /* Hide default radio button circles */
                .ios-task-cards input[type="radio"] {{
                    display: none !important;
                    opacity: 0 !important;
                    position: absolute !important;
                    width: 0 !important;
                    height: 0 !important;
                }}
                
                /* Style radio button text content */
                .ios-task-cards label span,
                .ios-task-cards .radio-option-text {{
                    font-size: 15px !important;
                    line-height: 1.4 !important;
                    color: #1D1D1F !important;
                    font-weight: 400 !important;
                    white-space: pre-line !important;
                    display: block !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
                }}
                
                /* Completed tasks - green accent */
                .completed-tasks label:hover {{
                    border-color: rgba(52, 199, 89, 0.2) !important;
                }}
                
                .completed-tasks input[type="radio"]:checked + label,
                .completed-tasks label:has(input[type="radio"]:checked) {{
                    background: rgba(52, 199, 89, 0.08) !important;
                    border-color: rgba(52, 199, 89, 0.3) !important;
                    box-shadow: 0 4px 16px rgba(52, 199, 89, 0.15), 
                                0 2px 6px rgba(52, 199, 89, 0.1) !important;
                }}
                
                /* Ongoing tasks - orange accent */
                .ongoing-tasks label:hover {{
                    border-color: rgba(255, 149, 0, 0.2) !important;
                }}
                
                .ongoing-tasks input[type="radio"]:checked + label,
                .ongoing-tasks label:has(input[type="radio"]:checked) {{
                    background: rgba(255, 149, 0, 0.08) !important;
                    border-color: rgba(255, 149, 0, 0.3) !important;
                    box-shadow: 0 4px 16px rgba(255, 149, 0, 0.15), 
                                0 2px 6px rgba(255, 149, 0, 0.1) !important;
                }}
                
                /* iOS action buttons */
                .ios-action-btn {{
                    background: rgba(0, 122, 255, 0.1) !important;
                    border: 1px solid rgba(0, 122, 255, 0.2) !important;
                    border-radius: 8px !important;
                    padding: 8px 16px !important;
                    font-size: 14px !important;
                    font-weight: 500 !important;
                    color: #007AFF !important;
                    transition: all 0.2s ease !important;
                    backdrop-filter: blur(10px) !important;
                    -webkit-backdrop-filter: blur(10px) !important;
                }}
                
                .ios-action-btn:hover {{
                    background: rgba(0, 122, 255, 0.15) !important;
                    border-color: rgba(0, 122, 255, 0.3) !important;
                    transform: translateY(-0.5px) !important;
                }}
                
                /* Sidebar Quick Chats Radio Styling */
                .sidebar-quick-chats {{
                    padding: 8px 0 !important;
                    overflow: visible !important;
                    max-height: none !important;
                    height: auto !important;
                }}
                
                /* Hide scrollbar for sidebar quick chats - aggressive targeting */
                .sidebar-quick-chats,
                .sidebar-quick-chats *,
                .sidebar-quick-chats .wrap,
                .sidebar-quick-chats fieldset,
                .gradio-sidebar .sidebar-quick-chats,
                .gradio-sidebar .sidebar-quick-chats *,
                div[class*="sidebar-quick-chats"],
                div[class*="sidebar-quick-chats"] * {{
                    overflow: hidden !important;
                    scrollbar-width: none !important;
                    -ms-overflow-style: none !important;
                }}
                
                .sidebar-quick-chats::-webkit-scrollbar,
                .sidebar-quick-chats *::-webkit-scrollbar,
                .sidebar-quick-chats .wrap::-webkit-scrollbar,
                .sidebar-quick-chats fieldset::-webkit-scrollbar,
                .gradio-sidebar .sidebar-quick-chats::-webkit-scrollbar,
                .gradio-sidebar .sidebar-quick-chats *::-webkit-scrollbar,
                div[class*="sidebar-quick-chats"]::-webkit-scrollbar,
                div[class*="sidebar-quick-chats"] *::-webkit-scrollbar {{
                    display: none !important;
                    width: 0px !important;
                    height: 0px !important;
                    background: transparent !important;
                }}
                
                /* Force no scrollbar on any gradio radio component in sidebar */
                .gradio-sidebar .gradio-radio,
                .gradio-sidebar .gradio-radio *,
                .gradio-sidebar fieldset {{
                    overflow: hidden !important;
                    scrollbar-width: none !important;
                    -ms-overflow-style: none !important;
                }}
                
                .gradio-sidebar .gradio-radio::-webkit-scrollbar,
                .gradio-sidebar .gradio-radio *::-webkit-scrollbar,
                .gradio-sidebar fieldset::-webkit-scrollbar {{
                    display: none !important;
                    width: 0px !important;
                    height: 0px !important;
                }}
                
                /* Style radio buttons as iOS-style cards */
                .sidebar-quick-chats .wrap,
                .sidebar-quick-chats fieldset {{
                    border: none !important;
                    padding: 0 !important;
                    margin: 0 !important;
                    background: transparent !important;
                }}
                
                .sidebar-quick-chats label {{
                    display: block !important;
                    background: rgba(248, 250, 252, 0.95) !important;
                    backdrop-filter: blur(20px) !important;
                    -webkit-backdrop-filter: blur(20px) !important;
                    border: 1px solid rgba(0, 0, 0, 0.02) !important;
                    border-radius: 8px !important;
                    padding: 12px !important;
                    margin: 4px 0 !important;
                    cursor: pointer !important;
                    transition: all 0.2s ease !important;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.01) !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
                }}
                
                .sidebar-quick-chats label:hover {{
                    background: rgba(0, 122, 255, 0.02) !important;
                    border-color: rgba(0, 122, 255, 0.03) !important;
                    transform: translateY(-0.5px) !important;
                    box-shadow: 0 2px 6px rgba(0, 122, 255, 0.02) !important;
                }}
                
                /* Selected state for sidebar radio */
                .sidebar-quick-chats input[type="radio"]:checked + label,
                .sidebar-quick-chats label:has(input[type="radio"]:checked) {{
                    background: rgba(0, 122, 255, 0.03) !important;
                    border-color: rgba(0, 122, 255, 0.05) !important;
                    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.03) !important;
                }}
                
                /* Hide default radio circles */
                .sidebar-quick-chats input[type="radio"] {{
                    display: none !important;
                    opacity: 0 !important;
                }}
                
                /* Style radio text content */
                .sidebar-quick-chats label span {{
                    font-size: 13px !important;
                    line-height: 1.3 !important;
                    color: #1D1D1F !important;
                    font-weight: 400 !important;
                    display: block !important;
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                    white-space: nowrap !important;
                }}
                
                /* Sidebar heading for Latest Chats */
                .gradio-sidebar h2:has(+ .sidebar-quick-chats) {{
                    font-size: 14px !important;
                    font-weight: 600 !important;
                    color: #6B7280 !important;
                    margin: 16px 0 8px 0 !important;
                    padding: 0 !important;
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
                
                // Removed JavaScript task management functions - now using proper Gradio components
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
        
        def handle_mode_change(selected_mode):
            """Handle mode change and update textbox styling, placeholder, and markdown header."""
            if selected_mode == "DeepResearch":
                # Return updates for wrapper styling, textbox placeholder, and markdown header
                return (
                    gr.update(elem_classes=["scout-textbox-wrapper", "mode-deepresearch"]),
                    gr.update(placeholder="Create a website based on my vibes"),
                    gr.update(value="Drop Research Ideas")
                )
            elif selected_mode == "AgenticSearch": 
                # Return updates for wrapper styling, textbox placeholder, and markdown header
                return (
                    gr.update(elem_classes=["scout-textbox-wrapper", "mode-agenticsearch"]),
                    gr.update(placeholder="Create a website based on my vibes"),
                    gr.update(value="Drop Agent Ideas")
                )
            else:  # Scout mode (default)
                # Return default styling, placeholder, and markdown header
                return (
                    gr.update(elem_classes=["scout-textbox-wrapper"]),
                    gr.update(placeholder="Create a website based on my vibes"),
                    gr.update(value="Drop Ideas")
                )
        
        # Task management functions
        def get_task_cards():
            """Fetch tasks from FastAPI and format as iOS-styled cards."""
            try:
                import requests
                response = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
                response.raise_for_status()
                tasks = response.json()
                
                if not tasks:
                    return "<div class='task-cards-container'><div class='no-tasks-message'>No tasks yet. Start a chat to see tasks here!</div></div>"
                
                cards_html = "<div class='task-cards-container'>"
                
                for task_id, task_data in tasks.items():
                    status = task_data["status"]
                    event_count = task_data["event_count"]
                    error = task_data.get("error", "")
                    
                    # Status emoji and color
                    if status == "completed":
                        status_emoji = "✅"
                        status_class = "completed"
                    elif status == "running":
                        status_emoji = "🔄"
                        status_class = "running"
                    elif status == "failed":
                        status_emoji = "❌"
                        status_class = "failed"
                    elif status == "cancelled":
                        status_emoji = "⏹️"
                        status_class = "cancelled"
                    else:
                        status_emoji = "⏳"
                        status_class = "pending"
                    
                    # Get task details
                    try:
                        detail_response = requests.get(f"{API_BASE_URL}/chat/{task_id}/status", timeout=3)
                        task_details = detail_response.json()
                        message = "Unknown task"
                        if task_details.get("events"):
                            for event in task_details["events"]:
                                if event.get("type") == "SystemMessage" and "data" in event:
                                    # Extract original message from context if available
                                    break
                        # For now, use task_id as identifier
                        message = f"Task {task_id[:8]}..."
                    except:
                        message = f"Task {task_id[:8]}..."
                    
                    # Action buttons
                    action_buttons = ""
                    if status in ["running", "pending"]:
                        action_buttons += f'<button class="task-action-btn stop-btn" onclick="stopTask(\'{task_id}\')">Stop</button>'
                    action_buttons += f'<button class="task-action-btn delete-btn" onclick="deleteTask(\'{task_id}\')">Delete</button>'
                    
                    card_html = f"""
                    <div class='task-card {status_class}' data-task-id='{task_id}'>
                        <div class='task-card-header'>
                            <span class='task-status-badge'>{status_emoji} {status.title()}</span>
                            <span class='task-id'>#{task_id[:8]}</span>
                        </div>
                        <div class='task-message'>{message}</div>
                        <div class='task-stats'>
                            <span class='event-count'>📝 {event_count} events</span>
                            {f'<span class="error-indicator">⚠️ {error[:50]}...</span>' if error else ''}
                        </div>
                        <div class='task-actions'>
                            {action_buttons}
                        </div>
                    </div>
                    """
                    cards_html += card_html
                
                cards_html += "</div>"
                return cards_html
                
            except Exception as e:
                return f"<div class='task-cards-container'><div class='error-message'>Failed to load tasks: {str(e)}</div></div>"
        
        def update_task_radios():
            """Update task radio buttons with rich iOS card-style choices."""
            import datetime
            try:
                import requests
                response = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
                response.raise_for_status()
                sessions_data = response.json()
                
                # Separate sessions by status
                completed_choices = []
                ongoing_choices = []
                
                for session_id, session_data in sessions_data.items():
                    status = session_data["status"]
                    total_events = session_data["total_events"]
                    latest_message = session_data.get("latest_message", "")
                    
                    # Create iOS card-style display text
                    display_message = latest_message[:45] + "..." if len(latest_message) > 45 else latest_message
                    if not display_message:
                        display_message = f"Session #{session_id[:8]}"
                    
                    # Mock timing data for iOS-style display
                    start_time = datetime.datetime.now() - datetime.timedelta(minutes=30)
                    date_str = start_time.strftime("%b %d")
                    time_str = start_time.strftime("%I:%M %p")
                    
                    if status in ["completed", "failed"]:
                        end_time = datetime.datetime.now() - datetime.timedelta(minutes=10)
                        duration_mins = 20
                        status_line = f"✅ Completed • {duration_mins}m"
                    else:
                        duration_mins = (datetime.datetime.now() - start_time).seconds // 60
                        status_line = f"🔄 Running • {duration_mins}m"
                    
                    # Create iOS-style multi-line card text
                    choice_text = f"{display_message}\n{date_str} • {time_str} • {total_events} events\n{status_line}"
                    choice_value = session_id
                    
                    if status == "completed":
                        completed_choices.append((choice_text, choice_value))
                    elif status in ["running", "pending"]:
                        ongoing_choices.append((choice_text, choice_value))
                
                # Update dropdowns
                completed_update = gr.update(choices=completed_choices, value=None)
                ongoing_update = gr.update(choices=ongoing_choices, value=None)
                
                return completed_update, ongoing_update
                
            except Exception as e:
                error_msg = f"❌ Failed to load: {str(e)}"
                error_update = gr.update(choices=[(error_msg, None)], value=None)
                return error_update, error_update
        
        def handle_completed_task_selection(selected_task_id):
            """Handle selection of a completed task - show action buttons."""
            if selected_task_id:
                return (
                    gr.update(visible=True),    # show action buttons
                    selected_task_id            # update selected task state
                )
            else:
                return (
                    gr.update(visible=False),   # hide action buttons
                    None                        # clear selected task state
                )
        
        def handle_ongoing_task_selection(selected_task_id):
            """Handle selection of an ongoing task - show action buttons."""
            if selected_task_id:
                return (
                    gr.update(visible=True),    # show action buttons  
                    selected_task_id            # update selected task state
                )
            else:
                return (
                    gr.update(visible=False),   # hide action buttons
                    None                        # clear selected task state
                )
        
        def view_task_events(session_id):
            """View events for selected task - connects to existing get_task_events function."""
            return get_task_events(session_id)
        
        def stop_task_action(task_id):
            """Stop a specific task."""
            if not task_id:
                return update_task_display()
            
            try:
                import requests
                response = requests.post(f"{API_BASE_URL}/chat/{task_id}/stop", timeout=5)
                response.raise_for_status()
                gr.Info(f"Task {task_id[:8]} stop requested")
            except Exception as e:
                gr.Warning(f"Failed to stop task: {str(e)}")
            
            return update_task_display()
        
        def delete_task_action(task_id):
            """Delete a specific task."""
            if not task_id:
                return update_task_display()
            
            try:
                import requests
                response = requests.delete(f"{API_BASE_URL}/tasks/{task_id}", timeout=5)
                response.raise_for_status()
                gr.Info(f"Task {task_id[:8]} deleted")
            except Exception as e:
                gr.Warning(f"Failed to delete task: {str(e)}")
            
            return update_task_display()
        
        # Event streaming functions
        def get_task_events(session_id):
            """Get events for a specific session from API."""
            if not session_id:
                return gr.update(value="*No task selected*", visible=True), gr.update(value="", visible=False), gr.update(visible=False)
            
            try:
                import requests
                # Get all tasks to find the ones for this session
                response = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
                response.raise_for_status()
                sessions_data = response.json()
                
                if session_id not in sessions_data:
                    return gr.update(value=f"*Session {session_id[:8]}... not found*", visible=True), gr.update(value="", visible=False), gr.update(visible=False)
                
                session_info = sessions_data[session_id]
                tasks = session_info.get("tasks", [])
                
                # Collect all events from all tasks in this session
                all_events = []
                combined_status = session_info.get("status", "unknown")
                
                for task_info in tasks:
                    task_id = task_info["task_id"]
                    try:
                        task_response = requests.get(f"{API_BASE_URL}/chat/{task_id}/status", timeout=3)
                        task_response.raise_for_status()
                        task_data = task_response.json()
                        
                        events = task_data.get("events", [])
                        for event in events:
                            event["_task_id"] = task_id[:8]  # Add task reference
                            all_events.append(event)
                    except Exception as e:
                        logger.debug(f"Failed to get events for task {task_id}: {e}")
                        continue
                
                session_id_short = session_id[:8]
                
                # Build event display
                if not all_events:
                    events_content = f"**Session {session_id_short}** ({combined_status})\n\n*No events yet...*"
                    stats_content = ""
                else:
                    # Header with session info
                    events_content = f"**Session {session_id_short}** ({combined_status})\n\n"
                    
                    # Process each event
                    for i, event in enumerate(all_events, 1):
                        event_type = event.get('type', 'Unknown')
                        task_ref = event.get('_task_id', 'unknown')
                        
                        events_content += f"**{i}.** `{event_type}` *[{task_ref}]*\n"
                        
                        # Handle different event types
                        if event_type == 'SystemMessage':
                            data = event.get('data', {})
                            if data.get('session_id'):
                                events_content += f"   📝 Session: `{data['session_id'][:8]}...`\n"
                            if data.get('original_message'):
                                msg = data['original_message'][:100] + "..." if len(data['original_message']) > 100 else data['original_message']
                                events_content += f"   💬 Message: *{msg}*\n"
                                
                        elif event_type == 'AssistantMessage':
                            content = event.get('content', [])
                            if content and len(content) > 0:
                                if content[0].get('text'):
                                    text = content[0]['text'][:150] + "..." if len(content[0]['text']) > 150 else content[0]['text']
                                    events_content += f"   💭 Response: *{text}*\n"
                                elif content[0].get('name'):
                                    tool_name = content[0]['name']
                                    events_content += f"   🔧 Tool: `{tool_name}`\n"
                                    
                        elif event_type == 'ToolResultMessage':
                            tool_result = event.get('content', [])
                            if tool_result and len(tool_result) > 0:
                                result_text = str(tool_result[0]).get('text', '')[:100] if tool_result[0] else ''
                                if result_text:
                                    events_content += f"   ✅ Result: *{result_text}...*\n"
                                else:
                                    events_content += f"   ✅ Tool completed\n"
                                    
                        elif event_type == 'ResultMessage':
                            result = event.get('result', '')[:150]
                            if result:
                                events_content += f"   🎯 Final: *{result}...*\n"
                        
                        events_content += "\n"
                
                # Build stats
                total_tasks = len(tasks)
                stats_content = f"📊 **{len(all_events)} events** from **{total_tasks} tasks** • Status: **{combined_status}**"
                if session_info.get('error'):
                    stats_content += f" • ⚠️ Error: {session_info['error'][:50]}..."
                
                return (
                    gr.update(value="", visible=False), # Hide header
                    gr.update(value=events_content, visible=True), # Show events
                    gr.update(value=stats_content, visible=True)   # Show stats
                )
                
            except Exception as e:
                error_msg = f"❌ Failed to load events: {str(e)}"
                return (
                    gr.update(value=error_msg, visible=True),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False)
                )
        
        def select_task_for_events(task_index, selected_task_state):
            """Select a session to view its events."""
            try:
                import requests
                response = requests.get(f"{API_BASE_URL}/tasks", timeout=3)
                response.raise_for_status()
                sessions_data = response.json()
                session_list = list(sessions_data.items())
                
                if task_index < len(session_list):
                    session_id = session_list[task_index][0]  # This is actually session_id
                    return get_task_events(session_id) + (session_id,)  # Add session_id to state
                    
            except Exception as e:
                error_msg = f"❌ Failed to select task: {str(e)}"
                return (
                    gr.update(value=error_msg, visible=True),
                    gr.update(value="", visible=False), 
                    gr.update(value="", visible=False),
                    None
                )
            
            return (
                gr.update(value="*No task found*", visible=True),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False), 
                None
            )
        
        def refresh_selected_task_events(selected_session_id):
            """Refresh events for the currently selected session."""
            return get_task_events(selected_session_id)
        
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
        
        def load_latest_chats_radio():
            """Load and format latest 10 chats for radio buttons."""
            chats = chat_manager.get_chats()
            if not chats:
                return gr.update(choices=[], value=None)
            
            # Get latest 10 chats
            latest_chats = chats[:10]
            choices = []
            for chat in latest_chats:
                title = chat['title'][:40] + "..." if len(chat['title']) > 40 else chat['title']
                choices.append((title, chat['id']))
            
            return gr.update(choices=choices, value=None)
        
        def create_new_chat(task_map):
            """Create a new chat and update the UI."""
            chat_id = chat_manager.create_chat()
            chats = chat_manager.get_chats()
            if not chats:
                updated_dropdown = gr.update(choices=[], value=None)
                updated_radio = gr.update(choices=[], value=None)
            else:
                choices = []
                for chat in chats:
                    title = chat['title'][:50] + "..." if len(chat['title']) > 50 else chat['title']
                    choices.append((title, chat['id']))
                updated_dropdown = gr.update(choices=choices, value=None)
                
                # Update radio with latest 10 chats
                latest_chats = chats[:10]
                radio_choices = []
                for chat in latest_chats:
                    title = chat['title'][:40] + "..." if len(chat['title']) > 40 else chat['title']
                    radio_choices.append((title, chat['id']))
                updated_radio = gr.update(choices=radio_choices, value=None)
            
            return (
                updated_dropdown,  # Update chat dropdown with None value
                updated_radio,     # Update radio buttons with latest chats
                [],  # Clear chatbot
                gr.update(visible=False),  # Hide chatbot
                gr.update(visible=True),   # Show placeholder
                gr.update(visible=True),   # Show flexible_spacer
                None,  # Reset current_chat_id to start fresh
                None,  # Reset current_session_id
                task_map  # Keep existing task map
            )
        
        def load_selected_chat(chat_id, task_map):
            """Load a selected chat and check for completed tasks."""
            if not chat_id:
                return [], gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), None, None, task_map
            
            logger.info(f"📂 Loading chat {chat_id}")
            chat = chat_manager.get_chat(chat_id)
            if chat:
                messages = chat['messages']
                session_id = chat['session_id']
                logger.info(f"📂 Loaded chat {chat_id} with session_id: {session_id}")
                
                # Check for completed tasks and update messages
                updated_chatbot, updated_task_map = update_chat_with_results(chat_id, task_map)
                if updated_chatbot.get('value') is not None:
                    messages = updated_chatbot['value']
                
                if messages:
                    return (
                        messages,  # Load chat history (potentially updated)
                        gr.update(visible=True),   # Show chatbot
                        gr.update(visible=False),  # Hide placeholder
                        gr.update(visible=False),  # Hide flexible_spacer
                        session_id,  # Update session_id
                        chat_id,  # Update current_chat_id
                        updated_task_map  # Updated task map
                    )
            return [], gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), None, None, task_map
        
        def delete_selected_chat(chat_id, task_map):
            """Delete the selected chat."""
            if chat_id:
                chat_manager.delete_chat(chat_id)
                # Also remove task map entries for this chat
                if task_map and chat_id in task_map:
                    updated_task_map = task_map.copy()
                    del updated_task_map[chat_id]
                else:
                    updated_task_map = task_map
            else:
                updated_task_map = task_map
                
            chats = chat_manager.get_chats()
            if not chats:
                updated_dropdown = gr.update(choices=[], value=None)
                updated_radio = gr.update(choices=[], value=None)
            else:
                choices = []
                for chat in chats:
                    title = chat['title'][:50] + "..." if len(chat['title']) > 50 else chat['title']
                    choices.append((title, chat['id']))
                updated_dropdown = gr.update(choices=choices, value=None)
                
                # Update radio with latest 10 chats
                latest_chats = chats[:10]
                radio_choices = []
                for chat in latest_chats:
                    title = chat['title'][:40] + "..." if len(chat['title']) > 40 else chat['title']
                    radio_choices.append((title, chat['id']))
                updated_radio = gr.update(choices=radio_choices, value=None)
                
            return (
                updated_dropdown,  # Update chat dropdown
                updated_radio,     # Update radio buttons with latest chats
                [],  # Clear chatbot
                gr.update(visible=False),  # Hide chatbot
                gr.update(visible=True),   # Show placeholder
                gr.update(visible=True),   # Show flexible_spacer
                None,  # Reset current_chat_id
                None,  # Reset current_session_id
                updated_task_map  # Updated task map
            )
        
        # Store ongoing task IDs per session to avoid creating multiple tasks
        session_task_map = gr.State({})
        
        # Store active task IDs for each chat
        chat_task_map = gr.State({})
        
        # Function to fetch and update chat with completed task results
        def update_chat_with_results(chat_id, task_map):
            """Check if any tasks for this chat are completed and update the chat history."""
            if not chat_id or not task_map or chat_id not in task_map:
                return gr.update(), task_map
            
            try:
                import requests
                chat = chat_manager.get_chat(chat_id)
                if not chat:
                    return gr.update(), task_map
                
                current_history = chat['messages']
                updated_history = current_history.copy()
                updated_task_map = task_map.copy()
                
                # Check each task for this chat
                tasks_to_remove = []
                for i, task_id in enumerate(task_map[chat_id]):
                    try:
                        response = requests.get(f"{API_BASE_URL}/chat/{task_id}/status", timeout=5)
                        response.raise_for_status()
                        task_data = response.json()
                        
                        if task_data["status"] in ["completed", "failed", "cancelled"]:
                            # Task is done, update chat history with results
                            if task_data["status"] == "completed" and task_data.get("events"):
                                # Process events to build response messages
                                response_messages = []
                                current_content = ""
                                
                                for event in task_data["events"]:
                                    event_type = event.get('type', 'unknown')
                                    
                                    if event_type == 'AssistantMessage':
                                        content = event.get('content', [])
                                        if content and len(content) > 0:
                                            if content[0].get('text'):
                                                current_content += content[0]['text']
                                            elif content[0].get('name'):
                                                tool_name = content[0]['name']
                                                tool_input = content[0].get('input', {})
                                                response_messages.append(
                                                    gr.ChatMessage(
                                                        role="assistant",
                                                        content=f"Using {tool_name}...\n```json\n{json.dumps(tool_input, indent=2)}\n```",
                                                        metadata={"title": f"🔧 {tool_name}", "status": "done"},
                                                    )
                                                )
                                    
                                    elif event_type == 'ResultMessage':
                                        final_result = event.get('result', '')
                                        if final_result and final_result.strip():
                                            current_content = final_result
                                
                                # Add main response if we have content
                                if current_content:
                                    response_messages.append(gr.ChatMessage(role="assistant", content=current_content))
                                
                                # Replace the "processing" message with actual results
                                # Find and replace the last assistant message that contains "background processing"
                                for j in range(len(updated_history) - 1, -1, -1):
                                    if (updated_history[j].get("role") == "assistant" and 
                                        "background processing" in updated_history[j].get("content", "")):
                                        # Replace with actual results
                                        if response_messages:
                                            # Replace the processing message with the first real message
                                            updated_history[j] = response_messages[0]
                                            # Add any additional messages
                                            for k, msg in enumerate(response_messages[1:], 1):
                                                updated_history.insert(j + k, msg)
                                        break
                            
                            elif task_data["status"] == "failed":
                                error_msg = task_data.get("error", "Unknown error")
                                # Replace processing message with error
                                for j in range(len(updated_history) - 1, -1, -1):
                                    if (updated_history[j].get("role") == "assistant" and 
                                        "background processing" in updated_history[j].get("content", "")):
                                        updated_history[j] = gr.ChatMessage(
                                            role="assistant", 
                                            content=f"⚠️ Task failed: {error_msg}"
                                        )
                                        break
                            
                            # Mark task for removal
                            tasks_to_remove.append(i)
                    
                    except Exception as e:
                        logger.debug(f"Error checking task {task_id}: {e}")
                        continue
                
                # Remove completed tasks
                for i in reversed(tasks_to_remove):
                    updated_task_map[chat_id].pop(i)
                
                # Clean up empty task lists
                if not updated_task_map[chat_id]:
                    del updated_task_map[chat_id]
                
                # Update chat if history changed
                if updated_history != current_history:
                    chat_manager.update_chat(chat_id, updated_history)
                    return gr.update(value=updated_history), updated_task_map
                
                return gr.update(), updated_task_map
                
            except Exception as e:
                logger.error(f"Error updating chat results: {e}")
                return gr.update(), task_map
        
        # Custom chat function wrapper  
        def handle_chat(message, history, chat_id_state, session_id_state, selected_mode, cwd_value, append_prompt_value, task_map, active_task_id, active_session_id):
            if not message.strip():
                return history, "", gr.update(), gr.update(), gr.update(), chat_id_state, session_id_state, task_map, active_task_id, active_session_id
            
            # Add user message to history
            new_history = history + [{"role": "user", "content": message}]
            
            # Make chatbot visible and hide placeholder when first message is sent
            chatbot_update = gr.update(visible=True)
            placeholder_update = gr.update(visible=False)
            flexible_spacer_update = gr.update(visible=False)
            
            # Create new chat if needed
            current_chat_id = chat_id_state
            current_session_id = session_id_state
            
            if not current_chat_id:
                current_chat_id = chat_manager.create_chat()
            
            # Call the non-blocking chat function
            logger.info(f"🔄 Sending message with session_id: {current_session_id}")
            response_messages, updated_session_id, task_id = chat_function_sync(
                message, history, current_session_id, selected_mode, cwd_value, append_prompt_value
            )
            
            # Update session_id if we got a new one
            logger.info(f"📨 Got back session_id: {updated_session_id}, task_id: {task_id}")
            if updated_session_id and updated_session_id != current_session_id:
                logger.info(f"🔄 Session ID changed: {current_session_id} → {updated_session_id}")
                current_session_id = updated_session_id
            
            # Store task_id for this chat
            updated_task_map = task_map.copy() if task_map else {}
            if task_id:
                if current_chat_id not in updated_task_map:
                    updated_task_map[current_chat_id] = []
                updated_task_map[current_chat_id].append(task_id)
            
            # Build complete history: previous conversation + user message + response
            complete_history = new_history + response_messages
            
            # Save to database (generate title from first message if new chat)
            if len(new_history) == 1:  # First message
                title = message[:50] + "..." if len(message) > 50 else message
                chat_manager.update_chat(current_chat_id, complete_history, title)
            else:
                chat_manager.update_chat(current_chat_id, complete_history)
            
            # Always update the database with the real session_id when we have it
            if current_session_id:
                logger.info(f"💾 Saving session_id {current_session_id} to chat {current_chat_id}")
                chat_manager.update_session_id(current_chat_id, current_session_id)
            
            # Return immediately - no more blocking!
            return complete_history, "", chatbot_update, placeholder_update, flexible_spacer_update, current_chat_id, current_session_id, updated_task_map, task_id, current_session_id
        
        # Connect send button and textbox submit
        send_button.click(
            fn=handle_chat,
            inputs=[scout_textbox, chatbot, current_chat_id, current_session_id, current_mode, cwd_textbox, append_system_prompt_textbox, chat_task_map, current_active_task_id, current_active_session_id],
            outputs=[chatbot, scout_textbox, chatbot, placeholder_md, flexible_spacer, current_chat_id, current_session_id, chat_task_map, current_active_task_id, current_active_session_id],
        )
        
        scout_textbox.submit(
            fn=handle_chat,
            inputs=[scout_textbox, chatbot, current_chat_id, current_session_id, current_mode, cwd_textbox, append_system_prompt_textbox, chat_task_map, current_active_task_id, current_active_session_id],
            outputs=[chatbot, scout_textbox, chatbot, placeholder_md, flexible_spacer, current_chat_id, current_session_id, chat_task_map, current_active_task_id, current_active_session_id],
        )
        
        # Connect chat dropdown to load selected chat
        chat_dropdown.change(
            fn=load_selected_chat,
            inputs=[chat_dropdown, chat_task_map],
            outputs=[chatbot, chatbot, placeholder_md, flexible_spacer, current_session_id, current_chat_id, chat_task_map]
        )
        
        # Connect latest chats radio to load selected chat
        latest_chats_radio.change(
            fn=load_selected_chat,
            inputs=[latest_chats_radio, chat_task_map],
            outputs=[chatbot, chatbot, placeholder_md, flexible_spacer, current_session_id, current_chat_id, chat_task_map]
        )
        
        # Connect sidebar buttons
        new_chat_btn.click(
            fn=create_new_chat,
            inputs=[chat_task_map],
            outputs=[chat_dropdown, latest_chats_radio, chatbot, chatbot, placeholder_md, flexible_spacer, current_chat_id, current_session_id, chat_task_map],
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
            inputs=[current_chat_id, chat_task_map],
            outputs=[chat_dropdown, latest_chats_radio, chatbot, chatbot, placeholder_md, flexible_spacer, current_chat_id, current_session_id, chat_task_map]
        )
        
        # Connect refresh chat results button
        def refresh_current_chat(chat_id, task_map):
            """Refresh the current chat with completed task results."""
            if not chat_id:
                return gr.update(), task_map
            updated_chatbot, updated_task_map = update_chat_with_results(chat_id, task_map)
            return updated_chatbot, updated_task_map
        
        refresh_chat_btn.click(
            fn=refresh_current_chat,
            inputs=[current_chat_id, chat_task_map],
            outputs=[chatbot, chat_task_map]
        )
        
        # Connect directory search functionality
        cwd_textbox.change(
            fn=update_directory_and_cards,
            inputs=[cwd_textbox],
            outputs=[cwd_textbox, combined_info]
        )
        
        # Connect task management buttons
        refresh_tasks_btn.click(
            fn=update_task_radios,
            inputs=[],
            outputs=[completed_tasks_radio, ongoing_tasks_radio]
        )
        
        # Connect radio selection handlers
        completed_tasks_radio.change(
            fn=handle_completed_task_selection,
            inputs=[completed_tasks_radio],
            outputs=[completed_actions, selected_task_id]
        )
        
        ongoing_tasks_radio.change(
            fn=handle_ongoing_task_selection,
            inputs=[ongoing_tasks_radio],
            outputs=[ongoing_actions, selected_task_id]
        )
        
        # Connect action buttons to their functions
        completed_view_btn.click(
            fn=view_task_events,
            inputs=[selected_task_id],
            outputs=[event_header, events_display, event_stats]
        )
        
        ongoing_view_btn.click(
            fn=view_task_events,
            inputs=[selected_task_id], 
            outputs=[event_header, events_display, event_stats]
        )
        
        completed_delete_btn.click(
            fn=delete_task_action,
            inputs=[selected_task_id],
            outputs=[]  # Will refresh via auto-refresh
        )
        
        ongoing_delete_btn.click(
            fn=delete_task_action,
            inputs=[selected_task_id],
            outputs=[]  # Will refresh via auto-refresh  
        )
        
        ongoing_stop_btn.click(
            fn=stop_task_action,
            inputs=[selected_task_id],
            outputs=[]  # Will refresh via auto-refresh
        )
        
        # Set up auto-refresh timer (every 2 seconds when enabled)
        auto_refresh_timer = gr.Timer(value=2, active=True)
        auto_refresh_timer.tick(
            fn=lambda checkbox_value: update_task_radios() if checkbox_value else (gr.update(), gr.update()),
            inputs=[auto_refresh_checkbox],
            outputs=[completed_tasks_radio, ongoing_tasks_radio]
        )
        
        # Set up auto-refresh timer for events (every 3 seconds when enabled and task selected)
        events_refresh_timer = gr.Timer(value=3, active=True)
        events_refresh_timer.tick(
            fn=lambda events_checkbox, task_id: refresh_selected_task_events(task_id) if (events_checkbox and task_id) else (gr.update(), gr.update(), gr.update()),
            inputs=[auto_refresh_events, selected_task_id],
            outputs=[event_header, events_display, event_stats]
        )
        
        # Connect sidebar event handlers to share the same task selection and event display
        # When a task is selected in workspace, update sidebar as well
        selected_task_id.change(
            fn=lambda task_id: refresh_selected_task_events(task_id) if task_id else (gr.update(value="*Select a task to view its live events*", visible=True), gr.update(value="", visible=False), gr.update(value="", visible=False)),
            inputs=[selected_task_id],
            outputs=[sidebar_event_header, sidebar_events_display, sidebar_event_stats]
        )
        
        # Set up auto-refresh timer for sidebar events (same as workspace, using shared task_id)
        sidebar_events_refresh_timer = gr.Timer(value=3, active=True)
        sidebar_events_refresh_timer.tick(
            fn=lambda sidebar_events_checkbox, task_id: refresh_selected_task_events(task_id) if (sidebar_events_checkbox and task_id) else (gr.update(), gr.update(), gr.update()),
            inputs=[sidebar_auto_refresh_events, selected_task_id],
            outputs=[sidebar_event_header, sidebar_events_display, sidebar_event_stats]
        )
        
        # Connect current active session to both workspace and sidebar Live Tasks
        # When a new chat task is created, automatically show it in Live Tasks using session_id
        def update_live_tasks_from_active_session(session_id):
            """Update both workspace and sidebar Live Tasks when active session changes."""
            if session_id:
                # Get task events and update sidebar
                sidebar_updates = refresh_selected_task_events(session_id)
                return (session_id, *sidebar_updates)
            else:
                # Clear displays if no session
                return (
                    session_id,
                    gr.update(value="*Select a task to view its live events*", visible=True), 
                    gr.update(value="", visible=False), 
                    gr.update(value="", visible=False)
                )
        
        current_active_session_id.change(
            fn=update_live_tasks_from_active_session,
            inputs=[current_active_session_id],
            outputs=[selected_task_id, sidebar_event_header, sidebar_events_display, sidebar_event_stats]
        )
        
        # Set up auto-refresh timer for chat (every 3 seconds when enabled)
        chat_refresh_timer = gr.Timer(value=3, active=True)
        chat_refresh_timer.tick(
            fn=lambda checkbox_value, chat_id, task_map: refresh_current_chat(chat_id, task_map) if (checkbox_value and chat_id) else (gr.update(), task_map),
            inputs=[auto_refresh_chat_checkbox, current_chat_id, chat_task_map],
            outputs=[chatbot, chat_task_map]
        )
        
        # Connect mode radio to current_mode state and update textbox styling, placeholder, and markdown header
        mode_radio.change(
            fn=lambda mode: (mode, *handle_mode_change(mode)),
            inputs=[mode_radio],
            outputs=[current_mode, textbox_wrapper, scout_textbox, placeholder_md]
        )
        
        # Load chat list and info cards on startup
        def initialize_ui():
            chat_list = load_chat_list()
            combined_update = update_info_cards("")
            return chat_list, combined_update
        
        def initialize_ui_with_tasks():
            chat_list = load_chat_list()
            latest_chats = load_latest_chats_radio()
            combined_update = update_info_cards("")
            completed_update, ongoing_update = update_task_radios()
            return chat_list, latest_chats, combined_update, completed_update, ongoing_update
        
        demo.load(
            fn=initialize_ui_with_tasks,
            inputs=[],
            outputs=[chat_dropdown, latest_chats_radio, combined_info, completed_tasks_radio, ongoing_tasks_radio]
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
