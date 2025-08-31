# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "sse-starlette",
#     "claude-code-sdk",
# ]

# ///
import asyncio
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator

from ssa import claude_code

app = FastAPI()

async def chat_stream_with_heartbeat(message: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """Stream chat responses with periodic heartbeat messages"""
    last_activity = time.time()
    heartbeat_count = 0
    
    try:
        claude_stream = claude_code(message, session_id=session_id)
        
        # Process events with heartbeat fallback
        async for event in claude_stream:
            last_activity = time.time()
            yield json.dumps({'type': 'claude_event', 'data': event})
        
        # Send completion
        yield json.dumps({'type': 'complete'})
        
    except Exception as e:
        yield json.dumps({'type': 'error', 'message': str(e)})

@app.get("/")
async def get_chat_ui():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Scout - AI Code Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Inter, ui-sans-serif, sans-serif; 
            background: #FFFFFF; 
            color: #1F2937; 
            overflow-x: hidden;
        }
        
        .app-container { display: flex; height: 100vh; }
        
        /* Sidebar styling matching original Scout */
        .sidebar {
            width: 280px;
            background: #FFFFFF;
            border-right: 1px solid #F2F4F6;
            padding: 20px;
            overflow-y: auto;
        }
        
        .sidebar h2 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            color: #1F2937;
        }
        
        .new-chat-btn {
            width: 100%;
            background: #57BAFF;
            color: white;
            border: none;
            padding: 12px 16px;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            margin-bottom: 16px;
            transition: all 0.15s ease;
        }
        
        .new-chat-btn:hover {
            background: #4AABF0;
            transform: translateY(-1px);
        }
        
        .chat-item {
            padding: 10px 12px;
            border-radius: 6px;
            margin-bottom: 4px;
            cursor: pointer;
            font-size: 14px;
            color: #6B7280;
            transition: all 0.15s ease;
        }
        
        .chat-item:hover {
            background: #EAF6FF;
            color: #57BAFF;
        }
        
        .chat-item.active {
            background: #EAF6FF;
            color: #57BAFF;
            font-weight: 500;
        }
        
        /* Main chat area */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #FFFFFF;
        }
        
        .chat-header {
            padding: 20px 24px;
            border-bottom: 1px solid #F2F4F6;
            text-align: center;
        }
        
        .chat-header h1 {
            font-size: 24px;
            font-weight: 600;
            color: #1F2937;
        }
        
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .message {
            display: flex;
            flex-direction: column;
            max-width: 80%;
        }
        
        .user-message {
            align-self: flex-end;
        }
        
        .assistant-message {
            align-self: flex-start;
        }
        
        .message-content {
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        
        .user-message .message-content {
            background: #57BAFF;
            color: white;
        }
        
        .assistant-message .message-content {
            background: transparent;
            color: #1F2937;
            border: none;
        }
        
        .thinking .message-content {
            background: #EAF6FF;
            color: #57BAFF;
            font-style: italic;
        }
        
        .error .message-content {
            background: #FEE2E2;
            color: #DC2626;
        }
        
        /* Scout-style input area */
        .input-container {
            padding: 24px;
            border-top: 1px solid #F2F4F6;
        }
        
        .scout-textbox-wrapper {
            border: 1px solid rgba(0, 0, 0, 0.06);
            border-radius: 18px;
            padding: 14px 18px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            box-shadow: 
                0 1px 3px rgba(0, 0, 0, 0.05),
                0 4px 12px rgba(0, 0, 0, 0.02),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
            margin: 0;
            overflow: hidden;
        }
        
        .scout-main-input {
            width: 100%;
            border: none;
            background: transparent;
            font-size: 16px;
            line-height: 1.5;
            resize: none;
            outline: none;
            margin-bottom: 6px;
            font-family: inherit;
        }
        
        .scout-button-row {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        
        .scout-context-btn {
            background: rgba(0, 122, 255, 0.08);
            border: 1px solid rgba(0, 122, 255, 0.12);
            color: #007AFF;
            font-size: 14px;
            font-weight: 500;
            padding: 8px 14px;
            border-radius: 12px;
            transition: all 0.15s ease;
            cursor: pointer;
        }
        
        .scout-context-btn:hover {
            background: rgba(0, 122, 255, 0.12);
            border-color: rgba(0, 122, 255, 0.18);
            transform: translateY(-0.5px);
        }
        
        .scout-send-btn {
            background: #007AFF;
            border: none;
            color: white;
            font-size: 18px;
            font-weight: 600;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s ease;
            box-shadow: 
                0 1px 3px rgba(0, 122, 255, 0.3),
                0 2px 6px rgba(0, 122, 255, 0.15);
        }
        
        .scout-send-btn:hover {
            background: #0056CC;
            transform: scale(1.05);
            box-shadow: 
                0 2px 6px rgba(0, 122, 255, 0.4),
                0 4px 12px rgba(0, 122, 255, 0.2);
        }
        
        .scout-send-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .status {
            text-align: center;
            font-size: 12px;
            color: #6B7280;
            margin: 8px 0;
        }
        
        .heartbeat { color: #10B981; }
        .processing { color: #F59E0B; }
        
        /* Scrollbar styling */
        .chat-history::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-history::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .chat-history::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        .chat-history::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <h2>🔍 Scout Chats</h2>
            <button class="new-chat-btn" onclick="createNewChat()">➕ New Chat</button>
            <div id="chat-list">
                <div class="chat-item active">New Chat (2025-08-31 16:19)</div>
            </div>
        </div>
        
        <!-- Main content -->
        <div class="main-content">
            <div class="chat-header">
                <h1>🔍 Scout - AI Code Assistant</h1>
            </div>
            
            <div id="chat-history" class="chat-history">
                <div class="message assistant-message">
                    <div class="message-content">
                        Hi! How can I help you with your code today?
                    </div>
                </div>
            </div>
            
            <div class="status" id="status">Ready</div>
            
            <div class="input-container">
                <div class="scout-textbox-wrapper">
                    <textarea 
                        id="message-input" 
                        class="scout-main-input" 
                        placeholder="Create a website based on my vibes"
                        rows="1"
                    ></textarea>
                    <div class="scout-button-row">
                        <button class="scout-context-btn" onclick="addContext()">@ Add context</button>
                        <div style="flex: 1;"></div>
                        <button id="send-btn" class="scout-send-btn" onclick="sendMessage()">↑</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentSessionId = null;
        let isProcessing = false;
        
        function addMessage(role, content, className = '', id = '') {
            const chatHistory = document.getElementById('chat-history');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}-message ${className}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = content;
            if (id) contentDiv.id = id;
            
            messageDiv.appendChild(contentDiv);
            chatHistory.appendChild(messageDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
            return contentDiv;
        }
        
        function updateStatus(text, className = '') {
            const statusEl = document.getElementById('status');
            statusEl.textContent = text;
            statusEl.className = `status ${className}`;
        }
        
        function setProcessing(processing) {
            isProcessing = processing;
            const sendBtn = document.getElementById('send-btn');
            const input = document.getElementById('message-input');
            sendBtn.disabled = processing;
            input.disabled = processing;
        }
        
        function createNewChat() {
            // Reset session
            currentSessionId = null;
            // Clear chat history except initial message
            const chatHistory = document.getElementById('chat-history');
            chatHistory.innerHTML = `
                <div class="message assistant-message">
                    <div class="message-content">
                        Hi! How can I help you with your code today?
                    </div>
                </div>
            `;
            updateStatus('Ready');
            document.getElementById('message-input').focus();
        }
        
        function addContext() {
            // Placeholder for context functionality
            updateStatus('Context feature coming soon...');
        }
        
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            if (!message || isProcessing) return;
            
            setProcessing(true);
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Add thinking indicator
            const thinkingContent = addMessage('assistant', '🤔 Thinking...', 'thinking', 'thinking-indicator');
            
            updateStatus('Connecting...', 'processing');
            
            try {
                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, session_id: currentSessionId })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMessage = '';
                let heartbeatCount = 0;
                let assistantElement = null;
                let buffer = '';  // Buffer for incomplete chunks
                
                updateStatus('Receiving response...', 'processing');
                
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    // Add to buffer and process complete lines
                    buffer += decoder.decode(value);
                    const lines = buffer.split('\\n');
                    
                    // Keep last line in buffer (might be incomplete)
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ') && line.trim().length > 6) {
                            try {
                                const jsonStr = line.slice(6).trim();
                                if (!jsonStr) continue;
                                const data = JSON.parse(jsonStr);
                                
                                if (data.type === 'claude_event') {
                                    // Remove thinking indicator when we get real content
                                    const thinkingEl = document.getElementById('thinking-indicator');
                                    if (thinkingEl) {
                                        thinkingEl.parentElement.remove();
                                    }
                                    
                                    const event = data.data;
                                    const eventType = event.type;
                                    
                                    if (eventType === 'AssistantMessage') {
                                        const content = event.content?.[0]?.text;
                                        if (content) {
                                            assistantMessage += content;
                                            
                                            if (!assistantElement) {
                                                assistantElement = addMessage('assistant', assistantMessage, '', 'streaming-response');
                                            } else {
                                                assistantElement.innerHTML = assistantMessage;
                                            }
                                        }
                                    } else if (eventType === 'ResultMessage') {
                                        const finalResult = event.result;
                                        if (finalResult?.trim()) {
                                            if (!assistantElement) {
                                                assistantElement = addMessage('assistant', finalResult);
                                            } else {
                                                assistantElement.innerHTML = finalResult;
                                                assistantElement.id = '';
                                            }
                                        }
                                    }
                                    
                                } else if (data.type === 'heartbeat') {
                                    heartbeatCount++;
                                    updateStatus(`Processing... (${heartbeatCount} heartbeats)`, 'heartbeat');
                                    
                                } else if (data.type === 'complete') {
                                    updateStatus('Ready');
                                    if (assistantElement) assistantElement.id = '';
                                    setProcessing(false);
                                    return;
                                    
                                } else if (data.type === 'error') {
                                    const thinkingEl = document.getElementById('thinking-indicator');
                                    if (thinkingEl) thinkingEl.parentElement.remove();
                                    addMessage('assistant', `⚠️ ${data.message}`, 'error');
                                    updateStatus('Error');
                                    setProcessing(false);
                                    return;
                                }
                                
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }
                
            } catch (error) {
                console.error('Stream error:', error);
                const thinkingEl = document.getElementById('thinking-indicator');
                if (thinkingEl) thinkingEl.parentElement.remove();
                addMessage('assistant', `⚠️ Connection error: ${error.message}`, 'error');
                updateStatus('Connection failed');
            } finally {
                setProcessing(false);
            }
        }
        
        // Auto-resize textarea and handle Enter key
        const messageInput = document.getElementById('message-input');
        
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
        
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey && !isProcessing) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Focus input on load
        messageInput.focus();
    </script>
</body>
</html>
    """)

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    """Stream chat responses with heartbeat to prevent timeout"""
    data = await request.json()
    message = data.get('message', '')
    session_id = data.get('session_id')
    
    if not message.strip():
        return {"error": "Empty message"}
    
    return EventSourceResponse(
        chat_stream_with_heartbeat(message, session_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)