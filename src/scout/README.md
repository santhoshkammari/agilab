## Optimizations
- [x] Plan/Action Modes
- [x] mcp websearch over builtin WebSearch
- [ ] Leverage Local/Gemini/Qwen cli's/Models
- [ ] Enforce MultiEdit for Max Model Efficiency

## Unlocking Model Capabilities


## WebSearch Comparison: MCP vs Claude Built-in
---
*Query: "find top5 rag papers, with 10 parallel search requests"*

| **Metric** | **MCP WebSearch** | **Claude Built-in WebSearch** | **Improvement** |
|------------|-------------------|-------------------------------|-----------------|
| **Duration** | 28,012 ms (28 sec) | 101,649 ms (102 sec) | **3.6x faster** |
| **API Duration** | 24,073 ms | 327,266 ms | **13.6x faster** |
| **Total Cost** | $0.095 USD | $1.515 USD | **94% cost reduction** |
| **Input Tokens** | 20,031 | 30,048 | **33% fewer tokens** |
| **Output Tokens** | 479 | 444 | Similar output |
| **Cache Creation Tokens** | 5,391 | 5,393 | Similar |
| **Cache Read Tokens** | 14,633 | 14,532 | Similar |
| **Web Search Requests** | 10 parallel searches | 10 parallel searches | Similar |
| **Total Turns** | 22 | 22 | Same complexity |

### Performance Summary:

**🚀 Speed Gains:**
- Overall execution: **3.6x faster**
- API processing: **13.6x faster**

**💰 Cost Efficiency:**
- **$1.42 savings per query** (94% reduction)
- **33% fewer input tokens** required

**📊 Resource Usage:**
- Similar output quality with significantly lower resource consumption
- More efficient search strategy vs multiple parallel requests


RoadMAP:
- Fix background task features and UI elements
- Enhanced session management and UI
- Introduced FastAPI backend for async tasks and workspace UI.
- Implemented glassmorphism UI for tabs and refined layout.
- Added tabs, settings sidebar, and context-aware info cards.
- Improved initial UI with styled placeholder and refined tools.
- Refactored Agent for simpler creation and added new examples.
- Implemented universal logger and improved chat UI/tool handling.
- Introduced Plan mode and advanced web fetching tools.


=========================
FASTAPI README
=========================
# Scout FastAPI Backend

This FastAPI backend provides background chat processing with task management for the Scout AI assistant.

## Quick Start

Start both servers:
```bash
python run_servers.py
```

- FastAPI: http://localhost:8000
- Gradio UI: http://localhost:7860

## API Endpoints

### Chat Processing

**Start Chat Task**
```bash
POST /chat
Content-Type: application/json

{
  "message": "Your question or request",
  "session_id": "optional-session-id",
  "mode": "Scout",
  "cwd": "/optional/working/directory",
  "append_system_prompt": "Additional instructions"
}
```

Response:
```json
{
  "task_id": "uuid-string",
  "status": "pending"
}
```

**Check Task Status**
```bash
GET /chat/{task_id}/status
```

Response:
```json
{
  "task_id": "uuid-string",
  "status": "completed|running|failed|cancelled",
  "events": [...],
  "error": "error message if failed"
}
```

**Stream Events (SSE)**
```bash
GET /chat/{task_id}/stream
```

Real-time Server-Sent Events stream showing task progress.

**Stop Running Task**
```bash
POST /chat/{task_id}/stop
```

Cancels a running task immediately.

### Task Management

**List All Tasks**
```bash
GET /tasks
```

**Delete Task**
```bash
DELETE /tasks/{task_id}
```

**Health Check**
```bash
GET /health
```

## Example Usage

```bash
# Start a task
TASK_ID=$(curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "List files in current directory"}' | jq -r .task_id)

# Check status
curl http://localhost:8000/chat/$TASK_ID/status

# Stream events in real-time
curl -N http://localhost:8000/chat/$TASK_ID/stream

# Stop if needed
curl -X POST http://localhost:8000/chat/$TASK_ID/stop

# Clean up
curl -X DELETE http://localhost:8000/tasks/$TASK_ID
```

## Features

- ✅ Background task processing
- ✅ Multiple concurrent tasks
- ✅ Real-time streaming (SSE)
- ✅ Task cancellation
- ✅ Session management
- ✅ Full claude-code integration
- ✅ Memory cleanup

## Architecture

- **api.py** - FastAPI server with task management
- **chat.py** - Modified Gradio interface using FastAPI endpoints
- **run_servers.py** - Launches both servers together

Tasks run independently in background threads, allowing the UI to remain responsive while processing multiple requests simultaneously.

