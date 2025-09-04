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